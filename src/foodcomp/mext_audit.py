"""Evidence-preserving MEXT inspection. This module never writes ML partitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from .util import sha256_file, write_csv, write_json


MEXT_URL = "https://www.mext.go.jp/a_menu/syokuhinseibun/mext_00001.html"
FAQ_URL = "https://fooddb.mext.go.jp/help.html"


@dataclass(frozen=True)
class Table:
    prefix: str
    name: str
    tag_row: int
    basis: str


TABLES = [
    Table("01", "main", 11, "100g_edible_food"),
    Table("03", "amino_acids_mass", 4, "100g_edible_food"),
    Table("04", "amino_acids_per_reference_nitrogen", 4, "1g_reference_nitrogen"),
    Table("05", "amino_acids_per_residue_protein", 4, "1g_amino_acid_residue_protein"),
    Table("06", "amino_acids_per_nitrogen_protein", 4, "1g_nitrogen_based_protein"),
    Table("08", "fatty_acids_mass", 4, "100g_edible_food"),
    Table("09", "fatty_acids_relative", 4, "100g_total_fatty_acids"),
    Table("10", "fatty_acids_per_lipid", 4, "1g_total_lipid"),
    Table("12", "carbohydrates", 4, "100g_edible_food"),
    Table("13", "dietary_fibre", 7, "100g_edible_food"),
    Table("14", "organic_acids", 4, "100g_edible_food"),
]
EQUIVALENTS = {"FATNLEA", "CHOAVLM", "CARTBEQ", "VITA_RAE", "NE", "NACL_EQ"}
DEFINITION_NOTES = {
    "NA": "Sodium; literal NA is a component identifier, not a missing-data marker.",
    "PROT-": "Nitrogen-derived protein; retain nitrogen factor and analytical definition.",
    "PROTCAA": "Protein calculated from amino acid residues; distinct from nitrogen-derived protein.",
    "FAT-": "Extractable lipid; retain assay-method definition, not identical to fatty acid sum.",
    "FATNLEA": "Triacylglycerol equivalents calculated from fatty acids; not direct total lipid mass.",
    "CHOAVLM": "Available carbohydrate as monosaccharide equivalents; exclude from chemical mass modality.",
    "CHOAVL": "Available carbohydrate mass sum; distinct from monosaccharide equivalents and by-difference carbohydrate.",
    "CHOAVLDF-": "Available carbohydrate by difference; calculated mass, not independent chemical measurement.",
    "CHOCDF-": "Carbohydrate by difference; calculated mass and algebraically coupled to proximate components.",
    "CYS": "Cystine (not free cysteine); preserve the hydrolysed amino-acid analytical definition.",
    "VITK": "Usually K1 + menaquinone-4; specified fermented foods include MK-7 converted by 444.7/649.0. Hold pending food-specific definition.",
    "VITD": "Normally D2 + D3; food notes identify nine egg/milk records containing active metabolites. Those rows are held separately.",
    "FIB-": "Main-table dietary fibre combines method-dependent values; do not merge with method-specific fibre automatically.",
    "FIBSOL": "Soluble dietary fibre, modified Prosky method.",
    "FIBINS": "Insoluble dietary fibre, modified Prosky method.",
    "FIBTG": "Total dietary fibre, modified Prosky method.",
    "FIB-SDFS": "Low-molecular-weight soluble dietary fibre, AOAC 2011.25.",
    "FIB-SDFP": "High-molecular-weight soluble dietary fibre, AOAC 2011.25.",
    "FIB-IDF": "Insoluble dietary fibre, AOAC 2011.25.",
    "FIB-TDF": "Total dietary fibre, AOAC 2011.25.",
    "STARES": "Resistant starch in the AOAC 2011.25 fibre table; distinct from total starch.",
    "FAUN": "Unidentified substances in the fatty acid assay; insufficient chemical identity for a target.",
    "NACL_EQ": "Salt equivalent calculated from sodium; not measured sodium chloride.",
    "NE": "Niacin equivalents; not measured niacin mass.",
}


def text(value) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def parse_mext_value(raw, prescribed_method_annotation_verified: bool = False) -> dict:
    token = text(raw)
    parse_token = token[:-1] if token.endswith("\u2020") and prescribed_method_annotation_verified else token
    estimated = parse_token.startswith("(") and parse_token.endswith(")")
    inner = parse_token[1:-1].strip() if estimated else parse_token
    number = np.nan
    if inner == "":
        status = "missing_blank"
    elif inner in {"-", "\u2014", "\u2013", "\u2212"}:
        status = "unmeasured_dash"
    elif inner.casefold() == "tr":
        status = "estimated_trace" if estimated else "trace_interval"
    elif inner == "*":
        status = "footnote_only_value_unresolved"
    elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", inner):
        number = float(inner)
        if number < 0:
            status = "invalid_negative"
        elif number == 0:
            status = "estimated_zero" if estimated else "reported_zero_censored_or_nondetect"
        else:
            status = "estimated_positive" if estimated else "reported_positive_origin_unresolved"
    else:
        status = "unparsed_requires_review"
    return {"raw_token": token, "value_status": status, "numeric_value": number,
            "parenthesized_estimate": estimated, "is_exact_observed_zero": False,
            "prescribed_method_annotation_verified": prescribed_method_annotation_verified and token.endswith("\u2020")}


def main_unit(column: int) -> str:
    if column == 4:
        return "%"
    if column in {5, 6}:
        return "kJ" if column == 5 else "kcal"
    if column in {7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 21, 22, 59, 60}:
        return "g/100 g"
    if column == 11 or 23 <= column <= 31 or 44 <= column <= 47 or 49 <= column <= 53 or column in {56, 58}:
        return "mg/100 g"
    if 33 <= column <= 43 or column in {48, 54, 55, 57}:
        return "ug/100 g"
    raise ValueError(f"Unreviewed MEXT main-table unit at column {column}")


def modality(tag: str, basis: str) -> str:
    if basis != "100g_edible_food":
        return "relative_denominator_excluded"
    if tag in {"ENERC", "ENERC_KCAL"}:
        return "energy_excluded"
    if tag == "REFUSE":
        return "refuse_fraction_metadata"
    if tag in EQUIVALENTS:
        return "equivalent_expression_excluded"
    if tag == "VITK":
        return "mixed_mass_equivalent_definition_hold"
    if tag == "FAUN":
        return "unidentified_substances_excluded"
    return "mass_per_100g_edible_food"


def read_table(path: Path, table: Table):
    # pandas' default NA vocabulary would erase the authoritative sodium tag NA.
    frame = pd.read_excel(path, sheet_name="表全体", header=None, keep_default_na=False)
    ids = frame[1].map(text)
    data = frame[ids.str.fullmatch(r"\d{5}")]
    if data[1].astype(str).duplicated().any():
        raise ValueError(f"Repeated food IDs inside the full table: {path}")
    columns = [(index, text(tag)) for index, tag in enumerate(frame.iloc[table.tag_row])
               if index > 3 and re.fullmatch(r"[A-Z][A-Z0-9_\-]*", text(tag))]
    if not columns or data.empty:
        raise ValueError(f"MEXT table schema not recognized: {path}")
    component_rows, measurement_rows, food_rows = [], [], []
    for index, tag in columns:
        unit = main_unit(index) if table.name == "main" else text(frame.iat[table.tag_row + 1, index])
        mod = modality(tag, table.basis)
        mass_factor = {"g/100 g": 1.0, "mg/100 g": 1e-3, "ug/100 g": 1e-6}.get(unit, np.nan)
        if mod == "mass_per_100g_edible_food" and not np.isfinite(mass_factor):
            raise ValueError(f"Unreviewed mass unit: {table.name} {tag} {unit}")
        if table.name == "main":
            names = [text(frame.iat[row, index]) for row in range(1, 10) if text(frame.iat[row, index])]
            original_name = " | ".join(names)
        else:
            original_name = text(frame.iat[table.tag_row - 1, index])
            if table.name == "dietary_fibre":
                original_name = " | ".join(text(frame.iat[row, index]) for row in range(3, 7) if text(frame.iat[row, index]))
        key = f"mext:{tag}:{table.basis}"
        component_rows.append({"component_key": key, "table": table.name, "tag": tag,
                               "original_name_ja": original_name, "raw_unit": unit, "basis": table.basis,
                               "modality": mod, "conversion_factor": mass_factor,
                               "definition_review": DEFINITION_NOTES.get(tag, "Official MEXT identifier retained; cross-database identity and analytical-method mapping not approved."),
                               "definition_evidence_url": MEXT_URL,
                               "unit_evidence": f"{path.name}, sheet full table, header column {index + 1}",
                               "cross_database_merge_approved": False})
        for row_index, row in data.iterrows():
            note = text(row.iloc[-1])
            prescribed = text(row[1]) == "03032" and "\u2020は規定法による測定値" in note
            parsed = parse_mext_value(row[index], prescribed_method_annotation_verified=prescribed)
            row_modality = mod
            definition_hold = ""
            if tag == "VITD" and "ビタミンD活性代謝物を含む" in note:
                row_modality = "food_specific_vitamin_D_active_metabolite_expression_hold"
                definition_hold = "Food note includes active metabolites; do not treat main-table value as D2+D3 mass."
            if tag == "FE" and "鉄: Trであるが" in note:
                parsed["value_status"] = "trace_with_numeric_display_not_exact_point_label"
                definition_hold = "Iron is explicitly trace despite a printed numeric convenience value."
            convertible = row_modality == "mass_per_100g_edible_food" and np.isfinite(parsed["numeric_value"])
            value = parsed["numeric_value"] * mass_factor if convertible else np.nan
            candidate = parsed["value_status"] == "reported_positive_origin_unresolved" and convertible and 0 < value <= 100
            measurement_rows.append({"table": table.name, "food_id": text(row[1]), "component_key": key,
                                     "tag": tag, "source_file": path.name, "excel_row": row_index + 1,
                                     "excel_column": index + 1, "raw_unit": unit, "basis": table.basis,
                                     "modality": row_modality, **parsed, "converted_g_per_100g": value,
                                     "food_specific_definition_hold": definition_hold,
                                     "positive_mass_candidate": candidate,
                                     "invalid_mass_fraction": bool(convertible and (value < 0 or value > 100)),
                                     "strict_validation_approved": False})
    for row_index, row in data.iterrows():
        note = text(row.iloc[-1])
        food_rows.append({"table": table.name, "food_id": text(row[1]), "food_group_id": text(row[0]).zfill(2),
                          "original_food_name_ja": text(row[3]), "source_note_ja": note,
                          "source_file": path.name, "excel_row": row_index + 1,
                          "foreign_source_note": bool(re.search("米国|アメリカ|豪州|オーストラリア|英国|イギリス|デンマーク|カナダ|外国", note)),
                          "recipe_or_estimation_note": bool(re.search("推計|推定|原材料|配合割合|計算", note)),
                          "referenced_food_ids_candidate": ";".join(sorted(set(re.findall(r"(?<!\d)\d{5}(?!\d)", note))))})
    return pd.DataFrame(component_rows), pd.DataFrame(measurement_rows), pd.DataFrame(food_rows)


def audit_mext(root: Path, output: Path) -> dict:
    raw = root / "data/raw/expansion_2026_09_06/mext"
    output.mkdir(parents=True, exist_ok=True)
    registries, measurements, food_tables, inventory = [], [], [], []
    for table in TABLES:
        paths = list(raw.glob(table.prefix + "_*.xlsx"))
        if len(paths) != 1:
            raise ValueError(f"Expected one {table.name} workbook, got {paths}")
        path = paths[0]
        components, values, foods = read_table(path, table)
        registries.append(components)
        measurements.append(values)
        food_tables.append(foods)
        sheets = pd.ExcelFile(path).sheet_names
        inventory.append({"table": table.name, "file": path.name, "sha256": sha256_file(path),
                          "basis": table.basis, "foods": len(foods), "columns": len(components),
                          "cells": len(values), "sheet_read": "表全体", "group_sheets_not_counted_again": len(sheets) - 1})
        print(f"MEXT {table.name}: {len(foods)} foods, {len(components)} columns", flush=True)
    components = pd.concat(registries, ignore_index=True)
    values = pd.concat(measurements, ignore_index=True)
    foods = pd.concat(food_tables, ignore_index=True)
    group = values.groupby(["food_id", "component_key"], sort=False)
    duplicates = group.agg(table_count=("table", "size"), raw_token_variants=("raw_token", "nunique"),
                           tables=("table", lambda x: ";".join(x)), raw_tokens=("raw_token", lambda x: ";".join(dict.fromkeys(x))))
    duplicates = duplicates[duplicates.table_count.gt(1)].reset_index()
    semantics = group.agg(numeric_variants=("numeric_value", "nunique"),
                           value_status_variants=("value_status", "nunique")).reset_index()
    duplicates = duplicates.merge(semantics, on=["food_id", "component_key"], validate="one_to_one")
    duplicates["review_type"] = np.select(
        [duplicates.numeric_variants.gt(1), duplicates.value_status_variants.gt(1), duplicates.raw_token_variants.gt(1)],
        ["different_numeric_values", "different_origin_or_missingness_notation", "formatting_only_difference"],
        default="identical_repeated_record",
    )
    conflict_keys = duplicates[duplicates.review_type.isin(["different_numeric_values", "different_origin_or_missingness_notation"])][["food_id", "component_key"]]
    values = values.merge(conflict_keys.assign(cross_table_raw_token_conflict=True), on=["food_id", "component_key"], how="left", validate="many_to_one")
    values["cross_table_raw_token_conflict"] = values.cross_table_raw_token_conflict.eq(True)
    main_ids = set(foods.loc[foods.table.eq("main"), "food_id"])
    orphan_foods = foods[~foods.food_id.isin(main_ids)].copy()
    orphan_foods["decision"] = "hold_subtable_only_identifier_until_official_identity_resolution"
    values["food_id_in_main_table"] = values.food_id.isin(main_ids)
    eligible = values[values.positive_mass_candidate & ~values.cross_table_raw_token_conflict & values.food_id_in_main_table]
    unique = eligible.drop_duplicates(["food_id", "component_key"])
    stats = values.groupby(["table", "component_key"], sort=False).agg(
        foods=("food_id", "nunique"), numeric_tokens=("numeric_value", "count"),
        positive_mass_candidates=("positive_mass_candidate", "sum"),
        parenthesized_tokens=("parenthesized_estimate", "sum"),
        raw_min=("numeric_value", "min"), raw_max=("numeric_value", "max"),
        mass_min=("converted_g_per_100g", "min"), mass_max=("converted_g_per_100g", "max"),
    ).reset_index()
    components = components.merge(stats, on=["table", "component_key"], validate="one_to_one")
    components["positive_mass_coverage_within_reported_table"] = components.positive_mass_candidates / components.foods
    candidate_stats = unique.groupby("component_key").agg(unique_positive_foods=("food_id", "nunique"),
                                                           candidate_min_g_per_100g=("converted_g_per_100g", "min"),
                                                           candidate_max_g_per_100g=("converted_g_per_100g", "max")).reset_index()
    main_foods = foods[foods.table.eq("main")].copy()
    if not components[components.table.eq("main")].tag.eq("NA").any():
        raise AssertionError("Sodium NA tag was lost")
    english = root / "data/raw/validation_mext_evidence_2026_09_06/mext/english_2015_main.xlsx"
    if english.exists():
        write_json({"status": "historical_official_alias_source_only_not_2023_identity_verification",
                    "file": str(english.relative_to(root)), "sha256": sha256_file(english),
                    "sheets": pd.ExcelFile(english).sheet_names,
                    "rule": "Never import 2015 values or infer unchanged food facets solely from shared food IDs."}, output / "english_name_review_status.json")
    errata = next(raw.glob("16_*.xlsx"))
    errata_rows = []
    for sheet in pd.ExcelFile(errata).sheet_names:
        frame = pd.read_excel(errata, sheet_name=sheet, header=None, keep_default_na=False)
        for index, row in frame.iterrows():
            nonempty = [text(cell) for cell in row if text(cell)]
            if nonempty:
                errata_rows.append({"sheet": sheet, "excel_row": index + 1, "cells": " | ".join(nonempty)})
    write_csv(pd.DataFrame(errata_rows), output / "mext_errata_transcription.csv")
    write_csv(pd.DataFrame(inventory), output / "mext_table_inventory.csv")
    write_csv(components, output / "mext_component_evidence_registry.csv")
    write_csv(foods, output / "mext_food_provenance.csv.gz")
    write_csv(orphan_foods, output / "mext_subtable_only_food_ids.csv")
    write_csv(values, output / "mext_raw_cell_semantics.csv.gz")
    write_csv(duplicates, output / "mext_repeated_cells_and_conflicts.csv.gz")
    write_csv(unique, output / "mext_positive_mass_candidates_NOT_TRAINING.csv.gz")
    write_csv(candidate_stats, output / "mext_unique_candidate_axis_support.csv")
    write_csv(values.groupby(["table", "modality", "value_status"]).size().reset_index(name="cells"), output / "mext_value_status_counts.csv")
    write_csv(values[values.value_status.isin(["unparsed_requires_review", "invalid_negative"]) | values.invalid_mass_fraction], output / "mext_unparsed_or_invalid_cells.csv")
    write_csv(values[values.food_specific_definition_hold.ne("") | values.value_status.eq("footnote_only_value_unresolved")], output / "mext_food_specific_definition_exceptions.csv")
    main_mass = values[values.table.eq("main") & values.modality.eq("mass_per_100g_edible_food")]
    summary = {
        "release_reviewed": "MEXT 8th revised edition supplement 2023; corrected workbooks 2026-03-27",
        "official_download_url": MEXT_URL, "numeric_semantics_url": FAQ_URL,
        "foods_in_main_table": len(main_foods), "main_component_columns": int(components.table.eq("main").sum()),
        "subtable_only_food_ids_held": sorted(set(orphan_foods.food_id)),
        "main_mass_component_columns": components.loc[components.table.eq("main") & components.modality.eq("mass_per_100g_edible_food"), "tag"].nunique(),
        "main_mass_cells": len(main_mass), "main_mass_status_counts": main_mass.value_status.value_counts().to_dict(),
        "table_and_basis_specific_columns": len(components), "all_table_cells": len(values),
        "relative_denominator_cells_excluded": int(values.modality.eq("relative_denominator_excluded").sum()),
        "duplicate_food_component_groups": len(duplicates), "duplicate_extra_cells": int((duplicates.table_count - 1).sum()),
        "cross_table_raw_notation_difference_groups": int(duplicates.raw_token_variants.gt(1).sum()),
        "semantic_conflict_groups_quarantined": len(conflict_keys),
        "repeated_cell_review_types": duplicates.review_type.value_counts().to_dict(),
        "unique_positive_mass_candidate_cells": len(unique), "candidate_foods": unique.food_id.nunique(),
        "candidate_axes_with_any_positive_value": unique.component_key.nunique(),
        "candidate_axes_with_at_least_50_positive_foods": int(candidate_stats.unique_positive_foods.ge(50).sum()),
        "candidate_axes_with_at_least_100_positive_foods": int(candidate_stats.unique_positive_foods.ge(100).sum()),
        "foods_with_explicit_foreign_source_notes": foods.loc[foods.foreign_source_note, "food_id"].nunique(),
        "foods_with_recipe_or_estimation_notes": foods.loc[foods.recipe_or_estimation_note, "food_id"].nunique(),
        "unparsed_tokens": values.loc[values.value_status.eq("unparsed_requires_review"), "raw_token"].value_counts().to_dict(),
        "invalid_mass_cells": int(values.invalid_mass_fraction.sum()),
        "food_specific_vitamin_D_active_metabolite_cells_held": int(values.modality.eq("food_specific_vitamin_D_active_metabolite_expression_hold").sum()),
        "numeric_display_but_explicit_trace_cells_held": int(values.value_status.eq("trace_with_numeric_display_not_exact_point_label").sum()),
        "prescribed_method_annotation_cells_verified": int(values.prescribed_method_annotation_verified.sum()),
        "strict_validation_labels_approved": 0, "new_training_rows_committed": 0,
        "zero_rule": "Reported 0 is rounding/censoring or nondetection, not certified absence. Parenthesized 0 is inferred absence. Neither is an ordinary exact-zero label.",
        "candidate_limitation": "Unparenthesized positive mass values pass syntactic/unit checks only. They are not certified independent assays, verified food concepts, or a model-ready dataset.",
        "name_review": "Japanese official food names preserved. Full name/facet and cross-source lineage review remains open.",
        "errata_review": "Corrected workbooks used; full errata transcription supplied. Every historical correction has not been independently adjudicated.",
    }
    write_json(summary, output / "MEXT_AUDIT.json")
    return summary

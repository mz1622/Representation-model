#!/usr/bin/env python3
"""Quantify BLS and MEXT candidates without claiming cross-source identity review."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.schema import parse_value
from foodcomp.util import normalize_text, read_component_csv, sha256_file, write_csv, write_json

BASE_FOLDER = "BLS_4_0_2025_DE"


def audit_bls(root, output, current_components, current_foods):
    raw = root / "data/raw/expansion_evidence_2026_09_07"
    manifest = json.loads((raw / "download_manifest.json").read_text())
    for item in manifest["files"]:
        if sha256_file(raw / item["path"]) != item["sha256"]:
            raise ValueError("An expansion raw snapshot changed")
    with ZipFile(raw / "bls/archive_00.zip") as archive:
        components = pd.read_excel(BytesIO(archive.read(f"{BASE_FOLDER}/BLS_4_0_Components_DE_EN.xlsx")), keep_default_na=False)
        data = pd.read_excel(BytesIO(archive.read(f"{BASE_FOLDER}/BLS_4_0_Daten_2025_DE.xlsx")), keep_default_na=False)
    components = components[components.iloc[:, 1].ne("")].copy()
    components.columns = ["index", "source_code", "original_name_de", "official_name_en", "unit", "source_group_de", "source_group_en", "formula", "formula_application"]
    if len(data) != 7140 or len(components) != 138 or data["BLS Code"].duplicated().any() or components.source_code.duplicated().any():
        raise ValueError("Unexpected BLS edition dimensions or duplicate source keys")
    foods = data.iloc[:, :3].copy()
    foods.columns = ["source_food_id", "original_name_de", "official_name_en"]
    foods["name_authority"] = "BLS 4.0 official bilingual workbook; not project-generated translation"
    foods["source_row_1based"] = np.arange(len(foods)) + 2
    components["unit_factor_to_g"] = components.unit.map({"g": 1.0, "mg": 1e-3, "\u00b5g": 1e-6})
    components["expression_status"] = "mass_expression_candidate"
    components.loc[components.source_code.isin(["ENERCJ", "ENERCC"]), "expression_status"] = "energy_excluded"
    components.loc[components.source_code.isin(["VITA", "VITAA", "NIAEQ", "FOL", "NACL"]), "expression_status"] = "equivalent_expression_excluded"
    components.loc[components.source_code.eq("VITE"), "expression_status"] = "duplicate_alpha_tocopherol_alias_hold"
    components.loc[components.source_code.eq("FAX"), "expression_status"] = "unspecified_other_fatty_acids_hold"
    components["identity_review_status"] = "official_source_local_identity_only_cross_database_review_pending"
    components["definition_cautions"] = ""
    cautions = {
        "PROT625": "N x 6.25, not interchangeable with food-specific nitrogen factors or amino-acid residue mass",
        "FOL": "FOLFD + 1.7*FOLAC is an activity equivalent; the same-looking FOL code does not prove chemical equivalence",
        "VITE": "Official formula VITE=TOCPHA; mass alpha-tocopherol here, not automatic mapping to INFOODS vitamin-E activity",
        "ID": "Official English label is iodide; verify elemental reporting definition before merging with iodine",
        "CYSTE": "Cysteine; not automatically interchangeable with MEXT cystine",
        "ASP": "Aspartic acid including asparagine; not free aspartic acid",
        "GLU": "Glutamic acid including glutamine; not free glutamic acid",
        "STARCH": "Starch, glycogen, dextrins are included in this expression",
    }
    for code, caution in cautions.items():
        components.loc[components.source_code.eq(code), "definition_cautions"] = caution
    parts = []
    for component in components.itertuples(index=False):
        code = component.source_code
        columns = [name for name in data if name.startswith(code + " ") and "[" in name]
        if len(columns) != 1:
            raise ValueError(f"Expected one BLS value column: {code}: {columns}")
        raw_values = data[columns[0]]
        numeric = pd.to_numeric(raw_values, errors="coerce")
        statuses = pd.Series(np.where(numeric.notna(), np.where(numeric.eq(0), "reported_zero_requires_lod_review", "reported_positive"), "unparsed"), index=data.index)
        for value in raw_values[numeric.isna()].unique():
            parsed = parse_value(value)
            statuses.loc[raw_values.eq(value)] = parsed["value_status"]
        origin = data[code + " Datenherkunft"]
        refs = data[code + " Referenz"]
        physical = numeric * component.unit_factor_to_g
        mass = component.expression_status == "mass_expression_candidate"
        candidate = mass & origin.eq("Analyse") & statuses.eq("reported_positive") & physical.between(0, 100, inclusive="right")
        part = pd.DataFrame({"source_food_id": foods.source_food_id, "source_component_id": code,
                             "source_row_1based": foods.source_row_1based, "source_value_column": columns[0],
                             "raw_value": raw_values, "source_origin": origin, "source_reference": refs,
                             "raw_unit": component.unit, "raw_basis": "100 g edible food",
                             "value_status": statuses, "numeric_value": numeric,
                             "expression_status": component.expression_status,
                             "converted_g_per_100g": physical,
                             "analysis_positive_mass_candidate": candidate,
                             "strict_validation_approved": False, "training_admitted": False})
        parts.append(part)
    cells = pd.concat(parts, ignore_index=True)
    if len(cells) != len(foods) * len(components) or cells.duplicated(["source_food_id", "source_component_id"]).any():
        raise ValueError("BLS cell count or uniqueness check failed")
    candidate = cells[cells.analysis_positive_mass_candidate].copy()
    candidate_counts = candidate.groupby("source_food_id").size()
    foods["positive_analysis_mass_cells"] = foods.source_food_id.map(candidate_counts).fillna(0).astype(int)
    foods["recipe_origin_cells"] = foods.source_food_id.map(cells[cells.source_origin.eq("Rezeptberechnung")].groupby("source_food_id").size()).fillna(0).astype(int)
    food_names = current_foods.assign(normalized_name=current_foods.canonical_name.map(normalize_text))
    exact = foods.assign(normalized_name=foods.official_name_en.map(normalize_text)).merge(
        food_names[["food_concept_id", "normalized_name", "partition", "canonical_name"]], on="normalized_name", how="inner")
    exact["status"] = "exact_name_candidate_only_no_merge_or_independence_claim"
    source_support = candidate.groupby("source_component_id").agg(
        positive_foods=("source_food_id", "nunique"), min_g_per_100g=("converted_g_per_100g", "min"), max_g_per_100g=("converted_g_per_100g", "max")).reset_index()
    components = components.merge(source_support, left_on="source_code", right_on="source_component_id", how="left", validate="one_to_one")
    components["positive_foods"] = components.positive_foods.fillna(0).astype(int)
    write_csv(foods, output / "bls_food_candidates.csv")
    write_csv(components, output / "bls_component_evidence.csv")
    write_csv(cells, output / "bls_all_cell_provenance_NOT_TRAINING.csv.gz")
    write_csv(candidate, output / "bls_analysis_positive_mass_candidates_NOT_TRAINING.csv.gz")
    write_csv(exact, output / "bls_exact_name_candidates_NOT_MERGES.csv")
    write_csv(cells.groupby(["source_origin", "value_status"]).size().reset_index(name="cells"), output / "bls_origin_status_counts.csv")
    invalid = cells[cells.expression_status.eq("mass_expression_candidate") & cells.numeric_value.notna() & ~cells.converted_g_per_100g.between(0, 100)]
    write_csv(invalid, output / "bls_invalid_mass_cells.csv")
    write_csv(cells[cells.value_status.isin(["invalid", "unparsed"])], output / "bls_unparsed_cells.csv")
    reference_counts = candidate.source_reference.value_counts(dropna=False).rename_axis("reference").reset_index(name="cells")
    write_csv(reference_counts, output / "bls_analysis_reference_availability.csv")
    origin_counts = cells.source_origin.value_counts().to_dict()
    return {"raw_foods": len(foods), "raw_component_columns": len(components), "raw_cells": len(cells),
            "origin_counts": origin_counts, "mass_expression_columns_after_definition_holds": int(components.expression_status.eq("mass_expression_candidate").sum()),
            "positive_analysis_mass_candidate_cells": len(candidate),
            "foods_with_positive_analysis_mass_candidates": candidate.source_food_id.nunique(),
            "foods_with_at_least_three_positive_analysis_mass_candidates": int(foods.positive_analysis_mass_cells.ge(3).sum()),
            "axes_with_positive_analysis_mass_candidates": candidate.source_component_id.nunique(),
            "analysis_candidate_references_missing": int(candidate.source_reference.isin(["", "-"]).sum()),
            "mass_candidates_out_of_physical_range": len(invalid),
            "exact_name_candidate_foods_vs_current_partitions": exact.source_food_id.nunique(),
            "exact_name_candidate_foods_vs_validation": exact.loc[exact.partition.eq("validation"), "source_food_id"].nunique(),
            "license": "CC BY 4.0; official download page archived",
            "admitted_new_training_foods": 0,
            "interpretation": "7140 includes recipes, assumptions and copies. Analyse is the official internal analytical-project category; result-level method/sample/reference details remain unavailable in this download. All name matches are candidates only."}


def audit_mext(root, output, current_components):
    prior = root / "reports/validation_mext_evidence_2026_09_06"
    candidates = pd.read_csv(prior / "mext_positive_mass_candidates_NOT_TRAINING.csv.gz", keep_default_na=False, low_memory=False, dtype={"food_id": str})
    axes = read_component_csv(prior / "mext_unique_candidate_axis_support.csv")
    axes["source_tag"] = axes.component_key.str.split(":").str[1]
    current = current_components[current_components.authority_namespace.eq("INFOODS")]
    proposals = axes.merge(current[["component_concept_id", "authority_id", "canonical_name", "definition", "training_role"]],
                            left_on="source_tag", right_on="authority_id", how="left")
    proposals["review_status"] = np.where(proposals.authority_id.notna(),
        "same_literal_code_only_definition_and_method_review_required", "requires_source_specific_identity_review")
    proposals["training_admitted"] = False
    write_csv(proposals, output / "mext_tag_review_candidates_NOT_APPROVED.csv")
    return {"positive_mass_candidate_cells": len(candidates), "candidate_axis_expressions": len(axes),
            "axes_with_same_literal_INFOODS_code_in_current_retained_registry": proposals.loc[proposals.authority_id.notna(), "component_key"].nunique(),
            "admitted_new_training_foods": 0, "cross_database_mappings_approved_by_this_script": 0,
            "interpretation": "Same code is not sufficient for merging; Japanese/English identity, food facets, value origin and dependency masking remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "data/processed/scientific_food_composition_v3/release"
    output = root / "reports/scientific_food_composition_v3/expansion"
    output.mkdir(parents=True, exist_ok=True)
    components = read_component_csv(release / "component_concept.csv.gz")
    components = components[components.training_role.ne("excluded")]
    foods = pd.read_csv(release / "ml_partition.csv", low_memory=False)
    summary = {"bls": audit_bls(root, output, components, foods), "mext": audit_mext(root, output, components),
               "validation_outcomes_opened": False, "net_training_increment_this_audit": 0}
    write_json(summary, output / "EXPANSION_AUDIT.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

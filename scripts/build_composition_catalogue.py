#!/usr/bin/env python3
"""Build a provenance-first catalogue for active FoodNutriGPT composition axes.

The catalogue intentionally separates a canonical model axis from its raw
source records.  This prevents a FooDB compound-table observation that is
accepted as, for example, calcium from being described as a USDA nutrient
record.  Generated natural-language names expand abbreviations for reading;
the source-original names remain unchanged in the source-record sheet.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data/processed/multisource_nutrition_v6_relative/axis_registry.csv"
DEFAULT_CNF = ROOT / "data/raw/cnf_2026/extracted/Nutrient_Name.csv"
DEFAULT_SR = ROOT / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04/nutrient.csv"
DEFAULT_FOODB_NUTRIENT = ROOT / "foodb_2020_04_07_csv/Nutrient.csv"
DEFAULT_FOODB_COMPOUND = ROOT / "foodb_2020_04_07_csv/Compound.csv"
DEFAULT_FOODB_CROSSWALK = ROOT / "data/audits/foodb_component_crosswalk_review.csv"
DEFAULT_VMH = ROOT / "data/audits/vmh_nutrition_schema_2026_08/vmh_to_local_crosswalk.csv"
DEFAULT_OUTPUT = ROOT / "outputs/composition_catalogue_2026_09"


FOODB_NUTRIENT_FULL_NAMES = {
    "Fat": "total fat (total lipid content)",
    "Proteins": "total protein",
    "Carbohydrate": "total carbohydrate",
    "Fatty acids": "total fatty acids",
    "Fiber (dietary)": "total dietary fiber",
    "Energy": "food energy",
    "Ash": "ash (inorganic mineral residue)",
}

DIRECT_FULL_NAMES = {
    "Total lipid (fat)": "total fat (total lipid content)",
    "Carbohydrate, by difference": "total carbohydrate calculated by difference",
    "Fiber, total dietary": "total dietary fiber",
    "Calcium, Ca": "calcium",
    "Iron, Fe": "iron",
    "Magnesium, Mg": "magnesium",
    "Phosphorus, P": "phosphorus",
    "Potassium, K": "potassium",
    "Sodium, Na": "sodium",
    "Zinc, Zn": "zinc",
    "Copper, Cu": "copper",
    "Fluoride, F": "fluoride",
    "Manganese, Mn": "manganese",
    "Selenium, Se": "selenium",
    "Vitamin E (alpha-tocopherol)": "vitamin E as alpha-tocopherol",
    "Vitamin C, total ascorbic acid": "total vitamin C as ascorbic acid",
    "Vitamin B-6": "vitamin B6",
    "Vitamin B-12": "vitamin B12",
    "Vitamin D (D2 + D3)": "total vitamin D, ergocalciferol plus cholecalciferol",
    "Vitamin D2 (ergocalciferol)": "vitamin D2 (ergocalciferol)",
    "Vitamin D3 (cholecalciferol)": "vitamin D3 (cholecalciferol)",
    "Vitamin K (phylloquinone)": "vitamin K1 (phylloquinone)",
    "Vitamin K (Menaquinone-4)": "vitamin K2 (menaquinone-4)",
    "Vitamin K (Dihydrophylloquinone)": "dihydrovitamin K1 (dihydrophylloquinone)",
    "Carotene, beta": "beta-carotene (provitamin A carotenoid)",
    "Carotene, alpha": "alpha-carotene (provitamin A carotenoid)",
    "Cryptoxanthin, beta": "beta-cryptoxanthin (provitamin A carotenoid)",
    "Tocopherol, alpha": "alpha-tocopherol (vitamin E form)",
    "Tocopherol, beta": "beta-tocopherol (vitamin E form)",
    "Tocopherol, gamma": "gamma-tocopherol (vitamin E form)",
    "Tocopherol, delta": "delta-tocopherol (vitamin E form)",
    "Tocotrienol, alpha": "alpha-tocotrienol (vitamin E form)",
    "Tocotrienol, beta": "beta-tocotrienol (vitamin E form)",
    "Tocotrienol, gamma": "gamma-tocotrienol (vitamin E form)",
    "Tocotrienol, delta": "delta-tocotrienol (vitamin E form)",
    "PUFA 22:6 n-3 (DHA)": "docosahexaenoic acid (DHA; C22:6 omega-3 polyunsaturated fatty acid)",
    "PUFA 20:5 n-3 (EPA)": "eicosapentaenoic acid (EPA; C20:5 omega-3 polyunsaturated fatty acid)",
    "PUFA 18:3 n-3 c,c,c (ALA)": "alpha-linolenic acid (ALA; C18:3 omega-3 polyunsaturated fatty acid)",
    "PUFA 18:2 CLAs": "conjugated linoleic acid isomers (C18:2)",
    "Phytosterols": "total phytosterols (plant sterols)",
}

SATURATED_COMMON = {
    "4:0": "butyric acid", "6:0": "hexanoic acid (caproic acid)",
    "8:0": "octanoic acid (caprylic acid)", "10:0": "decanoic acid (capric acid)",
    "12:0": "dodecanoic acid (lauric acid)", "13:0": "tridecanoic acid",
    "14:0": "tetradecanoic acid (myristic acid)", "15:0": "pentadecanoic acid",
    "16:0": "hexadecanoic acid (palmitic acid)", "17:0": "heptadecanoic acid",
    "18:0": "octadecanoic acid (stearic acid)", "20:0": "eicosanoic acid (arachidic acid)",
    "22:0": "docosanoic acid (behenic acid)", "24:0": "tetracosanoic acid (lignoceric acid)",
}

UNSATURATED_COMMON = {
    "14:1": "tetradecenoic acid (myristoleic acid)", "15:1": "pentadecenoic acid",
    "16:1": "hexadecenoic acid (palmitoleic acid)", "17:1": "heptadecenoic acid",
    "18:1": "octadecenoic acid", "18:2": "octadecadienoic acid (linoleic-acid family)",
    "18:3": "octadecatrienoic acid (linolenic-acid family)", "18:4": "octadecatetraenoic acid",
    "20:1": "eicosenoic acid", "20:2": "eicosadienoic acid", "20:3": "eicosatrienoic acid",
    "20:4": "eicosatetraenoic acid (arachidonic-acid family)", "20:5": "eicosapentaenoic acid",
    "21:5": "heneicosapentaenoic acid", "22:1": "docosenoic acid", "22:2": "docosadienoic acid",
    "22:3": "docosatrienoic acid", "22:4": "docosatetraenoic acid (adrenic-acid family)",
    "22:5": "docosapentaenoic acid", "22:6": "docosahexaenoic acid", "24:1": "tetracosenoic acid (nervonic-acid family)",
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def parse_source_key(source_key: str) -> tuple[str, str]:
    source_kind, source_id = source_key.split(":", 1)
    return source_kind, source_id


def fatty_acid_name(value: str) -> str | None:
    """Expand shorthand fatty-acid names without turning them into unjustified IUPAC names."""
    text = clean(value)
    chain_match = re.search(r"(\d{1,2}:\d)", text)
    if not chain_match:
        return None
    chain = chain_match.group(1)
    prefix = text.upper().split()[0] if text.upper().split() else ""
    if prefix == "SFA" or chain.endswith(":0"):
        base = SATURATED_COMMON.get(chain, f"C{chain} saturated fatty acid")
        return f"{base} (C{chain} saturated fatty acid)"
    if prefix == "TFA" or " trans" in text.lower() or re.search(r"\b\d+t\b", text.lower()):
        kind = "trans unsaturated fatty acid"
    elif prefix == "MUFA" or chain.split(":")[1] == "1":
        kind = "monounsaturated fatty acid"
    elif prefix == "PUFA" or int(chain.split(":")[1]) >= 2:
        kind = "polyunsaturated fatty acid"
    else:
        return None
    base = UNSATURATED_COMMON.get(chain, f"C{chain} unsaturated fatty acid")
    # A bare ``c`` or ``t`` is an isomer label, not a positional common name.
    # Avoid incorrectly calling a trans isomer "palmitoleic acid", for example.
    bare_config = re.search(rf"{re.escape(chain)}\s+([ct])\b", text.lower())
    if bare_config:
        systematic_base = re.sub(r" \([^)]*\)", "", base)
        configuration = "cis" if bare_config.group(1) == "c" else "trans"
        return f"{configuration}-{systematic_base} (C{chain} {kind}; configuration specified, double-bond position not specified)"
    details = []
    omega = re.search(r"n-(\d)", text.lower())
    if omega:
        details.append(f"omega-{omega.group(1)}")
    cis_positions = re.findall(r"(\d+)c", text.lower())
    trans_positions = re.findall(r"(\d+)t", text.lower())
    if cis_positions:
        details.append("cis double bond" + ("s" if len(cis_positions) > 1 else "") + " at carbon " + ", ".join(cis_positions))
    if trans_positions:
        details.append("trans double bond" + ("s" if len(trans_positions) > 1 else "") + " at carbon " + ", ".join(trans_positions))
    suffix = "; " + "; ".join(details) if details else ""
    return f"{base} (C{chain} {kind}{suffix})"


def natural_name(axis_name: str, axis_id: str) -> str:
    name = clean(axis_name)
    if name in DIRECT_FULL_NAMES:
        return DIRECT_FULL_NAMES[name]
    if name in FOODB_NUTRIENT_FULL_NAMES:
        return FOODB_NUTRIENT_FULL_NAMES[name]
    generated_fatty_acid = fatty_acid_name(name)
    if generated_fatty_acid:
        return generated_fatty_acid
    replacements = {
        "SFA": "saturated fatty acids",
        "MUFA": "monounsaturated fatty acids",
        "PUFA": "polyunsaturated fatty acids",
        "TFA": "trans fatty acids",
        "CLAs": "conjugated linoleic acid isomers",
    }
    for abbreviation, expanded in replacements.items():
        name = re.sub(rf"\b{re.escape(abbreviation)}\b", expanded, name)
    return name


def current_definition(row: pd.Series) -> str:
    natural = row["generated_natural_language_name"]
    if row["axis_class"] == "macro_nutrient":
        return f"Core macronutrient: mass concentration of {natural} per 100 g edible food."
    if row["axis_class"] == "micro_nutrient":
        return f"Core micronutrient: mass concentration of {natural} per 100 g edible food."
    if row["mask_policy"] == "context_only":
        return f"Nutrient chemical form retained only as observed context; not randomly masked or evaluated because positive support is too limited or degenerate. Measures {natural} per 100 g edible food."
    return f"Nutrient chemical form: mass concentration of {natural} per 100 g edible food; predicted during Stage 2."


def source_definition(source_kind: str, raw: pd.Series) -> tuple[str, str, str, str]:
    """Return source, source-table definition, raw name, and raw description."""
    if source_kind == "cnf_nutrient":
        original_name = clean(raw["Nutrient_Name_EN"])
        unit = clean(raw["Nutrient_Unit"])
        definition = (
            "Nutrition: CNF Nutrient_Name/Nutrient_Amount record; "
            f"CNF nutrient code {clean(raw['Nutrient_Code'])}, symbol {clean(raw['Nutrient_Symbol'])}, unit {unit}."
        )
        description = f"CNF English nutrient name: {original_name}. Tag name: {clean(raw['Tagname'])}."
        return "CNF 2026", definition, original_name, description
    if source_kind == "sr_nutrient":
        original_name = clean(raw["name"])
        unit = clean(raw["unit_name"])
        definition = (
            "Nutrition: USDA FoodData Central SR Legacy nutrient table record; "
            f"USDA nutrient number {clean(raw['nutrient_nbr'])}, unit {unit}."
        )
        description = f"USDA SR Legacy nutrient label: {original_name}."
        return "USDA SR Legacy", definition, original_name, description
    if source_kind == "foodb_nutrient":
        original_name = clean(raw["name"])
        description = clean(raw["description"])
        definition = "Nutrition: FooDB Nutrient table record."
        return "FooDB", definition, original_name, description
    if source_kind == "foodb_compound":
        original_name = clean(raw["name"])
        chemistry = "; ".join(part for part in [
            clean(raw["kingdom"]), clean(raw["superklass"]), clean(raw["klass"]), clean(raw["subklass"])
        ] if part)
        definition = "Compound: FooDB Compound table record."
        description = clean(raw["description"])
        if chemistry:
            description = f"FooDB chemical taxonomy: {chemistry}. " + description
        return "FooDB", definition, original_name, description
    raise ValueError(f"Unsupported source key: {source_kind}")


def source_records_for_axis(
    axis: pd.Series,
    source_tables: dict[str, pd.DataFrame],
    foodb_crosswalk: pd.DataFrame,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    crosswalk_by_pair = {
        (str(row.foodb_axis_kind), str(row.foodb_axis_name)): row
        for row in foodb_crosswalk.itertuples(index=False)
    }
    for key in str(axis["source_axis_keys"]).split("; "):
        source_kind, source_id = parse_source_key(key)
        table = source_tables[source_kind]
        raw = table.loc[source_id]
        source, definition, original_name, description = source_definition(source_kind, raw)
        mapping_relation = "Native source-specific axis"
        if source_kind.startswith("foodb_"):
            pair = ("nutrient" if source_kind == "foodb_nutrient" else "compound", original_name)
            review = crosswalk_by_pair.get(pair)
            if review is not None:
                mapping_relation = f"FooDB-to-USDA mapping: {review.mapping_relation}; review status: {review.review_status}."
        elif axis["axis_id"].startswith("fdc:"):
            mapping_relation = "Same canonical USDA nutrient identity across source databases."
        records.append({
            "axis_id": axis["axis_id"],
            "canonical_axis_name": axis["axis_name"],
            "generated_natural_language_name": axis["generated_natural_language_name"],
            "source": source,
            "source_table_role": definition.split(":", 1)[0],
            "source_table_definition": definition,
            "source_axis_key": key,
            "source_original_name": original_name,
            "source_record_description": description,
            "mapping_relation": mapping_relation,
        })
    return records


def ensure_unique_index(frame: pd.DataFrame, index_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[index_col] = frame[index_col].astype(str)
    if frame[index_col].duplicated().any():
        raise ValueError(f"Duplicate raw source ID in {index_col}.")
    return frame.set_index(index_col, drop=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(DEFAULT_REGISTRY).sort_values(["training_layer", "axis_class", "axis_name"], kind="stable").reset_index(drop=True)
    registry["generated_natural_language_name"] = [natural_name(name, axis_id) for name, axis_id in zip(registry["axis_name"], registry["axis_id"])]
    registry["current_composition_definition"] = registry.apply(current_definition, axis=1)
    registry["in_current_training"] = True
    registry["normalized_unit"] = "g/100 g edible food"

    cnf = ensure_unique_index(pd.read_csv(DEFAULT_CNF), "Nutrient_Code")
    sr = ensure_unique_index(pd.read_csv(DEFAULT_SR), "id")
    foodb_nutrient = ensure_unique_index(pd.read_csv(DEFAULT_FOODB_NUTRIENT, low_memory=False), "id")
    foodb_compound = ensure_unique_index(pd.read_csv(DEFAULT_FOODB_COMPOUND, low_memory=False), "id")
    source_tables = {
        "cnf_nutrient": cnf,
        "sr_nutrient": sr,
        "foodb_nutrient": foodb_nutrient,
        "foodb_compound": foodb_compound,
    }
    foodb_crosswalk = pd.read_csv(DEFAULT_FOODB_CROSSWALK)

    source_rows = []
    for _, axis in registry.iterrows():
        source_rows.extend(source_records_for_axis(axis, source_tables, foodb_crosswalk))
    source_records = pd.DataFrame(source_rows).sort_values(["axis_id", "source", "source_axis_key"], kind="stable")

    source_summary = source_records.groupby("axis_id", sort=False).agg(
        source_database_count=("source", "nunique"),
        source_databases=("source", lambda values: "; ".join(dict.fromkeys(values))),
        source_definition_summary=("source_table_definition", lambda values: " | ".join(dict.fromkeys(values))),
        source_original_names=("source_original_name", lambda values: " | ".join(dict.fromkeys(values))),
    ).reset_index()
    catalogue = registry.merge(source_summary, on="axis_id", how="left", validate="one_to_one")

    vmh = pd.read_csv(DEFAULT_VMH)
    vmh_fields = vmh[["axis_id", "vmh_nutrient_id", "vmh_common_name", "vmh_category", "vmh_unit", "recommended_role"]].dropna(subset=["axis_id"]).drop_duplicates("axis_id")
    catalogue = catalogue.merge(vmh_fields, on="axis_id", how="left", validate="one_to_one")
    catalogue["vmh_status"] = catalogue["vmh_nutrient_id"].notna().map({True: "VMH identifier linked", False: "No direct VMH identifier link"})

    catalogue_columns = [
        "axis_index", "axis_id", "canonical_axis_name", "generated_natural_language_name",
        "current_composition_definition", "axis_class", "training_layer", "target_kind", "mask_policy",
        "normalized_unit", "observed_foods", "positive_foods", "explicit_zero_foods", "source_database_count",
        "source_databases", "source_axis_keys", "source_original_names", "source_definition_summary",
        "classification_evidence", "mask_family", "vmh_status", "vmh_nutrient_id", "vmh_common_name",
        "vmh_category", "vmh_unit", "recommended_role",
    ]
    catalogue = catalogue.rename(columns={"axis_name": "canonical_axis_name"})
    catalogue = catalogue[catalogue_columns]

    overview = pd.DataFrame([
        ["Catalogue scope", "192 active composition axes in multisource_nutrition_v6_relative."],
        ["Sources", "FooDB, USDA FoodData Central SR Legacy, and Canadian Nutrient File (CNF) 2026."],
        ["Canonical model definition", "Each row is one current model axis after reviewed cross-source identity mapping."],
        ["FooDB record types", "FooDB entries are explicitly labelled as either Nutrient-table or Compound-table records in Source Records."],
        ["USDA / CNF record types", "USDA SR Legacy and CNF entries are nutrient records; their raw labels and codes are preserved."],
        ["Natural-language names", "Generated names expand abbreviations for readability; source-original labels remain authoritative."],
        ["Unit", "All active model targets use mass quantities converted to g/100 g edible food."],
        ["Missing values", "Absent values remain missing. Explicit zero values are recorded separately and are not inferred from absence."],
        ["Training roles", "Core nutrition: Stage 1 targets. Nutrient chemical forms: Stage 2 targets. Context-only axes are visible but never masked or scored."],
    ], columns=["Topic", "Definition"])

    source_glossary = pd.DataFrame([
        ["FooDB", "Nutrient", "FooDB Nutrient.csv", "FooDB curated nutrient label; source original name and description preserved."],
        ["FooDB", "Compound", "FooDB Compound.csv", "FooDB chemical compound record; only reviewed mappings or retained nutrient chemical forms appear in the active catalogue."],
        ["USDA SR Legacy", "Nutrition", "FoodData Central SR Legacy nutrient.csv", "USDA nutrient table record identified by nutrient number and unit."],
        ["CNF 2026", "Nutrition", "CNF Nutrient_Name.csv / Nutrient_Amount.csv", "Canadian Nutrient File nutrient definition identified by CNF nutrient code, symbol, English name, and unit."],
    ], columns=["Source", "Source-internal role", "Raw definition table", "How it is represented in this workbook"])

    catalogue.to_csv(args.output_dir / "current_composition_catalogue.csv", index=False)
    source_records.to_csv(args.output_dir / "composition_source_records.csv", index=False)
    overview.to_csv(args.output_dir / "catalogue_overview.csv", index=False)
    source_glossary.to_csv(args.output_dir / "source_definition_glossary.csv", index=False)
    catalogue.to_json(args.output_dir / "current_composition_catalogue.json", orient="records", indent=2)
    source_records.to_json(args.output_dir / "composition_source_records.json", orient="records", indent=2)
    overview.to_json(args.output_dir / "catalogue_overview.json", orient="records", indent=2)
    source_glossary.to_json(args.output_dir / "source_definition_glossary.json", orient="records", indent=2)
    (args.output_dir / "catalogue_manifest.json").write_text(json.dumps({
        "active_axes": int(len(catalogue)),
        "source_records": int(len(source_records)),
        "source_counts": source_records["source"].value_counts().to_dict(),
        "axis_class_counts": catalogue["axis_class"].value_counts().to_dict(),
        "mask_policy_counts": catalogue["mask_policy"].value_counts().to_dict(),
    }, indent=2) + "\n")

    print(f"Active axes: {len(catalogue)}")
    print(f"Source records: {len(source_records)}")
    print(source_records["source"].value_counts().to_string())
    print(f"Wrote catalogue CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()

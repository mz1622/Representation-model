#!/usr/bin/env python3
"""Audit mass-scale core micronutrient coverage across current and candidate sources.

The audit deliberately distinguishes a chemical mass form from activity
equivalents.  It does not create a training matrix and never treats a trace or
``not known`` marker as a numeric zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SR_DIR = ROOT / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04"
CNF_DIR = ROOT / "data/raw/cnf_2026/extracted"
FOODB_COMPOUND_DIR = ROOT / "data/processed/fooddb_raw_mass_compounds_v2"
COFID_PATH = ROOT / "data/raw/candidates/cofid_2021/cofid_2021.xlsx"
AFCD_PATH = ROOT / "data/raw/candidates/afcd_release_3/nutrient_profiles.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "data/audits/core_micronutrient_coverage_2026_08"


MASS_FACTORS = {
    "g": 1.0,
    "gram": 1.0,
    "mg": 1e-3,
    "milligram": 1e-3,
    "ug": 1e-6,
    "µg": 1e-6,
    "microgram": 1e-6,
}

# A mass-only training schema. Vitamin A is represented by preformed retinol;
# RAE, DFE, IU and similar activity equivalents require a separate unit-aware
# objective and are not interchangeable with chemical mass measurements.
CORE_MICROS = [
    ("fdc:301", "Calcium", "mineral", "301"),
    ("fdc:302", "Chloride", "mineral", "302"),
    ("fdc:303", "Iron", "trace_mineral", "303"),
    ("fdc:304", "Magnesium", "mineral", "304"),
    ("fdc:305", "Phosphorus", "mineral", "305"),
    ("fdc:306", "Potassium", "mineral", "306"),
    ("fdc:307", "Sodium", "mineral", "307"),
    ("fdc:309", "Zinc", "trace_mineral", "309"),
    ("fdc:310", "Chromium", "trace_mineral", "310"),
    ("fdc:312", "Copper", "trace_mineral", "312"),
    ("fdc:313", "Fluoride", "trace_mineral", "313"),
    ("fdc:314", "Iodine", "trace_mineral", "314"),
    ("fdc:315", "Manganese", "trace_mineral", "315"),
    ("fdc:316", "Molybdenum", "trace_mineral", "316"),
    ("fdc:317", "Selenium", "trace_mineral", "317"),
    ("fdc:319", "Retinol (preformed vitamin A)", "vitamin", "319"),
    ("fdc:323", "Alpha-tocopherol (vitamin E)", "vitamin", "323"),
    ("fdc:328", "Vitamin D (D2 + D3)", "vitamin", "328"),
    ("fdc:401", "Vitamin C", "vitamin", "401"),
    ("fdc:404", "Thiamin (B1)", "vitamin", "404"),
    ("fdc:405", "Riboflavin (B2)", "vitamin", "405"),
    ("fdc:406", "Niacin (B3)", "vitamin", "406"),
    ("fdc:410", "Pantothenic acid (B5)", "vitamin", "410"),
    ("fdc:415", "Vitamin B6", "vitamin", "415"),
    ("fdc:416", "Biotin (B7)", "vitamin", "416"),
    ("fdc:417", "Total folate (B9)", "vitamin", "417"),
    ("fdc:418", "Vitamin B12", "vitamin", "418"),
    ("fdc:421", "Total choline", "vitamin_like", "421"),
    ("fdc:430", "Vitamin K1 (phylloquinone)", "vitamin", "430"),
]

FOODB_COMPOUND_NAMES = {
    "Calcium": "fdc:301", "Chloride": "fdc:302", "Iron": "fdc:303", "Copper": "fdc:312",
    "Fluoride": "fdc:313", "Iodine": "fdc:314", "Manganese": "fdc:315", "Molybdenum": "fdc:316",
    "Selenium": "fdc:317", "Retinol": "fdc:319", "alpha-Tocopherol": "fdc:323",
    "L-Ascorbic acid": "fdc:401", "Thiamine": "fdc:404", "Riboflavine": "fdc:405",
    "Pantothenic acid": "fdc:410", "Biotin": "fdc:416", "Choline": "fdc:421",
    "Phosphorus": "fdc:305", "Potassium": "fdc:306", "Sodium": "fdc:307", "Zinc": "fdc:309",
}

COFID_COLUMNS = {
    "Sodium (mg)": "fdc:307", "Potassium (mg)": "fdc:306", "Calcium (mg)": "fdc:301",
    "Magnesium (mg)": "fdc:304", "Phosphorus (mg)": "fdc:305", "Iron (mg)": "fdc:303",
    "Copper (mg)": "fdc:312", "Zinc (mg)": "fdc:309", "Chloride (mg)": "fdc:302",
    "Manganese (mg)": "fdc:315", "Selenium (µg)": "fdc:317", "Iodine (µg)": "fdc:314",
    "Retinol (µg)": "fdc:319", "Vitamin D (µg)": "fdc:328", "Thiamin (mg)": "fdc:404",
    "Riboflavin (mg)": "fdc:405", "Niacin (mg)": "fdc:406", "Vitamin B6 (mg)": "fdc:415",
    "Vitamin B12 (µg)": "fdc:418", "Folate (µg)": "fdc:417", "Pantothenate (mg)": "fdc:410",
    "Biotin (µg)": "fdc:416", "Vitamin C (mg)": "fdc:401", "Vitamin K1 (µg)": "fdc:430",
    "Alpha-tocopherol (mg)": "fdc:323",
}

AFCD_COLUMNS = {
    "Calcium (Ca) \n(mg)": "fdc:301", "Chromium (Cr) \n(ug)": "fdc:310",
    "Chloride (Cl) \n(mg)": "fdc:302", "Copper (Cu) \n(mg)": "fdc:312",
    "Fluoride (F) \n(ug)": "fdc:313", "Iodine (I) \n(ug)": "fdc:314",
    "Iron (Fe) \n(mg)": "fdc:303", "Magnesium (Mg) \n(mg)": "fdc:304",
    "Manganese (Mn) \n(mg)": "fdc:315", "Molybdenum (Mo) \n(ug)": "fdc:316",
    "Phosphorus (P) \n(mg)": "fdc:305", "Potassium (K) \n(mg)": "fdc:306",
    "Selenium (Se) \n(ug)": "fdc:317", "Sodium (Na) \n(mg)": "fdc:307",
    "Zinc (Zn) \n(mg)": "fdc:309", "Retinol (preformed vitamin A) \n(ug)": "fdc:319",
    "Alpha tocopherol \n(mg)": "fdc:323", "Thiamin (B1) \n(mg)": "fdc:404",
    "Riboflavin (B2) \n(mg)": "fdc:405", "Niacin (B3) \n(mg)": "fdc:406",
    "Pantothenic acid (B5) \n(mg)": "fdc:410", "Pyridoxine (B6) \n(mg)": "fdc:415",
    "Biotin (B7) \n(ug)": "fdc:416", "Cobalamin (B12) \n(ug)": "fdc:418",
    "Total folates \n(ug)": "fdc:417", "Vitamin C \n(mg)": "fdc:401",
}


def numeric_count(values: pd.Series) -> tuple[int, int]:
    numeric = pd.to_numeric(values.replace({"Tr": np.nan, "N": np.nan}), errors="coerce")
    return int(numeric.notna().sum()), int((numeric.dropna() > 0).sum())


def summarize(source: str, axis_id: str, values: pd.Series, direct_merge: bool, note: str) -> dict[str, object]:
    observed, positive = numeric_count(values)
    return {
        "source": source,
        "axis_id": axis_id,
        "raw_observed_foods": observed,
        "observed_foods": observed,
        "positive_foods": positive,
        "direct_mass_merge_candidate": direct_merge,
        "note": note,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def audit_sr() -> list[dict[str, object]]:
    nutrients = pd.read_csv(SR_DIR / "nutrient.csv")
    nutrients["code"] = nutrients["nutrient_nbr"].map(lambda value: format(float(value), "g") if pd.notna(value) else "")
    values = pd.read_csv(SR_DIR / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    data = values.merge(nutrients[["id", "code"]], left_on="nutrient_id", right_on="id", how="inner")
    return [
        summarize("SR Legacy", f"fdc:{code}", group.groupby("fdc_id")["amount"].first(), True, "Official USDA nutrient code and mass unit.")
        for code, group in data.groupby("code") if f"fdc:{code}" in {item[0] for item in CORE_MICROS}
    ]


def audit_cnf() -> list[dict[str, object]]:
    values = pd.read_csv(CNF_DIR / "Nutrient_Amount.csv")
    values["code"] = pd.to_numeric(values["Nutrient_Code"], errors="coerce").map(lambda value: format(value, "g") if pd.notna(value) else "")
    return [
        summarize("CNF 2026", f"fdc:{code}", group.groupby("Food_Code")["Nutrient_Amount"].first(), True, "Same USDA nutrient code; mass unit recorded by CNF.")
        for code, group in values.groupby("code") if f"fdc:{code}" in {item[0] for item in CORE_MICROS}
    ]


def audit_foodb() -> list[dict[str, object]]:
    axes = pd.read_csv(FOODB_COMPOUND_DIR / "compound_axis_registry.csv")
    values = pd.read_csv(FOODB_COMPOUND_DIR / "observed_compound_values.csv")
    data = values.merge(axes[["foodb_compound_id", "axis_name"]], left_on="source_id", right_on="foodb_compound_id", how="inner")
    rows = []
    for name, axis_id in FOODB_COMPOUND_NAMES.items():
        group = data[data["axis_name"].eq(name)]
        if not group.empty:
            per_food = group.groupby("food_id")["value_g_per_100g"].agg(["count", "min", "max", "median"])
            per_food["relative_range"] = np.where(
                per_food["max"].eq(0), 0.0, (per_food["max"] - per_food["min"]) / per_food["max"]
            )
            retainable = per_food[per_food["relative_range"].le(0.05)]
            row = summarize(
                "FooDB compounds", axis_id, retainable["median"], True,
                "Exact chemical-name candidate. Same-food repeated records are retained only when their range is within 5%."
            )
            row["raw_observed_foods"] = int(len(per_food))
            row["dropped_within_food_collision_foods"] = int(len(per_food) - len(retainable))
            rows.append(row)
    return rows


def read_cofid_sheet(sheet: str) -> pd.DataFrame:
    # Row 0 contains the display headers; rows 1-2 contain INFOODS tags and
    # descriptions, so actual foods begin at row 3.
    return pd.read_excel(COFID_PATH, sheet_name=sheet, header=0).iloc[3:].copy()


def audit_cofid() -> list[dict[str, object]]:
    sheets = {
        "1.4 Inorganics": {key: value for key, value in COFID_COLUMNS.items() if key in {
            "Sodium (mg)", "Potassium (mg)", "Calcium (mg)", "Magnesium (mg)", "Phosphorus (mg)", "Iron (mg)", "Copper (mg)", "Zinc (mg)", "Chloride (mg)", "Manganese (mg)", "Selenium (µg)", "Iodine (µg)"}},
        "1.5 Vitamins": {key: value for key, value in COFID_COLUMNS.items() if key not in {"Alpha-tocopherol (mg)"} and key not in {
            "Sodium (mg)", "Potassium (mg)", "Calcium (mg)", "Magnesium (mg)", "Phosphorus (mg)", "Iron (mg)", "Copper (mg)", "Zinc (mg)", "Chloride (mg)", "Manganese (mg)", "Selenium (µg)", "Iodine (µg)"}},
        "1.6 Vitamin Fractions": {"Alpha-tocopherol (mg)": "fdc:323"},
    }
    rows = []
    for sheet, mapping in sheets.items():
        frame = read_cofid_sheet(sheet)
        for column, axis_id in mapping.items():
            if column in frame:
                rows.append(summarize("CoFID 2021", axis_id, frame[column], True, f"Per-100g table; trace (Tr) and unknown (N) are retained as missing, not zero. Sheet: {sheet}."))
    return rows


def audit_afcd() -> list[dict[str, object]]:
    frame = pd.read_excel(AFCD_PATH, sheet_name="All solids & liquids per 100 g", header=2)
    rows = []
    for column, axis_id in AFCD_COLUMNS.items():
        if column in frame:
            rows.append(summarize("AFCD Release 3", axis_id, frame[column], True, "Per-100g candidate. Retain food-level derivation (analysed/recipe/borrowed/etc.) before training."))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema = pd.DataFrame(CORE_MICROS, columns=["axis_id", "canonical_name", "class", "usda_code"])
    coverage = pd.DataFrame(audit_sr() + audit_cnf() + audit_foodb() + audit_cofid() + audit_afcd())
    coverage = schema.merge(coverage, on="axis_id", how="left")
    coverage["raw_observed_foods"] = coverage["raw_observed_foods"].fillna(0).astype(int)
    coverage["observed_foods"] = coverage["observed_foods"].fillna(0).astype(int)
    coverage["positive_foods"] = coverage["positive_foods"].fillna(0).astype(int)
    coverage.to_csv(args.output_dir / "core_micronutrient_source_coverage.csv", index=False)

    matrix = coverage.pivot_table(index=["axis_id", "canonical_name", "class"], columns="source", values="observed_foods", aggfunc="sum", fill_value=0).reset_index()
    matrix["sources_with_observations"] = (matrix.drop(columns=["axis_id", "canonical_name", "class"]) > 0).sum(axis=1)
    matrix.to_csv(args.output_dir / "core_micronutrient_coverage_matrix.csv", index=False)

    current_sources = {"SR Legacy", "CNF 2026", "FooDB compounds"}
    current = coverage[coverage["source"].isin(current_sources)].groupby("axis_id")["observed_foods"].sum()
    all_sources = coverage.groupby("axis_id")["observed_foods"].sum()
    schema["current_source_observed_rows"] = schema["axis_id"].map(current).fillna(0).astype(int)
    schema["including_downloaded_candidates_observed_rows"] = schema["axis_id"].map(all_sources).fillna(0).astype(int)
    schema["current_status"] = np.select(
        [schema["current_source_observed_rows"].eq(0), schema["current_source_observed_rows"].lt(20)],
        ["absent", "below current support threshold"],
        default="available",
    )
    schema.to_csv(args.output_dir / "core_micronutrient_schema_status.csv", index=False)

    unavailable = schema[schema["current_source_observed_rows"].eq(0)][["canonical_name", "axis_id"]]
    gained = schema[(schema["current_source_observed_rows"].lt(20)) & (schema["including_downloaded_candidates_observed_rows"].ge(20))][["canonical_name", "axis_id"]]
    report = f"""# Core Micronutrient Coverage Audit

## Scope

This audit defines **29 mass-scale core micronutrient axes**: 15 minerals/trace minerals and 14 vitamins or vitamin-like nutrients. It uses chemical mass forms only. Activity-equivalent measures such as vitamin A RAE, folate DFE, vitamin E TE, and IU are deliberately excluded from this mass-only schema because they are not chemically interchangeable with `g/100g` values.

The current v4 registry has 26 micro-labelled axes, but that is not 26 independent core nutrients: it includes three folate forms (total folate, food folate, and folic acid) while leaving several essential mineral axes unmapped or under the compound label.

## Current Three-Source Coverage

Of the 29 core axes, {int((schema['current_source_observed_rows'] >= 20).sum())} have at least 20 **source-level, collision-screened candidate records** when SR Legacy, CNF, and exact-name FooDB compound candidates are counted. This is not final trainable coverage: cross-source food deduplication and definition checks still remain. The currently absent axes are:

{markdown_table(unavailable)}

Axes that become available at 20 or more numeric food records after downloading CoFID and AFCD candidates are:

{markdown_table(gained)}

## Candidate-Source Decision

CoFID 2021 is a 2,886-food UK composition table with direct values for chloride, iodine, biotin, vitamin K1 and the standard vitamins/minerals. AFCD Release 3 has 1,588 food rows and direct values for chromium, chloride, fluoride, iodine, molybdenum and the standard micronutrients. Both are valuable candidates, but neither should be concatenated into training yet: food preparation basis, trace/unknown notation, and derivation quality must be preserved and audited. CoFID uses `Tr` for a trace amount and `N` for present-but-not-quantified; these are not zero labels. AFCD records a food-level derivation category, including analysed, recipe, borrowed, imputed, label and estimated values.

## Required Schema Corrections Before Retraining

1. Map exact FooDB compound records for calcium, iron, copper, iodine, molybdenum, selenium, zinc, B vitamins, retinol, alpha-tocopherol and ascorbic acid to their canonical nutrient axes only after within-food collision checks.
2. Promote fluoride, iodine and molybdenum to micronutrients. Add chloride and chromium once CoFID/AFCD are admitted through a source-aware pipeline.
3. Keep one primary B9 target, **total folate**, and retain food folate/folic acid only as non-target chemical-form context if their simultaneous visibility is controlled. Otherwise they leak the masked total-folate target.
4. Do not add RAE, DFE, IU, TE or other activity equivalents to this mass-only objective. They require a separate unit-aware task.
5. Reclassify ash as a proximate/mineral-residue component, not a compound.

The full source-by-axis counts are in `core_micronutrient_source_coverage.csv`; the compact matrix is in `core_micronutrient_coverage_matrix.csv`.
"""
    # Avoid a runtime-only optional formatting dependency.
    report = report.replace("<table>", "")
    (args.output_dir / "CORE_MICRONUTRIENT_COVERAGE_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit whether reviewed FooDB compound records can share numeric labels.

This script intentionally evaluates only reviewed mappings on 3- or 4-source
axes. It accepts the directly declared ``mg/100g`` and ``mg/100 g`` records
for unit conversion, then checks whether a FooDB food-level aggregate would
collapse contradictory values from different original food records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "audits"


def normalize_unit(value: object) -> str:
    return str(value).replace(" ", "").lower()


def convert_from_mg_per_100g(values: pd.Series, target_unit: str) -> tuple[pd.Series, str]:
    if target_unit == "Gram":
        return values / 1000.0, "mg/100g -> g/100g: divide by 1,000"
    if target_unit == "Milligram":
        return values, "mg/100g -> mg/100g: identity"
    if target_unit == "Microgram":
        return values * 1000.0, "mg/100g -> ug/100g: multiply by 1,000"
    raise ValueError(f"Unsupported canonical unit: {target_unit}")


def main() -> None:
    axes = pd.read_csv(AUDIT_DIR / "shared_training_axis_candidates.csv")
    focus = axes[axes["source_count"].isin([3, 4]) & axes["foodb_mapping_count"].gt(0)].copy()
    review = pd.read_csv(AUDIT_DIR / "foodb_component_crosswalk_review.csv")
    review = review[
        review["foodb_axis_kind"].eq("compound") & review["canonical_usda_code"].isin(focus["canonical_usda_code"])
    ].copy()
    compounds = pd.read_csv(ROOT / "foodb_2020_04_07_csv/Compound.csv", usecols=["id", "name"], low_memory=False)
    review = review.merge(compounds, left_on="foodb_axis_name", right_on="name", how="left", validate="one_to_one")
    if review["id"].isna().any():
        raise ValueError("A reviewed compound mapping did not resolve to exactly one FooDB compound ID")

    content = pd.read_csv(
        ROOT / "foodb_2020_04_07_csv/Content.csv",
        usecols=["source_id", "source_type", "food_id", "orig_source_id", "orig_food_id", "orig_food_part", "preparation_type", "standard_content", "orig_unit"],
        low_memory=False,
    )
    content["standard_content"] = pd.to_numeric(content["standard_content"], errors="coerce")
    content = content[
        content["source_type"].eq("Compound")
        & content["source_id"].isin(review["id"])
        & content["standard_content"].notna()
    ].merge(
        review[["id", "canonical_usda_code", "canonical_name", "foodb_axis_name"]],
        left_on="source_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )
    target_units = focus.set_index("canonical_usda_code")["canonical_unit"].to_dict()
    content["normalized_orig_unit"] = content["orig_unit"].map(normalize_unit)
    compatible = content[content["normalized_orig_unit"].eq("mg/100g")].copy()

    converted_parts = []
    conversion_rules: dict[int, str] = {}
    for code, group in compatible.groupby("canonical_usda_code"):
        converted, rule = convert_from_mg_per_100g(group["standard_content"], target_units[code])
        group = group.assign(value_per_100g=converted)
        converted_parts.append(group)
        conversion_rules[code] = rule
    compatible = pd.concat(converted_parts, ignore_index=True)

    rows = []
    for code, axis in focus.set_index("canonical_usda_code").iterrows():
        raw = content[content["canonical_usda_code"].eq(code)]
        usable = compatible[compatible["canonical_usda_code"].eq(code)]
        pair = usable.groupby("food_id")["value_per_100g"].agg(["size", "min", "max"])
        conflicting = (pair["min"].eq(0) & pair["max"].gt(0))
        rows.append(
            {
                "canonical_usda_code": code,
                "canonical_name": axis.canonical_name,
                "canonical_unit_per_100g": axis.canonical_unit,
                "foodb_compound": axis.foodb_mappings,
                "all_numeric_records": len(raw),
                "recognized_mg_per_100g_records": len(usable),
                "recognized_mg_per_100g_foods": int(usable["food_id"].nunique()),
                "food_level_repeated_foods": int((pair["size"] > 1).sum()),
                "food_level_zero_positive_conflicts": int(conflicting.sum()),
                "food_level_conflict_percent": round(float(conflicting.mean() * 100), 2) if len(pair) else None,
                "conversion_rule": conversion_rules.get(code, "no compatible record"),
                "unit_normalization_possible": bool(len(usable)),
                "food_level_training_decision": "defer",
                "decision_reason": "A FooDB food ID aggregates distinct original foods/preparations; do not collapse contradictory content records into one label.",
            }
        )
    result = pd.DataFrame(rows).sort_values("canonical_usda_code")
    result.to_csv(AUDIT_DIR / "foodb_compound_unit_compatibility.csv", index=False)

    lines = [
        "# FooDB Compound Unit Compatibility Audit",
        "",
        "## Scope",
        "",
        "This audit covers the 16 reviewed FooDB compound mappings on canonical axes present in three or four datasets. It deliberately excludes single-source axes and name-only mapping candidates.",
        "",
        "## Unit Result",
        "",
        f"All {int(result.unit_normalization_possible.sum())}/{len(result)} focus compounds have directly declared `mg/100g`-style FooDB records. They can be converted to the official external per-100-g unit using the rule recorded per axis in the CSV.",
        "",
        "## Label Result",
        "",
        "None of these values should yet be used as a FooDB food-level target. `Content.food_id` is an upper-level FooDB entity, while its records combine different original foods, parts, and preparations. A zero and a positive value within one FooDB food ID therefore often refer to different original observations rather than repeated measurements of one item. Mean, median, or a zero-precedence rule would create synthetic labels.",
        "",
        "## Decision",
        "",
        "- Use the 116 three/four-source axes as the current shared reference vocabulary, with missing values absent from tokens and loss.",
        "- Do **not** merge FooDB compound values into that food-level matrix in the first run, even though unit conversion is available.",
        "- To use FooDB compounds later, create an observation-level FooDB food table keyed by a reviewed original-food/provenance identifier and preparation state; aggregate only records proven to describe the same underlying observation. Then reapply the unit conversion and duplicate-consistency audit.",
        "",
        "## Per-Axis Results",
        "",
        "| Code | Axis | Target unit / 100 g | FooDB foods with convertible records | Zero/positive conflicts within FooDB food ID | Decision |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.canonical_usda_code} | {row.canonical_name} | {row.canonical_unit_per_100g} | "
            f"{row.recognized_mg_per_100g_foods} | {row.food_level_zero_positive_conflicts} ({row.food_level_conflict_percent}%) | {row.food_level_training_decision} |"
        )
    (AUDIT_DIR / "FOODB_COMPOUND_UNIT_COMPATIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} compound compatibility rows")
    print("Unit-normalizable:", int(result["unit_normalization_possible"].sum()))
    print("Food-level usable now: 0")


if __name__ == "__main__":
    main()

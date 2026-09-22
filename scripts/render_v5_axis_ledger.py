#!/usr/bin/env python3
"""Render a source-by-source evidence ledger for every v5 axis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data/audits/multisource_nutrition_v5"
DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"


def join_unique(values: pd.Series) -> str:
    return "; ".join(sorted({str(value) for value in values if pd.notna(value) and str(value)}))


def main() -> None:
    classification = pd.read_csv(AUDIT_DIR / "all_axis_classification.csv")
    coverage = pd.read_csv(AUDIT_DIR / "axis_source_coverage_pre_row_dedup.csv")
    retained = pd.read_csv(DATA_DIR / "axis_registry.csv")
    source_summary = coverage.groupby(["axis_id", "source"], as_index=False).agg(
        source_observed_food_rows=("source_observed_food_rows", "sum"),
        source_positive_food_rows=("source_positive_food_rows", "sum"),
        source_zero_food_rows=("source_zero_food_rows", "sum"),
        source_axis_keys=("source_axis_keys", join_unique),
    )
    wide_parts = []
    for source in ("foodb", "sr_legacy", "cnf"):
        subset = source_summary[source_summary["source"].eq(source)].set_index("axis_id")
        renamed = subset.rename(columns={
            "source_observed_food_rows": f"{source}_source_food_rows_pre_row_dedup",
            "source_positive_food_rows": f"{source}_positive_food_rows_pre_row_dedup",
            "source_zero_food_rows": f"{source}_zero_food_rows_pre_row_dedup",
            "source_axis_keys": f"{source}_source_axis_keys",
        })[[
            f"{source}_source_food_rows_pre_row_dedup", f"{source}_positive_food_rows_pre_row_dedup",
            f"{source}_zero_food_rows_pre_row_dedup", f"{source}_source_axis_keys",
        ]]
        wide_parts.append(renamed)
    ledger = classification.set_index("axis_id")
    for part in wide_parts:
        ledger = ledger.join(part, how="left")
    ledger = ledger.reset_index().merge(
        retained[["axis_id", "observed_foods", "positive_foods", "explicit_zero_foods", "mask_policy", "axis_index"]],
        on="axis_id", how="left", validate="one_to_one",
    )
    numeric_columns = [column for column in ledger.columns if column.endswith("_food_rows_pre_row_dedup") or column in {"observed_foods", "positive_foods", "explicit_zero_foods", "axis_index"}]
    ledger[numeric_columns] = ledger[numeric_columns].fillna(0).astype(int)
    for source in ("foodb", "sr_legacy", "cnf"):
        ledger[f"{source}_source_axis_keys"] = ledger[f"{source}_source_axis_keys"].fillna("")
    ledger["retained_in_v5"] = ledger["axis_index"].ne(0) | ledger["axis_id"].eq(retained.iloc[0]["axis_id"])
    ledger = ledger.sort_values(["retained_in_v5", "training_layer", "axis_class", "axis_id"], ascending=[False, True, True, True], kind="stable")
    ledger.to_csv(AUDIT_DIR / "v5_axis_evidence_ledger.csv", index=False)

    retained_counts = ledger[ledger["retained_in_v5"]].groupby(["training_layer", "axis_class"]).size().sort_values(ascending=False)
    excluded_counts = ledger[~ledger["retained_in_v5"]].groupby("axis_class").size().sort_values(ascending=False)
    lines = [
        "# v5 Axis Evidence Ledger Guide",
        "",
        "`v5_axis_evidence_ledger.csv` contains one row for every mass axis found in the three raw sources. The source-specific `*_source_food_rows_pre_row_dedup` columns count source food rows before cross-source duplicate resolution. `*_source_axis_keys` gives the original table identifier that produced those values. The final `observed_foods`, `positive_foods`, and `mask_policy` columns apply only to retained v5 axes after row deduplication and quality filters.",
        "",
        "## Retained v5 Axes",
        "",
        "| Training layer / class | Axes |",
        "|---|---:|",
        *[f"| {layer} / {axis_class} | {count:,} |" for (layer, axis_class), count in retained_counts.items()],
        "",
        "## Excluded Axes",
        "",
        "| Exclusion class | Axes |",
        "|---|---:|",
        *[f"| {axis_class} | {count:,} |" for axis_class, count in excluded_counts.items()],
        "",
        "The ledger is evidence, not a claim that every listed food value is analytically identical across databases. Definition-compatible axes are merged by exact USDA code or an explicit FooDB chemical mapping; source collisions and food duplicates are resolved by the v5 build audit.",
    ]
    (AUDIT_DIR / "V5_AXIS_EVIDENCE_LEDGER_GUIDE.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {AUDIT_DIR / 'v5_axis_evidence_ledger.csv'}")


if __name__ == "__main__":
    main()

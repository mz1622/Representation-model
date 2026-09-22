#!/usr/bin/env python3
"""Audit source-native food records that have no mapped prediction-axis values.

This is an audit-only report. It never imputes values, changes an axis mapping,
or treats a missing source value as zero.

Colab:
    python scripts/audit_food_axis_coverage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import write_csv  # noqa: E402


DEFAULT_AUDIT = ROOT / "data" / "processed" / "global_frozen_prediction_panel_food_audit_v2"
DEFAULT_OUTPUT = ROOT / "outputs" / "food_composition_atlas" / "audits"

# The audited staging files predate the global source registry. These aliases
# identify the same source release; they do not merge any food or measurement.
STAGING_TO_REGISTRY_KEY = {
    "afcd": "afcd_release_3",
    "ciqual": "ciqual_2025",
    "cofid": "cofid_2021",
    "fndds": "usda_fndds_2021_2023",
    "foodb": "foodb_2020",
    "frida": "frida_6_1",
    "norway": "norwegian_fcdb",
}


def _source_coverage(
    observations: pd.DataFrame,
    mapped: pd.DataFrame,
    direct: pd.DataFrame,
) -> pd.DataFrame:
    mapped_foods = (
        mapped.loc[:, ["food_observation_id", "source_key", "target_axis_id"]]
        .drop_duplicates()
        .groupby(["source_key", "food_observation_id"], as_index=False)
        .agg(
            mapped_source_axis_count=("target_axis_id", "nunique"),
        )
    )
    direct_foods = (
        direct.loc[:, ["food_observation_id", "source_key", "target_axis_id"]]
        .drop_duplicates()
        .groupby(["source_key", "food_observation_id"], as_index=False)
        .agg(model_eligible_axis_count=("target_axis_id", "nunique"))
    )
    result = observations.merge(
        mapped_foods,
        how="left",
        on=["source_key", "food_observation_id"],
        validate="one_to_one",
    ).merge(
        direct_foods,
        how="left",
        on=["source_key", "food_observation_id"],
        validate="one_to_one",
    )
    result["mapped_source_axis_count"] = result["mapped_source_axis_count"].fillna(0).astype(int)
    result["model_eligible_axis_count"] = result["model_eligible_axis_count"].fillna(0).astype(int)
    result["has_mapped_prediction_axis"] = result["mapped_source_axis_count"].gt(0)
    result["has_no_mapped_prediction_axis"] = ~result["has_mapped_prediction_axis"]
    result["source_registry_key"] = result["source_key"].replace(STAGING_TO_REGISTRY_KEY)
    return (
        result.groupby(["source_key", "source_registry_key"], as_index=False)
        .agg(
            source_food_observations=("food_observation_id", "size"),
            foods_with_mapped_prediction_axis=("has_mapped_prediction_axis", "sum"),
            foods_without_mapped_prediction_axis=("has_no_mapped_prediction_axis", "sum"),
            median_mapped_source_axis_count=("mapped_source_axis_count", "median"),
            mean_mapped_source_axis_count=("mapped_source_axis_count", "mean"),
            foods_with_model_eligible_axis=("model_eligible_axis_count", lambda values: int(values.gt(0).sum())),
            median_model_eligible_axis_count=("model_eligible_axis_count", "median"),
        )
        .assign(
            food_records_without_axis_percent=lambda frame: (
                100
                * frame["foods_without_mapped_prediction_axis"]
                / frame["source_food_observations"]
            )
        )
        .sort_values(
            ["food_records_without_axis_percent", "source_food_observations"],
            ascending=[False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_audit(audit_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    observation_path = audit_dir / "food_observation_to_exact_name_group.csv.gz"
    mapped_path = audit_dir / "mapped_numeric_target_measurements.csv.gz"
    direct_path = audit_dir / "direct_mass_label_candidate_measurements.csv.gz"
    ledger_path = audit_dir / "global_source_ingestion_ledger.csv"
    for path in (observation_path, mapped_path, direct_path, ledger_path):
        if not path.exists():
            raise FileNotFoundError(f"Required audited input is missing: {path}")

    observations = pd.read_csv(observation_path, keep_default_na=False, low_memory=False)
    mapped = pd.read_csv(mapped_path, keep_default_na=False, low_memory=False)
    direct = pd.read_csv(direct_path, keep_default_na=False, low_memory=False)
    ledger = pd.read_csv(ledger_path, keep_default_na=False)
    source_coverage = _source_coverage(observations, mapped, direct).merge(
        ledger.loc[
            :,
            [
                "source_key",
                "source_name",
                "region",
                "access_status",
                "ingestion_status",
                "ingestion_reason",
                "mapped_numeric_measurements_retained",
                "frozen_prediction_axes_with_exact_crosswalk",
            ],
        ],
        how="left",
        left_on="source_registry_key",
        right_on="source_key",
        validate="one_to_one",
        suffixes=("", "_registry"),
    )
    source_coverage = source_coverage.drop(columns=["source_key_registry"])

    observed_ids = set(mapped["food_observation_id"])
    unmapped_foods = observations.loc[
        ~observations["food_observation_id"].isin(observed_ids),
        [
            "food_observation_id",
            "source_key",
            "source_food_id",
            "original_name",
            "scientific_name",
            "food_group",
            "food_subgroup",
            "food_type",
            "part",
            "processing",
            "cooking",
            "recipe_status",
            "source_table",
            "source_lineage",
            "source_record_url",
            "exact_name_group_id",
        ],
    ].copy()
    unmapped_foods = unmapped_foods.merge(
        ledger.loc[:, ["source_key", "source_name", "ingestion_status", "ingestion_reason"]],
        how="left",
        on="source_key",
        validate="many_to_one",
    ).sort_values(["source_key", "original_name"], kind="stable")

    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "source_food_prediction_axis_coverage.csv"
    missing_path = output_dir / "food_observations_without_mapped_prediction_axes.csv.gz"
    summary_path = output_dir / "food_axis_coverage_audit.md"
    write_csv(source_coverage, coverage_path)
    write_csv(unmapped_foods, missing_path)

    total = len(observations)
    no_axis = len(unmapped_foods)
    summary_path.write_text(
        "# Food Observation to Prediction-Axis Coverage Audit\n\n"
        f"- Source-native food observations audited: {total:,}\n"
        f"- Food observations with no mapped source prediction-axis value: {no_axis:,} "
        f"({100 * no_axis / total:.2f}%)\n"
        "- Missing source measurements are not converted to zero. This report only identifies "
        "records that need acquisition, source-adapter, or axis-mapping review.\n\n"
        "## Interpretation\n\n"
        "A food record can have no mapped target value because its source has not been numerically "
        "integrated, its locally available artifact contains catalogue metadata rather than values, "
        "or its reported components have not yet been mapped to the 386 proposed axes. A source value can "
        "be mapped yet remain outside the current model-label gate; this is reported separately. This report "
        "does not claim that the food has zero composition data.\n",
        encoding="utf-8",
    )
    return coverage_path, missing_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    coverage_path, missing_path = build_audit(args.audit_dir, args.output_dir)
    print(f"Wrote source coverage: {coverage_path}")
    print(f"Wrote unmapped food records: {missing_path}")


if __name__ == "__main__":
    main()

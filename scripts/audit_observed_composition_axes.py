#!/usr/bin/env python3
"""Create an Atlas axis registry containing only axes with source observations.

An observed axis has at least one finite mapped numerical source value with a
status of ``observed``, ``explicit_zero``, or a numeric bracketed estimate.
Missing, censored, range-only, assumed-zero, unresolved-zero, and invalid
records do not establish that an axis has an observation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import sha256_file, write_csv, write_json  # noqa: E402


SOURCE_PANEL = ROOT / "data/processed/proposed_prediction_axis_panel_v2/proposed_prediction_axis_registry.csv"
SOURCE_AUDIT = ROOT / "data/processed/global_frozen_prediction_panel_food_audit_v6"
DEFAULT_OUTPUT = ROOT / "data/processed/audited_prediction_axis_panel_v1"
DEFAULT_REPORT = ROOT / "reports/audited_prediction_axis_panel_v1"
OBSERVED_VALUE_STATUSES = {
    "observed",
    "explicit_zero",
    "bracketed_or_parenthesized_estimate",
}


def build_observed_axis_panel(
    source_panel: Path,
    source_audit: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, int]:
    """Write a new, non-destructive panel after auditing source observations."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audited axis panel: {output_dir}")
    panel = pd.read_csv(source_panel, keep_default_na=False)
    required_panel = {"target_axis_id", "canonical_name", "axis_family", "recommended_training_stage"}
    missing_panel = sorted(required_panel - set(panel.columns))
    if missing_panel:
        raise ValueError(f"Source panel is missing required columns: {missing_panel}")
    if panel.target_axis_id.duplicated().any():
        raise ValueError("Source panel has duplicate target-axis IDs")

    mapped_path = source_audit / "mapped_numeric_target_measurements.csv.gz"
    direct_path = source_audit / "direct_mass_label_candidate_measurements.csv.gz"
    mapped = pd.read_csv(mapped_path, keep_default_na=False, low_memory=False)
    direct = pd.read_csv(direct_path, keep_default_na=False, low_memory=False)
    required_measurement = {
        "target_axis_id", "measurement_id", "source_key", "food_observation_id", "numeric_value", "value_status",
    }
    missing_measurement = sorted(required_measurement - set(mapped.columns))
    if missing_measurement:
        raise ValueError(f"Mapped measurement table is missing required columns: {missing_measurement}")

    mapped["numeric_value"] = pd.to_numeric(mapped.numeric_value, errors="coerce")
    mapped["is_source_observation"] = (
        mapped.numeric_value.notna() & mapped.value_status.isin(OBSERVED_VALUE_STATUSES)
    )
    mapped_summary = (
        mapped.groupby("target_axis_id", as_index=False)
        .agg(
            mapped_record_count=("measurement_id", "size"),
            finite_numeric_record_count=("numeric_value", lambda values: int(values.notna().sum())),
            source_observation_count=("is_source_observation", "sum"),
            source_count=("source_key", "nunique"),
            food_observation_count=("food_observation_id", "nunique"),
            value_statuses=("value_status", lambda values: "; ".join(sorted(set(values.astype(str))))),
        )
    )
    direct_summary = (
        direct.groupby("target_axis_id", as_index=False)
        .agg(
            direct_mass_label_count=("measurement_id", "size"),
            direct_mass_source_count=("source_key", "nunique"),
            direct_mass_food_count=("food_observation_id", "nunique"),
        )
    )
    audit = (
        panel.merge(mapped_summary, how="left", on="target_axis_id", validate="one_to_one")
        .merge(direct_summary, how="left", on="target_axis_id", validate="one_to_one")
    )
    count_columns = [
        "mapped_record_count", "finite_numeric_record_count", "source_observation_count",
        "source_count", "food_observation_count", "direct_mass_label_count",
        "direct_mass_source_count", "direct_mass_food_count",
    ]
    for column in count_columns:
        audit[column] = audit[column].fillna(0).astype(int)
    audit["value_statuses"] = audit.value_statuses.fillna("")
    audit["audit_decision"] = audit.source_observation_count.gt(0).map(
        {True: "retain_has_source_observation", False: "remove_no_source_observation"}
    )
    audit["audit_rule"] = (
        "finite numeric value with status observed, explicit_zero, or bracketed_or_parenthesized_estimate"
    )

    retained = audit.loc[audit.audit_decision.eq("retain_has_source_observation"), panel.columns].copy()
    removed = audit.loc[
        audit.audit_decision.eq("remove_no_source_observation"),
        [
            "target_axis_id", "canonical_name", "axis_family", "recommended_training_stage",
            "mapped_record_count", "finite_numeric_record_count", "source_observation_count",
            "value_statuses", "audit_decision", "audit_rule",
        ],
    ].copy()
    if retained.empty:
        raise ValueError("Axis observation audit removed every target axis")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    write_csv(retained, output_dir / "proposed_prediction_axis_registry.csv")
    write_csv(audit, output_dir / "axis_observation_audit.csv")
    write_csv(removed, output_dir / "removed_zero_observation_axes.csv")
    write_csv(
        retained.groupby(["recommended_training_stage", "axis_family"], as_index=False)
        .agg(axis_count=("target_axis_id", "size")),
        output_dir / "axis_family_summary.csv",
    )
    manifest = {
        "panel_version": "audited_prediction_axis_panel_v1",
        "source_panel": str(source_panel),
        "source_panel_sha256": sha256_file(source_panel),
        "source_audit": str(source_audit),
        "mapped_measurement_sha256": sha256_file(mapped_path),
        "direct_measurement_sha256": sha256_file(direct_path),
        "observation_rule": "A finite numeric mapped source value with status observed, explicit_zero, or bracketed_or_parenthesized_estimate.",
        "retained_axis_count": int(len(retained)),
        "removed_zero_observation_axis_count": int(len(removed)),
        "retained_by_stage": {
            str(stage): int(count)
            for stage, count in retained.recommended_training_stage.value_counts().sort_index().items()
        },
        "removed_by_stage": {
            str(stage): int(count)
            for stage, count in removed.recommended_training_stage.value_counts().sort_index().items()
        },
    }
    write_json(manifest, output_dir / "panel_manifest.json")
    (report_dir / "AXIS_OBSERVATION_AUDIT.md").write_text(
        "# Composition-Axis Observation Audit\n\n"
        "Only axes with at least one finite, mapped source value labelled `observed`, `explicit_zero`, "
        "or `bracketed_or_parenthesized_estimate` are retained in the public Atlas registry. "
        "Missing, censored, range-only, assumed-zero, unresolved-zero, and invalid records do not count as observations.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, default=SOURCE_PANEL)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    manifest = build_observed_axis_panel(
        args.source_panel, args.source_audit, args.output_dir, args.report_dir
    )
    print(manifest)


if __name__ == "__main__":
    main()

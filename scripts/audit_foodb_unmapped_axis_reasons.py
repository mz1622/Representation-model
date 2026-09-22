#!/usr/bin/env python3
"""Explain why FooDB food observations have no mapped proposed-axis value.

The audit is source-native: it inspects the original FooDB Content rows and
the fixed source-axis crosswalk. It neither maps a new chemical automatically
nor treats an unresolved FooDB source ID as a zero value.

Colab:
    python scripts/audit_foodb_unmapped_axis_reasons.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import stable_id, write_csv  # noqa: E402


DEFAULT_AUDIT = ROOT / "data" / "processed" / "global_frozen_prediction_panel_food_audit_v3"
DEFAULT_CONTENT = ROOT / "foodb_2020_04_07_csv" / "Content.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "global_frozen_prediction_panel_food_audit_v3" / "food_axis_coverage"


def _food_observation_ids(chunk: pd.DataFrame) -> pd.Series:
    lineage_food = chunk["orig_food_id"].where(chunk["orig_food_id"].notna(), chunk["food_id"])
    part = chunk["orig_food_part"].fillna("").astype(str).str.strip()
    preparation = chunk["preparation_type"].fillna("").astype(str).str.strip()
    citation = chunk["citation"].fillna("FOODB").astype(str).str.strip()
    keys = citation + "|" + lineage_food.astype(str) + "|" + part + "|" + preparation
    return keys.map(lambda key: stable_id("foodobs", "foodb", key))


def build_audit(audit_dir: Path, content_path: Path, output_dir: Path) -> tuple[Path, Path]:
    if not content_path.exists():
        raise FileNotFoundError(f"FooDB Content export is required: {content_path}")
    observations = pd.read_csv(
        audit_dir / "food_observation_to_exact_name_group.csv.gz",
        usecols=["food_observation_id", "source_key", "source_food_id", "original_name", "part", "processing"],
        keep_default_na=False,
        low_memory=False,
    )
    foodb = observations.loc[observations.source_key.eq("foodb")].copy()
    mapped = pd.read_csv(
        audit_dir / "mapped_numeric_target_measurements.csv.gz",
        usecols=["food_observation_id", "source_key"],
        keep_default_na=False,
        low_memory=False,
    )
    mapped_foods = set(mapped.loc[mapped.source_key.eq("foodb"), "food_observation_id"])
    pending_ids = set(foodb.loc[~foodb.food_observation_id.isin(mapped_foods), "food_observation_id"])
    source_axes = pd.read_csv(
        audit_dir / "global_source_axis_to_frozen_prediction_axis.csv.gz",
        usecols=["source_key", "source_axis_id"],
        keep_default_na=False,
    )
    mapped_source_axes = set(source_axes.loc[source_axes.source_key.eq("foodb"), "source_axis_id"])

    parts = []
    usecols = [
        "food_id", "orig_food_id", "orig_food_part", "citation", "preparation_type",
        "source_type", "source_id", "orig_content",
    ]
    for chunk in pd.read_csv(content_path, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk["food_observation_id"] = _food_observation_ids(chunk)
        chunk = chunk.loc[chunk.food_observation_id.isin(pending_ids)].copy()
        if chunk.empty:
            continue
        chunk["source_axis_id"] = chunk.source_type.astype(str) + ":" + chunk.source_id.astype(str)
        chunk["is_unresolved_component_zero"] = chunk.source_axis_id.eq("Compound:0")
        chunk["is_mapped_source_axis"] = chunk.source_axis_id.isin(mapped_source_axes)
        chunk["has_numeric_raw_value"] = pd.to_numeric(chunk.orig_content, errors="coerce").notna()
        parts.append(chunk.loc[:, [
            "food_observation_id", "source_axis_id", "is_unresolved_component_zero",
            "is_mapped_source_axis", "has_numeric_raw_value",
        ]])
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["food_observation_id", "source_axis_id", "is_unresolved_component_zero", "is_mapped_source_axis", "has_numeric_raw_value"]
    )
    summary = raw.groupby("food_observation_id", as_index=False).agg(
        source_content_rows=("source_axis_id", "size"),
        source_axis_count=("source_axis_id", "nunique"),
        numeric_content_rows=("has_numeric_raw_value", "sum"),
        unresolved_compound_zero_rows=("is_unresolved_component_zero", "sum"),
        mapped_source_axis_rows=("is_mapped_source_axis", "sum"),
    )
    result = foodb.loc[foodb.food_observation_id.isin(pending_ids)].merge(
        summary,
        how="left",
        on="food_observation_id",
        validate="one_to_one",
    )
    for column in (
        "source_content_rows", "source_axis_count", "numeric_content_rows",
        "unresolved_compound_zero_rows", "mapped_source_axis_rows",
    ):
        result[column] = result[column].fillna(0).astype(int)
    result["reason"] = "unmapped_source_components_outside_current_prediction_panel"
    result.loc[result.source_content_rows.eq(0), "reason"] = "no_source_content_row_found"
    result.loc[
        result.source_content_rows.gt(0)
        & result.unresolved_compound_zero_rows.eq(result.source_content_rows),
        "reason",
    ] = "all_source_content_rows_reference_unresolved_FooDB_Compound_0"
    result.loc[
        result.source_content_rows.gt(0)
        & result.mapped_source_axis_rows.gt(0),
        "reason",
    ] = "mapped_source_axis_present_but_no_mapped_numeric_measurement_retained"
    result["policy"] = (
        "No value is inferred. Unresolved FooDB component ID 0 is an invalid component reference, "
        "not an analyte identity or a numeric zero."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "foodb_foods_without_mapped_prediction_axes_explained.csv.gz"
    count_path = output_dir / "foodb_unmapped_axis_reason_summary.csv"
    write_csv(result.sort_values(["reason", "original_name"], kind="stable"), detail_path)
    write_csv(
        result.groupby("reason", as_index=False).agg(
            food_observations=("food_observation_id", "size"),
            median_raw_content_rows=("source_content_rows", "median"),
            median_unresolved_compound_zero_rows=("unresolved_compound_zero_rows", "median"),
        ).sort_values("food_observations", ascending=False, kind="stable"),
        count_path,
    )
    return detail_path, count_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--content", type=Path, default=DEFAULT_CONTENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    detail, counts = build_audit(args.audit_dir, args.content, args.output_dir)
    print(f"Wrote FooDB unmapped-food detail: {detail}")
    print(f"Wrote FooDB reason summary: {counts}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the frozen V8/V2 corpus against the source-aware V9 training plan.

The audit is read-only. It does not alter data, splits, normalizers, masks, or
model outputs. Exact-name groups are treated only as cross-source comparison
candidates; they are not asserted to be reviewed biological food concepts.

Colab:
    !python scripts/audit_v8_source_aware_source_free.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
DEFAULT_TRAINER = ROOT / "scripts/train_global_foodnutrigpt_v8_single_stage.py"
DEFAULT_OUTPUT_DIR = ROOT / "reports/v8_source_aware_source_free_audit_2026_09_22_v4"


PROFILE_COLUMNS = [
    "profile_id", "source_key", "exact_name_group_id", "partition", "food_text",
]
TOKEN_COLUMNS = [
    "profile_id", "source_key", "target_axis_id", "canonical_name", "loss_group",
    "measurement_id", "normalized_value_g_per_100g", "log1p_value", "value_status",
    "loss_eligible", "within_profile_axis_observation_count",
    "within_profile_axis_loss_weight",
]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def bool_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].astype(bool).sum())


def audit_code(trainer: Path) -> pd.DataFrame:
    source = trainer.read_text(encoding="utf-8")
    checks = [
        (
            "source_embedding_enters_encoder",
            'self.source_embedding(batch["source"]).unsqueeze(1)' in source,
            "Current V8 encoder concatenates a profile source embedding into every training sequence.",
            "Fail for V9: source must be confined to a training-only calibration branch.",
        ),
        (
            "validation_forces_unknown_source",
            "valid_dataset = MaskedProfileDataset(corpus, valid_positions, config, training=False, force_unknown_source=True)" in source,
            "Validation replaces source with UNKNOWN before encoder input.",
            "Pass for source-free validation input, but labels are still source-native profiles.",
        ),
        (
            "test_forces_unknown_source",
            "force_unknown_source=True, cover_all_evaluation_families=True" in source,
            "Complete-test input replaces source with UNKNOWN before encoder input.",
            "Pass for source-free test input, but not for a source-free test target.",
        ),
        (
            "profile_axis_only_loss_weights",
            "within_profile_axis_loss_weight" in source,
            "Current loss weights only duplicate measurements within one source-native profile-axis cell.",
            "Fail for V9: cross-source observations of an exact-name candidate are not equalized together.",
        ),
        (
            "text_only_augmentation_present",
            "text_only_probability" in source,
            "The trainer supports text-only augmentation by masking all numeric context for a sample.",
            "This is an auxiliary zero-shot task, not removal of food text embeddings.",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "present_in_v8", "observed_behavior", "v9_assessment"])


def source_cell_table(tokens: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate records within source/profile before cross-source comparison."""
    joined = tokens.merge(
        profiles[["profile_id", "source_key", "exact_name_group_id", "partition"]],
        on=["profile_id", "source_key"], how="inner", validate="many_to_one",
    )
    if len(joined) != len(tokens):
        raise ValueError("Every token must join to exactly one active food profile.")
    if joined["normalized_value_g_per_100g"].lt(0).any():
        raise ValueError("Negative normalized mass value found.")
    if not joined["value_status"].isin(["observed", "explicit_zero"]).all():
        raise ValueError("Tokens include non-observed value statuses.")

    return (
        joined.groupby(
            [
                "exact_name_group_id", "target_axis_id", "canonical_name", "loss_group",
                "source_key", "partition", "profile_id",
            ],
            as_index=False,
        )
        .agg(
            source_profile_axis_measurements=("measurement_id", "size"),
            source_profile_axis_value_g_per_100g=("normalized_value_g_per_100g", "median"),
            source_profile_axis_log1p=("log1p_value", "median"),
            source_profile_axis_loss_weight_sum=("within_profile_axis_loss_weight", "sum"),
        )
    )


def cross_source_candidate_cells(source_cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact-name candidates without asserting identity equivalence."""
    per_source = (
        source_cells.groupby(
            ["exact_name_group_id", "target_axis_id", "canonical_name", "loss_group", "source_key"],
            as_index=False,
        )
        .agg(
            source_profiles=("profile_id", "nunique"),
            source_value_g_per_100g=("source_profile_axis_value_g_per_100g", "median"),
            source_log1p=("source_profile_axis_log1p", "median"),
        )
    )
    candidates = (
        per_source.groupby(
            ["exact_name_group_id", "target_axis_id", "canonical_name", "loss_group"],
            as_index=False,
        )
        .agg(
            independent_sources=("source_key", "nunique"),
            source_profiles=("source_profiles", "sum"),
            source_value_min_g_per_100g=("source_value_g_per_100g", "min"),
            source_value_median_g_per_100g=("source_value_g_per_100g", "median"),
            source_value_max_g_per_100g=("source_value_g_per_100g", "max"),
            source_log1p_min=("source_log1p", "min"),
            source_log1p_median=("source_log1p", "median"),
            source_log1p_max=("source_log1p", "max"),
        )
    )
    candidates["source_log1p_span"] = candidates["source_log1p_max"] - candidates["source_log1p_min"]
    candidates["cross_source_candidate"] = candidates["independent_sources"].ge(2)
    return candidates.sort_values(
        ["cross_source_candidate", "source_log1p_span"], ascending=[False, False], kind="stable",
    )


def markdown_report(
    manifest: dict[str, object], code_checks: pd.DataFrame, source_summary: pd.DataFrame,
    conflict_summary: pd.DataFrame, split_summary: pd.DataFrame,
) -> str:
    def table(frame: pd.DataFrame) -> str:
        def cell(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value).replace("|", "\\|")

        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(cell(value) for value in row) + " |")
        return "\n".join(lines)

    return f"""# V8 Source-Aware / Source-Free Audit

## Scope

This read-only audit examines `{manifest['dataset_version']}` against the planned
V9 protocol: **source-aware training with source-free inference and source-free
benchmark inputs**. The current V8/V2 corpus remains an immutable source-native
baseline. Exact-name groups in this report are only candidates for a repeated
food identity; no row is treated as a confirmed cross-source merge.

## Active Corpus

- Food profiles: {manifest['food_profiles']:,}
- Source-native numeric tokens: {manifest['source_native_tokens']:,}
- Active axes: {manifest['active_axes']:,}
- Masked-loss axes: {manifest['loss_axes']:,}
- Complete-test profiles: {manifest['complete_test_profiles']:,}
- Complete-test axes: {manifest['complete_test_axes']:,}

All active tokens are non-negative values expressed as `g/100 g`; missing values
are absent tokens and explicit zeros remain observed tokens.

## Split Audit

{table(split_summary)}

The current split has no exact-name-group overlap among active train,
validation, and complete-test partitions. The separately reported excluded
Foundation-overlap rows are not benchmark profiles. Exact-name grouping still
does not certify semantic equivalence for different food names, translations,
or food facets.

## Cross-Source Candidate Audit

{table(source_summary)}

`cross_source_candidate` means that an exact-name candidate has at least two
source keys for the same axis after within-profile duplicate records are reduced.
It is an audit signal, not permission to pool values. `source_log1p_span` is the
difference between the highest and lowest source median on `log1p(g/100 g)`.

{table(conflict_summary)}

## Training-Code Audit

{table(code_checks)}

## Required V9 Changes

1. Keep raw source-native observations and provenance in the build layer.
2. Remove the source embedding from the Transformer sequence. Source must not
   shape the food embedding used at inference.
3. Add a training-only, zero-centred `source × axis` residual calibration head:
   `prediction_known_source = base_prediction + source_axis_residual`.
4. At inference, validation, and complete testing, emit the source-free base
   prediction only; source must not be an input feature.
5. For a repeated exact-name candidate and axis, make each independent source
   contribute equal total loss. Do not let duplicate records or a high-volume
   source multiply its supervision.
6. Keep the test split at the exact-name-group level. For source-free testing,
   generate one prediction per food-axis and aggregate its error equally across
   available raw source labels offline. Source identity must not appear in the
   test tensor or model output.
7. Preserve text-only training as a separately reported zero-shot objective.
   It must not be described as removing food text from the model.

## Interpretation

V8/V2 is a useful frozen baseline for source-native supervision with unknown
source evaluation. It is not yet a source-aware/source-free benchmark because a
source embedding enters the training encoder and each source observation remains
an independent profile-level target. V9 should be built in a new versioned
directory; it must not overwrite V8/V2 data, masks, or results.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data_dir, split_dir, trainer, output_dir = (
        args.data_dir.resolve(), args.split_dir.resolve(), args.trainer.resolve(), args.output_dir.resolve(),
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_dir}")

    profiles = pd.read_csv(data_dir / "food_profiles.csv.gz", usecols=PROFILE_COLUMNS, low_memory=False)
    tokens = pd.read_csv(data_dir / "source_native_axis_tokens.csv.gz", usecols=TOKEN_COLUMNS, low_memory=False)
    axes = pd.read_csv(data_dir / "axis_registry.csv", low_memory=False)
    manifest = json.loads((data_dir / "build_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((split_dir / "splits.json").read_text(encoding="utf-8"))

    if profiles["profile_id"].duplicated().any():
        raise ValueError("Food profiles are not unique by profile_id.")
    if not set(splits["train"]).issubset(set(profiles["profile_id"])):
        raise ValueError("Split contains a train profile absent from the corpus.")
    if int(manifest["food_profiles"]) != len(profiles) or int(manifest["source_native_tokens"]) != len(tokens):
        raise ValueError("Build manifest counts do not match the active corpus files.")

    active_partitions = {"train", "validation", "test_complete_axis_panel"}
    active_profiles = profiles[profiles["partition"].isin(active_partitions)].copy()
    active_partition_per_group = active_profiles.groupby("exact_name_group_id")["partition"].nunique()
    mixed_active_partition_groups = int(active_partition_per_group.gt(1).sum())
    excluded_overlap_groups = set(profiles.loc[
        profiles["partition"].eq("excluded_foundation_exact_name_overlap"), "exact_name_group_id"
    ]) & set(active_profiles["exact_name_group_id"])
    source_cells = source_cell_table(tokens, profiles)
    candidates = cross_source_candidate_cells(source_cells)
    cross_source = candidates[candidates["cross_source_candidate"]].copy()

    source_summary = pd.DataFrame([
        {
            "source_native_profiles": int(len(profiles)),
            "source_native_tokens": int(len(tokens)),
            "sources": int(profiles["source_key"].nunique()),
            "exact_name_groups": int(profiles["exact_name_group_id"].nunique()),
            "exact_name_groups_with_multiple_profiles": int(
                profiles.groupby("exact_name_group_id")["profile_id"].nunique().gt(1).sum()
            ),
            "candidate_food_axis_cells": int(len(candidates)),
            "cross_source_candidate_food_axis_cells": int(len(cross_source)),
            "cross_source_candidate_fraction": float(len(cross_source) / max(len(candidates), 1)),
        }
    ])
    conflict_summary = (
        cross_source.groupby("loss_group", as_index=False)
        .agg(
            candidate_food_axis_cells=("target_axis_id", "size"),
            axes=("target_axis_id", "nunique"),
            exact_name_groups=("exact_name_group_id", "nunique"),
            median_independent_sources=("independent_sources", "median"),
            median_source_log1p_span=("source_log1p_span", "median"),
            p90_source_log1p_span=("source_log1p_span", lambda values: float(np.quantile(values, 0.90))),
            max_source_log1p_span=("source_log1p_span", "max"),
        )
        .sort_values("loss_group", kind="stable")
    )
    if conflict_summary.empty:
        conflict_summary = pd.DataFrame(columns=[
            "loss_group", "candidate_food_axis_cells", "axes", "exact_name_groups",
            "median_independent_sources", "median_source_log1p_span", "p90_source_log1p_span",
            "max_source_log1p_span",
        ])
    split_summary = (
        profiles.groupby("partition", as_index=False)
        .agg(
            source_native_profiles=("profile_id", "nunique"),
            exact_name_groups=("exact_name_group_id", "nunique"),
            sources=("source_key", "nunique"),
            food_texts=("food_text", "nunique"),
        )
        .sort_values("partition", kind="stable")
    )
    split_summary["mixed_active_partition_exact_name_groups"] = mixed_active_partition_groups
    split_summary["excluded_overlap_exact_name_groups"] = len(excluded_overlap_groups)
    code_checks = audit_code(trainer)

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(source_summary, output_dir / "source_native_corpus_summary.csv")
    write_csv(split_summary, output_dir / "split_audit.csv")
    write_csv(conflict_summary, output_dir / "cross_source_candidate_summary.csv")
    write_csv(cross_source, output_dir / "cross_source_exact_name_candidate_cells.csv.gz")
    write_csv(code_checks, output_dir / "training_code_audit.csv")

    audit_manifest = {
        "audit_name": "v8_source_aware_source_free_audit_2026_09_22",
        "dataset_version": manifest["dataset_version"],
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "trainer": str(trainer),
        "food_profiles": int(len(profiles)),
        "source_native_tokens": int(len(tokens)),
        "active_axes": int(len(axes)),
        "loss_axes": int(bool_count(axes, "loss_eligible")),
        "complete_test_profiles": int(manifest["complete_test_profiles"]),
        "complete_test_axes": int(manifest["complete_test_axes"]),
        "mixed_active_partition_exact_name_groups": mixed_active_partition_groups,
        "excluded_overlap_exact_name_groups": len(excluded_overlap_groups),
        "cross_source_candidate_food_axis_cells": int(len(cross_source)),
        "definition": "Exact-name candidate groups are audit candidates only; no cross-source identity merge is asserted.",
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")
    (output_dir / "V8_SOURCE_AWARE_SOURCE_FREE_AUDIT.md").write_text(
        markdown_report(audit_manifest, code_checks, source_summary, conflict_summary, split_summary),
        encoding="utf-8",
    )
    print(json.dumps(audit_manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the current global axis panel and plan a source-isolated test.

This is a read-only audit.  It neither edits the Atlas nor creates a train/
validation/test split.  The output distinguishes source-native pretraining
eligibility from eligibility for a source-isolated USDA Foundation test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# These are minimum *food-observation* counts, never raw measurement counts.
# They are eligibility gates for a regression loss, not scientific inclusion
# criteria: a scientifically valid but sparse axis remains in the registry.
MIN_PRETRAIN_DIRECT_FOODS = 20
MIN_PRETRAIN_POSITIVE_FOODS = 20
MIN_EXTERNAL_TEST_FOODS = 30
MIN_EXTERNAL_TEST_POSITIVE_FOODS = 10
MIN_EXTERNAL_TRAIN_FOODS = 100
MIN_EXTERNAL_TRAIN_POSITIVE_FOODS = 20


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(str).str.casefold().isin({"true", "1", "yes"})


def _food_axis_summary(measurements: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Summarize at source-native food-axis granularity, not record granularity."""
    if measurements.empty:
        return pd.DataFrame(columns=["target_axis_id"])
    values = measurements.copy()
    values["positive"] = values["normalized_value_g_per_100g"].gt(0)
    values["explicit_zero"] = values["value_status"].eq("explicit_zero")
    grouped = values.groupby("target_axis_id", dropna=False)
    summary = grouped.agg(
        **{
            f"{prefix}_measurement_records": ("measurement_id", "size"),
            f"{prefix}_food_observations": ("food_observation_id", "nunique"),
            f"{prefix}_exact_name_groups": ("exact_name_group_id", "nunique"),
            f"{prefix}_sources": ("source_key", "nunique"),
            f"{prefix}_positive_records": ("positive", "sum"),
            f"{prefix}_explicit_zero_records": ("explicit_zero", "sum"),
            f"{prefix}_positive_foods": (
                "food_observation_id", lambda x: x[values.loc[x.index, "positive"]].nunique()
            ),
            f"{prefix}_zero_foods": (
                "food_observation_id", lambda x: x[values.loc[x.index, "explicit_zero"]].nunique()
            ),
            f"{prefix}_min_g_per_100g": ("normalized_value_g_per_100g", "min"),
            f"{prefix}_median_g_per_100g": ("normalized_value_g_per_100g", "median"),
            f"{prefix}_max_g_per_100g": ("normalized_value_g_per_100g", "max"),
            f"{prefix}_sd_log1p_g_per_100g": (
                "normalized_value_g_per_100g", lambda x: float(np.std(np.log1p(x), ddof=0))
            ),
        }
    ).reset_index()
    return summary


def _fill_counts(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for prefix in prefixes:
        for column in result.columns:
            if column.startswith(prefix + "_") and (
                column.endswith("records")
                or column.endswith("observations")
                or column.endswith("groups")
                or column.endswith("sources")
                or column.endswith("foods")
            ):
                result[column] = result[column].fillna(0).astype(int)
    return result


def _pretraining_role(row: pd.Series) -> tuple[str, str]:
    if row["mapped_numeric_measurement_records"] == 0:
        return "exclude_no_numeric_mapping", "No numeric source measurement maps to this frozen axis."
    if row["direct_mass_measurement_records"] == 0:
        return "exclude_non_mass_only", "Mapped values exist but none are direct fresh-weight mass values."
    if row["direct_mass_food_observations"] < MIN_PRETRAIN_DIRECT_FOODS:
        return "context_only_sparse", f"Fewer than {MIN_PRETRAIN_DIRECT_FOODS} direct-mass food observations."
    if row["direct_mass_positive_foods"] < MIN_PRETRAIN_POSITIVE_FOODS:
        return "context_only_degenerate", f"Fewer than {MIN_PRETRAIN_POSITIVE_FOODS} positive food observations; regression would be dominated by zeros."
    if not np.isfinite(row["direct_mass_sd_log1p_g_per_100g"]) or row["direct_mass_sd_log1p_g_per_100g"] == 0:
        return "context_only_degenerate", "The direct-mass labels have zero log-scale variation."
    if row["direct_mass_sources"] == 1:
        return "pretrain_loss_single_source", "Eligible for masked loss, but observed in one source only; no cross-source claim is supported."
    return "pretrain_loss_eligible", "Direct-mass support and positive-value variation meet the pre-registered regression gate."


def _foundation_test_role(row: pd.Series) -> tuple[str, str]:
    if not row["pretraining_role"].startswith("pretrain_loss"):
        return "not_testable_pretraining_ineligible", "Not eligible for the main pretraining regression loss."
    if row["foundation_food_observations"] < MIN_EXTERNAL_TEST_FOODS:
        return "not_testable_foundation_support", f"USDA Foundation has fewer than {MIN_EXTERNAL_TEST_FOODS} direct-mass foods."
    if row["foundation_positive_foods"] < MIN_EXTERNAL_TEST_POSITIVE_FOODS:
        return "not_testable_foundation_positive_support", f"USDA Foundation has fewer than {MIN_EXTERNAL_TEST_POSITIVE_FOODS} positive foods."
    if row["nonfoundation_food_observations"] < MIN_EXTERNAL_TRAIN_FOODS:
        return "not_testable_train_support", f"Non-Foundation training pool has fewer than {MIN_EXTERNAL_TRAIN_FOODS} direct-mass foods."
    if row["nonfoundation_positive_foods"] < MIN_EXTERNAL_TRAIN_POSITIVE_FOODS:
        return "not_testable_train_positive_support", f"Non-Foundation training pool has fewer than {MIN_EXTERNAL_TRAIN_POSITIVE_FOODS} positive foods."
    return "foundation_external_test_eligible", "Supports a fully source-isolated Foundation evaluation."


def _source_test_role(row: pd.Series) -> str:
    """Return generic test eligibility for a source withheld before training."""
    if not row["pretraining_regression_eligible"]:
        return "not_testable_pretraining_ineligible"
    if row["test_source_food_observations"] < MIN_EXTERNAL_TEST_FOODS:
        return "not_testable_test_source_support"
    if row["test_source_positive_foods"] < MIN_EXTERNAL_TEST_POSITIVE_FOODS:
        return "not_testable_test_source_positive_support"
    if row["remaining_train_food_observations"] < MIN_EXTERNAL_TRAIN_FOODS:
        return "not_testable_remaining_train_support"
    if row["remaining_train_positive_foods"] < MIN_EXTERNAL_TRAIN_POSITIVE_FOODS:
        return "not_testable_remaining_train_positive_support"
    return "source_isolated_test_eligible"


def _source_axis_support(direct: pd.DataFrame) -> pd.DataFrame:
    """Count each source-native food once per axis, including zero/positive state."""
    cell = (
        direct.assign(
            _positive=direct["normalized_value_g_per_100g"].gt(0),
            _zero=direct["value_status"].eq("explicit_zero"),
        )
        .groupby(["source_key", "target_axis_id", "food_observation_id"], as_index=False)
        .agg(_positive=("_positive", "max"), _zero=("_zero", "max"))
    )
    return (
        cell.groupby(["source_key", "target_axis_id"], as_index=False)
        .agg(
            test_source_food_observations=("food_observation_id", "nunique"),
            test_source_positive_foods=("_positive", "sum"),
            test_source_zero_foods=("_zero", "sum"),
        )
    )


def _join_unique(series: pd.Series) -> str:
    values = sorted({str(value).strip() for value in series.dropna() if str(value).strip()})
    return " | ".join(values)


def _observed_evidence_summary(mapped: pd.DataFrame) -> pd.DataFrame:
    """Retain source-level facts explaining why an observed axis is not targetable."""
    return (
        mapped.groupby("target_axis_id", as_index=False)
        .agg(
            observed_source_keys=("source_key", _join_unique),
            observed_raw_units=("raw_unit", _join_unique),
            observed_raw_bases=("raw_basis", _join_unique),
            observed_measurement_modalities=("measurement_modality", _join_unique),
            observed_conversion_statuses=("conversion_status", _join_unique),
            observed_mapping_statuses=("mapping_status", _join_unique),
        )
    )


def _write_observed_non_targets(output_dir: Path, result: pd.DataFrame, mapped: pd.DataFrame) -> None:
    """Write the complete, reviewable list of observed but non-target axes."""
    evidence = _observed_evidence_summary(mapped)
    non_targets = result.loc[
        result["mapped_numeric_measurement_records"].gt(0)
        & ~result["pretraining_regression_eligible"]
    ].merge(evidence, on="target_axis_id", how="left", validate="one_to_one")
    non_targets["recommended_model_treatment"] = np.select(
        [
            non_targets["pretraining_role"].eq("context_only_sparse"),
            non_targets["pretraining_role"].eq("context_only_degenerate"),
            non_targets["pretraining_role"].eq("exclude_non_mass_only"),
        ],
        [
            "Retain only as an observed context token; never sample it as a masked target.",
            "Exclude from masked loss because it has insufficient non-zero variation; do not use it to claim prediction capability.",
            "Exclude from the direct-mass model; retain in the Atlas pending a separate, compatible measurement-modality task.",
        ],
        default="Review required.",
    )
    columns = [
        "target_axis_id", "canonical_name", "recommended_training_stage", "axis_family",
        "pretraining_role", "pretraining_decision_reason", "recommended_model_treatment",
        "mapped_numeric_measurement_records", "mapped_numeric_food_observations", "mapped_numeric_sources",
        "direct_mass_measurement_records", "direct_mass_food_observations", "direct_mass_sources",
        "direct_mass_positive_foods", "direct_mass_zero_foods", "direct_mass_min_g_per_100g",
        "direct_mass_median_g_per_100g", "direct_mass_max_g_per_100g", "direct_mass_sd_log1p_g_per_100g",
        "observed_source_keys", "observed_raw_units", "observed_raw_bases",
        "observed_measurement_modalities", "observed_conversion_statuses", "observed_mapping_statuses",
        "measurement_modality_required", "selection_reason", "scientific_basis",
    ]
    non_targets = non_targets[columns].sort_values(
        ["pretraining_role", "recommended_training_stage", "axis_family", "canonical_name"], kind="stable"
    )
    non_targets.to_csv(output_dir / "observed_non_target_axes.csv", index=False)

    short_columns = [
        "canonical_name", "recommended_training_stage", "axis_family", "pretraining_role",
        "mapped_numeric_food_observations", "direct_mass_food_observations", "direct_mass_positive_foods",
        "direct_mass_zero_foods", "observed_source_keys", "observed_raw_units", "pretraining_decision_reason",
    ]
    lines = [
        "# Observed Axes Excluded From the Current Prediction Target Set",
        "",
        "This document lists every frozen scientific axis with at least one mapped numeric observation that is not eligible for the current direct-mass masked-regression loss.",
        "",
        "## Decision Rules",
        "",
        f"- `context_only_sparse`: fewer than {MIN_PRETRAIN_DIRECT_FOODS} source-native foods with compatible direct-mass values.",
        f"- `context_only_degenerate`: fewer than {MIN_PRETRAIN_POSITIVE_FOODS} positive foods or zero log-scale variation.",
        "- `exclude_non_mass_only`: observed values exist, but none are compatible fresh-weight direct mass labels; they must not be mixed with g/100 g regression.",
        "",
    ]
    for role, group in non_targets.groupby("pretraining_role", sort=False):
        lines.extend([f"## {role} ({len(group)} axes)", ""])
        fields = short_columns
        header = "| " + " | ".join(fields) + " |"
        divider = "| " + " | ".join("---" for _ in fields) + " |"
        lines.extend([header, divider])
        for row in group[fields].fillna("").astype(str).itertuples(index=False, name=None):
            lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
        lines.append("")
    lines.extend([
        "The CSV companion contains raw basis, conversion status, mapping status, scientific selection basis, and all numerical support fields.",
        "",
    ])
    (output_dir / "OBSERVED_NON_TARGET_AXES.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown(output: Path, summary: dict[str, int], roles: pd.DataFrame,
              family_summary: pd.DataFrame, source_test_summary: pd.DataFrame,
              source_audit_version: str) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a compact Markdown table without requiring optional tabulate."""
        columns = list(frame.columns)
        rows = frame.fillna("").astype(str).values.tolist()
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows]
        return "\n".join([header, divider, *body])

    role_rows = roles.groupby("pretraining_role", sort=False).size().reset_index(name="axes")
    test_rows = roles.groupby("foundation_test_role", sort=False).size().reset_index(name="axes")
    lines = [
        "# Current Axis Training Audit and Independent-Test Plan",
        "",
        "## Scope",
        "",
        f"This is a read-only analysis of `{source_audit_version}`. "
        "It does not alter an axis, select a source value, construct a split, or train a model.",
        "",
        "The frozen registry's historical `recommended_training_stage` is retained only as a scientific grouping. "
        "The current model uses one training run with two jointly optimized macro-axis losses, `nutrition` and "
        "`food_metabolome`; it does not transfer a checkpoint between sequential stages.",
        "",
        "## Headline Counts",
        "",
        *[f"- {label}: {value:,}" for label, value in summary.items()],
        "",
        "## Axis Eligibility",
        "",
        "Scientific inclusion and numerical training eligibility are distinct. An axis can remain scientifically important "
        "while being excluded from this version's numerical loss because it has no compatible mass labels or cannot "
        "support a stable train-only scale.",
        "",
        markdown_table(role_rows),
        "",
        "### USDA Foundation External-Test Eligibility",
        "",
        "Foundation is proposed as a source-isolated test source: it must be entirely absent from tokenizer fitting, "
        "normalization fitting, pretraining, model selection, augmentation, and hyperparameter selection. "
        "A testable axis requires at least 30 Foundation food observations, 10 positive Foundation foods, 100 "
        "non-Foundation training foods, and 20 positive non-Foundation foods. These gates assess estimation power, "
        "not biological importance.",
        "",
        markdown_table(test_rows),
        "",
        "## Recommended Three-Way Protocol",
        "",
        "1. **Training corpus:** all source-native food observations except USDA Foundation and the later frozen food-family test blocks. Each source record remains a separate profile; missing is absent, zero is observed zero.",
        "2. **Validation corpus:** grouped food-family blocks sampled only from the non-Foundation training sources. Use this for early stopping, loss weights, architecture, and all baseline tuning.",
        "3. **Independent test A, source isolated:** every USDA Foundation food observation is excluded before any fitting. Evaluate only the axes marked `foundation_external_test_eligible` and keep the panel sealed until one final model is selected.",
        "4. **Independent test B, food-family isolated:** reserve whole duplicate/lineage/near-duplicate blocks from all non-Foundation sources before fitting. This tests unseen foods while retaining broad axis coverage; sources are seen, but test food profiles and labels are not.",
        "",
        "The two tests answer different questions. Test A measures source/domain transfer. Test B measures generalization to unseen foods. Neither test row, text, numeric label, scaler contribution, or augmentation may enter pretraining.",
        "",
        "## Split Blocking Rules",
        "",
        "- The atomic split unit is a connected food block, not a row: exact-name groups, stable source lineage, FoodOn/FoodEx2/LanguaL identity where available, and conservative high-similarity name candidates are connected for blocking only.",
        "- Candidate similarity never merges foods or values. It only prevents potential duplicates from crossing partitions.",
        "- Every source-native observation linked to a held food block is held. Multiple source values remain multiple test labels; they are not averaged.",
        "- Axis vocabulary and authoritative names may be fixed from the registry, but all fitted statistics, text preprocessing learned from data, normalizers, samplers, and checkpoints use training rows only.",
        "- Mask families must be sampled after partitioning. Aggregate and child axes in the same algebraic family are jointly hidden whenever leaving one visible would deterministically reveal another.",
        "",
        "## Evaluation Unit",
        "",
        "For source-native test labels, score each observed `(source, food observation, axis)` cell, report source-stratified results, and use a food-block cluster bootstrap. Do not manufacture a pooled cross-source 'truth'. For Foundation Test A, one source provides the reference labels, which makes the target unambiguous while still testing external transfer.",
        "",
        "## Axis-Family Summary",
        "",
        markdown_table(family_summary),
        "",
        "The complete per-axis evidence and eligibility decision is in `axis_training_eligibility.csv`.",
        "",
        "## Source-Isolated Test Feasibility",
        "",
        "The following table is a pre-blocking feasibility screen. A final split must also remove every cross-source exact-name, "
        "formal-lineage, and conservative near-duplicate block connected to the held source. Therefore final training support will "
        "be lower than shown here.",
        "",
        markdown_table(source_test_summary),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "data/processed/global_frozen_prediction_panel_food_audit_v8",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/global_axis_training_test_audit_2026_09_19",
    )
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")

    source_manifest = json.loads((audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    source_audit_version = str(source_manifest["audit_version"])
    registry = pd.read_csv(audit_dir / "frozen_prediction_axis_registry.csv", keep_default_na=False)
    mapped = pd.read_csv(audit_dir / "mapped_numeric_target_measurements.csv.gz", low_memory=False)
    direct = pd.read_csv(audit_dir / "direct_mass_label_candidate_measurements.csv.gz", low_memory=False)
    # A mapped numeric expression may be energy, an activity equivalent, or a
    # percentage and therefore have no g/100 g conversion.  Preserve it here
    # to distinguish "no numerical evidence" from "not compatible with the
    # main direct-mass regression task".  Its numeric value is only used for
    # presence/positive counts, never reported as a mass distribution.
    mapped["numeric_value"] = pd.to_numeric(mapped["numeric_value"], errors="coerce")
    mapped = mapped.dropna(subset=["numeric_value"]).copy()
    mapped["normalized_value_g_per_100g"] = mapped["numeric_value"]
    direct["normalized_value_g_per_100g"] = pd.to_numeric(
        direct["normalized_value_g_per_100g"], errors="coerce"
    )
    direct = direct.dropna(subset=["normalized_value_g_per_100g"]).copy()

    all_mapped = _food_axis_summary(mapped, "mapped_numeric")
    all_direct = _food_axis_summary(direct, "direct_mass")
    foundation = _food_axis_summary(direct[direct["source_key"].eq("usda_foundation")], "foundation")
    nonfoundation = _food_axis_summary(direct[~direct["source_key"].eq("usda_foundation")], "nonfoundation")
    result = registry.merge(all_mapped, on="target_axis_id", how="left", validate="one_to_one")
    for summary in (all_direct, foundation, nonfoundation):
        result = result.merge(summary, on="target_axis_id", how="left", validate="one_to_one")
    result = _fill_counts(result, ["mapped_numeric", "direct_mass", "foundation", "nonfoundation"])

    roles = result.apply(_pretraining_role, axis=1, result_type="expand")
    result[["pretraining_role", "pretraining_decision_reason"]] = roles
    test_roles = result.apply(_foundation_test_role, axis=1, result_type="expand")
    result[["foundation_test_role", "foundation_test_decision_reason"]] = test_roles
    result["pretraining_regression_eligible"] = result["pretraining_role"].str.startswith("pretrain_loss")
    result["foundation_external_test_eligible"] = result["foundation_test_role"].eq(
        "foundation_external_test_eligible"
    )

    # Screen every source as a potential completely unseen test domain. This
    # does not create a split and deliberately does not yet remove name/lineage
    # blocks shared with the candidate test source.
    source_support = _source_axis_support(direct)
    source_keys = sorted(source_support["source_key"].unique())
    source_grid = (
        pd.MultiIndex.from_product([source_keys, result["target_axis_id"]], names=["source_key", "target_axis_id"])
        .to_frame(index=False)
        .merge(source_support, on=["source_key", "target_axis_id"], how="left")
        .merge(
            result[[
                "target_axis_id", "canonical_name", "recommended_training_stage", "axis_family",
                "pretraining_regression_eligible", "direct_mass_food_observations", "direct_mass_positive_foods",
            ]],
            on="target_axis_id", how="left", validate="many_to_one",
        )
    )
    for column in ["test_source_food_observations", "test_source_positive_foods", "test_source_zero_foods"]:
        source_grid[column] = source_grid[column].fillna(0).astype(int)
    source_grid["remaining_train_food_observations"] = (
        source_grid["direct_mass_food_observations"] - source_grid["test_source_food_observations"]
    )
    source_grid["remaining_train_positive_foods"] = (
        source_grid["direct_mass_positive_foods"] - source_grid["test_source_positive_foods"]
    )
    source_grid["source_test_role"] = source_grid.apply(_source_test_role, axis=1)
    source_grid["source_isolated_test_eligible"] = source_grid["source_test_role"].eq(
        "source_isolated_test_eligible"
    )
    source_test_summary = (
        source_grid.groupby("source_key", as_index=False)
        .agg(
            direct_mass_axes_available=("test_source_food_observations", lambda x: int(x.gt(0).sum())),
            source_isolated_test_axes=("source_isolated_test_eligible", "sum"),
            nutrition_source_isolated_test_axes=(
                "source_isolated_test_eligible",
                lambda x: int(x[source_grid.loc[x.index, "recommended_training_stage"].eq("Stage 1 nutrition composition")].sum()),
            ),
            food_metabolome_source_isolated_test_axes=(
                "source_isolated_test_eligible",
                lambda x: int(x[source_grid.loc[x.index, "recommended_training_stage"].eq("Stage 2 food metabolome")].sum()),
            ),
        )
        .sort_values(["source_isolated_test_axes", "source_key"], ascending=[False, True], kind="stable")
    )
    result = result.sort_values(["recommended_training_stage", "axis_family", "canonical_name"], kind="stable")

    result["loss_group"] = np.where(
        result["recommended_training_stage"].eq("Stage 1 nutrition composition"),
        "nutrition",
        "food_metabolome",
    )
    family_summary = (
        result.groupby(["loss_group", "axis_family"], dropna=False)
        .agg(
            registered_axes=("target_axis_id", "size"),
            mapped_numeric_axes=("mapped_numeric_measurement_records", lambda x: int(x.gt(0).sum())),
            direct_mass_axes=("direct_mass_measurement_records", lambda x: int(x.gt(0).sum())),
            pretraining_loss_axes=("pretraining_regression_eligible", "sum"),
            foundation_external_test_axes=("foundation_external_test_eligible", "sum"),
        )
        .reset_index()
        .sort_values(["loss_group", "axis_family"], kind="stable")
    )
    summary = {
        "Registered frozen candidate axes": len(result),
        "Axes with any mapped numeric value": int(result.mapped_numeric_measurement_records.gt(0).sum()),
        "Axes with any compatible direct-mass value": int(result.direct_mass_measurement_records.gt(0).sum()),
        "Axes eligible for source-native pretraining loss": int(result.pretraining_regression_eligible.sum()),
        "Axes eligible for Foundation external test": int(result.foundation_external_test_eligible.sum()),
        "Mapped numeric source-native measurements": len(mapped),
        "Direct-mass source-native measurements": len(direct),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(output_dir / "axis_training_eligibility.csv", index=False)
    family_summary.to_csv(output_dir / "axis_family_training_summary.csv", index=False)
    source_grid.to_csv(output_dir / "axis_source_isolated_test_eligibility.csv", index=False)
    source_test_summary.to_csv(output_dir / "source_isolated_test_feasibility.csv", index=False)
    _write_observed_non_targets(output_dir, result, mapped)
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "audit_version": "global_axis_training_test_audit_2026_09_19",
                "source_audit": str(audit_dir),
                "source_axis_registry": str(audit_dir / "frozen_prediction_axis_registry.csv"),
                "read_only": True,
                "pretraining_gate": {
                    "minimum_direct_mass_food_observations": MIN_PRETRAIN_DIRECT_FOODS,
                    "minimum_positive_food_observations": MIN_PRETRAIN_POSITIVE_FOODS,
                },
                "foundation_external_test_gate": {
                    "minimum_foundation_food_observations": MIN_EXTERNAL_TEST_FOODS,
                    "minimum_foundation_positive_food_observations": MIN_EXTERNAL_TEST_POSITIVE_FOODS,
                    "minimum_nonfoundation_food_observations": MIN_EXTERNAL_TRAIN_FOODS,
                    "minimum_nonfoundation_positive_food_observations": MIN_EXTERNAL_TRAIN_POSITIVE_FOODS,
                },
                "summary": summary,
                "source_isolated_test_feasibility": "Pre-blocking only; exact-name, lineage, and near-duplicate split blocks are not yet removed.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _markdown(
        output_dir / "AXIS_TRAINING_AND_TEST_PLAN.md", summary, result, family_summary, source_test_summary,
        source_audit_version,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote audit to {output_dir}")


if __name__ == "__main__":
    main()

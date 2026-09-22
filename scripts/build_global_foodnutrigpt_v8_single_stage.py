#!/usr/bin/env python3
"""Build the source-native v8 corpus for one-stage FoodNutriGPT training.

The builder preserves every direct-mass source measurement as an input token.
It never pools, averages, or selects a cross-source winner.  USDA Foundation is
held out before text embedding, normalizer fitting, training, or early stopping.

Colab:
    python scripts/build_global_foodnutrigpt_v8_single_stage.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import sha256_file, write_csv, write_json  # noqa: E402


VERSION = "global_foodnutrigpt_v8_single_stage_v2_complete_test"
SOURCE_AUDIT = ROOT / "data/processed/global_frozen_prediction_panel_food_audit_v8"
AXIS_AUDIT = ROOT / "reports/global_axis_training_test_audit_2026_09_20_v2"
DEFAULT_DATA_DIR = ROOT / f"data/processed/{VERSION}"
DEFAULT_SPLIT_DIR = ROOT / f"data/splits/{VERSION}"
# The original v1 axis registry freezes the current scientific target set.  The
# new split changes food allocation only; it must not change the 187 targets in
# response to test coverage.
REFERENCE_AXIS_REGISTRY = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v1/axis_registry.csv"
FOUNDATION_SOURCE = "usda_foundation"
VALIDATION_FRACTION = 0.15
SPLIT_SEED = 20260920
MIN_TRAIN_PROFILES_PER_LOSS_AXIS = 20
MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS = 20
MIN_COMPLETE_TEST_PROFILES_PER_LOSS_AXIS = 5
MIN_COMPLETE_TEST_POSITIVE_PROFILES_PER_LOSS_AXIS = 5


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(f"{SPLIT_SEED}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_food_text(row: pd.Series) -> str:
    """Use source-provided food fields without inserting the database name."""
    fragments = [str(row.original_name).strip()]
    fields = [
        ("scientific name", "scientific_name"),
        ("group", "food_group"),
        ("subgroup", "food_subgroup"),
        ("type", "food_type"),
        ("part", "part"),
        ("processing", "processing"),
        ("cooking", "cooking"),
        ("preservation", "preservation"),
        ("physical state", "physical_state"),
    ]
    for label, field in fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip() and str(value).strip().casefold() != "nan":
            fragments.append(f"{label}: {str(value).strip()}")
    return ". ".join(fragments)


def profile_axis_support(tokens: pd.DataFrame, profile_ids: set[str]) -> pd.DataFrame:
    """Count source-native food profiles once per axis, despite duplicate rows."""
    subset = tokens[tokens["profile_id"].isin(profile_ids)].copy()
    cells = (
        subset.assign(_positive=subset["normalized_value_g_per_100g"].gt(0))
        .groupby(["target_axis_id", "profile_id"], as_index=False)
        .agg(_positive=("_positive", "max"))
    )
    return (
        cells.groupby("target_axis_id", as_index=False)
        .agg(
            train_profile_count=("profile_id", "nunique"),
            train_positive_profile_count=("_positive", "sum"),
        )
    )


def load_frozen_target_axis_ids() -> set[str]:
    """Read the existing 187-axis proposal without letting a new split alter it."""
    if not REFERENCE_AXIS_REGISTRY.exists():
        raise FileNotFoundError(
            "The frozen v1 target registry is required to preserve the current target definition: "
            f"{REFERENCE_AXIS_REGISTRY}"
        )
    reference = pd.read_csv(REFERENCE_AXIS_REGISTRY, low_memory=False)
    required = {"target_axis_id", "training_role"}
    if missing := required - set(reference.columns):
        raise ValueError(f"Frozen target registry lacks columns: {sorted(missing)}")
    target_ids = set(reference.loc[
        reference["training_role"].eq("masked_loss_target"), "target_axis_id"
    ])
    if len(target_ids) != 187:
        raise ValueError(f"Expected 187 frozen prediction axes, found {len(target_ids)}.")
    return target_ids


def group_axis_support(
    tokens: pd.DataFrame, profiles: pd.DataFrame, axis_ids: set[str],
) -> pd.DataFrame:
    """Return one observed/positive count per exact-name group and axis."""
    subset = tokens[
        tokens["profile_id"].isin(set(profiles["profile_id"]))
        & tokens["target_axis_id"].isin(axis_ids)
    ].copy()
    if subset["exact_name_group_id"].isna().any():
        raise ValueError("A token selected for group support lacks an exact-name group.")
    cells = (
        subset.assign(_positive=subset["normalized_value_g_per_100g"].gt(0))
        .groupby(["exact_name_group_id", "target_axis_id", "profile_id"], as_index=False)
        .agg(_positive=("_positive", "max"))
    )
    return (
        cells.groupby(["exact_name_group_id", "target_axis_id"], as_index=False)
        .agg(observed_profiles=("profile_id", "nunique"), positive_profiles=("_positive", "sum"))
    )


def select_axis_coverage_test_groups(
    group_support: pd.DataFrame,
    foundation_support: pd.DataFrame,
    target_axis_ids: set[str],
) -> tuple[set[str], pd.DataFrame]:
    """Greedily reserve non-Foundation name groups until every target is testable.

    The reservation always leaves the required observed and positive support in
    the non-test pool.  Foundation measurements count toward test coverage but
    never reduce the support available for training.
    """
    target_ids = sorted(target_axis_ids)
    total = group_support.groupby("target_axis_id", as_index=True)[
        ["observed_profiles", "positive_profiles"]
    ].sum().reindex(target_ids, fill_value=0)
    foundation = foundation_support.set_index("target_axis_id")[
        ["observed_profiles", "positive_profiles"]
    ].reindex(target_ids, fill_value=0)
    if (total["observed_profiles"] < MIN_TRAIN_PROFILES_PER_LOSS_AXIS).any() or (
        total["positive_profiles"] < MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS
    ).any():
        invalid = total.index[
            (total["observed_profiles"] < MIN_TRAIN_PROFILES_PER_LOSS_AXIS)
            | (total["positive_profiles"] < MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS)
        ].tolist()
        raise ValueError(f"Frozen target axes cannot retain minimum train support: {invalid[:5]}")

    records_by_group: dict[str, list[tuple[str, int, int]]] = {}
    groups_by_axis: dict[str, list[str]] = {axis_id: [] for axis_id in target_ids}
    for row in group_support.itertuples(index=False):
        group_id, axis_id = str(row.exact_name_group_id), str(row.target_axis_id)
        records_by_group.setdefault(group_id, []).append(
            (axis_id, int(row.observed_profiles), int(row.positive_profiles))
        )
        groups_by_axis[axis_id].append(group_id)
    for group_ids in groups_by_axis.values():
        group_ids.sort()

    supplemental_observed = {axis_id: 0 for axis_id in target_ids}
    supplemental_positive = {axis_id: 0 for axis_id in target_ids}
    selected_groups: set[str] = set()

    for _ in range(len(records_by_group) + 1):
        deficits: list[tuple[str, int, int]] = []
        for axis_id in target_ids:
            observed = int(foundation.loc[axis_id, "observed_profiles"]) + supplemental_observed[axis_id]
            positive = int(foundation.loc[axis_id, "positive_profiles"]) + supplemental_positive[axis_id]
            observed_deficit = max(0, MIN_COMPLETE_TEST_PROFILES_PER_LOSS_AXIS - observed)
            positive_deficit = max(0, MIN_COMPLETE_TEST_POSITIVE_PROFILES_PER_LOSS_AXIS - positive)
            if observed_deficit or positive_deficit:
                deficits.append((axis_id, observed_deficit, positive_deficit))
        if not deficits:
            break

        # Solve low-support axes first, while preferring groups that cover more
        # still-unmet axes.  This is deterministic because ties use group ID.
        deficits.sort(key=lambda item: (
            len(groups_by_axis[item[0]]), -(item[1] + item[2]),
            int(total.loc[item[0], "observed_profiles"]), item[0],
        ))
        selected_axis = deficits[0][0]
        candidates: list[tuple[float, str]] = []
        for group_id in groups_by_axis[selected_axis]:
            if group_id in selected_groups:
                continue
            records = records_by_group[group_id]
            if not all(
                supplemental_observed[axis_id] + observed <= int(total.loc[axis_id, "observed_profiles"])
                - MIN_TRAIN_PROFILES_PER_LOSS_AXIS
                and supplemental_positive[axis_id] + positive <= int(total.loc[axis_id, "positive_profiles"])
                - MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS
                for axis_id, observed, positive in records
            ):
                continue
            score = 0.0
            for axis_id, observed, positive in records:
                current_observed = int(foundation.loc[axis_id, "observed_profiles"]) + supplemental_observed[axis_id]
                current_positive = int(foundation.loc[axis_id, "positive_profiles"]) + supplemental_positive[axis_id]
                score += min(observed, max(0, MIN_COMPLETE_TEST_PROFILES_PER_LOSS_AXIS - current_observed))
                score += min(positive, max(0, MIN_COMPLETE_TEST_POSITIVE_PROFILES_PER_LOSS_AXIS - current_positive))
            candidates.append((score, group_id))
        if not candidates:
            name = selected_axis
            raise ValueError(
                f"Cannot reserve a complete test group for target axis {name} while retaining train support."
            )
        _, group_id = max(candidates, key=lambda item: (item[0], item[1]))
        selected_groups.add(group_id)
        for axis_id, observed, positive in records_by_group[group_id]:
            supplemental_observed[axis_id] += observed
            supplemental_positive[axis_id] += positive
    else:
        raise RuntimeError("Complete-test group selection did not converge.")

    coverage = pd.DataFrame({
        "target_axis_id": target_ids,
        "foundation_observed_profiles": [int(foundation.loc[axis_id, "observed_profiles"]) for axis_id in target_ids],
        "foundation_positive_profiles": [int(foundation.loc[axis_id, "positive_profiles"]) for axis_id in target_ids],
        "supplemental_observed_profiles": [supplemental_observed[axis_id] for axis_id in target_ids],
        "supplemental_positive_profiles": [supplemental_positive[axis_id] for axis_id in target_ids],
        "available_non_test_observed_profiles": [
            int(total.loc[axis_id, "observed_profiles"]) - supplemental_observed[axis_id]
            for axis_id in target_ids
        ],
        "available_non_test_positive_profiles": [
            int(total.loc[axis_id, "positive_profiles"]) - supplemental_positive[axis_id]
            for axis_id in target_ids
        ],
    })
    coverage["test_observed_profiles"] = (
        coverage["foundation_observed_profiles"] + coverage["supplemental_observed_profiles"]
    )
    coverage["test_positive_profiles"] = (
        coverage["foundation_positive_profiles"] + coverage["supplemental_positive_profiles"]
    )
    if (
        coverage["test_observed_profiles"] < MIN_COMPLETE_TEST_PROFILES_PER_LOSS_AXIS
    ).any() or (
        coverage["test_positive_profiles"] < MIN_COMPLETE_TEST_POSITIVE_PROFILES_PER_LOSS_AXIS
    ).any():
        raise AssertionError("Complete test panel does not cover every frozen target axis.")
    return selected_groups, coverage


def repair_train_support_after_validation(
    group_support: pd.DataFrame,
    group_assignment: pd.DataFrame,
    target_axis_ids: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Move validation groups back to train until all frozen axes retain support."""
    assignment = group_assignment.set_index("exact_name_group_id")["partition"].to_dict()
    records_by_group: dict[str, list[tuple[str, int, int]]] = {}
    groups_by_axis: dict[str, list[str]] = {axis_id: [] for axis_id in target_axis_ids}
    for row in group_support.itertuples(index=False):
        group_id, axis_id = str(row.exact_name_group_id), str(row.target_axis_id)
        records_by_group.setdefault(group_id, []).append(
            (axis_id, int(row.observed_profiles), int(row.positive_profiles))
        )
        groups_by_axis[axis_id].append(group_id)
    for group_ids in groups_by_axis.values():
        group_ids.sort()

    train_observed = {axis_id: 0 for axis_id in target_axis_ids}
    train_positive = {axis_id: 0 for axis_id in target_axis_ids}
    for group_id, records in records_by_group.items():
        if assignment.get(group_id) != "train":
            continue
        for axis_id, observed, positive in records:
            train_observed[axis_id] += observed
            train_positive[axis_id] += positive

    moved_groups: list[str] = []
    for _ in range(len(records_by_group) + 1):
        deficits = [
            axis_id for axis_id in sorted(target_axis_ids)
            if train_observed[axis_id] < MIN_TRAIN_PROFILES_PER_LOSS_AXIS
            or train_positive[axis_id] < MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS
        ]
        if not deficits:
            break
        deficits.sort(key=lambda axis_id: (len(groups_by_axis[axis_id]), axis_id))
        axis_id = deficits[0]
        candidates: list[tuple[float, str]] = []
        for group_id in groups_by_axis[axis_id]:
            if assignment.get(group_id) != "validation":
                continue
            score = 0.0
            for candidate_axis, observed, positive in records_by_group[group_id]:
                observed_gap = max(0, MIN_TRAIN_PROFILES_PER_LOSS_AXIS - train_observed[candidate_axis])
                positive_gap = max(0, MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS - train_positive[candidate_axis])
                score += min(observed, observed_gap) + min(positive, positive_gap)
            candidates.append((score, group_id))
        if not candidates:
            raise ValueError(
                f"Validation allocation leaves frozen axis {axis_id} without required training support."
            )
        _, group_id = max(candidates, key=lambda item: (item[0], item[1]))
        assignment[group_id] = "train"
        moved_groups.append(group_id)
        for candidate_axis, observed, positive in records_by_group[group_id]:
            train_observed[candidate_axis] += observed
            train_positive[candidate_axis] += positive
    else:
        raise RuntimeError("Validation support repair did not converge.")

    repaired = pd.DataFrame({
        "exact_name_group_id": sorted(assignment),
        "partition": [assignment[group_id] for group_id in sorted(assignment)],
    })
    return repaired, moved_groups


def fit_train_normalizer(tokens: pd.DataFrame, train_profiles: set[str], axis_ids: set[str]) -> pd.DataFrame:
    """Fit positive-value log scales on training profiles only."""
    train = tokens[
        tokens["profile_id"].isin(train_profiles)
        & tokens["target_axis_id"].isin(axis_ids)
        & tokens["normalized_value_g_per_100g"].gt(0)
    ].copy()
    train["log1p_value"] = np.log1p(train["normalized_value_g_per_100g"].to_numpy(dtype=np.float64))
    normalizer = (
        train.groupby("target_axis_id", as_index=False)["log1p_value"]
        .median()
        .rename(columns={"log1p_value": "median_log1p"})
    )
    centred = train.merge(normalizer, on="target_axis_id", validate="many_to_one")
    scales = (
        centred.assign(_squared_deviation=(centred["log1p_value"] - centred["median_log1p"]) ** 2)
        .groupby("target_axis_id", as_index=False)["_squared_deviation"]
        .mean()
        .assign(positive_median_baseline_log_rmse=lambda frame: np.sqrt(frame["_squared_deviation"]))
        .drop(columns=["_squared_deviation"])
    )
    normalizer = normalizer.merge(scales, on="target_axis_id", validate="one_to_one")
    normalizer["scale_log1p"] = normalizer["positive_median_baseline_log_rmse"].clip(lower=1e-4)
    return normalizer.sort_values("target_axis_id", kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--axis-audit", type=Path, default=AXIS_AUDIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    args = parser.parse_args()

    source_audit, axis_audit = args.source_audit.resolve(), args.axis_audit.resolve()
    data_dir, split_dir = args.data_dir.resolve(), args.split_dir.resolve()
    if data_dir.exists() or split_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing corpus or split directory.")

    manifest = json.loads((source_audit / "audit_manifest.json").read_text(encoding="utf-8"))
    if manifest["audit_version"] != "global_frozen_prediction_panel_food_audit_v8":
        raise ValueError(f"Expected v8 source audit, got {manifest['audit_version']!r}.")

    axis_audit_frame = pd.read_csv(axis_audit / "axis_training_eligibility.csv", low_memory=False)
    required_axis_columns = {
        "target_axis_id", "loss_group", "pretraining_regression_eligible",
        "foundation_external_test_eligible", "mask_family", "canonical_name",
    }
    missing_axis_columns = sorted(required_axis_columns - set(axis_audit_frame.columns))
    if missing_axis_columns:
        raise ValueError(f"Axis audit is missing required columns: {missing_axis_columns}")
    axis_audit_frame["pretraining_regression_eligible"] = axis_audit_frame[
        "pretraining_regression_eligible"
    ].astype(str).str.casefold().isin({"true", "1", "yes"})
    axis_audit_frame["foundation_external_test_eligible"] = axis_audit_frame[
        "foundation_external_test_eligible"
    ].astype(str).str.casefold().isin({"true", "1", "yes"})

    direct_columns = [
        "measurement_id", "source_key", "food_observation_id", "component_observation_id",
        "target_axis_id", "target_axis_name", "normalized_value_g_per_100g", "value_status",
        "exact_name_group_id", "raw_unit", "raw_basis", "source_value_origin",
    ]
    tokens = pd.read_csv(
        source_audit / "direct_mass_label_candidate_measurements.csv.gz",
        usecols=direct_columns,
        low_memory=False,
    )
    tokens["normalized_value_g_per_100g"] = pd.to_numeric(
        tokens["normalized_value_g_per_100g"], errors="raise"
    )
    if tokens["normalized_value_g_per_100g"].lt(0).any():
        raise ValueError("Direct-mass corpus contains a negative g/100 g value.")
    if not tokens["value_status"].isin(["observed", "explicit_zero"]).all():
        raise ValueError("Direct-mass corpus contains a non-observed/non-zero value status.")

    food_columns = [
        "food_observation_id", "source_key", "original_name", "scientific_name", "food_group",
        "food_subgroup", "food_type", "part", "processing", "cooking", "preservation",
        "physical_state", "exact_name_group_id",
    ]
    foods = pd.read_csv(
        source_audit / "food_observation_to_exact_name_group.csv.gz",
        usecols=food_columns,
        low_memory=False,
    )
    if foods.duplicated(["source_key", "food_observation_id"]).any():
        raise ValueError("Source audit food observations are not unique by source and observation ID.")
    foods["profile_id"] = "profile:" + foods["source_key"].astype(str) + ":" + foods["food_observation_id"].astype(str)
    foods["food_text"] = foods.apply(build_food_text, axis=1)

    tokens["profile_id"] = "profile:" + tokens["source_key"].astype(str) + ":" + tokens["food_observation_id"].astype(str)
    tokens = tokens.merge(
        foods[["profile_id", "exact_name_group_id"]].rename(columns={"exact_name_group_id": "food_exact_name_group_id"}),
        on="profile_id",
        how="left",
        validate="many_to_one",
    )
    if tokens["food_exact_name_group_id"].isna().any():
        raise ValueError("A direct-mass measurement could not be joined to its food profile.")
    if not tokens["exact_name_group_id"].eq(tokens["food_exact_name_group_id"]).all():
        raise ValueError("Measurement and food exact-name groups disagree.")
    tokens = tokens.drop(columns=["food_exact_name_group_id"])

    axes = axis_audit_frame[[
        "target_axis_id", "canonical_name", "mask_family", "loss_group",
        "pretraining_regression_eligible", "foundation_external_test_eligible",
    ]].copy()
    tokens = tokens.merge(axes, on="target_axis_id", how="inner", validate="many_to_one")
    if tokens.empty:
        raise ValueError("No direct-mass tokens survived the axis audit join.")

    frozen_target_axis_ids = load_frozen_target_axis_ids()
    missing_frozen_targets = frozen_target_axis_ids - set(axes["target_axis_id"])
    if missing_frozen_targets:
        raise ValueError(f"Frozen targets are absent from the current axis audit: {sorted(missing_frozen_targets)[:5]}")

    profiles = foods[foods["profile_id"].isin(tokens["profile_id"])].copy()
    foundation_profiles = profiles[profiles["source_key"].eq(FOUNDATION_SOURCE)].copy()
    if foundation_profiles.empty:
        raise ValueError("USDA Foundation has no direct-mass profiles in the v8 audit.")
    foundation_groups = set(foundation_profiles["exact_name_group_id"])
    blocked_nonfoundation = profiles[
        ~profiles["source_key"].eq(FOUNDATION_SOURCE)
        & profiles["exact_name_group_id"].isin(foundation_groups)
    ].copy()
    eligible_profiles = profiles[
        ~profiles["source_key"].eq(FOUNDATION_SOURCE)
        & ~profiles["exact_name_group_id"].isin(foundation_groups)
    ].copy()

    # A full external source is retained, then minimal additional name groups
    # are held out to make every frozen prediction axis testable.
    nonfoundation_group_support = group_axis_support(tokens, eligible_profiles, frozen_target_axis_ids)
    foundation_group_support = group_axis_support(tokens, foundation_profiles, frozen_target_axis_ids)
    foundation_support = (
        foundation_group_support.groupby("target_axis_id", as_index=False)[
            ["observed_profiles", "positive_profiles"]
        ].sum()
    )
    supplemental_test_groups, complete_test_coverage = select_axis_coverage_test_groups(
        nonfoundation_group_support, foundation_support, frozen_target_axis_ids,
    )
    supplemental_test_profiles = eligible_profiles[
        eligible_profiles["exact_name_group_id"].isin(supplemental_test_groups)
    ].copy()
    remaining_profiles = eligible_profiles[
        ~eligible_profiles["exact_name_group_id"].isin(supplemental_test_groups)
    ].copy()

    group_assignment = remaining_profiles[["exact_name_group_id"]].drop_duplicates().copy()
    group_assignment["partition"] = np.where(
        group_assignment["exact_name_group_id"].map(stable_fraction).lt(VALIDATION_FRACTION),
        "validation",
        "train",
    )
    remaining_group_support = nonfoundation_group_support[
        ~nonfoundation_group_support["exact_name_group_id"].isin(supplemental_test_groups)
    ].copy()
    group_assignment, repaired_validation_groups = repair_train_support_after_validation(
        remaining_group_support, group_assignment, frozen_target_axis_ids,
    )
    remaining_profiles = remaining_profiles.merge(
        group_assignment, on="exact_name_group_id", validate="many_to_one",
    )
    remaining_profiles["test_panel_origin"] = "not_test"
    foundation_profiles["partition"] = "test_complete_axis_panel"
    foundation_profiles["test_panel_origin"] = "foundation_source_holdout"
    supplemental_test_profiles["partition"] = "test_complete_axis_panel"
    supplemental_test_profiles["test_panel_origin"] = "axis_coverage_holdout"
    blocked_nonfoundation["partition"] = "excluded_foundation_exact_name_overlap"
    blocked_nonfoundation["test_panel_origin"] = "not_test"
    profiles = profiles.merge(
        pd.concat([
            remaining_profiles[["profile_id", "partition", "test_panel_origin"]],
            foundation_profiles[["profile_id", "partition", "test_panel_origin"]],
            supplemental_test_profiles[["profile_id", "partition", "test_panel_origin"]],
            blocked_nonfoundation[["profile_id", "partition", "test_panel_origin"]],
        ], ignore_index=True),
        on="profile_id",
        how="left",
        validate="one_to_one",
    )
    if profiles["partition"].isna().any():
        raise ValueError("A profile received no split partition.")

    train_profiles = set(profiles.loc[profiles["partition"].eq("train"), "profile_id"])
    support = profile_axis_support(tokens, train_profiles)
    axes = axes.merge(support, on="target_axis_id", how="left", validate="one_to_one")
    axes[["train_profile_count", "train_positive_profile_count"]] = axes[
        ["train_profile_count", "train_positive_profile_count"]
    ].fillna(0).astype(int)
    axes["loss_eligible"] = axes["target_axis_id"].isin(frozen_target_axis_ids)
    unsupported_frozen_targets = axes.loc[
        axes["loss_eligible"]
        & (
            axes["train_profile_count"].lt(MIN_TRAIN_PROFILES_PER_LOSS_AXIS)
            | axes["train_positive_profile_count"].lt(MIN_TRAIN_POSITIVE_PROFILES_PER_LOSS_AXIS)
        ), "target_axis_id"
    ].tolist()
    if unsupported_frozen_targets:
        raise ValueError(
            "The complete-test allocation violated frozen target train support: "
            f"{unsupported_frozen_targets[:5]}"
        )
    active_axis_ids = set(axes.loc[axes["train_profile_count"].gt(0), "target_axis_id"])
    axes = axes[axes["target_axis_id"].isin(active_axis_ids)].copy()
    axes["training_role"] = np.where(axes["loss_eligible"], "masked_loss_target", "context_only")
    axes = axes.sort_values(["loss_group", "canonical_name"], kind="stable").reset_index(drop=True)
    axes["axis_index"] = np.arange(len(axes), dtype=np.int64)

    tokens = tokens[tokens["target_axis_id"].isin(set(axes["target_axis_id"]))].copy()
    tokens = tokens.merge(
        axes[["target_axis_id", "axis_index", "loss_eligible", "training_role"]],
        on="target_axis_id",
        how="inner",
        validate="many_to_one",
    )
    duplicate_sizes = tokens.groupby(["profile_id", "target_axis_id"])["measurement_id"].transform("size")
    tokens["within_profile_axis_observation_count"] = duplicate_sizes.astype(np.int32)
    tokens["within_profile_axis_loss_weight"] = 1.0 / duplicate_sizes.to_numpy(dtype=np.float32)

    train_profiles = set(profiles.loc[profiles["partition"].eq("train"), "profile_id"])
    normalizer = fit_train_normalizer(tokens, train_profiles, set(axes["target_axis_id"]))
    axes = axes.merge(normalizer, on="target_axis_id", how="left", validate="one_to_one")
    missing_scale = axes["median_log1p"].isna()
    if axes.loc[missing_scale, "loss_eligible"].any():
        bad = axes.loc[missing_scale & axes["loss_eligible"], "canonical_name"].tolist()
        raise ValueError(f"Loss-eligible axes lack positive train values: {bad}")
    axes.loc[missing_scale, ["median_log1p", "positive_median_baseline_log_rmse", "scale_log1p"]] = [0.0, 1.0, 1.0]
    tokens = tokens.merge(
        axes[["target_axis_id", "median_log1p", "scale_log1p"]],
        on="target_axis_id",
        validate="many_to_one",
    )
    tokens["log1p_value"] = np.log1p(tokens["normalized_value_g_per_100g"].to_numpy(dtype=np.float64))
    tokens["normalized_log1p_value"] = (
        (tokens["log1p_value"] - tokens["median_log1p"]) / tokens["scale_log1p"]
    )
    tokens["is_positive"] = tokens["normalized_value_g_per_100g"].gt(0)

    train_source_keys = sorted(profiles.loc[profiles["partition"].eq("train"), "source_key"].unique())
    source_vocab = pd.DataFrame({"source_key": train_source_keys, "source_index": np.arange(1, len(train_source_keys) + 1)})
    source_vocab = pd.concat([
        pd.DataFrame({"source_key": ["__UNKNOWN__"], "source_index": [0]}),
        source_vocab,
    ], ignore_index=True)
    profiles = profiles.merge(source_vocab, on="source_key", how="left", validate="many_to_one")
    profiles["source_index"] = profiles["source_index"].fillna(0).astype(int)

    test_axes = set(frozen_target_axis_ids)
    train_test_group_overlap = set(profiles.loc[profiles.partition.eq("train"), "exact_name_group_id"]) & set(
        profiles.loc[profiles.partition.eq("test_complete_axis_panel"), "exact_name_group_id"]
    )
    if train_test_group_overlap:
        raise AssertionError("Complete-test exact-name groups leaked into training.")
    validation_test_group_overlap = set(profiles.loc[profiles.partition.eq("validation"), "exact_name_group_id"]) & set(
        profiles.loc[profiles.partition.eq("test_complete_axis_panel"), "exact_name_group_id"]
    )
    if validation_test_group_overlap:
        raise AssertionError("Complete-test exact-name groups leaked into validation.")

    data_dir.mkdir(parents=True, exist_ok=False)
    split_dir.mkdir(parents=True, exist_ok=False)
    profiles = profiles.sort_values("profile_id", kind="stable")
    tokens = tokens.sort_values(["profile_id", "axis_index", "measurement_id"], kind="stable")
    write_csv(profiles, data_dir / "food_profiles.csv.gz")
    write_csv(axes, data_dir / "axis_registry.csv")
    write_csv(tokens, data_dir / "source_native_axis_tokens.csv.gz")
    write_csv(normalizer, data_dir / "train_only_axis_normalization.csv")
    write_csv(source_vocab, data_dir / "source_vocabulary.csv")
    complete_test_coverage = complete_test_coverage.merge(
        axes[["target_axis_id", "canonical_name", "loss_group"]],
        on="target_axis_id", how="left", validate="one_to_one",
    ).sort_values(["loss_group", "canonical_name"], kind="stable")
    write_csv(complete_test_coverage, data_dir / "complete_test_axis_coverage.csv")
    write_csv(
        pd.DataFrame({"exact_name_group_id": sorted(supplemental_test_groups)}),
        data_dir / "supplemental_test_exact_name_groups.csv",
    )
    write_csv(
        profiles.groupby("partition", as_index=False).agg(
            source_native_profiles=("profile_id", "nunique"),
            exact_name_groups=("exact_name_group_id", "nunique"),
            sources=("source_key", "nunique"),
        ),
        data_dir / "partition_summary.csv",
    )
    splits = {
        "version": VERSION,
        "seed": SPLIT_SEED,
        "validation_fraction_by_exact_name_group": VALIDATION_FRACTION,
        "foundation_source_holdout": FOUNDATION_SOURCE,
        "train": profiles.loc[profiles.partition.eq("train"), "profile_id"].tolist(),
        "validation": profiles.loc[profiles.partition.eq("validation"), "profile_id"].tolist(),
        "test_complete_axis_panel": profiles.loc[
            profiles.partition.eq("test_complete_axis_panel"), "profile_id"
        ].tolist(),
        "foundation_source_holdout_profiles": profiles.loc[
            profiles.test_panel_origin.eq("foundation_source_holdout"), "profile_id"
        ].tolist(),
        "axis_coverage_holdout_profiles": profiles.loc[
            profiles.test_panel_origin.eq("axis_coverage_holdout"), "profile_id"
        ].tolist(),
        "excluded_foundation_exact_name_overlap": profiles.loc[
            profiles.partition.eq("excluded_foundation_exact_name_overlap"), "profile_id"
        ].tolist(),
        "test_axis_ids": sorted(test_axes),
        "minimum_test_observed_profiles_per_axis": MIN_COMPLETE_TEST_PROFILES_PER_LOSS_AXIS,
        "minimum_test_positive_profiles_per_axis": MIN_COMPLETE_TEST_POSITIVE_PROFILES_PER_LOSS_AXIS,
    }
    write_json(splits, split_dir / "splits.json")
    build_manifest = {
        "dataset_version": VERSION,
        "source_audit": str(source_audit),
        "source_audit_manifest_sha256": sha256_file(source_audit / "audit_manifest.json"),
        "axis_audit": str(axis_audit),
        "axis_audit_sha256": sha256_file(axis_audit / "axis_training_eligibility.csv"),
        "measurement_policy": "Every compatible source-native direct-mass record remains a token; no cross-source or within-cell numerical pooling is performed.",
        "unit_policy": "All model values are normalized_value_g_per_100g. Raw units and values remain in source_native_axis_tokens.csv.gz.",
        "missing_policy": "Missing values are absent tokens. Explicit zero remains an observed token.",
        "source_policy": "Source is a profile-level input only. Evaluation always uses UNKNOWN; USDA Foundation source IDs are not in the train vocabulary.",
        "split_policy": "USDA Foundation is source-isolated. A minimal deterministic set of additional non-Foundation exact-name groups is held out so every frozen target axis has at least five observed and five positive test profiles. No test exact-name group occurs in train or validation.",
        "food_profiles": int(len(profiles)),
        "source_native_tokens": int(len(tokens)),
        "active_axes": int(len(axes)),
        "loss_axes": int(axes["loss_eligible"].sum()),
        "nutrition_loss_axes": int((axes["loss_eligible"] & axes["loss_group"].eq("nutrition")).sum()),
        "food_metabolome_loss_axes": int((axes["loss_eligible"] & axes["loss_group"].eq("food_metabolome")).sum()),
        "complete_test_axes": int(len(test_axes)),
        "complete_test_profiles": int(profiles["partition"].eq("test_complete_axis_panel").sum()),
        "foundation_source_holdout_profiles": int(profiles["test_panel_origin"].eq("foundation_source_holdout").sum()),
        "axis_coverage_holdout_profiles": int(profiles["test_panel_origin"].eq("axis_coverage_holdout").sum()),
        "axis_coverage_holdout_exact_name_groups": int(len(supplemental_test_groups)),
        "validation_groups_reassigned_to_train_for_axis_support": int(len(repaired_validation_groups)),
        "foundation_exact_name_overlap_profiles_excluded": int(len(blocked_nonfoundation)),
    }
    write_json(build_manifest, data_dir / "build_manifest.json")
    print(json.dumps(build_manifest, indent=2))


if __name__ == "__main__":
    main()

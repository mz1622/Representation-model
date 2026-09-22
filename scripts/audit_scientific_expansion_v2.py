#!/usr/bin/env python3
"""Verify expansion, preserved identities, changed labels, and source evidence."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.benchmark import load_release
from foodcomp.dataset import FoodCompositionDataset
from foodcomp.util import sha256_file, write_csv, write_json

VERSION = "scientific_food_composition_v2"


def main():
    base = ROOT / "data/processed" / VERSION
    release = base / "release"
    old = ROOT / "data/processed/scientific_food_composition_v1/release"
    output = ROOT / "reports" / VERSION
    output.mkdir(parents=True, exist_ok=True)
    values, foods, components, _, _ = load_release(release)
    old_foods = pd.read_csv(old / "ml_partition.csv")
    measurements = pd.read_csv(release / "measurement_main_eligible.csv.gz", low_memory=False)
    observation = pd.read_csv(release / "food_observation.csv.gz", low_memory=False)
    profiles = pd.read_csv(release / "canonical_profile.csv.gz", low_memory=False)
    accepted = profiles[profiles.aggregation_status.eq("accepted") & profiles.component_concept_id.isin(components.component_concept_id)]
    locked = foods.partition.eq("validation")
    previous_locked = old_foods[old_foods.partition.eq("validation")]
    current_locked = foods[locked]
    for columns in (["food_concept_id", "validation_panel"], ["food_concept_id", "family_cluster_id"]):
        pd.testing.assert_frame_equal(previous_locked[columns].sort_values("food_concept_id").reset_index(drop=True), current_locked[columns].sort_values("food_concept_id").reset_index(drop=True))
    frozen = json.loads((base / "v1_immutable_snapshot.json").read_text())
    assert all(sha256_file(ROOT / path) == digest for path, digest in frozen.items())
    downloads = json.loads((ROOT / "data/raw/expansion_2026_09_06/download_manifest.json").read_text())
    assert all(sha256_file(ROOT / "data/raw/expansion_2026_09_06" / record["path"]) == record["sha256"] for record in downloads["files"])
    assert not measurements.raw_unit.fillna("").str.contains(r"/\s*g\s*N\b", case=False, regex=True).any()
    assert not measurements.value_status.isin(["reported_zero_unresolved", "assumed_zero", "left_censored_zero_encoding"]).any()
    selected_validation_ids = set(accepted.loc[accepted.partition.eq("validation"), "measurement_ids"].str.split(";").explode())
    used_val = measurements[measurements.measurement_id.isin(selected_validation_ids)]
    assert used_val.strict_validation_eligible.astype(str).str.casefold().eq("true").all()
    assert not used_val.source_key.isin(["afcd", "cofid", "norway"]).any()

    afcd_archive = pd.read_csv(base / "staging/afcd/measurement_part_0000.csv.gz", usecols=["food_observation_id", "method_expression", "data_layer", "main_value_eligible"])
    afcd_conflicts = afcd_archive[afcd_archive.data_layer.eq("source_metadata_conflict")]
    assert not afcd_conflicts.main_value_eligible.astype(str).str.casefold().eq("true").any()
    metadata_conflicts = afcd_conflicts.groupby(["food_observation_id", "method_expression"]).size().reset_index(name="archived_measurement_rows")
    metadata_conflicts = metadata_conflicts.merge(observation[["food_observation_id", "source_food_id", "original_name"]], on="food_observation_id", how="left", validate="many_to_one")
    write_csv(metadata_conflicts, output / "afcd_official_metadata_conflicts.csv")

    original_ids = set(old_foods.loc[old_foods.partition.eq("train"), "food_concept_id"])
    current_ids = set(foods.loc[foods.partition.eq("train"), "food_concept_id"])
    added, removed = current_ids - original_ids, original_ids - current_ids
    write_csv(foods[foods.food_concept_id.isin(added)], output / "newly_eligible_train_foods.csv.gz")
    exclusions = pd.read_csv(ROOT / "data/audits" / VERSION / "dataset/frozen_split_exclusion_ledger.csv")
    removed_rows = old_foods[old_foods.food_concept_id.isin(removed)].merge(exclusions[["food_concept_id", "reason"]], on="food_concept_id", how="left")
    removed_rows["reason"] = removed_rows["reason"].fillna("no_source_admitted_value_remains_under_v2_definition_and_quality_gates")
    write_csv(removed_rows, output / "removed_train_foods.csv.gz")
    lost = foods[locked & ~foods.text_task_eligible].copy()
    observed_count = np.isfinite(values).sum(axis=1)
    lost["remaining_observed_matrix_values"] = observed_count[lost.index]
    lost["reason"] = np.where(lost.remaining_observed_matrix_values.eq(0), "no_remaining_eligible_mass_label", "context_only_values_but_no_qualified_maskable_target")
    write_csv(lost, output / "validation_foods_without_maskable_target.csv")
    task_counts = foods.groupby(["partition", "validation_panel"], dropna=False).agg(
        registered_foods=("food_concept_id", "size"), scorable_text_foods=("text_task_eligible", "sum"),
        reconstruction_foods=("reconstruction_task_eligible", "sum"),
    ).reset_index()
    write_csv(task_counts, output / "effective_task_food_counts.csv")
    rows = []
    for source in sorted(observation.source_key.unique()):
        eligible = measurements[measurements.source_key.eq(source)]
        contribution = accepted[accepted.source_keys.str.split(";").map(lambda names: source in names)]
        rows.append({
            "source_key": source, "raw_food_observations": int(observation.source_key.eq(source).sum()),
            "foods_with_admitted_values_before_split": eligible.food_observation_id.nunique(),
            "admitted_measurements_before_split": len(eligible),
            "accepted_train_foods": contribution.loc[contribution.partition.eq("train"), "food_concept_id"].nunique(),
            "accepted_validation_foods": contribution.loc[contribution.partition.eq("validation"), "food_concept_id"].nunique(),
            "accepted_matrix_cells": len(contribution),
            "tier_C_measurements": int(eligible.quality_tier.eq("C").sum()),
        })
    source_summary = pd.DataFrame(rows)
    write_csv(source_summary, output / "source_contribution.csv")
    write_csv(measurements.groupby(["source_key", "data_layer", "quality_tier", "independent_evidence", "strict_validation_eligible"], dropna=False).size().reset_index(name="measurement_count"), output / "measurement_evidence_layers.csv")
    write_csv(exclusions, output / "split_exclusions.csv")

    masks = pd.read_csv(ROOT / "data/splits" / VERSION / "fixed_benchmark_masks.csv.gz")
    reconstruction = masks[masks.task.eq("family_reconstruction")]
    mask_foods = reconstruction.drop_duplicates(["food_concept_id", "visibility"])
    assert mask_foods.visible_target_count.gt(0).all()
    assert mask_foods.visible_target_count.lt(mask_foods.observed_target_count).all()
    visibility = mask_foods.groupby("visibility").agg(foods=("food_concept_id", "size"), mean_actual_visibility=("actual_target_visibility", "mean"), median_actual_visibility=("actual_target_visibility", "median"), min_visible_targets=("visible_target_count", "min")).reset_index()
    write_csv(visibility, output / "actual_mask_visibility.csv")

    dataset = FoodCompositionDataset(release, partition="train")
    for index in range(len(dataset)):
        item = dataset[index]
        target = item["target_mask"].astype(bool)
        context = item["context_mask"].astype(bool)
        assert not np.any(target & context)
        hidden_families = set(dataset.family_by_column[target])
        assert not set(dataset.family_by_column[context]) & hidden_families
        assert np.isfinite(item["context_values"]).all()
    registry = pd.read_csv(release / "component_concept.csv.gz", low_memory=False)
    retained = registry[registry.training_role.ne("excluded")]
    write_csv(retained, output / "retained_component_registry.csv.gz")
    roles = retained.groupby(["nutritional_role", "training_role", "nutritional_role_review_status"], dropna=False).size().reset_index(name="axis_count")
    write_csv(roles, output / "component_classification_review_status.csv")
    distribution = components[["component_concept_id", "canonical_name", "training_role", "train_raw_min", "train_raw_max", "train_raw_sd", "train_raw_q1", "train_raw_q3", "train_log_median", "train_robust_scale"]].copy()
    distribution["train_max_absolute_robust_z"] = np.maximum(
        np.abs(np.log1p(distribution.train_raw_min) - distribution.train_log_median),
        np.abs(np.log1p(distribution.train_raw_max) - distribution.train_log_median),
    ) / distribution.train_robust_scale.replace(0, np.nan)
    distribution["interpretation"] = "Training distribution diagnostic only; extreme standardized values do not automatically imply invalid chemistry"
    write_csv(distribution, output / "train_distribution_diagnostics.csv")
    summary = {
        "dataset_version": VERSION, "v1_artifacts_unchanged": True, "validation_identities_and_families_preserved": True,
        "train_foods": len(current_ids), "added_train_foods": len(added), "removed_train_foods": len(removed),
        "net_train_increase": len(current_ids) - len(original_ids), "validation_registered_foods": int(locked.sum()),
        "validation_text_scorable_foods": int((locked & foods.text_task_eligible).sum()),
        "validation_without_maskable_targets": len(lost), "retained_axes": len(retained),
        "maskable_axes": int(retained.training_role.eq("maskable_target").sum()),
        "unresolved_chemical_class_retained_axes": int(retained.chemical_class.eq("unresolved_chemical_class").sum()),
        "source_specific_retained_axes": int(retained.identity_status.ne("authority_verified").sum()),
        "interface_examples_checked": len(dataset), "interface_family_mask_checks_passed": True,
        "downloaded_files_hash_verified": len(downloads["files"]), "locked_validation_outcomes_opened": False,
        "afcd_official_food_metadata_conflicts_quarantined": len(metadata_conflicts),
        "environment": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "xlrd"]},
        "release_status": "candidate_not_final_scientific_release",
        "audited_code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in list((ROOT / "src/foodcomp").glob("*.py")) + [Path(__file__).resolve(), ROOT / "tests/test_scientific_foodcomp.py"]},
    }
    write_json(summary, output / "EXPANSION_VERIFICATION.json")
    write_json({
        "dataset_version": VERSION, "locked_validation_opened": False,
        "gates": [
            "Recover independently supported labels for validation foods currently lacking targets; do not replace their identities",
            "Expert assessment of unresolved food identities, mixed-provenance training records and compound classification",
            "Resolve high-heterogeneity profiles and validate biochemical dependency groups",
            "Complete dual-reviewer scoping review and full FAO database evaluation",
            "Obtain permission or confirm conditions for restricted-source transformations and redistribution",
            "Validate newly related frozen-family cases before claiming full unseen-family generalization",
        ],
    }, output / "OPEN_RELEASE_GATES.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

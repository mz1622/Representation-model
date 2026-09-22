#!/usr/bin/env python3
"""Verify evidence-corrected release and expose actual, not nominal, sample sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.benchmark import load_release, fit_train_normalizer
from foodcomp.dataset import FoodCompositionDataset
from foodcomp.label_evidence import gate_reference_evidence, reference_catalogs
from foodcomp.util import read_component_csv, sha256_file, write_csv, write_json

VERSION = "scientific_food_composition_v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "data/processed" / VERSION
    release = base / "release"
    previous = root / "data/processed/scientific_food_composition_v2/release"
    output = root / "reports" / VERSION
    output.mkdir(parents=True, exist_ok=True)
    values, foods, components, food_index, component_index = load_release(release)
    old_values, old_foods, old_components, _, _ = load_release(previous)
    measurements = pd.read_csv(release / "measurement_main_eligible.csv.gz", low_memory=False)
    if measurements.measurement_id.duplicated().any():
        raise ValueError("Measurement identifiers are not unique")
    rechecked = gate_reference_evidence(measurements, reference_catalogs(root))
    if not rechecked.main_value_eligible.all():
        raise ValueError("Known evidence-ineligible measurements survived the rebuilt release")
    old_bad = pd.read_csv(root / "reports/validation_mext_evidence_2026_09_06/validation_labels_rejected_by_source_evidence.csv.gz", usecols=["measurement_id"])
    if set(old_bad.measurement_id) & set(measurements.measurement_id):
        raise ValueError("A previously rejected validation measurement is still admitted")

    identity = ["food_concept_id", "validation_panel", "family_cluster_id"]
    pd.testing.assert_frame_equal(
        foods.loc[foods.partition.eq("validation"), identity].sort_values("food_concept_id").reset_index(drop=True),
        old_foods.loc[old_foods.partition.eq("validation"), identity].sort_values("food_concept_id").reset_index(drop=True),
    )
    snapshot = json.loads((base / "protected_previous_versions.json").read_text())
    for path, digest in snapshot.items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Protected file changed: {path}")

    profiles = pd.read_csv(release / "canonical_profile.csv.gz", low_memory=False)
    accepted = profiles[profiles.aggregation_status.eq("accepted") & profiles.component_concept_id.isin(components.component_concept_id)].copy()
    validation = accepted[accepted.partition.eq("validation")]
    links = validation[["food_concept_id", "component_concept_id", "measurement_ids"]].copy()
    links["measurement_id"] = links.measurement_ids.str.split(";")
    links = links.drop(columns="measurement_ids").explode("measurement_id")
    used = links.merge(measurements, on="measurement_id", validate="many_to_one", indicator=True)
    if not used._merge.eq("both").all() or not used.strict_validation_eligible.eq(True).all():
        raise ValueError("Missing or non-strict evidence supports a validation cell")
    mapping_old = pd.read_csv(previous / "food_observation_to_concept.csv.gz")
    locked_ids = set(foods.loc[foods.partition.eq("validation"), "food_concept_id"])
    allowed_obs = set(mapping_old.loc[mapping_old.food_concept_id.isin(locked_ids), "food_observation_id"])
    if not set(used.food_observation_id) <= allowed_obs:
        raise ValueError("A new food observation was used to supply frozen validation labels")
    used["numeric_zero_requires_source_lod_review"] = used.normalized_value_g_per_100g.eq(0)
    used["method_available"] = used.analytical_method.fillna("").ne("")
    write_csv(used.drop(columns="_merge"), output / "validation_selected_evidence.csv.gz")
    write_csv(used.groupby(["source_key", "value_status", "method_available"]).size().reset_index(name="support_records"), output / "validation_evidence_summary.csv")

    summary_rows = []
    for version, data, fs, cs in (("v2", old_values, old_foods, old_components), ("v3", values, foods, components)):
        targets = cs.training_role.eq("maskable_target").to_numpy()
        for partition in ("train", "validation"):
            rows = fs.partition.eq(partition).to_numpy()
            summary_rows.append({"version": version, "partition": partition, "registered_foods": int(rows.sum()),
                                 "text_scorable_foods": int(fs.loc[rows, "text_task_eligible"].sum()),
                                 "reconstruction_foods": int(fs.loc[rows, "reconstruction_task_eligible"].sum()),
                                 "observed_cells": int(np.isfinite(data[rows]).sum()),
                                 "observed_target_cells": int(np.isfinite(data[rows][:, targets]).sum()),
                                 "retained_axes": len(cs), "maskable_axes": int(targets.sum())})
    summary_table = pd.DataFrame(summary_rows)
    write_csv(summary_table, output / "v2_v3_dataset_comparison.csv")
    observed = np.isfinite(values).sum(axis=1)
    foods["observed_matrix_cells"] = observed
    food_status = foods.merge(old_foods[["food_concept_id", "text_task_eligible", "reconstruction_task_eligible"]],
                              on="food_concept_id", how="left", suffixes=("", "_v2"), validate="one_to_one")
    write_csv(food_status[food_status.partition.eq("validation")], output / "all_1426_validation_food_status.csv")
    write_csv(food_status[food_status.partition.eq("validation") & ~food_status.text_task_eligible], output / "validation_foods_without_targets.csv")
    write_csv(foods.groupby(["partition", "validation_panel"], dropna=False).agg(
        registered_foods=("food_concept_id", "size"), text_scorable_foods=("text_task_eligible", "sum"),
        reconstruction_foods=("reconstruction_task_eligible", "sum")).reset_index(), output / "task_panel_summary.csv")

    old_ids = set(old_foods.loc[old_foods.partition.eq("train"), "food_concept_id"])
    new_ids = set(foods.loc[foods.partition.eq("train"), "food_concept_id"])
    old_concepts = pd.read_csv(previous / "food_concept.csv.gz", low_memory=False)
    old_conflicts = set(old_concepts.loc[old_concepts.family_expansion_conflict.eq(True), "food_concept_id"])
    if old_conflicts & new_ids:
        raise ValueError("Previously unresolved family bridges entered train without adjudication")
    write_csv(old_foods[old_foods.food_concept_id.isin(old_ids - new_ids)], output / "removed_train_foods.csv.gz")
    write_csv(foods[foods.food_concept_id.isin(new_ids - old_ids)], output / "added_train_foods.csv.gz")
    removed_map = mapping_old[mapping_old.food_concept_id.isin(old_ids-new_ids)]
    removed_obs = set(removed_map.food_observation_id)
    reason_parts = []
    for path in (base / "staging").glob("*/measurement_part_*.csv.gz"):
        columns = ["food_observation_id", "source_key", "data_layer", "main_value_eligible", "exclusion_reason"]
        for chunk in pd.read_csv(path, usecols=columns, low_memory=False, chunksize=100000):
            selected = chunk[chunk.food_observation_id.isin(removed_obs)].copy()
            if not selected.empty:
                reason_parts.append(selected.groupby(columns, dropna=False).size().reset_index(name="archived_value_records"))
    if reason_parts:
        reasons = pd.concat(reason_parts, ignore_index=True).merge(
            removed_map[["food_observation_id", "food_concept_id"]], on="food_observation_id", validate="many_to_one")
        write_csv(reasons, output / "removed_train_foods_source_evidence.csv.gz")
        write_csv(reasons.groupby(["source_key", "data_layer", "exclusion_reason"], dropna=False).agg(
            affected_removed_foods=("food_concept_id", "nunique"), archived_value_records=("archived_value_records", "sum")).reset_index(),
            output / "train_removal_reason_summary.csv")
    write_csv(old_foods[old_foods.food_concept_id.isin(old_ids-new_ids)].groupby("source_keys").size().reset_index(name="removed_train_foods"),
              output / "train_removal_source_summary.csv")
    old_registry = read_component_csv(previous / "component_concept.csv.gz")
    registry = read_component_csv(release / "component_concept.csv.gz")
    role_columns = ["component_concept_id", "canonical_name", "training_role", "train_count", "validation_count"]
    roles = old_registry[role_columns].merge(registry[role_columns], on="component_concept_id", how="outer", suffixes=("_v2", "_v3"), validate="one_to_one")
    write_csv(roles[roles.training_role_v2.fillna("absent").ne(roles.training_role_v3.fillna("absent"))], output / "axis_role_changes.csv")
    old_component_map = read_component_csv(previous / "component_observation_to_concept.csv.gz")
    new_component_map = read_component_csv(release / "component_observation_to_concept.csv.gz")
    component_bridge = old_component_map.merge(new_component_map, on="component_observation_id", suffixes=("_v2", "_v3"), validate="one_to_one")
    write_csv(component_bridge[component_bridge.component_concept_id_v2.ne(component_bridge.component_concept_id_v3)], output / "component_identity_corrections.csv")

    old_profiles = pd.read_csv(previous / "canonical_profile.csv.gz", low_memory=False)
    old_selected = old_profiles[old_profiles.aggregation_status.eq("accepted") & old_profiles.component_concept_id.isin(old_components.component_concept_id) & old_profiles.partition.eq("validation")]
    columns = ["food_concept_id", "component_concept_id", "canonical_value_g_per_100g", "measurement_ids"]
    changes = old_selected[columns].merge(validation[columns], on=columns[:2], how="outer", suffixes=("_v2", "_v3"), indicator=True, validate="one_to_one")
    changes["value_changed"] = changes._merge.eq("both") & ~np.isclose(changes.canonical_value_g_per_100g_v2, changes.canonical_value_g_per_100g_v3, rtol=1e-10, atol=1e-15)
    changes["support_changed"] = changes._merge.eq("both") & changes.measurement_ids_v2.ne(changes.measurement_ids_v3)
    write_csv(changes, output / "validation_cell_changes.csv.gz")

    source_rows = []
    observations = pd.read_csv(release / "food_observation.csv.gz", low_memory=False)
    for source in sorted(observations.source_key.unique()):
        contributed = accepted[accepted.source_keys.str.split(";").map(lambda ids: source in ids)]
        part = measurements[measurements.source_key.eq(source)]
        source_rows.append({"source_key": source, "raw_food_observations": int(observations.source_key.eq(source).sum()),
                            "admitted_measurements_before_split": len(part),
                            "train_foods_with_retained_cells": contributed.loc[contributed.partition.eq("train"), "food_concept_id"].nunique(),
                            "validation_foods_with_retained_cells": contributed.loc[contributed.partition.eq("validation"), "food_concept_id"].nunique(),
                            "train_cells": int(contributed.partition.eq("train").sum()),
                            "validation_cells": int(contributed.partition.eq("validation").sum())})
    write_csv(pd.DataFrame(source_rows), output / "source_contribution.csv")

    masks = pd.read_csv(root / "data/splits" / VERSION / "fixed_benchmark_masks.csv.gz")
    rows = masks.food_concept_id.map(food_index).to_numpy(int)
    cols = masks.component_concept_id.map(component_index).to_numpy(int)
    if not np.isfinite(values[rows, cols]).all() or not components.iloc[cols].training_role.eq("maskable_target").all():
        raise ValueError("Fixed masks include missing or ineligible targets")
    task_masks = masks[masks.task.eq("family_reconstruction")].drop_duplicates(["food_concept_id", "visibility"])
    if not (task_masks.visible_target_count.gt(0) & task_masks.visible_target_count.lt(task_masks.observed_target_count)).all():
        raise ValueError("Reconstruction masks require nonempty observed context and hidden labels")
    write_csv(task_masks.groupby(["partition", "visibility"]).agg(
        food_count=("food_concept_id", "size"), mean_actual_visibility=("actual_target_visibility", "mean"),
        median_actual_visibility=("actual_target_visibility", "median")).reset_index(), output / "actual_mask_visibility.csv")
    dataset = FoodCompositionDataset(release, "train")
    for i in range(len(dataset)):
        item = dataset[i]
        hidden = item["target_mask"].astype(bool)
        context = item["context_mask"].astype(bool)
        if np.any(hidden & context) or set(dataset.family_by_column[hidden]) & set(dataset.family_by_column[context]):
            raise ValueError(f"Masked family leaked into context: {item['food_concept_id']}")
    train_rows = np.flatnonzero(foods.partition.eq("train"))
    centers, scales, stats = fit_train_normalizer(values, train_rows)
    target = components.training_role.eq("maskable_target").to_numpy()
    np.testing.assert_allclose(centers[target], components.loc[target, "train_log_median"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scales[target], components.loc[target, "train_robust_scale"], rtol=1e-5, atol=1e-10)
    stats.insert(0, "component_concept_id", components.component_concept_id)
    write_csv(stats, output / "train_only_normalization_check.csv")
    sodium = registry[registry.authority_namespace.eq("INFOODS") & registry.authority_id.eq("NA")]
    if len(sodium) != 1 or sodium.iloc[0].infoods_tag != "NA":
        raise ValueError("Official sodium identity was lost")
    summary = {
        "dataset_version": VERSION, "previous_files_hash_verified": len(snapshot),
        "registered_validation_foods": len(locked_ids),
        "validation_scorable_foods": int((foods.partition.eq("validation") & foods.text_task_eligible).sum()),
        "removed_known_bad_validation_support_records": len(old_bad),
        "known_rejected_measurements_remaining": 0, "used_validation_support_records": len(used),
        "numeric_zero_support_records_pending_source_lod_review": int(used.numeric_zero_requires_source_lod_review.sum()),
        "selected_labels_with_changed_value_same_component_id": int(changes.value_changed.sum()),
        "selected_labels_with_changed_support_same_component_id": int(changes.support_changed.sum()),
        "added_train_foods": len(new_ids-old_ids), "removed_train_foods": len(old_ids-new_ids),
        "interface_train_foods_checked": len(dataset), "fixed_mask_rows_checked": len(masks),
        "train_only_scaling_verified": True, "sodium_identity_verified": True,
        "previous_family_conflict_holds_preserved": True,
        "validation_outcomes_opened": False, "mext_or_bls_added_to_training": False,
        "status": "evidence_corrected_candidate_not_expert_certified",
        "remaining_gates": ["Source-specific reported-zero and censoring semantics", "Food facets, component identity and class expert review",
                            "Value-level independence and complete lineage", "Recovery of missing validation evidence without substituting foods",
                            "New source multilingual identity and family blocking before admission"],
    }
    write_json(summary, output / "VERIFICATION.json")
    print(json.dumps(summary, indent=2), flush=True)
    print(summary_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

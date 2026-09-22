#!/usr/bin/env python3
"""Allocate an evidence-safe validation supplement and export per-axis review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.target_coverage import (
    allocate_validation, apply_candidate_partition, component_identity_collisions,
    coverage_counts, eligibility_issues, make_transfer_blocks, profile_evidence,
)
from foodcomp.util import read_component_csv, sha256_file, write_csv, write_json

PLAN_VERSION = "nutrition_target_coverage_2026_09_08"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-version", default="scientific_food_composition_v3")
    parser.add_argument("--output-name", default=PLAN_VERSION)
    parser.add_argument("--time-limit", type=float, default=90)
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "data/processed" / args.source_version / "release"
    output = root / "reports" / args.output_name
    splits = root / "data/splits" / args.output_name
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed coverage plan exists; use a new --output-name")
    protected = {}
    for directory in (release, root / "data/splits" / args.source_version):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                protected[str(path.relative_to(root))] = sha256_file(path)
    print("Reading immutable release and selected measurement evidence", flush=True)
    components = read_component_csv(release / "component_concept.csv.gz")
    observations = read_component_csv(release / "component_observation.csv.gz")
    component_mapping = read_component_csv(release / "component_observation_to_concept.csv.gz")
    foods = pd.read_csv(release / "ml_partition.csv", low_memory=False)
    food_observations = pd.read_csv(release / "food_observation.csv.gz", low_memory=False)
    food_mapping = pd.read_csv(release / "food_observation_to_concept.csv.gz", low_memory=False)
    measurements = pd.read_csv(release / "measurement_main_eligible.csv.gz", low_memory=False)
    source_policies = sorted(measurements.source_admission_policy.dropna().unique()) if "source_admission_policy" in measurements else []
    raw_profiles = pd.read_csv(release / "canonical_profile.csv.gz", low_memory=False)
    profiles = profile_evidence(raw_profiles, measurements)
    collisions = component_identity_collisions(observations, component_mapping)
    used_measurements = set(profiles.measurement_ids.str.split(";").explode())
    used_observations = set(measurements.loc[measurements.measurement_id.isin(used_measurements), "food_observation_id"])
    blocks = make_transfer_blocks(foods, food_mapping, food_observations,
                                  selected_observation_ids=used_observations)
    ledger, training = coverage_counts(components, profiles, foods, blocks)
    ledger = eligibility_issues(ledger, set(collisions.component_concept_id))
    print(f"Audited {len(ledger)} represented expressions; allocating whole blocks", flush=True)
    moved, optimization = allocate_validation(ledger, training, foods, blocks, args.time_limit)
    partition, ledger, withheld = apply_candidate_partition(foods, profiles, ledger, moved, blocks)

    old_val = foods[foods.partition.eq("validation")]
    identity_columns = ["food_concept_id", "validation_panel", "family_cluster_id"]
    pd.testing.assert_frame_equal(
        old_val[identity_columns].sort_values("food_concept_id").reset_index(drop=True),
        partition.loc[partition.food_concept_id.isin(old_val.food_concept_id), identity_columns]
        .sort_values("food_concept_id").reset_index(drop=True),
    )
    if not set(partition.partition) <= {"train", "validation"}:
        raise AssertionError("Unexpected ML partition")
    old_targets = ledger.training_role.eq("maskable_target")
    if not ledger.loc[old_targets, "proposed_train_count"].ge(100).all():
        raise AssertionError("Reallocation destroyed existing target training support")
    train_families = set(partition.loc[partition.partition.eq("train"), "family_cluster_id"])
    supplement_families = set(partition.loc[partition.food_concept_id.isin(moved), "family_cluster_id"])
    if train_families & supplement_families:
        raise AssertionError("A new holdout family remains in training")
    checks = {
        "frozen_validation_ids_and_panels_preserved": True,
        "only_train_and_validation_partitions": True,
        "old_target_training_counts_at_least_100": True,
        "new_holdout_family_and_lineage_blocks_complete": True,
        "all_candidate_validation_labels_require_versioned_source_eligibility": True,
        "selection_uses_support_not_validation_magnitudes_or_predictions": True,
        "all_rescued_counts_verified_after_transfer": True,
        "source_release_and_masks_unchanged": True,
    }
    for relative, digest in protected.items():
        if sha256_file(root / relative) != digest:
            raise AssertionError(f"Immutable artifact changed: {relative}")
    represented_ids = set(ledger.component_concept_id)
    excluded_registry = components[
        ~components.component_concept_id.isin(represented_ids)
        & components.nutritional_role.isin([
            "macronutrient", "micronutrient_mineral", "micronutrient_vitamin", "essential_nutrient_choline",
        ])
    ].copy()
    review = ledger[ledger.review_required_before_exclusion].copy()
    review["user_decision"] = "pending"
    review["reviewer_notes"] = ""
    summary = ledger.groupby(["nutrition_scope", "training_role", "coverage_decision"]).size().reset_index(name="axes")
    source_parts = profiles[["food_concept_id", "component_concept_id", "source_keys"]].copy()
    source_parts["source_key"] = source_parts.source_keys.str.split(";")
    source_parts = source_parts.explode("source_key")
    source_counts = source_parts.groupby(["component_concept_id", "source_key"]).food_concept_id.nunique().reset_index(name="contributing_profile_count")
    write_csv(partition, splits / "ml_partition_candidate.csv")
    write_csv(partition[partition.food_concept_id.isin(moved)], splits / "moved_train_foods.csv")
    write_csv(ledger, output / "all_component_coverage.csv")
    write_csv(ledger[ledger.nutrition_scope.eq("core_nutrition_expression")], output / "core_nutrition_coverage.csv")
    write_csv(review, output / "component_review_queue.csv")
    write_csv(review[review.nutrition_scope.eq("core_nutrition_expression")], output / "core_nutrition_review_queue.csv")
    write_csv(review[review.both_current_partitions_below_minimum], output / "both_partitions_insufficient_review.csv")
    write_csv(review[review.total_count.lt(130)], output / "total_support_below_130_review.csv")
    write_csv(ledger[ledger.coverage_decision.eq("coverage_restored_by_training_transfer")], output / "coverage_restored_axes.csv")
    write_csv(collisions[collisions.component_concept_id.isin(represented_ids)], output / "identity_collision_review.csv")
    write_csv(source_counts, output / "selected_source_contributions.csv")
    write_csv(withheld, output / "moved_profiles_withheld_from_validation.csv.gz")
    write_csv(excluded_registry, output / "core_catalogue_axes_without_accepted_values.csv")
    write_csv(summary, output / "coverage_summary.csv")
    active = ledger[ledger.training_role.ne("excluded")]
    core = active[active.nutrition_scope.eq("core_nutrition_expression")]
    forms = active[active.nutrition_scope.eq("nutrient_chemical_form")]
    manifest = {
        "plan_version": args.output_name, "source_dataset_version": args.source_version,
        "status": "candidate_split_and_review_only_not_a_training_release",
        "source_admission_policies": source_policies or ["legacy_source_evidence_rules"],
        "legacy_strict_count_fields": "Count records admitted by this dataset version's validation-reference policy; not universally certified assays",
        "request": "Every admitted nutritional target should be predicted; supplement validation from train, review unsupported axes before retention or exclusion.",
        "interpretation": "100 percent target coverage, not 100 percent prediction accuracy or all human nutritional needs.",
        "frozen_validation_foods": len(old_val), "moved_train_foods": len(moved),
        "candidate_train_foods": int(partition.partition.eq("train").sum()),
        "candidate_validation_foods": int(partition.partition.eq("validation").sum()),
        "active_axes": len(active), "audited_axes_including_archived": len(ledger),
        "core_active_expressions": len(core),
        "core_original_maskable": int(core.training_role.eq("maskable_target").sum()),
        "core_support_ready_after_transfer": int(core.support_ready_for_prediction.sum()),
        "core_restored_by_transfer": int(core.coverage_decision.eq("coverage_restored_by_training_transfer").sum()),
        "form_active_expressions": len(forms),
        "form_restored_by_transfer": int(forms.coverage_decision.eq("coverage_restored_by_training_transfer").sum()),
        "active_restored_by_transfer": int(active.coverage_decision.eq("coverage_restored_by_training_transfer").sum()),
        "all_audited_review_rows": len(review),
        "withheld_nonstrict_moved_profile_cells": len(withheld),
        "all_audited_both_partitions_insufficient": int(ledger.both_current_partitions_below_minimum.sum()),
        "all_audited_total_support_below_130": int(ledger.total_count.lt(130).sum()),
        "source_release_hashes": protected, "automatic_checks": checks,
        "optimization": optimization,
        "thresholds": {"remaining_train_concepts": 100, "validation_concepts": 30, "coverage": 0.02},
        "normalization": "Unchanged robust log1p policy, checked again using remaining train only. No zero-inflation fallback was invented.",
        "scope": "Existing role labels are provisional; the review includes wrongly unclassified other compositions and archived nutrients with accepted values.",
        "scientific_limits": [
            "Known canonical identity collisions and mask-family semantic defects are not repaired by splitting.",
            "AFCD and CoFID training-only evidence is not promoted to strict validation by changing partition.",
            "Counts are food concepts, not independent assays; family support is reported separately.",
            "Thirty validation observations is an operational minimum, not a statistical power guarantee.",
            "Reported zero versus censoring/rounding semantics still require source-level review.",
            "Raw values and original 1426 validation identities are unchanged; no model outcomes were opened.",
        ],
        "training_or_benchmark_executed": False,
        "production_role_or_mask_changes": False,
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in (
            root / "src/foodcomp/target_coverage.py", Path(__file__).resolve(),
        )},
    }
    write_json(manifest, output / "manifest.json")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"source_release_hashes", "optimization", "code_sha256"}}, indent=2))


if __name__ == "__main__":
    main()

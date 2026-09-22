"""Build a source-excluded, FooDB-centred candidate without overwriting v4.

Colab: pip install -r requirements-colab.txt
       python scripts/review_foodb_centered_v5.py
       python scripts/build_scientific_food_composition_v5.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.foodb_centered import (
    ALLOWED_SOURCES, EXCLUDED_SOURCES, recover_orphans, prepare_identities,
    virtual_import_components, harmonization_view, merge_equivalent_concepts,
    foo_first_selection, name_key, retain_internal_reference_values,
)
from foodcomp.frozen_release import assign_frozen_partitions
from foodcomp.harmonize import aggregate_profiles, build_component_concepts, classify_training_roles, build_profile_matrix
from foodcomp.ontology import parse_infoods_tagnames
from foodcomp.source_policy import apply_source_policy, TRUSTED_REFERENCE_POLICY
from foodcomp.sources import gate_component_expressions
from foodcomp.util import read_component_csv, write_csv, write_json, sha256_file

VERSION = "scientific_food_composition_v5_1"


def read(path):
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])


def main(*, version=VERSION, revision=False, single_record=False):
    VERSION = version
    previous = ROOT / "data/processed/scientific_food_composition_v4/release"
    destination = ROOT / "data/processed" / VERSION
    output = destination / "release"
    audit = ROOT / "data/audits" / VERSION / "dataset"
    review = ROOT / "data/audits/scientific_food_composition_v5/review"
    if (destination / "build_manifest.json").exists():
        raise FileExistsError(f"Completed {VERSION} release exists; create a new version instead of overwriting it")
    snapshot = {str(p.relative_to(ROOT)): sha256_file(p) for p in previous.glob("*") if p.is_file()}
    infoods = parse_infoods_tagnames(ROOT / "data/raw/reference/infoods_tagnames_2022")
    metadata = read(review / "content_identity_metadata.csv.gz").fillna("")
    metadata["id"] = metadata.id.astype(str)
    metadata["orig_source_id"] = metadata.orig_source_id.astype(str).str.replace(r"\.0$", "", regex=True)
    inventory = pd.read_csv(review / "orphan_compound_inventory.csv", dtype=str, keep_default_na=False)
    original_foods = read(previous / "food_observation.csv.gz").fillna("")
    foods = original_foods[original_foods.source_key.isin(ALLOWED_SOURCES)].copy()
    mapping = read(previous / "food_observation_to_concept.csv.gz").fillna("")
    mapping = mapping[mapping.food_observation_id.isin(foods.food_observation_id)].copy()
    previous_partitions = read(previous / "ml_partition.csv").fillna("")
    locked_ids = set(previous_partitions.loc[previous_partitions.partition.eq("validation"), "food_concept_id"])
    concepts = read(previous / "food_concept.csv.gz").fillna("")
    concepts = concepts[concepts.food_concept_id.isin(set(mapping.food_concept_id) | locked_ids)].copy()
    links = mapping.merge(foods, on="food_observation_id", validate="one_to_one")
    foo_ids = set(links.loc[links.source_key.eq("foodb"), "food_concept_id"])
    concepts["dataset_layer"] = np.where(concepts.food_concept_id.isin(foo_ids), "foodb_anchor", "supplementary_food_extension")
    counts = links.groupby("food_concept_id").source_key.agg(lambda x: ";".join(sorted(set(x))))
    concepts["source_keys"] = concepts.food_concept_id.map(counts).fillna("")
    concepts["source_count"] = concepts.source_keys.map(lambda x: len(x.split(";")) if x else 0)
    concepts["observation_count"] = concepts.food_concept_id.map(links.groupby("food_concept_id").size()).fillna(0).astype(int)
    foo_labels = links[links.source_key.eq("foodb")].sort_values("food_observation_id").drop_duplicates("food_concept_id").set_index("food_concept_id")
    concepts["previous_canonical_name"] = concepts.canonical_name
    concepts["canonical_name"] = concepts.food_concept_id.map(foo_labels.original_name).fillna(concepts.canonical_name)
    concepts["canonical_name_basis"] = np.where(concepts.food_concept_id.isin(foo_ids), "original_FooDB_Content_name_and_existing_food_facets", "supplementary_official_source_name")
    # Remove excluded-source metadata, not only its numerical cells.
    source_order = {s: i for i, s in enumerate(["foodb", "usda_sr_legacy", "cnf", "frida", "ciqual", "usda_foundation"])}
    representatives = links.assign(priority=links.source_key.map(source_order)).sort_values(["priority", "food_observation_id"]).drop_duplicates("food_concept_id").set_index("food_concept_id")
    for column in ["food_group", "food_subgroup", "food_type"]:
        concepts[column] = concepts.food_concept_id.map(representatives[column]).fillna("")

    observations = read_component_csv(previous / "component_observation.csv.gz").fillna("")
    observations = observations[observations.source_key.isin(ALLOWED_SOURCES)].copy()
    observations, orphan_review = recover_orphans(observations, inventory, ROOT)
    observations = prepare_identities(observations, infoods)
    old_main = read(previous / "measurement_main_eligible.csv.gz")
    parts = [old_main[old_main.source_key.isin(ALLOWED_SOURCES - {"foodb", "cnf", "usda_sr_legacy"})].copy()]
    staging = ROOT / "data/processed/scientific_food_composition_v3/staging"
    dispositions = []
    paths = [p for s in ["foodb", "cnf", "usda_sr_legacy"] for p in sorted((staging / s).glob("measurement_part_*.csv.gz"))]
    for path in paths:
        frame = apply_source_policy(read(path), TRUSTED_REFERENCE_POLICY)
        frame = retain_internal_reference_values(frame)
        frame = gate_component_expressions(frame, observations, infoods)
        keep = frame.main_value_eligible
        dispositions.append(frame.groupby(["main_value_eligible", "exclusion_reason"], dropna=False).size().reset_index(name="records"))
        if keep.any():
            parts.append(frame[keep].copy())
    measurements = pd.concat(parts, ignore_index=True)
    if measurements.measurement_id.duplicated().any():
        raise ValueError("Duplicate measurement IDs before identity resolution")
    print(f"Allowed source measurement candidates: {len(measurements):,}", flush=True)
    measurements, observations = virtual_import_components(measurements, observations, metadata)
    observations = prepare_identities(observations, infoods)
    observations = observations[observations.component_observation_id.isin(measurements.component_observation_id)].copy()
    observations = observations.drop_duplicates("component_observation_id")
    source_view = harmonization_view(observations)
    component_concepts, component_mapping, evidence, component_review = build_component_concepts(source_view, ROOT, infoods)
    component_concepts, component_mapping = merge_equivalent_concepts(component_concepts, component_mapping, observations, infoods)
    if revision:
        from foodcomp.composition_revision import revise_compositions, ROLE_POLICY
        component_concepts, component_mapping, decisions, holds = revise_compositions(component_concepts, component_mapping)
        write_csv(decisions, audit / "composition_deduplication_ledger.csv")
        write_csv(holds, audit / "composition_exclusion_ledger.csv")
        held_obs = set(component_mapping.loc[component_mapping.component_concept_id.isin(holds.component_concept_id), "component_observation_id"])
        write_csv(measurements[measurements.component_observation_id.isin(held_obs)], audit / "explicitly_removed_measurements.csv.gz")
        measurements = measurements[~measurements.component_observation_id.isin(held_obs)].copy()
    write_csv(observations, output / "component_observation.csv.gz")
    write_csv(component_concepts, audit / "component_identity_before_filter.csv")
    write_csv(component_mapping, output / "component_observation_to_concept.csv.gz")
    write_csv(component_mapping.merge(observations, on="component_observation_id", validate="one_to_one"), audit / "component_identity_merge_ledger.csv.gz")
    unresolved = set(component_concepts.loc[component_concepts.identity_status.eq("unresolved_component_identity"), "component_concept_id"])
    bad_obs = set(component_mapping.loc[component_mapping.component_concept_id.isin(unresolved), "component_observation_id"])
    identity_held = measurements[measurements.component_observation_id.isin(bad_obs)]
    write_csv(identity_held, audit / "unresolved_component_measurements.csv.gz")
    if not revision:
        measurements = measurements[~measurements.component_observation_id.isin(bad_obs)].copy()
    # Apply the resolved expression (including original USDA codes) before
    # aggregation; this catches a mass-looking vitamin activity equivalent.
    measurements = gate_component_expressions(measurements, source_view, infoods)
    documented_hold_ids = set(observations.loc[observations.harmonization_exclusion_reason.ne(""), "component_observation_id"])
    definition_hold = measurements.exclusion_reason.isin(["activity_equivalent_not_chemical_mass", "nitrogen_denominator_not_food_mass"])
    definition_hold |= measurements.component_observation_id.isin(documented_hold_ids)
    measurements.loc[measurements.component_observation_id.isin(documented_hold_ids), "exclusion_reason"] = "documented_vitamer_equivalent_not_single_chemical_mass"
    write_csv(measurements[definition_hold], audit / "incompatible_definition_measurements.csv.gz")
    measurements = measurements[~definition_hold].copy()
    candidate_foods = set(mapping.loc[mapping.food_observation_id.isin(measurements.food_observation_id) & mapping.exclusion_flag.eq(""), "food_concept_id"])
    partitions, split_exclusions = assign_frozen_partitions(concepts, mapping, foods, candidate_foods, previous_partitions)
    frozen_val_obs = set(mapping.loc[mapping.food_concept_id.isin(locked_ids), "food_observation_id"])
    print(f"Resolved candidate components: {len(component_concepts):,}; selecting FooDB and supplemental cells", flush=True)
    if single_record:
        from foodcomp.cell_selection import SOURCE_PRIORITY, selection_policy
        selected = measurements.copy()
        write_json(selection_policy(), output / "cell_selection_policy.json")
    else:
        selected, selection_ledger = foo_first_selection(measurements, mapping, component_mapping, partitions, resolve_internal_copies=False)
        selected["main_value_eligible"] = True
        write_csv(selection_ledger, audit / "measurement_selection_ledger.csv.gz")
    print(f"Selected pre-aggregation records: {len(selected):,}", flush=True)
    profiles, conflicts = aggregate_profiles(selected, mapping, component_mapping, partitions, frozen_val_obs,
                                             source_priority=SOURCE_PRIORITY if single_record else None)
    registry, sensitivity = classify_training_roles(profiles, component_concepts, partitions, component_mapping, source_view, selected, mapping, foods, **(ROLE_POLICY if revision else {}))
    if single_record:
        selected = selected[selected.measurement_id.isin(profiles.measurement_ids)].copy()
        if len(conflicts) or not profiles.selected_measurement_count.eq(1).all():
            raise AssertionError("Single-record policy failed to resolve eligible cells")
        previous_conflicts = read(ROOT / "data/audits/scientific_food_composition_v6/dataset/unresolved_measurement_conflicts.csv")
        resolved = previous_conflicts[["food_concept_id", "component_concept_id"]].merge(
            profiles, on=["food_concept_id", "component_concept_id"], how="left", validate="one_to_one")
        if resolved.canonical_value_g_per_100g.isna().any():
            raise AssertionError("A previously unresolved cell has no selected value")
        write_csv(resolved, audit / "resolved_previous_heterogeneity_cells.csv")
    if revision:
        hold_reasons = holds.set_index("component_concept_id").reason
        held = registry.component_concept_id.isin(hold_reasons.index)
        registry.loc[held, "training_role"] = "excluded"
        registry.loc[held, "training_exclusion_reason"] = registry.loc[held, "component_concept_id"].map(hold_reasons)
        write_json(ROLE_POLICY, output / "training_role_policy.json")
    retained = registry[registry.training_role.ne("excluded")]
    if single_record:
        resolved["axis_retained"] = resolved.component_concept_id.isin(retained.component_concept_id)
        write_csv(resolved, audit / "resolved_previous_heterogeneity_cells.csv")
    duplicate_names = retained[retained.canonical_name.map(name_key).duplicated(False)]
    write_csv(duplicate_names, audit / "duplicate_names_requiring_review.csv")
    if len(duplicate_names):
        write_csv(registry, output / "component_concept.csv.gz")
        write_csv(profiles, output / "canonical_profile.csv.gz")
        raise ValueError("Retained composition names are not unique; inspect duplicate_names_requiring_review.csv")
    retained_values = profiles[profiles.aggregation_status.eq("accepted") & profiles.component_concept_id.isin(retained.component_concept_id)]
    active_ids = set(retained_values.food_concept_id)
    partitions["has_retained_values"] = partitions.food_concept_id.isin(active_ids)
    write_csv(partitions[~partitions.has_retained_values], audit / "empty_registered_foods.csv")
    partitions = partitions[partitions.has_retained_values | partitions.partition.eq("validation")].copy()
    target_ids = set(retained.loc[retained.training_role.eq("maskable_target"), "component_concept_id"])
    targets = retained_values[retained_values.component_concept_id.isin(target_ids)]
    family_counts = targets.merge(registry[["component_concept_id", "component_family"]], on="component_concept_id").groupby("food_concept_id").component_family.nunique()
    partitions["mask_family_count"] = partitions.food_concept_id.map(family_counts).fillna(0).astype(int)
    partitions["text_task_eligible"] = partitions.food_concept_id.isin(set(targets.food_concept_id))
    partitions["reconstruction_task_eligible"] = partitions.mask_family_count.ge(2)
    partitions["benchmark_eligible"] &= partitions.has_retained_values
    matrix = build_profile_matrix(profiles, partitions, registry, output / "canonical_profile_matrix.npz")

    for name, frame in {"food_observation": foods, "food_concept": concepts, "food_observation_to_concept": mapping,
                        "component_concept": registry, "canonical_profile": profiles, "measurement_main_eligible": selected}.items():
        write_csv(frame, output / f"{name}.csv.gz")
    write_csv(partitions, output / "ml_partition.csv")
    write_csv(previous_partitions[previous_partitions.partition.eq("validation")], destination / "frozen_validation_identity_manifest.csv")
    hierarchy = json.loads((previous / "food_hierarchy.json").read_text())
    hierarchy["dataset_version"] = VERSION
    hierarchy["provenance_note"] = "Existing family grouping retained; excluded sources cannot supply new food metadata or values. Historical hierarchy retained for split continuity."
    write_json(hierarchy, output / "food_hierarchy.json")
    write_csv(conflicts, audit / "unresolved_measurement_conflicts.csv")
    write_csv(split_exclusions, audit / "frozen_split_exclusion_ledger.csv")
    write_csv(sensitivity, audit / "coverage_threshold_sensitivity.csv")
    write_csv(evidence, audit / "component_evidence_ledger.csv.gz")
    write_csv(component_review, audit / "component_review_queue.csv")
    write_csv(registry, audit / "component_training_decision_ledger.csv.gz")
    actual = profiles[profiles.aggregation_status.eq("accepted") & profiles.food_concept_id.isin(partitions.food_concept_id) & profiles.component_concept_id.isin(retained.component_concept_id)]
    selected_ids = set(actual.measurement_ids.str.split(";").explode())
    actual_measurements = selected[selected.measurement_id.isin(selected_ids)]
    actual_measurements = actual_measurements.merge(mapping[["food_observation_id", "food_concept_id"]], on="food_observation_id")
    actual_measurements = actual_measurements.merge(component_mapping[["component_observation_id", "component_concept_id"]], on="component_observation_id")
    source_counts = actual_measurements.groupby("source_key").agg(foods=("food_concept_id", "nunique"), observations=("food_observation_id", "nunique"), axes=("component_concept_id", "nunique"), selected_records=("measurement_id", "size"))
    write_csv(source_counts.reset_index(), audit / "source_contribution_counts.csv")
    orphan_map = component_mapping.merge(orphan_review[["component_observation_id", "source_id"]], on="component_observation_id")
    orphan_map = orphan_map.merge(registry[["component_concept_id", "canonical_name", "training_role", "train_count", "validation_count", "training_exclusion_reason"]], on="component_concept_id")
    actual_orphan = actual_measurements.groupby("original_component_observation_id").agg(selected_records=("measurement_id", "size"), contributing_foods=("food_concept_id", "nunique"))
    orphan_review = orphan_review.merge(orphan_map, on=["component_observation_id", "source_id"], how="left")
    orphan_review = orphan_review.merge(actual_orphan, left_on="component_observation_id", right_index=True, how="left")
    write_csv(orphan_review, audit / "orphan_compound_final_decisions.csv")
    remaining_ids = set(partitions.food_concept_id)
    summary = {"dataset_version": VERSION, "status": "candidate_pending_domain_review", "anchor": "FooDB",
               "excluded_direct_sources": sorted(EXCLUDED_SOURCES), "allowed_sources": sorted(ALLOWED_SOURCES),
               "internal_citation_policy": "preserve_source_published_values_and_references_no_origin_readjudication",
               "food_observations": len(foods), "food_concepts": len(concepts), "ml_food_concepts": len(partitions),
               "nonempty_food_concepts": len(active_ids & remaining_ids),
               "foodb_anchor_nonempty_foods": len(active_ids & remaining_ids & foo_ids),
               "supplementary_nonempty_foods": len((active_ids & remaining_ids) - foo_ids),
               "train_food_concepts": int(partitions.partition.eq("train").sum()),
               "validation_identities_preserved": int(partitions.partition.eq("validation").sum()),
               "validation_nonempty_foods": int((partitions.partition.eq("validation") & partitions.has_retained_values).sum()),
               "maskable_targets": int(retained.training_role.eq("maskable_target").sum()),
               "context_only_components": int(retained.training_role.eq("context_only").sum()), "matrix": matrix,
               "selected_records": len(actual_measurements), "unresolved_conflicts": len(conflicts),
               "orphan_ids_in_inventory": len(inventory), "orphan_ids_with_selected_values": int(orphan_review.selected_records.fillna(0).gt(0).sum()),
               "orphan_selected_values": int(orphan_review.selected_records.fillna(0).sum()),
               "models_trained": False, "validation_outcomes_opened": False,
               "target_eligibility_policy": ROLE_POLICY if revision else "historical_v5_1",
               "unique_composition_names": True, "unique_food_component_cells": not actual.duplicated(["food_concept_id", "component_concept_id"]).any()}
    if single_record:
        summary["cell_selection_policy"] = selection_policy()
        summary["previous_heterogeneity_resolution"] = {
            "previous_groups": len(resolved), "resolved_groups": int(resolved.canonical_value_g_per_100g.notna().sum()),
            "retained_axis_groups": int(resolved.axis_retained.sum()),
            "winner_sources": resolved.source_keys.value_counts().to_dict(),
        }
        summary["single_record_selection"] = {
            "candidate_cells": len(profiles),
            "multiple_record_cells": int(profiles.candidate_record_count.gt(1).sum()),
            "multiple_source_cells": int(profiles.candidate_source_count.gt(1).sum()),
            "numerically_differing_cells": int(profiles.candidate_value_disagreement.sum()),
        }
    if set(actual_measurements.source_key) & EXCLUDED_SOURCES:
        raise AssertionError("Excluded numerical source remains")
    if set(partitions.loc[partitions.partition.eq("validation"), "food_concept_id"]) != locked_ids:
        raise AssertionError("Frozen validation identities changed")
    for relative, digest in snapshot.items():
        if sha256_file(ROOT / relative) != digest:
            raise AssertionError("Immutable v4 input changed: " + relative)
    write_json(summary, output / "dataset_summary.json")
    write_json({**summary, "protected_v4_hashes": snapshot, "build_script": str(Path(__file__).relative_to(ROOT))}, destination / "build_manifest.json")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(source_counts.to_string(), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit frozen validation label evidence and MEXT, without changing the release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.label_evidence import foundation_result_links, gate_reference_evidence, inspect_reference, reference_catalogs
from foodcomp.mext_audit import audit_mext
from foodcomp.util import sha256_file, write_csv, write_json


def read(path: Path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)


def validation_audit(root: Path, output: Path) -> dict:
    base = root / "data/processed/scientific_food_composition_v2"
    release = base / "release"
    foods = read(release / "ml_partition.csv")
    locked = foods[foods.partition.eq("validation")]
    if len(locked) != 1426 or not locked.food_concept_id.is_unique:
        raise ValueError("Frozen validation must contain the same 1,426 unique registered foods")
    components = read(release / "component_concept.csv.gz")
    retained = components[components.training_role.ne("excluded")]
    targets = set(retained.loc[retained.training_role.eq("maskable_target"), "component_concept_id"])
    profiles = read(release / "canonical_profile.csv.gz")
    selected = profiles[profiles.partition.eq("validation") & profiles.aggregation_status.eq("accepted") & profiles.component_concept_id.isin(retained.component_concept_id)].copy()
    selected["is_maskable_axis"] = selected.component_concept_id.isin(targets)
    links = selected[["food_concept_id", "component_concept_id", "is_maskable_axis", "measurement_ids"]].copy()
    links["measurement_id"] = links.measurement_ids.str.split(";")
    links = links.drop(columns="measurement_ids").explode("measurement_id")
    measurements = read(release / "measurement_main_eligible.csv.gz")
    collisions = measurements[measurements.duplicated("measurement_id", keep=False)]
    write_csv(collisions, output / "existing_measurement_id_collisions.csv")
    supporting = measurements[measurements.measurement_id.isin(links.measurement_id)]
    if set(supporting.measurement_id) != set(links.measurement_id):
        raise ValueError("A selected validation label has no underlying measurement evidence")
    labels = links.merge(supporting, on="measurement_id", validate="many_to_one")
    observations = read(release / "food_observation.csv.gz")
    labels = labels.merge(observations[["food_observation_id", "source_food_id", "original_name", "source_record_url"]], on="food_observation_id", validate="many_to_one")
    labels = labels.merge(retained[["component_concept_id", "canonical_name", "component_family", "identity_status"]], on="component_concept_id", validate="many_to_one")
    catalogs = reference_catalogs(root)
    evidence_rows = []
    for source in ("frida", "ciqual"):
        for ref in labels.loc[labels.source_key.eq(source), "source_reference"].unique():
            evidence_rows.append({"source_key": source, "source_reference": ref, **inspect_reference(source, ref, catalogs)})
    references = pd.DataFrame(evidence_rows)
    labels = labels.merge(references, on=["source_key", "source_reference"], how="left", validate="many_to_one")
    labels["evidence_issues"] = labels.evidence_issues.fillna("")
    sr_derived = labels.source_key.eq("usda_sr_legacy") & labels.method_expression.str.contains("Recipe|Based on physical composition", case=False, regex=True)
    labels.loc[sr_derived, "evidence_issues"] = "SR_recipe_or_physical_composition_not_direct_analysis"
    labels["strict_evidence_rejected"] = labels.evidence_issues.ne("")
    raw = pd.to_numeric(labels.numeric_value, errors="coerce")
    factor = pd.to_numeric(labels.conversion_factor, errors="coerce")
    converted = pd.to_numeric(labels.normalized_value_g_per_100g, errors="coerce")
    labels["exact_conversion_matches"] = np.isclose(raw * factor, converted, rtol=1e-10, atol=1e-15)
    labels["sample_count_available"] = pd.to_numeric(labels.sample_count, errors="coerce").gt(0)
    labels["numeric_zero_needs_source_specific_lod_review"] = raw.eq(0)
    labels["identity_review_pending"] = labels.identity_status.ne("authority_verified")
    labels["current_strict_flag_is_not_expert_certification"] = True
    foundation_links = foundation_result_links(root / "data/raw/usda/foundation_2026_04_30")
    nutrient_obs = read(release / "component_observation.csv.gz")
    foundation = labels[labels.source_key.eq("usda_foundation")].merge(
        nutrient_obs[["component_observation_id", "source_component_id"]], on="component_observation_id", validate="many_to_one")
    foundation["parent_fdc_id"] = foundation.source_food_id.astype(int)
    foundation["nutrient_id"] = foundation.source_component_id.astype(int)
    foundation = foundation.merge(foundation_links, on=["parent_fdc_id", "nutrient_id"], how="left", validate="many_to_one")
    foundation["catalogue_method_list_is_not_measurement_specific"] = True
    method_lookup = foundation.set_index("measurement_id").actual_linked_method_labels.to_dict()
    labels["actual_linked_method_labels"] = labels.measurement_id.map(method_lookup).fillna("")
    labels["measurement_specific_method_available"] = labels.actual_linked_method_labels.ne("")
    bad = labels[labels.strict_evidence_rejected].copy()
    rejected_cells = bad[["food_concept_id", "component_concept_id"]].drop_duplicates()
    kept = selected.merge(rejected_cells.assign(rejected_on_reference_check=True), on=["food_concept_id", "component_concept_id"], how="left", validate="one_to_one")
    kept = kept[~kept.rejected_on_reference_check.eq(True)]
    current_scorable = set(selected.loc[selected.is_maskable_axis, "food_concept_id"])
    after_scorable = set(kept.loc[kept.is_maskable_axis, "food_concept_id"])
    counts = labels.groupby("food_concept_id").agg(
        selected_measurement_records=("measurement_id", "size"),
        reference_rejected_records=("strict_evidence_rejected", "sum"),
        source_zero_review_records=("numeric_zero_needs_source_specific_lod_review", "sum"),
        measurements_with_reported_sample_count=("sample_count_available", "sum"),
        measurements_with_actual_method_links=("measurement_specific_method_available", "sum"),
    ).reset_index()
    full_foods = locked.merge(counts, on="food_concept_id", how="left", validate="one_to_one")
    for name in counts.columns[1:]:
        full_foods[name] = full_foods[name].fillna(0).astype(int)
    full_foods["currently_scorable"] = full_foods.food_concept_id.isin(current_scorable)
    full_foods["still_scorable_after_reference_rejections_only"] = full_foods.food_concept_id.isin(after_scorable)
    full_foods["counterfactual_only_release_not_modified"] = True
    write_csv(references, output / "validation_reference_evidence.csv")
    write_csv(labels, output / "validation_selected_label_evidence.csv.gz")
    write_csv(bad, output / "validation_labels_rejected_by_source_evidence.csv.gz")
    write_csv(foundation, output / "foundation_actual_result_method_links.csv.gz")
    write_csv(full_foods, output / "validation_all_1426_foods_evidence.csv")
    write_csv(labels.groupby(["source_key", "evidence_issues", "value_status"]).size().reset_index(name="records"), output / "validation_source_evidence_counts.csv")

    lost = read(root / "reports/scientific_food_composition_v2/validation_foods_without_maskable_target.csv")
    food_map = read(release / "food_observation_to_concept.csv.gz")
    mapped = lost[["food_concept_id", "canonical_name", "remaining_observed_matrix_values", "reason"]].merge(food_map[["food_observation_id", "food_concept_id"]], on="food_concept_id").merge(observations, on="food_observation_id", validate="many_to_one")
    afcd_details = pd.read_excel(root / "data/raw/expansion_2026_09_06/afcd/00_AFCD Release 3 - Food Details.xlsx", sheet_name="Food details", header=2, keep_default_na=False)
    details = afcd_details.set_index("Public Food Key")
    mapped["sampling_evidence"] = mapped.description
    afcd = mapped.source_key.eq("afcd")
    mapped.loc[afcd, "sampling_evidence"] = mapped.loc[afcd, "source_food_id"].map(details["Sampling Details"])
    mapped["food_level_analysed"] = False
    mapped.loc[afcd, "food_level_analysed"] = mapped.loc[afcd, "source_food_id"].map(details.Derivation).eq("Analysed")
    for name, pattern in {
        "mentions_imputation": r"imput|estimat|borrow", "mentions_usda": r"USDA|FDC",
        "mentions_dry_matter_adjustment": r"dry matter adjust", "mentions_recipe": r"recipe|calculated",
        "mentions_mixed_food_pool": r"pooled|mixed.*samples|various.*cuts",
        "mentions_ST19036": r"Dunlop.*2022|purchased.*2021.*2022",
    }.items():
        mapped[name] = mapped.sampling_evidence.str.contains(pattern, case=False, regex=True, na=False)
    mapped["assessment"] = np.where(mapped.remaining_observed_matrix_values.gt(0),
        "Remaining context-only axes do not meet predeclared target support/scale criteria; do not lower thresholds.",
        "Recover original same-food same-component analytical evidence; a food-level label alone does not certify every value.")
    mapped.loc[afcd, "assessment"] = "Analysed food with mixed component provenance; resolve per-component sampling, calculations and imputed zeros before label restoration."
    mapped["automatic_label_restoration_approved"] = False
    write_csv(mapped, output / "validation_missing_target_food_source_ledger.csv")
    write_csv(mapped[afcd], output / "afcd_143_food_sampling_evidence.csv")
    component_map = read(release / "component_observation_to_concept.csv.gz")
    missing_cells = []
    missing_obs_ids = set(mapped.food_observation_id)
    for source in mapped.source_key.unique():
        for path in sorted((base / "staging" / source).glob("measurement_part_*.csv.gz")):
            for chunk in pd.read_csv(path, keep_default_na=False, low_memory=False, chunksize=100000):
                part = chunk[chunk.food_observation_id.isin(missing_obs_ids)]
                if not part.empty:
                    missing_cells.append(part)
    archived_labels = pd.concat(missing_cells, ignore_index=True)
    archived_labels = archived_labels.merge(mapped[["food_observation_id", "food_concept_id", "source_food_id"]], on="food_observation_id", validate="many_to_one")
    archived_labels = archived_labels.merge(component_map[["component_observation_id", "component_concept_id"]], on="component_observation_id", how="left", validate="many_to_one")
    archived_labels = archived_labels.merge(components[["component_concept_id", "canonical_name", "training_role", "training_exclusion_reason"]], on="component_concept_id", how="left", validate="many_to_one")
    archived_labels["restore_approved"] = False
    write_csv(archived_labels, output / "validation_missing_foods_component_evidence.csv.gz")
    rejected_evidence = gate_reference_evidence(measurements.copy(), catalogs)
    changed = rejected_evidence[measurements.strict_validation_eligible.astype(str).str.casefold().eq("true") & ~rejected_evidence.strict_validation_eligible.astype(str).str.casefold().eq("true")]
    write_csv(changed.groupby(["source_key", "exclusion_reason"]).size().reset_index(name="affected_measurements_before_partition"), output / "future_reader_gate_impact.csv")
    summary = {
        "frozen_validation_foods": len(locked), "current_scorable_foods": len(current_scorable),
        "current_foods_without_maskable_target": len(lost), "missing_foods_with_AFCD_records": int(mapped.loc[afcd, "food_concept_id"].nunique()),
        "missing_foods_with_AFCD_analysed_food_flag": int(mapped.loc[afcd & mapped.food_level_analysed, "food_concept_id"].nunique()),
        "missing_foods_archived_component_records_traced": len(archived_labels),
        "selected_retained_validation_cells": len(selected), "selected_supporting_measurement_records": len(labels),
        "existing_measurement_id_collisions_outside_selected_validation": collisions.measurement_id.nunique(),
        "reference_rejected_measurement_records": len(bad), "reference_rejected_matrix_cells": len(rejected_cells),
        "reference_rejected_maskable_cells": len(bad[bad.is_maskable_axis][["food_concept_id", "component_concept_id"]].drop_duplicates()),
        "foods_affected_by_reference_rejection": bad.food_concept_id.nunique(),
        "reference_rejected_by_source": bad.groupby("source_key").size().to_dict(),
        "counterfactual_scorable_foods_after_reference_rejections_only": len(after_scorable),
        "counterfactual_newly_unscorable_foods": len(current_scorable - after_scorable),
        "counterfactual_is_not_a_new_approved_benchmark": True,
        "conversion_mismatch_records": int((~labels.exact_conversion_matches).sum()),
        "source_numeric_zero_records_needing_lod_semantics_review": int(labels.numeric_zero_needs_source_specific_lod_review.sum()),
        "foundation_supporting_records": len(foundation),
        "foundation_records_with_actual_sample_method_link": int(foundation.actual_linked_result_count.fillna(0).gt(0).sum()),
        "foundation_records_without_actual_sample_method_link": int(foundation.actual_linked_result_count.isna().sum()),
        "labels_restored_to_release": 0, "validation_model_outcomes_opened": False,
        "findings_scope": "Full selected-label metadata/reference check plus targeted primary-document verification; not full expert or laboratory certification.",
    }
    write_json(summary, output / "VALIDATION_LABEL_EVIDENCE_AUDIT.json")
    return summary


def mince_candidate_check(root: Path, output: Path):
    evidence = root / "data/raw/validation_mext_evidence_2026_09_06/afcd/mince_2014.xml"
    if not evidence.exists():
        raise FileNotFoundError(f"Primary mince paper snapshot missing: {evidence}")
    table = ET.parse(evidence).find(".//table-wrap[@id='nutrients-06-02217-t004']/table")
    if table is None:
        raise ValueError("The expected primary-paper Table 4 is absent")
    rows = {"".join(row[0].itertext()).strip(): ["".join(cell.itertext()).strip() for cell in row] for row in table.findall(".//tbody/tr")}
    source_labels = {"Moisture (g/100 g)": "Moisture (water)", "Protein (g/100 g)": "Protein", "Total Fat (g/100 g)": "Fat, total"}
    release = root / "data/processed/scientific_food_composition_v2/release"
    measurements = read(release / "measurement_main_eligible.csv.gz")
    observations = read(release / "food_observation.csv.gz")
    mapping = read(release / "food_observation_to_concept.csv.gz")
    partition = read(release / "ml_partition.csv")
    components = read(release / "component_observation.csv.gz")
    selected = measurements[measurements.source_key.eq("afcd")].merge(observations[["food_observation_id", "source_food_id", "original_name"]], on="food_observation_id")
    selected = selected.merge(components[["component_observation_id", "original_name"]].rename(columns={"original_name": "component_original_name"}), on="component_observation_id")
    selected = selected.merge(mapping[["food_observation_id", "food_concept_id"]], on="food_observation_id").merge(partition[["food_concept_id", "partition"]], on="food_concept_id", how="left")
    candidates = []
    for food_id, column, category in (("F000666", 1, "Raw low-fat mince"), ("F000655", 3, "Raw high-fat mince")):
        for paper_label, database_label in source_labels.items():
            match = selected[selected.source_food_id.eq(food_id) & selected.component_original_name.eq(database_label)]
            if len(match) != 1:
                raise ValueError(f"Expected exactly one AFCD measurement: {food_id} {database_label}; got {len(match)}")
            raw_paper_cell = rows[paper_label][column]
            paper_value = float(re.match(r"^[\d.]+", raw_paper_cell).group())
            row = match.iloc[0]
            current_value = float(row.normalized_value_g_per_100g)
            candidates.append({"food_concept_id": row.food_concept_id, "source_food_id": food_id, "food_name": row.original_name,
                               "partition": row.partition, "component_name": database_label, "measurement_id": row.measurement_id,
                               "afcd_value_g_per_100g": current_value, "paper_value_g_per_100g": paper_value,
                               "paper_table_cell": raw_paper_cell, "paper_table": f"Table 4, {category}",
                               "equal_at_published_precision": bool(np.isclose(current_value, paper_value, rtol=0, atol=1e-12)),
                               "evidence_url": "https://doi.org/10.3390/nu6062217", "evidence_sha256": sha256_file(evidence),
                               "same_primary_sampling_record_confirmed": False,
                               "unresolved_issue": "AFCD describes 2011 sampling with 15/9 samples; the paper describes 2010 retail sampling and 61 raw samples. Confirm underlying MLA dataset/subset linkage.",
                               "variance_not_reused": "Paper labels dispersion mean +/- SE; do not infer an independent n or reuse variance until subgroup linkage is confirmed.",
                               "restore_approved": False})
    write_csv(pd.DataFrame(candidates), output / "primary_paper_candidate_label_check_NOT_APPROVED.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-mext", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "reports/validation_mext_evidence_2026_09_06"
    output.mkdir(parents=True, exist_ok=True)
    protected = [path for directory in (root / "data/processed/scientific_food_composition_v2/release", root / "data/splits/scientific_food_composition_v2") for path in directory.rglob("*") if path.is_file()]
    snapshot = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    snapshot_path = output / "v2_immutable_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError("v2 release/split changed since the evidence audit began")
    write_json(snapshot, snapshot_path)
    summary = validation_audit(root, output)
    mince_candidate_check(root, output)
    if not args.skip_mext:
        audit_mext(root, output)
    assert snapshot == {str(path.relative_to(root)): sha256_file(path) for path in protected}
    write_json({"v2_files_unchanged": len(snapshot), "validation_food_ids_preserved": 1426,
                "new_training_rows": 0, "restored_validation_labels": 0, "model_outcomes_opened": False,
                "new_source_reader_gates_apply_to_future_builds_only": True}, output / "IMMUTABILITY_VERIFICATION.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

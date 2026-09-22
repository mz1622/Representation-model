#!/usr/bin/env python3
"""Apply the approved SR/CNF/FooDB trust policy to immutable v3 source records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.audit import run as run_audit
from foodcomp.harmonize import run_harmonization
from foodcomp.source_policy import TRUSTED_REFERENCE_POLICY, policy_metadata
from foodcomp.target_coverage import component_identity_collisions
from foodcomp.util import read_component_csv, sha256_file, write_csv, write_json

VERSION = "scientific_food_composition_v4"


def verify_snapshot(root: Path, snapshot: dict) -> None:
    for relative, digest in snapshot.items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"Protected input changed: {relative}")


def source_effects(disposition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in disposition.groupby("source_key"):
        flags = {
            key: group[key].astype(str).str.casefold().eq("true")
            for key in ["source_gate_previous_eligible", "main_value_eligible",
                        "source_gate_previous_validation_eligible", "validation_reference_eligible"]
        }
        before, after = flags["source_gate_previous_eligible"], flags["main_value_eligible"]
        count = group.measurement_count
        rows.append({
            "source_key": source, "staged_records": int(count.sum()),
            "before_main_eligible": int(count[before].sum()),
            "after_main_eligible": int(count[after].sum()),
            "newly_admitted_records": int(count[~before & after].sum()),
            "newly_held_records": int(count[before & ~after].sum()),
            "before_legacy_validation_eligible": int(count[flags["source_gate_previous_validation_eligible"]].sum()),
            "after_reference_validation_eligible": int(count[flags["validation_reference_eligible"]].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    previous = root / "data/processed/scientific_food_composition_v3"
    target = root / "data/processed" / VERSION
    audit = root / "data/audits" / VERSION / "dataset"
    reports = root / "reports" / VERSION
    manifest_path = target / "build_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Completed build exists: {manifest_path}; never overwrite a completed release")
    if not (previous / "build_manifest.json").exists():
        raise FileNotFoundError("A completed immutable v3 build is required")

    snapshot_path = target / "protected_previous_versions.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        paths = []
        for version in ("scientific_food_composition_v1", "scientific_food_composition_v2", "scientific_food_composition_v3"):
            for folder in [root / "data/processed" / version / "release", root / "data/splits" / version]:
                paths.extend(path for path in folder.rglob("*") if path.is_file())
        paths.extend(path for path in (previous / "staging").rglob("*") if path.is_file())
        snapshot = {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)}
        write_json(snapshot, snapshot_path)
    verify_snapshot(root, snapshot)
    locked = pd.read_csv(previous / "release/ml_partition.csv")
    locked = locked[locked.partition.eq("validation")]
    if len(locked) != 1426 or locked.food_concept_id.duplicated().any():
        raise ValueError("Expected the original 1,426 unique validation identities")
    write_csv(locked, target / "frozen_validation_identity_manifest.csv")
    write_json(policy_metadata(), target / "source_admission_policy.json")
    print("Applying source policy; original staging, values and validation identities remain protected.", flush=True)
    run_harmonization(
        root, previous / "staging", target / "release", audit,
        frozen_release=previous / "release", dataset_version=VERSION, as_of_date="2026-09-08",
        source_admission_policy=TRUSTED_REFERENCE_POLICY,
    )
    partition = pd.read_csv(target / "release/ml_partition.csv")
    identity = ["food_concept_id", "validation_panel", "family_cluster_id"]
    pd.testing.assert_frame_equal(
        locked[identity].sort_values("food_concept_id").reset_index(drop=True),
        partition.loc[partition.partition.eq("validation"), identity].sort_values("food_concept_id").reset_index(drop=True),
    )
    checks = run_audit(root, target / "release", audit)
    if checks["critical_failures"]:
        raise RuntimeError(f"Data checks failed: {checks['failed_check_ids']}")
    effects = source_effects(pd.read_csv(audit / "source_policy/source_policy_disposition.csv"))
    write_csv(effects, reports / "source_admission_changes.csv")
    components = read_component_csv(target / "release/component_concept.csv.gz")
    observations = read_component_csv(target / "release/component_observation.csv.gz")
    mapping = read_component_csv(target / "release/component_observation_to_concept.csv.gz")
    collisions = component_identity_collisions(observations, mapping)
    write_csv(collisions, reports / "known_component_identity_review.csv")
    counts = components[components.training_role.ne("excluded")].groupby(
        ["nutritional_role", "training_role"], dropna=False,
    ).size().reset_index(name="axis_count")
    write_csv(counts, reports / "provisional_component_role_counts.csv")
    before_components = read_component_csv(previous / "release/component_concept.csv.gz")
    comparison_columns = ["component_concept_id", "canonical_name", "nutritional_role", "training_role",
                          "train_count", "validation_count", "training_exclusion_reason"]
    comparison = before_components[comparison_columns].merge(
        components[comparison_columns], on="component_concept_id", how="outer", suffixes=("_v3", "_v4"),
        validate="one_to_one",
    )
    write_csv(comparison, reports / "component_qualification_changes.csv")
    verify_snapshot(root, snapshot)
    code = list((root / "src/foodcomp").glob("*.py")) + [Path(__file__).resolve()]
    write_json({
        "dataset_version": VERSION, "status": "candidate_source_policy_review_not_benchmark_ready",
        "source_admission_policy": policy_metadata(), "validation_food_identities": len(locked),
        "previous_versions_unchanged": True, "protected_files": len(snapshot),
        "validation_outcomes_opened": False, "models_trained": False, "new_benchmark_masks_created": False,
        "automatic_audit": checks,
        "unresolved_decimal_identity_collision_concepts": int(collisions.component_concept_id.nunique()),
        "benchmark_release_blocks": [
            "Known component-identity collisions, nutritional classification and mask-family reviews remain unresolved",
            "Trusted reference values are not guaranteed individual analytical ground truth",
            "Copied origins, reported-zero interpretation and dataset licences still require review",
        ],
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in code},
    }, manifest_path)
    print(effects.to_string(index=False), flush=True)
    print(f"Source-policy candidate completed: {target}; no models or locked-validation outcomes run.", flush=True)


if __name__ == "__main__":
    main()

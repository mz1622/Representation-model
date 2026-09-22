#!/usr/bin/env python3
"""Reaggregate evidence-corrected labels without modifying frozen v1/v2 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.audit import run as run_audit
from foodcomp.benchmark import load_release, make_fixed_masks
from foodcomp.harmonize import run_harmonization
from foodcomp.sources import SOURCE_LOADERS, write_source_staging
from foodcomp.util import sha256_file, write_csv, write_json

VERSION = "scientific_food_composition_v3"


def verify_snapshot(root, snapshot):
    for relative, digest in snapshot.items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"Protected artifact changed: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--skip-staging", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    previous = root / "data/processed/scientific_food_composition_v2"
    target = root / "data/processed" / VERSION
    audit = root / "data/audits" / VERSION / "dataset"
    manifest_path = target / "build_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Completed build already exists: {manifest_path}; do not overwrite it")
    snapshot_path = target / "protected_previous_versions.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        snapshot = {}
        for version in ("scientific_food_composition_v1", "scientific_food_composition_v2"):
            for base in (root / "data/processed" / version / "release", root / "data/splits" / version):
                for path in base.rglob("*"):
                    if path.is_file():
                        snapshot[str(path.relative_to(root))] = sha256_file(path)
        write_json(snapshot, snapshot_path)
    verify_snapshot(root, snapshot)
    partitions = pd.read_csv(previous / "release/ml_partition.csv")
    locked = partitions[partitions.partition.eq("validation")]
    if len(locked) != 1426 or locked.food_concept_id.duplicated().any():
        raise ValueError("Expected 1,426 distinct frozen validation food identities")
    write_csv(locked, target / "frozen_validation_identity_manifest.csv")
    staging_manifest = target / "staging_manifest.json"
    if not args.skip_staging:
        write_source_staging(root, target / "staging", list(SOURCE_LOADERS))
        foodb_source = previous / "staging/foodb"
        if not list(foodb_source.glob("measurement_part_*.csv.gz")):
            raise FileNotFoundError(f"Immutable FooDB archive missing: {foodb_source}")
        (target / "staging/foodb").mkdir(parents=True, exist_ok=True)
        for path in foodb_source.glob("*.csv.gz"):
            shutil.copy2(path, target / "staging/foodb" / path.name)
        write_json({str(path.relative_to(root)): sha256_file(path)
                    for path in (target / "staging").glob("*/*.csv.gz")}, staging_manifest)
    elif not staging_manifest.exists():
        raise FileNotFoundError("Cannot skip staging without a completed staging manifest")
    verify_snapshot(root, json.loads(staging_manifest.read_text()))
    if args.stage_only:
        verify_snapshot(root, snapshot)
        return
    run_harmonization(root, target / "staging", target / "release", audit,
                      frozen_release=previous / "release", dataset_version=VERSION, as_of_date="2026-09-07")
    values, foods, components, _, _ = load_release(target / "release")
    identity = ["food_concept_id", "validation_panel", "family_cluster_id"]
    pd.testing.assert_frame_equal(
        locked[identity].sort_values("food_concept_id").reset_index(drop=True),
        foods.loc[foods.partition.eq("validation"), identity].sort_values("food_concept_id").reset_index(drop=True),
    )
    checks = run_audit(root, target / "release", audit)
    if checks["critical_failures"]:
        raise RuntimeError(f"Data audit failed: {checks['failed_check_ids']}")
    masks = make_fixed_masks(values, foods, components, root / "data/splits" / VERSION,
                             dataset_version=VERSION)
    verify_snapshot(root, snapshot)
    code = list((root / "src/foodcomp").glob("*.py")) + [Path(__file__).resolve()]
    write_json({
        "dataset_version": VERSION, "status": "candidate_pending_domain_review",
        "built_as_of": "2026-09-07", "validation_food_identities": len(locked),
        "previous_versions_unchanged": True, "validation_outcomes_opened": False,
        "new_external_sources_admitted": [], "mext_status": "separate_candidate_not_admitted",
        "fixed_mask_rows": len(masks), "automatic_audit": checks,
        "policy": "Rebuild all admitted sources with reference gates; reaggregate remaining evidence. Keep validation foods and panels, but version corrected labels, axis eligibility and masks. No new source supplies frozen validation labels.",
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in code},
    }, manifest_path)
    print(f"Evidence-corrected candidate completed: {target}", flush=True)


if __name__ == "__main__":
    main()

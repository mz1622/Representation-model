#!/usr/bin/env python3
"""Rebuild a separate candidate without changing v1 or its validation foods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.harmonize import run_harmonization
from foodcomp.sources import SOURCE_LOADERS, write_source_staging
from foodcomp.util import sha256_file, write_csv, write_json

VERSION = "scientific_food_composition_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-staging", action="store_true")
    parser.add_argument("--sources", nargs="*", help="Restage selected sources in the v2 directory only")
    parser.add_argument("--skip-harmonization", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    previous = root / "data/processed/scientific_food_composition_v1/release"
    target = root / "data/processed" / VERSION
    audit = root / "data/audits" / VERSION / "dataset"
    snapshot_path = target / "v1_immutable_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        paths = list(previous.glob("*")) + list((root / "data/splits/scientific_food_composition_v1").glob("*"))
        snapshot = {str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()}
        write_json(snapshot, snapshot_path)
    for relative, digest in snapshot.items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"Frozen v1 artifact changed: {relative}")
    partitions = pd.read_csv(previous / "ml_partition.csv")
    locked = partitions[partitions["partition"].eq("validation")]
    if len(locked) != 1426:
        raise ValueError(f"Expected original 1426 validation identities, found {len(locked)}")
    write_csv(locked, target / "frozen_validation_identity_manifest.csv")
    if not args.skip_staging:
        write_source_staging(root, target / "staging", args.sources or list(SOURCE_LOADERS))
        # FooDB remains a provenance-only archive with zero admitted labels.
        # Copy the already staged archive byte-for-byte; no borrowed value is
        # promoted to an independent observation during this rebuild.
        foodb = target / "staging/foodb"
        foodb.mkdir(parents=True, exist_ok=True)
        for source in (previous.parent / "staging/foodb").glob("*.csv.gz"):
            shutil.copy2(source, foodb / source.name)
    if not args.skip_harmonization:
        run_harmonization(root, target / "staging", target / "release", audit,
                          frozen_release=previous, dataset_version=VERSION, as_of_date="2026-09-06")
        current = pd.read_csv(target / "release/ml_partition.csv")
        current_locked = current[current["partition"].eq("validation")]
        identity_columns = ["food_concept_id", "validation_panel"]
        old_identity = locked[identity_columns].sort_values("food_concept_id").reset_index(drop=True)
        new_identity = current_locked[identity_columns].sort_values("food_concept_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(old_identity, new_identity)
    unchanged = all(sha256_file(root / relative) == digest for relative, digest in snapshot.items())
    if not unchanged:
        raise RuntimeError("A protected v1 artifact changed during the v2 build")
    code = list((root / "src/foodcomp").glob("*.py")) + [Path(__file__).resolve()]
    write_json({
        "dataset_version": VERSION, "status": "candidate_pending_expert_review",
        "v1_unchanged": unchanged, "validation_food_identities": len(locked),
        "validation_outcomes_opened": False,
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in code},
        "policy": "New and curated reference data may train; no new sources enter frozen validation labels. Corrections may remove invalid labels without replacing foods.",
    }, target / "build_manifest.json")


if __name__ == "__main__":
    main()

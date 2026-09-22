#!/usr/bin/env python3
"""Package the active FoodNutriGPT corpus for a gated dataset release.

The package copies only the files required to reproduce the source-native V8
baseline and writes SHA-256 checksums. It never changes the source corpus.

Colab:
    !python scripts/package_foodnutrigpt_dataset_release.py \
        --output-dir /content/foodnutrigpt_v8_source_native_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"

DATA_FILES = [
    "axis_registry.csv",
    "build_manifest.json",
    "complete_test_axis_coverage.csv",
    "food_profiles.csv.gz",
    "partition_summary.csv",
    "source_native_axis_tokens.csv.gz",
    "source_vocabulary.csv",
    "supplemental_test_exact_name_groups.csv",
    "train_only_axis_normalization.csv",
]
SPLIT_FILES = ["splits.json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, release_root: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(f"Required release file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.relative_to(release_root).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir, split_dir, output_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite release package: {output_dir}")

    build_manifest = json.loads((data_dir / "build_manifest.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, object]] = []
    for name in DATA_FILES:
        files.append(copy_file(data_dir / name, output_dir / "data" / name, output_dir))
    for name in SPLIT_FILES:
        files.append(copy_file(split_dir / name, output_dir / "splits" / name, output_dir))

    release_manifest = {
        "release_name": "foodnutrigpt_v8_source_native_baseline",
        "dataset_version": build_manifest["dataset_version"],
        "release_scope": "Source-native V8 baseline only; not the planned V9 source-aware/source-free benchmark.",
        "values": "normalized_value_g_per_100g; missing values are absent tokens; explicit zero is observed.",
        "redistribution_note": "Confirm every upstream source licence before publishing this package publicly.",
        "files": files,
    }
    (output_dir / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# FoodNutriGPT V8 Source-Native Baseline\n\n"
        "This package reproduces the V8 source-native corpus and its frozen split. "
        "It is not a source-free inference benchmark. See the code repository "
        "for the V9 training plan and audit criteria.\n",
        encoding="utf-8",
    )
    print(json.dumps(release_manifest, indent=2))


if __name__ == "__main__":
    main()

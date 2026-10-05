#!/usr/bin/env python3
"""Build the fixed equal-weight v3-seeds plus ReGLU Transformer ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.no_foodname_ensemble import equal_log_ensemble  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    numeric_only_training_view, score_subset,
)


DEFAULT_MEMBERS = (
    ROOT / "output/no_foodname_masked_axis_20ep_20261005",
    ROOT / "output/no_foodname_masked_axis_20ep_seed20261006",
    ROOT / "output/no_foodname_masked_axis_20ep_seed20261007",
    ROOT / "output/no_foodname_v7_reglu_20ep_20261005",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=Path, nargs="+", default=DEFAULT_MEMBERS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if len(args.members) < 2 or len(set(map(str, args.members))) != len(args.members):
        raise ValueError("members must contain at least two distinct runs")

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    expected_data_hash = digest(args.data_dir / "manifest.json")
    member_manifests = []
    member_predictions = []
    for directory in args.members:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete ensemble member: {directory}")
        if manifest.get("data_sha256") != expected_data_hash:
            raise ValueError(f"data hash mismatch: {directory}")
        if manifest.get("food_name_model_input") is not False:
            raise ValueError(f"food-name input is not false: {directory}")
        if manifest.get("complete_test_opened") is not False:
            raise ValueError(f"complete test policy mismatch: {directory}")
        if list(map(int, manifest.get("axes", []))) != list(map(int, data.targets)):
            raise ValueError(f"axis panel mismatch: {directory}")
        member_manifests.append(manifest)
        member_predictions.append(pd.read_parquet(
            directory / "validation_predictions.parquet"))

    started = time.monotonic()
    prediction = equal_log_ensemble(member_predictions, data.scale)
    axes = list(map(int, data.targets))
    metrics, per_axis = score_subset(
        data, prediction, axes, validation_rows=data.validation)
    args.output_dir.mkdir(parents=True)
    prediction.to_parquet(
        args.output_dir / "validation_predictions.parquet", index=False)
    per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    source_files = (
        ROOT / "src/foodcomp/no_foodname_ensemble.py",
        ROOT / "src/foodcomp/research_no_foodname_v1.py",
        ROOT / "scripts/ensemble_no_foodname_transformers.py",
    )
    manifest = {
        "status": "complete",
        "version": "no_foodname_v11_equal_log_transformer_ensemble",
        "method": "equal mean in log1p(raw / training_axis_scale) space",
        "members": [str(path.relative_to(ROOT)) for path in args.members],
        "member_versions": [item["version"] for item in member_manifests],
        "member_seeds": [item["config"]["seed"] for item in member_manifests],
        "weights": [1 / len(args.members)] * len(args.members),
        "data_sha256": expected_data_hash,
        "code_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_files
        },
        "axes": axes,
        "validation_profiles": len(data.validation),
        "complete_test_opened": False,
        "food_name_model_input": False,
        "source_model_input": False,
        "continuous_weight_search": False,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"equal_log_transformer_ensemble complete: "
          f"{metrics['all']['scaled_log_mae']:.6f}", flush=True)


if __name__ == "__main__":
    main()

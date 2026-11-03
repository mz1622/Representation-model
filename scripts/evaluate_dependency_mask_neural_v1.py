#!/usr/bin/env python3
"""Evaluate a validation-selected dependency-mask V5+CCE or V5+MVC checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import load_dependency_model  # noqa: E402
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    attach_complete_test, predict_transformer,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    model, checkpoint, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, device
    )
    if checkpoint.get("stage") not in {"cce", "mvc"}:
        raise ValueError("official checkpoint must be a CCE- or MVC-stage checkpoint")
    validation_jobs = data.jobs[["profile_index", "axis_index", "target"]].copy()
    _, test_jobs, panel = attach_complete_test(data, args.frozen_data_dir, args.split_path)
    policy = DependencyMaskPolicy(data.axes)
    manifest = {
        "status": "running", "version": "dependency_mask_v1_neural_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_stage": checkpoint["stage"],
        "checkpoint_epoch": checkpoint["completed_epoch"],
        "checkpoint_selection": (
            f"lowest validation macro scaled-log MAE within {checkpoint['stage'].upper()} stage"
        ),
        "code_sha256": {
            "scripts/evaluate_dependency_mask_neural_v1.py": digest(Path(__file__)),
            "src/foodcomp/dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/dependency_mask_v1.py"
            ),
            "src/foodcomp/research_dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/research_dependency_mask_v1.py"
            ),
        },
        "config": asdict(config), "complete_test": panel,
        "food_name_model_input": False, "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        for name, jobs in (("validation", validation_jobs), ("complete_test", test_jobs)):
            prediction = predict_transformer(
                model, data, policy, jobs, device, args.batch_size, two_pass=True
            )
            metrics, per_axis, _ = score_profile_axis_macro(data, jobs, prediction)
            prediction.to_parquet(args.output_dir / f"{name}_predictions.parquet", index=False)
            per_axis.to_csv(args.output_dir / f"{name}_axis_metrics.csv", index=False)
            manifest[name] = metrics
            print(f"{name}: {metrics}", flush=True)
        pd.DataFrame([
            {"partition": name, **manifest[name]}
            for name in ("validation", "complete_test")
        ]).to_csv(args.output_dir / "metrics.csv", index=False)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

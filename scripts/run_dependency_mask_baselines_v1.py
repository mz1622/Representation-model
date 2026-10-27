#!/usr/bin/env python3
"""Fit KNN, RF, and XGBoost under the frozen dependency-mask protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy, VERSION as MASK_VERSION  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    attach_complete_test, fit_predict_baseline, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


VERSION = "dependency_mask_v1_baselines"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="knn,rf,xgb")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--xgb-trees", type=int, default=300)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or not set(methods).issubset({"knn", "rf", "xgb"}):
        raise ValueError("methods must be selected from knn,rf,xgb")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = replace(
        Config(), seed=args.seed, n_jobs=args.n_jobs,
        rf_trees=args.rf_trees, xgb_trees=args.xgb_trees,
    )
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    validation_jobs = data.jobs[["profile_index", "axis_index", "target"]].copy()
    validation_jobs["partition"] = "validation"
    _, test_jobs, panel_audit = attach_complete_test(
        data, args.frozen_data_dir, args.split_path
    )
    test_jobs["partition"] = "complete_test"
    combined_jobs = pd.concat((validation_jobs, test_jobs), ignore_index=True)
    policy = DependencyMaskPolicy(data.axes)
    manifest = {
        "status": "running", "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "frozen_build_manifest_sha256": digest(args.frozen_data_dir / "build_manifest.json"),
        "frozen_split_sha256": digest(args.split_path),
        "mask_policy_version": MASK_VERSION,
        "mask_policy_manifest_sha256": digest(args.policy_manifest),
        "food_name_model_input": False, "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "config": asdict(config), "methods": {},
        "validation_jobs": int(len(validation_jobs)),
        "complete_test": panel_audit,
        "metric": "direct profile-axis error, mean within axis, equal-weight macro over 187 axes",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        for method in methods:
            method_start = time.monotonic()
            prediction, support = fit_predict_baseline(
                data, policy, method, combined_jobs, config, train_rows=data.train
            )
            merged = combined_jobs[["profile_index", "axis_index", "partition"]].merge(
                prediction, on=["profile_index", "axis_index"], validate="one_to_one"
            )
            target = args.output_dir / method
            target.mkdir()
            method_record = {}
            for partition, jobs in (
                ("validation", validation_jobs), ("complete_test", test_jobs)
            ):
                part_prediction = merged[merged.partition.eq(partition)][
                    ["profile_index", "axis_index", "prediction"]
                ]
                metrics, per_axis, scored = score_profile_axis_macro(
                    data, jobs, part_prediction
                )
                part_prediction.to_parquet(
                    target / f"{partition}_predictions.parquet", index=False
                )
                per_axis.to_csv(target / f"{partition}_axis_metrics.csv", index=False)
                method_record[partition] = metrics
            support.to_csv(target / "fit_support.csv", index=False)
            method_record["elapsed_seconds"] = time.monotonic() - method_start
            manifest["methods"][method] = method_record
            write_json(args.output_dir / "manifest.json", manifest)
            print(
                f"{method} complete: validation="
                f"{method_record['validation']['macro_scaled_log_mae']:.6f}, "
                f"test={method_record['complete_test']['macro_scaled_log_mae']:.6f}",
                flush=True,
            )
        rows = []
        for method, record in manifest["methods"].items():
            rows.append({
                "method": method,
                **{f"validation_{k}": v for k, v in record["validation"].items()},
                **{f"complete_test_{k}": v for k, v in record["complete_test"].items()},
            })
        pd.DataFrame(rows).sort_values("complete_test_macro_scaled_log_mae").to_csv(
            args.output_dir / "metrics.csv", index=False
        )
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

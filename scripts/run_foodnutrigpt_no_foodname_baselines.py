#!/usr/bin/env python3
"""Run the three composition-only baselines on the unified 187-axis protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config,
    baseline_predictions,
    numeric_only_training_view,
    score_subset,
)
from run_foodnutrigpt_no_foodname_v1 import select_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes", help="comma-separated target axis IDs; default all 187")
    parser.add_argument("--max-train-profiles", type=int, default=0)
    parser.add_argument("--max-validation-profiles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--rf-trees", type=int, default=Config.rf_trees)
    parser.add_argument("--xgb-trees", type=int, default=Config.xgb_trees)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.rf_trees, args.xgb_trees, args.n_jobs) < 1:
        raise ValueError("tree counts and jobs must be positive")

    config = replace(
        Config(),
        seed=args.seed,
        rf_trees=args.rf_trees,
        xgb_trees=args.xgb_trees,
        n_jobs=args.n_jobs,
    )
    data = ResearchData(args.data_dir, "quarantined")
    train_rows = select_rows(data.train, args.max_train_profiles, config.seed)
    validation_rows = select_rows(
        data.validation, args.max_validation_profiles, config.seed + 1
    )
    data = numeric_only_training_view(data, train_rows)
    axes = (
        list(map(int, args.axes.split(",")))
        if args.axes
        else list(map(int, data.targets))
    )
    if (
        not axes
        or len(set(axes)) != len(axes)
        or not set(axes).issubset(set(data.targets))
    ):
        raise ValueError("axes must be distinct supervised IDs")
    if len(axes) != 187 and not args.axes:
        raise ValueError(f"expected 187 default target axes, found {len(axes)}")
    selected_jobs = data.jobs[
        data.jobs.axis_index.isin(axes)
        & data.jobs.profile_index.isin(validation_rows)
    ]
    if set(selected_jobs.axis_index) != set(axes):
        raise ValueError("validation subset does not cover every selected axis")

    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    source_files = (
        ROOT / "src/foodcomp/research_no_foodname_v1.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_baselines.py",
    )
    manifest = {
        "status": "running",
        "version": "no_foodname_baselines_unified187_v1",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "code_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_files
        },
        "config": asdict(config),
        "axes": axes,
        "axis_count": len(axes),
        "train_profiles": len(train_rows),
        "validation_profiles": len(validation_rows),
        "validation_jobs": len(selected_jobs),
        "methods": ["rf", "xgb", "knn"],
        "training_scope": "187 independent per-axis regressors under one shared protocol",
        "primary_metric": "all187 macro-axis scaled_log_mae",
        "food_name_model_input": False,
        "source_model_input": False,
        "complete_test_opened": False,
        "pilot_subset": bool(
            args.axes or args.max_train_profiles or args.max_validation_profiles
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        results = {}
        for method in manifest["methods"]:
            method_started = time.monotonic()
            predictions, fit = baseline_predictions(
                data,
                method,
                axes,
                config,
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
            metrics, per_axis = score_subset(
                data, predictions, axes, validation_rows=validation_rows
            )
            target = args.output_dir / method
            target.mkdir()
            predictions.to_parquet(
                target / "validation_predictions.parquet", index=False
            )
            fit.to_csv(target / "fit_support.csv", index=False)
            per_axis.to_csv(target / "axis_metrics.csv", index=False)
            write_json(target / "metrics.json", metrics)
            results[method] = {
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - method_started,
            }
            print(
                f"{method} complete: all187="
                f"{metrics['all']['scaled_log_mae']:.6f}",
                flush=True,
            )
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            results=results,
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

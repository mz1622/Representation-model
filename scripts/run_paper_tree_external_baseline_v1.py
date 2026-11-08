#!/usr/bin/env python3
"""Train current RF/XGBoost baselines on paper data and evaluate zero-shot."""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.paper_dae_baseline_v1 import build_paper_axis_crosswalk  # noqa: E402
from foodcomp.paper_tree_transfer_v1 import (  # noqa: E402
    VERSION,
    build_external_feature_view,
    build_paper_training_view,
)
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    attach_complete_test,
    dense_features,
    make_baseline,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config,
    inverse_values,
    numeric_only_training_view,
)
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from run_paper_dae_external_baseline_v1 import (  # noqa: E402
    load_paper_data,
    paper_split_and_mask,
)


def _method_label(method: str) -> str:
    return {"rf": "Paper-data RF", "xgb": "Paper-data XGBoost"}[method]


def _load_or_fit_axis(
    *, method: str, axis: int, axis_number: int, axis_total: int,
    paper_view, paper_train: np.ndarray, external_view, policy,
    validation_jobs: pd.DataFrame, test_jobs: pd.DataFrame,
    config: Config, method_dir: Path,
):
    prediction_path = method_dir / f"axis_{axis:03d}_predictions.parquet"
    support_path = method_dir / f"axis_{axis:03d}_support.json"
    if prediction_path.exists() and support_path.exists():
        prediction = pd.read_parquet(prediction_path)
        support = json.loads(support_path.read_text(encoding="utf-8"))
        expected = int(validation_jobs.axis_index.eq(axis).sum() + test_jobs.axis_index.eq(axis).sum())
        if len(prediction) != expected or not prediction.axis_index.eq(axis).all():
            raise ValueError(f"stale cached prediction for {method} axis {axis}")
        print(f"{method} axis {axis_number}/{axis_total} cached", flush=True)
        return prediction, support

    started = time.monotonic()
    train_axis = np.full(len(paper_train), axis, dtype=np.int64)
    x_train = dense_features(paper_view, paper_train, train_axis, policy)
    y_train = paper_view.values[paper_train, axis]
    model = make_baseline(method, axis, config)
    model.fit(x_train, y_train)
    if method == "rf":
        model.n_jobs = 1

    pieces = []
    for partition, jobs in (
        ("validation", validation_jobs),
        ("complete_test", test_jobs),
    ):
        part = jobs[jobs.axis_index.eq(axis)]
        rows = part.profile_index.to_numpy(dtype=np.int64)
        eval_axis = np.full(len(rows), axis, dtype=np.int64)
        x_eval = dense_features(external_view, rows, eval_axis, policy)
        normalized = model.predict(x_eval)
        prediction = inverse_values(normalized, paper_view.scale[axis])
        pieces.append(pd.DataFrame({
            "partition": partition,
            "profile_index": rows,
            "axis_index": axis,
            "prediction": prediction,
        }))
    result = pd.concat(pieces, ignore_index=True)
    support = {
        "axis_index": axis,
        "train_profiles": int(len(paper_train)),
        "validation_jobs": int(validation_jobs.axis_index.eq(axis).sum()),
        "complete_test_jobs": int(test_jobs.axis_index.eq(axis).sum()),
        "feature_count": int(x_train.shape[1]),
        "mask_size": int(len(policy.mask_indices(axis))),
        "paper_positive_scale_g_per_100g": float(paper_view.scale[axis]),
        "fit_and_predict_seconds": float(time.monotonic() - started),
    }
    result.to_parquet(prediction_path, index=False)
    write_json(support_path, support)
    del model, x_train
    gc.collect()
    print(
        f"{method} axis {axis_number}/{axis_total} complete "
        f"({support['fit_and_predict_seconds']:.1f}s)",
        flush=True,
    )
    return result, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--paper-dae-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--methods", nargs="+", choices=["rf", "xgb"], default=["rf", "xgb"]
    )
    args = parser.parse_args()

    paper_value, paper_columns, _, nutrient, paper_value_path = load_paper_data(
        args.paper_repo
    )
    paper_train, paper_holdout, _, _ = paper_split_and_mask(paper_value, args.seed)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    validation_jobs = data.jobs[["profile_index", "axis_index", "target"]].copy()
    _, test_jobs, complete_audit = attach_complete_test(
        data, args.frozen_data_dir, args.split_path
    )
    policy = DependencyMaskPolicy(data.axes)
    links, crosswalk = build_paper_axis_crosswalk(
        paper_columns, nutrient, data.axes
    )
    target_axes = sorted(link.axis_index for link in links if link.loss_eligible)
    if len(target_axes) != 127:
        raise ValueError(f"expected 127 strict common targets, found {len(target_axes)}")
    validation_jobs = validation_jobs[
        validation_jobs.axis_index.isin(target_axes)
    ].reset_index(drop=True)
    test_jobs = test_jobs[test_jobs.axis_index.isin(target_axes)].reset_index(drop=True)

    paper_view = build_paper_training_view(
        paper_value, links, len(data.axes), paper_train
    )
    external_view = build_external_feature_view(
        data, paper_view.scale, [link.axis_index for link in links]
    )
    config = Config(seed=args.seed, n_jobs=args.n_jobs)

    if args.output_dir.exists():
        if not (args.output_dir / "manifest.json").exists():
            raise FileExistsError("output directory exists without a resumable manifest")
        manifest = json.loads(
            (args.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("version") != VERSION or manifest.get("seed") != args.seed:
            raise ValueError("output manifest is incompatible with this run")
    else:
        args.output_dir.mkdir(parents=True)
        manifest = {
            "status": "running",
            "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "methods": args.methods,
            "paper_data_sha256": digest(paper_value_path),
            "current_data_sha256": digest(args.data_dir / "manifest.json"),
            "paper_training_rows": int(len(paper_train)),
            "paper_holdout_rows_unused": int(len(paper_holdout)),
            "strict_common_axes_total": int(len(links)),
            "strict_common_supervised_axes": int(len(target_axes)),
            "validation_jobs": int(len(validation_jobs)),
            "complete_test_jobs": int(len(test_jobs)),
            "complete_test_original_panel": complete_audit,
            "training": {
                "labels": "paper average-value dataset only",
                "normalization": "per-axis positive median of paper training rows",
                "masking": "current dependency-closure mask applied to paper inputs",
                "feature_schema": "252 normalized axis values + 252 visibility flags",
                "unmapped_current_axes": "always hidden",
                "current_training_labels_used": False,
                "current_validation_used_for_selection": False,
                "current_metric_scale_used_for_scoring_only": True,
                "model_checkpoints_saved": False,
            },
            "hyperparameters": {
                "RF": {
                    "trees": config.rf_trees,
                    "min_samples_leaf": config.rf_leaf,
                    "max_features": 0.5,
                    "criterion": "squared_error",
                    "bootstrap": True,
                },
                "XGBoost": {
                    "trees": config.xgb_trees,
                    "max_depth": config.xgb_depth,
                    "learning_rate": 0.05,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                },
                "n_jobs": args.n_jobs,
                "axis_seed": "20261005 + current axis index",
            },
            "results": {},
            "progress": {},
        }
        crosswalk.to_csv(
            args.output_dir / "paper_to_current_axis_crosswalk.csv", index=False
        )
        pd.DataFrame({
            "axis_index": np.arange(len(data.axes)),
            "canonical_name": data.axes.canonical_name,
            "mapped_from_paper": np.isin(
                np.arange(len(data.axes)), [link.axis_index for link in links]
            ),
            "paper_positive_scale_g_per_100g": paper_view.scale,
        }).to_csv(args.output_dir / "paper_training_scales.csv", index=False)
        write_json(args.output_dir / "manifest.json", manifest)

    started = time.monotonic()
    comparison_rows = []
    for method in args.methods:
        method_dir = args.output_dir / method
        method_dir.mkdir(exist_ok=True)
        predictions, supports = [], []
        for number, axis in enumerate(target_axes, start=1):
            result, support = _load_or_fit_axis(
                method=method,
                axis=axis,
                axis_number=number,
                axis_total=len(target_axes),
                paper_view=paper_view,
                paper_train=paper_train,
                external_view=external_view,
                policy=policy,
                validation_jobs=validation_jobs,
                test_jobs=test_jobs,
                config=config,
                method_dir=method_dir,
            )
            predictions.append(result)
            supports.append(support)
            manifest["progress"][method] = number
            write_json(args.output_dir / "manifest.json", manifest)

        predictions = pd.concat(predictions, ignore_index=True)
        pd.DataFrame(supports).sort_values("axis_index").to_csv(
            method_dir / "fit_support.csv", index=False
        )
        for partition, jobs in (
            ("validation", validation_jobs),
            ("complete_test", test_jobs),
        ):
            part = predictions[predictions.partition.eq(partition)].drop(
                columns="partition"
            )
            part.to_parquet(method_dir / f"{partition}_predictions.parquet", index=False)
            metrics, per_axis, _ = score_profile_axis_macro(data, jobs, part)
            per_axis.to_csv(method_dir / f"{partition}_axis_metrics.csv", index=False)
            label = _method_label(method)
            comparison_rows.append({"method": label, "partition": partition, **metrics})
            manifest["results"].setdefault(label, {})[partition] = metrics
            print(
                f"{label} {partition}: SLMAE={metrics['macro_scaled_log_mae']:.6f}",
                flush=True,
            )
        write_json(args.output_dir / "manifest.json", manifest)

    new_comparison = pd.DataFrame(comparison_rows)
    new_comparison.to_csv(args.output_dir / "paper_tree_metrics.csv", index=False)
    prior = pd.read_csv(args.paper_dae_result_dir / "common_127_axis_metrics.csv")
    combined = pd.concat((prior, new_comparison), ignore_index=True).sort_values(
        ["partition", "macro_scaled_log_mae", "method"]
    )
    combined.to_csv(args.output_dir / "all_methods_common_127_axis_metrics.csv", index=False)
    source_names = {
        "Paper DAE (no name)",
        "Paper-data DAE + FoodText",
        "Paper-data RF",
        "Paper-data XGBoost",
    }
    combined[combined.method.isin(source_names)].to_csv(
        args.output_dir / "source_only_common_127_axis_metrics.csv", index=False
    )
    manifest.update(
        status="complete",
        elapsed_seconds=float(manifest.get("elapsed_seconds", 0.0) + time.monotonic() - started),
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()

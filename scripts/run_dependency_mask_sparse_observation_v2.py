#!/usr/bin/env python3
"""Sparse-context comparison restricted to V5+CCE and three formal baselines."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    checkpoint_metadata, dependency_visible, load_dependency_model, predict_normalized,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import dense_features, make_baseline, score_profile_axis_macro  # noqa: E402
from foodcomp.research_no_foodname_v1 import inverse_values, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261027
BUDGETS = (5, 10, 20, 40, "all")


def sparse_visible(full, budget, seed):
    if budget == "all":
        return full.copy()
    result = np.zeros_like(full)
    rng = np.random.default_rng(seed)
    for row in range(len(full)):
        available = np.flatnonzero(full[row])
        chosen = rng.choice(available, min(int(budget), len(available)), replace=False)
        result[row, chosen] = True
    return result


def baseline_predict(models, method, data, rows, axes, visible):
    features = np.concatenate((
        np.where(visible, data.values[rows], 0), visible.astype(np.float32)
    ), axis=1).astype(np.float32)
    prediction = np.zeros(len(rows), dtype=np.float32)
    for axis in np.unique(axes):
        position = np.flatnonzero(axes == axis)
        model = models[int(axis)]
        if method == "knn":
            distance, neighbor = model[0].kneighbors(features[position])
            y, weight = model[1], model[2]
            importance = weight[neighbor] / (distance + 1e-3)
            prediction[position] = (
                (importance * y[neighbor]).sum(1) / importance.sum(1)
            )
        else:
            prediction[position] = model.predict(features[position])
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    started = time.monotonic()
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    policy = DependencyMaskPolicy(data.axes)
    model, checkpoint, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, device
    )

    support = data.observed[np.ix_(data.validation, data.targets)].sum(0)
    selected_axes = np.asarray(data.targets, dtype=np.int64)[
        np.argsort(-support, kind="stable")[:12]
    ]
    rng = np.random.default_rng(SEED)
    jobs = []
    for axis in selected_axes:
        candidates = data.validation[data.observed[data.validation, axis]]
        visible = dependency_visible(
            data, policy, candidates, np.full(len(candidates), axis, dtype=np.int64)
        )
        candidates = candidates[visible.sum(1) >= 5]
        chosen = np.sort(rng.choice(candidates, 200, replace=False))
        jobs.append(pd.DataFrame({
            "profile_index": chosen, "axis_index": int(axis),
            "target": data.raw[chosen, axis],
        }))
    jobs = pd.concat(jobs, ignore_index=True).sort_values(
        ["profile_index", "axis_index"]
    ).reset_index(drop=True)
    jobs.to_csv(args.output_dir / "jobs.csv", index=False)
    rows = jobs.profile_index.to_numpy(dtype=np.int64)
    axes = jobs.axis_index.to_numpy(dtype=np.int64)
    full_visible = dependency_visible(data, policy, rows, axes)

    baseline_models = {label: {} for label in ("RF", "XGBoost", "KNN")}
    for method, label in (("rf", "RF"), ("xgb", "XGBoost"), ("knn", "KNN")):
        for axis in selected_axes:
            train = data.train[data.observed[data.train, axis]]
            train_axis = np.full(len(train), axis, dtype=np.int64)
            x_train = dense_features(data, train, train_axis, policy)
            y = data.values[train, axis]
            weight = data.weights[train, axis]
            estimator = make_baseline(method, int(axis), config)
            if method == "knn":
                estimator.set_params(n_neighbors=min(config.knn_neighbors, len(train)))
                estimator.fit(x_train)
                baseline_models[label][int(axis)] = (estimator, y, weight)
            else:
                estimator.fit(x_train, y, sample_weight=weight / weight.mean())
                if hasattr(estimator, "n_jobs"):
                    estimator.n_jobs = 1
                baseline_models[label][int(axis)] = estimator
            print(f"fit sparse {label} axis {axis}", flush=True)

    records, predictions = [], []
    for budget in BUDGETS:
        repeats = (-1,) if budget == "all" else (0, 1, 2)
        for repeat in repeats:
            seed = SEED + 1_000 * (repeat + 2) + (0 if budget == "all" else int(budget))
            visible = sparse_visible(full_visible, budget, seed)
            normalized = predict_normalized(
                model, data, rows, axes, visible, device, 512, two_pass=True
            )
            for label, values in [("V5_CCE", normalized)]:
                raw = inverse_values(values, data.scale[axes])
                frame = jobs[["profile_index", "axis_index"]].copy()
                frame["prediction"] = raw
                metrics, _, _ = score_profile_axis_macro(data, jobs, frame)
                records.append({
                    "method": label, "visible_budget": str(budget), "repeat": repeat,
                    "mean_visible": float(visible.sum(1).mean()), **metrics,
                })
                predictions.append(frame.assign(method=label, visible_budget=str(budget), repeat=repeat))
            for label, method in (("RF", "rf"), ("XGBoost", "xgb"), ("KNN", "knn")):
                values = baseline_predict(
                    baseline_models[label], method, data, rows, axes, visible
                )
                raw = inverse_values(values, data.scale[axes])
                frame = jobs[["profile_index", "axis_index"]].copy()
                frame["prediction"] = raw
                metrics, _, _ = score_profile_axis_macro(data, jobs, frame)
                records.append({
                    "method": label, "visible_budget": str(budget), "repeat": repeat,
                    "mean_visible": float(visible.sum(1).mean()), **metrics,
                })
                predictions.append(frame.assign(method=label, visible_budget=str(budget), repeat=repeat))
            print(f"sparse budget={budget} repeat={repeat} complete", flush=True)

    runs = pd.DataFrame(records)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    summary = runs.groupby(["method", "visible_budget"], as_index=False).agg(
        macro_scaled_log_mae_mean=("macro_scaled_log_mae", "mean"),
        macro_scaled_log_mae_sd=("macro_scaled_log_mae", "std"),
        macro_scaled_log_rmse_mean=("macro_scaled_log_rmse", "mean"),
        macro_raw_mae_mean=("macro_raw_mae", "mean"),
        mean_visible=("mean_visible", "mean"), repeats=("repeat", "size"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest = {
        "status": "complete", "version": "dependency_mask_sparse_observation_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "profiles": int(jobs.profile_index.nunique()), "jobs": len(jobs),
        "targets": len(selected_axes), "jobs_per_target": 200,
        "budgets": list(BUDGETS), "finite_budget_repeats": 3,
        "methods": ["V5_CCE", "RF", "XGBoost", "KNN"],
        "mask_order": "dependency target mask first, then random context thinning",
        "baseline_training": "formal dependency-mask features; no mask-mix augmentation",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint": checkpoint_metadata(args.checkpoint, checkpoint),
        "summary": json.loads(summary.to_json(orient="records")),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

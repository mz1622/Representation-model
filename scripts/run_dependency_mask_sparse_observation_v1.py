#!/usr/bin/env python3
"""Sparse-observation completion audit for dependency-mask v1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    checkpoint_metadata, dependency_visible, load_dependency_model,
    predict_normalized,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    dense_features, make_baseline, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    inverse_values, numeric_only_training_view,
)
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


def baseline_predict(models, method, data, rows, axes, visible, config):
    values = np.where(visible, data.values[rows], 0)
    features = np.concatenate((values, visible.astype(np.float32)), axis=1).astype(np.float32)
    prediction = np.zeros(len(rows), dtype=np.float32)
    for axis in np.unique(axes):
        position = np.flatnonzero(axes == axis)
        model = models[int(axis)]
        if method == "knn":
            distance, neighbor = model[0].kneighbors(features[position])
            y, weight = model[1], model[2]
            importance = weight[neighbor] / (distance + 1e-3)
            prediction[position] = (importance * y[neighbor]).sum(1) / importance.sum(1)
        else:
            prediction[position] = model.predict(features[position])
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "models").mkdir()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    started = time.monotonic()
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    policy = DependencyMaskPolicy(data.axes)
    support = data.observed[np.ix_(data.validation, data.targets)].sum(0)
    selected_axes = np.asarray(data.targets, dtype=np.int64)[
        np.argsort(-support, kind="stable")[:12]
    ]
    job_parts = []
    rng = np.random.default_rng(SEED)
    for axis in selected_axes:
        candidates = data.validation[data.observed[data.validation, axis]]
        target_axes = np.full(len(candidates), axis, dtype=np.int64)
        visible = dependency_visible(data, policy, candidates, target_axes)
        candidates = candidates[visible.sum(1) >= 5]
        if len(candidates) < 200:
            raise ValueError(f"axis {axis} has fewer than 200 eligible validation rows")
        chosen = np.sort(rng.choice(candidates, 200, replace=False))
        job_parts.append(pd.DataFrame({
            "profile_index": chosen, "axis_index": int(axis),
            "target": data.raw[chosen, axis],
        }))
    jobs = pd.concat(job_parts, ignore_index=True).sort_values(
        ["profile_index", "axis_index"]
    ).reset_index(drop=True)
    rows = jobs.profile_index.to_numpy(dtype=np.int64)
    axes = jobs.axis_index.to_numpy(dtype=np.int64)
    full_visible = dependency_visible(data, policy, rows, axes)
    jobs.to_csv(args.output_dir / "jobs.csv", index=False)
    data.axes[data.axes.axis_index.isin(selected_axes)][
        ["axis_index", "canonical_name"]
    ].to_csv(args.output_dir / "target_axes.csv", index=False)

    checkpoints = {
        "V4": (args.checkpoint_dir / "v4_best_validation.pt", False),
        "V5": (args.checkpoint_dir / "v5_best_validation.pt", True),
        "V5_CCE": (args.checkpoint_dir / "cce_best_validation.pt", True),
    }
    models, checkpoint_info = {}, {}
    for name, (path, two_pass) in checkpoints.items():
        model, checkpoint, config = load_dependency_model(
            path, data, args.policy_manifest, device
        )
        models[name] = (model, two_pass)
        checkpoint_info[name] = checkpoint_metadata(path, checkpoint)

    baseline_models = {"RF": {}, "KNN": {}}
    for method, key in (("rf", "RF"), ("knn", "KNN")):
        for axis in selected_axes:
            train = data.train[data.observed[data.train, axis]]
            target_axes = np.full(len(train), axis, dtype=np.int64)
            x_train = dense_features(data, train, target_axes, policy)
            y = data.values[train, axis]
            weight = data.weights[train, axis]
            estimator = make_baseline(method, int(axis), config)
            if method == "knn":
                estimator.set_params(n_neighbors=min(config.knn_neighbors, len(train)))
                estimator.fit(x_train)
                baseline_models[key][int(axis)] = (estimator, y, weight)
            else:
                estimator.fit(x_train, y, sample_weight=weight / weight.mean())
                estimator.n_jobs = 1
                baseline_models[key][int(axis)] = estimator
                joblib.dump(estimator, args.output_dir / "models" / f"rf_{axis}.joblib")
            print(f"fit sparse baseline {key} axis {axis}", flush=True)

    results = []
    prediction_parts = []
    for budget in BUDGETS:
        repeats = (-1,) if budget == "all" else (0, 1, 2)
        for repeat in repeats:
            visible = sparse_visible(full_visible, budget, SEED + 1_000 * (repeat + 2) + (0 if budget == "all" else int(budget)))
            for name, (model, two_pass) in models.items():
                normalized = predict_normalized(
                    model, data, rows, axes, visible, device, 512,
                    two_pass=two_pass,
                )
                raw = inverse_values(normalized, data.scale[axes])
                prediction = pd.DataFrame({
                    "profile_index": rows, "axis_index": axes, "prediction": raw,
                })
                metrics, _, _ = score_profile_axis_macro(data, jobs, prediction)
                results.append({
                    "method": name, "visible_budget": str(budget), "repeat": repeat,
                    "mean_visible": float(visible.sum(1).mean()), **metrics,
                })
                tagged = prediction.assign(method=name, visible_budget=str(budget), repeat=repeat)
                prediction_parts.append(tagged)
            for name in ("RF", "KNN"):
                normalized = baseline_predict(
                    baseline_models[name], name.lower(), data, rows, axes,
                    visible, config,
                )
                raw = inverse_values(normalized, data.scale[axes])
                prediction = pd.DataFrame({
                    "profile_index": rows, "axis_index": axes, "prediction": raw,
                })
                metrics, _, _ = score_profile_axis_macro(data, jobs, prediction)
                results.append({
                    "method": name, "visible_budget": str(budget), "repeat": repeat,
                    "mean_visible": float(visible.sum(1).mean()), **metrics,
                })
                prediction_parts.append(prediction.assign(
                    method=name, visible_budget=str(budget), repeat=repeat
                ))
            print(f"sparse budget {budget} repeat {repeat} complete", flush=True)
    result = pd.DataFrame(results)
    result.to_csv(args.output_dir / "runs.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    summary = result.groupby(["method", "visible_budget"], as_index=False).agg(
        macro_scaled_log_mae_mean=("macro_scaled_log_mae", "mean"),
        macro_scaled_log_mae_sd=("macro_scaled_log_mae", "std"),
        macro_scaled_log_rmse_mean=("macro_scaled_log_rmse", "mean"),
        macro_raw_mae_mean=("macro_raw_mae", "mean"),
        mean_visible=("mean_visible", "mean"),
        repeats=("repeat", "size"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest = {
        "status": "complete", "version": "dependency_mask_sparse_observation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "profiles": int(jobs.profile_index.nunique()), "jobs": len(jobs),
        "targets": len(selected_axes), "jobs_per_target": 200,
        "budgets": list(BUDGETS), "finite_budget_repeats": 3,
        "mask_order": "dependency target mask first, then random context thinning",
        "baseline_training": "standard dependency-mask training features; no sparse mask-mix augmentation",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoints": checkpoint_info,
        "summary": json.loads(summary.to_json(orient="records")),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

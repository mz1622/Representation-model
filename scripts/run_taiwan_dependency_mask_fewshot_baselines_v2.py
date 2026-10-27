#!/usr/bin/env python3
"""Compare frozen V5+CCE with Taiwan few-shot RF and XGBoost fits."""
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

from foodcomp.dependency_mask_analysis_v1 import attach_taiwan_panel, load_dependency_model  # noqa: E402
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    dense_features, make_baseline, predict_transformer, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import inverse_values, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SHOTS = (8, 32, 128)
SEEDS = (20261027, 20261028, 20261029)
SPLIT_SEED = 20261026
TEST_PROFILES = 400


def coverage_nested_order(pool, axes, observed, seed):
    """Put a label-coverage set first, then append a random remaining order."""
    rng = np.random.default_rng(seed)
    remaining = np.asarray(pool, dtype=np.int64).copy()
    uncovered = np.ones(len(axes), dtype=bool)
    selected = []
    while uncovered.any():
        score = observed[np.ix_(remaining, axes)][:, uncovered].sum(1)
        best = score.max(initial=0)
        if best <= 0:
            raise ValueError("adaptation pool cannot cover all Taiwan axes")
        tied = np.flatnonzero(score == best)
        position = int(rng.choice(tied))
        row = int(remaining[position])
        selected.append(row)
        uncovered &= ~observed[row, axes]
        remaining = np.delete(remaining, position)
    if len(selected) > min(SHOTS):
        raise ValueError(f"axis coverage requires {len(selected)} profiles, more than 8 shots")
    tail = rng.permutation(remaining)
    return np.concatenate((np.asarray(selected, dtype=np.int64), tail)), len(selected)


def fit_predict(method, data, policy, config, train_rows, test_jobs):
    test_rows = test_jobs.profile_index.to_numpy(dtype=np.int64)
    test_axes = test_jobs.axis_index.to_numpy(dtype=np.int64)
    prediction = np.zeros(len(test_jobs), dtype=np.float32)
    support = []
    for axis in np.sort(test_jobs.axis_index.unique()):
        labelled = train_rows[data.observed[train_rows, axis]]
        if len(labelled) < 1:
            raise ValueError(f"{method} axis {axis} has no Taiwan label")
        train_axis = np.full(len(labelled), axis, dtype=np.int64)
        x_train = dense_features(data, labelled, train_axis, policy)
        estimator = make_baseline(method, int(axis), config)
        estimator.fit(x_train, data.values[labelled, axis])
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        position = np.flatnonzero(test_axes == axis)
        x_test = dense_features(
            data, test_rows[position], np.full(len(position), axis, dtype=np.int64), policy
        )
        prediction[position] = estimator.predict(x_test)
        support.append({"axis_index": int(axis), "training_labels": len(labelled)})
    raw = inverse_values(prediction, data.scale[test_axes])
    frame = test_jobs[["profile_index", "axis_index"]].copy()
    frame["prediction"] = raw
    return frame, pd.DataFrame(support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, novel_rows, panel, axes = attach_taiwan_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    foundation_model, checkpoint, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, device
    )
    order = np.random.default_rng(SPLIT_SEED).permutation(novel_rows)
    test_rows = np.sort(order[:TEST_PROFILES])
    adaptation_pool = np.sort(order[TEST_PROFILES:])
    test_jobs = jobs[jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    if test_jobs.axis_index.nunique() != 80:
        raise ValueError("fixed Taiwan test does not cover all 80 axes")
    split = panel[["food_id", "profile_index", "english_name_seen_in_pretraining"]].copy()
    split["fewshot_partition"] = np.where(
        split.profile_index.isin(test_rows), "fixed_test",
        np.where(split.profile_index.isin(adaptation_pool), "adaptation_pool", "excluded"),
    )
    split.to_csv(args.output_dir / "profile_split.csv", index=False)

    frozen = predict_transformer(
        foundation_model, data, policy, test_jobs, device, 512, two_pass=True
    ).sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    del foundation_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frozen_metrics, frozen_axis, _ = score_profile_axis_macro(data, test_jobs, frozen)
    frozen_axis.to_csv(args.output_dir / "frozen_zero_shot_axis_metrics.csv", index=False)
    write_json(args.output_dir / "frozen_zero_shot_metrics.json", frozen_metrics)

    records, supports, axis_parts, prediction_parts = [], [], [], []
    manifest = {
        "status": "running", "version": "taiwan_dependency_mask_fewshot_baselines_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES, "test_jobs": len(test_jobs), "test_axes": 80,
        "adaptation_pool_profiles": len(adaptation_pool),
        "frozen_zero_shot_metrics": frozen_metrics,
        "tree_protocol": (
            "For each axis, fit the unchanged formal RF or XGBoost estimator only on "
            "the selected Taiwan adaptation profiles; evaluate on one fixed Taiwan test."
        ),
        "source_pretraining_used_by_tree_fewshot": False,
        "transformer_finetuning_performed": False,
        "subset_sampling": "missingness-only greedy axis coverage, then random; 8 subset 32 subset 128",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "runs": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            train_rows = np.sort(nested[:shots])
            for method, label in (("rf", "RF"), ("xgb", "XGBoost")):
                run_started = time.monotonic()
                print(f"Taiwan few-shot {label}: seed={seed} shots={shots}", flush=True)
                prediction, support = fit_predict(
                    method, data, policy, config, train_rows, test_jobs
                )
                metrics, per_axis, _ = score_profile_axis_macro(data, test_jobs, prediction)
                record = {
                    "method": label, "shots": shots, "seed": seed, **metrics,
                    "train_profiles": len(train_rows),
                    "train_jobs": int(data.observed[np.ix_(train_rows, axes)].sum()),
                    "active_training_axes": int((support.training_labels > 0).sum()),
                    "min_axis_labels": int(support.training_labels.min()),
                    "max_axis_labels": int(support.training_labels.max()),
                    "coverage_profiles": coverage_profiles,
                    "elapsed_seconds": time.monotonic() - run_started,
                }
                records.append(record)
                manifest["runs"].append(record)
                support.insert(0, "seed", seed)
                support.insert(0, "shots", shots)
                support.insert(0, "method", label)
                supports.append(support)
                per_axis.insert(0, "seed", seed)
                per_axis.insert(0, "shots", shots)
                per_axis.insert(0, "method", label)
                axis_parts.append(per_axis)
                prediction_parts.append(prediction.assign(method=label, shots=shots, seed=seed))
                write_json(args.output_dir / "manifest.json", manifest)
                print(f"  SLMAE={metrics['macro_scaled_log_mae']:.6f}", flush=True)

    runs = pd.DataFrame(records)
    runs.to_csv(args.output_dir / "fewshot_runs.csv", index=False)
    pd.concat(supports, ignore_index=True).to_csv(
        args.output_dir / "training_support.csv", index=False
    )
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    pd.concat(prediction_parts, ignore_index=True).to_parquet(
        args.output_dir / "test_predictions.parquet", index=False
    )
    metrics = ["macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"]
    summary = runs.groupby(["method", "shots"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sd": (metric, "std") for metric in metrics},
        min_axis_labels=("min_axis_labels", "min"),
        max_axis_labels=("max_axis_labels", "max"),
    )
    frozen_rows = []
    for shots in SHOTS:
        frozen_rows.append({
            "method": "V5+CCE frozen", "shots": shots,
            **{f"{metric}_mean": frozen_metrics[metric] for metric in metrics},
            **{f"{metric}_sd": 0.0 for metric in metrics},
            "min_axis_labels": 0, "max_axis_labels": 0,
        })
    summary = pd.concat([summary, pd.DataFrame(frozen_rows)], ignore_index=True)
    summary.to_csv(args.output_dir / "fewshot_summary.csv", index=False)
    manifest.update(
        status="complete", elapsed_seconds=time.monotonic() - started,
        completed_runs=len(runs), summary=summary.to_dict(orient="records"),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

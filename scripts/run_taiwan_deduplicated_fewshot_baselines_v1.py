#!/usr/bin/env python3
"""Taiwan-only RF/XGBoost few-shot fits on the deduplicated external split."""
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

from foodcomp.dependency_mask_analysis_v1 import load_dependency_model  # noqa: E402
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    dense_features, make_baseline, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    inverse_values, numeric_only_training_view,
)
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.taiwan_deduplicated_v1 import (  # noqa: E402
    SEEDS, SHOTS, SPLIT_SEED, TEST_PROFILES, attach_clean_panel,
    coverage_nested_order, group_aware_split, row_hash, split_frame,
)


def fit_predict(method, data, policy, config, train_rows, test_jobs):
    test_rows = test_jobs.profile_index.to_numpy(dtype=np.int64)
    test_axes = test_jobs.axis_index.to_numpy(dtype=np.int64)
    prediction = np.zeros(len(test_jobs), dtype=np.float32)
    support = []
    for axis in np.sort(test_jobs.axis_index.unique()):
        labelled = train_rows[data.observed[train_rows, axis]]
        if not len(labelled):
            raise ValueError(f"{method} axis {axis} has no Taiwan label")
        train_axes = np.full(len(labelled), axis, dtype=np.int64)
        estimator = make_baseline(method, int(axis), config)
        estimator.fit(
            dense_features(data, labelled, train_axes, policy),
            data.values[labelled, axis],
        )
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        position = np.flatnonzero(test_axes == axis)
        normalized = estimator.predict(dense_features(
            data, test_rows[position],
            np.full(len(position), axis, dtype=np.int64), policy,
        ))
        prediction[position] = normalized
        support.append({"axis_index": int(axis), "training_labels": len(labelled)})
    frame = test_jobs[["profile_index", "axis_index"]].copy()
    frame["prediction"] = inverse_values(prediction, data.scale[test_axes])
    return frame, pd.DataFrame(support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--overlap-ledger", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zero-shot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, clean_jobs, clean, panel, axes = attach_clean_panel(
        data, args.audit_dir, args.overlap_ledger
    )
    test_rows, adaptation_pool = group_aware_split(clean)
    policy = DependencyMaskPolicy(data.axes)
    test_jobs = clean_jobs[clean_jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    split = split_frame(panel, clean, test_rows, adaptation_pool)
    expected_split = pd.read_csv(args.zero_shot_dir / "profile_split.csv")[[
        "profile_index", "fewshot_partition"
    ]].sort_values("profile_index").reset_index(drop=True)
    actual_split = split[["profile_index", "fewshot_partition"]].sort_values(
        "profile_index"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_split, actual_split, check_dtype=False)
    split.to_csv(args.output_dir / "profile_split.csv", index=False)

    model, _, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, torch.device("cpu")
    )
    del model
    zero_metrics = pd.read_csv(args.zero_shot_dir / "metrics.csv")
    frozen = zero_metrics[
        zero_metrics.panel.eq("fixed_test") & zero_metrics.method.eq("V5_CCE")
    ]
    if len(frozen) != 1:
        raise ValueError("missing fixed-test frozen V5+CCE metric")
    frozen_metrics = frozen.iloc[0].to_dict()

    manifest = {
        "status": "running", "version": "taiwan_deduplicated_fewshot_baselines_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES, "test_jobs": len(test_jobs),
        "test_axes": len(axes), "adaptation_pool_profiles": len(adaptation_pool),
        "identity_group_aware_split": True,
        "tree_protocol": "fit unchanged formal estimator per axis on selected Taiwan profiles only",
        "source_pretraining_used_by_tree_fewshot": False,
        "subset_sampling": "missingness-only axis coverage then random; nested 8 subset 32 subset 128",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "overlap_ledger_sha256": digest(args.overlap_ledger),
        "zero_shot_manifest_sha256": digest(args.zero_shot_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "runs": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    records, supports, axis_parts = [], [], []
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            train_rows = np.sort(nested[:shots])
            for method, label in (("rf", "RF Taiwan-only"), ("xgb", "XGBoost Taiwan-only")):
                run_started = time.monotonic()
                print(f"{label}: seed={seed} shots={shots}", flush=True)
                prediction, support = fit_predict(
                    method, data, policy, config, train_rows, test_jobs
                )
                metrics, per_axis, _ = score_profile_axis_macro(data, test_jobs, prediction)
                record = {
                    "method": label, "shots": shots, "seed": seed, **metrics,
                    "train_profiles": len(train_rows),
                    "train_jobs": int(data.observed[np.ix_(train_rows, axes)].sum()),
                    "min_axis_labels": int(support.training_labels.min()),
                    "max_axis_labels": int(support.training_labels.max()),
                    "coverage_profiles": coverage_profiles,
                    "selected_profile_rows_sha256": row_hash(train_rows),
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
                write_json(args.output_dir / "manifest.json", manifest)

    runs = pd.DataFrame(records)
    metrics = ["macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"]
    summary = runs.groupby(["method", "shots"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sd": (metric, "std") for metric in metrics},
        min_axis_labels=("min_axis_labels", "min"),
        max_axis_labels=("max_axis_labels", "max"),
    )
    frozen_rows = [{
        "method": "V5+CCE frozen", "shots": shots,
        **{f"{metric}_mean": frozen_metrics[metric] for metric in metrics},
        **{f"{metric}_sd": 0.0 for metric in metrics},
        "min_axis_labels": 0, "max_axis_labels": 0,
    } for shots in SHOTS]
    summary = pd.concat((summary, pd.DataFrame(frozen_rows)), ignore_index=True)
    runs.to_csv(args.output_dir / "fewshot_runs.csv", index=False)
    summary.to_csv(args.output_dir / "fewshot_summary.csv", index=False)
    pd.concat(supports, ignore_index=True).to_csv(
        args.output_dir / "training_support.csv", index=False
    )
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    manifest.update(
        status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started, completed_runs=len(runs),
        summary=summary.to_dict(orient="records"),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()

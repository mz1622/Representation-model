#!/usr/bin/env python3
"""Rescore frozen Taiwan zero-shot predictions after conservative deduplication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.research_dependency_mask_v1 import score_profile_axis_macro  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.taiwan_deduplicated_v1 import (  # noqa: E402
    SPLIT_SEED, TEST_PROFILES, attach_clean_panel, group_aware_split, split_frame,
)


SEED = 20261027


def bootstrap_matrix(scored, profiles, axes):
    profile_lookup = {value: index for index, value in enumerate(profiles)}
    axis_lookup = {value: index for index, value in enumerate(axes)}
    total = np.zeros((len(profiles), len(axes)), dtype=np.float64)
    count = np.zeros_like(total)
    rows = scored.profile_index.map(profile_lookup).to_numpy(dtype=np.int64)
    columns = scored.axis_index.map(axis_lookup).to_numpy(dtype=np.int64)
    np.add.at(total, (rows, columns), scored.scaled_log_abs_error.to_numpy(float))
    np.add.at(count, (rows, columns), 1.0)
    return total, count


def paired_bootstrap(scored_by_method, draws=2_000):
    methods = sorted(scored_by_method)
    reference = scored_by_method[methods[0]][["profile_index", "axis_index"]]
    for method in methods[1:]:
        if not reference.equals(scored_by_method[method][["profile_index", "axis_index"]]):
            raise ValueError("paired zero-shot cells differ across methods")
    profiles = np.sort(reference.profile_index.unique())
    axes = np.sort(reference.axis_index.unique())
    matrices = {
        method: bootstrap_matrix(frame, profiles, axes)
        for method, frame in scored_by_method.items()
    }
    rng = np.random.default_rng(SEED + 77)
    weights = np.stack([
        np.bincount(rng.integers(len(profiles), size=len(profiles)), minlength=len(profiles))
        for _ in range(draws)
    ]).astype(np.float64)
    estimates = {}
    for method, (total, count) in matrices.items():
        numerator = weights @ total
        denominator = weights @ count
        estimates[method] = np.nanmean(np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        ), axis=1)
    rows = []
    for comparator in methods:
        if comparator == "V5_CCE":
            continue
        difference = estimates["V5_CCE"] - estimates[comparator]
        rows.append({
            "method": "V5_CCE", "comparator": comparator, "draws": draws,
            "mean_difference": float(difference.mean()),
            "ci_low": float(np.quantile(difference, 0.025)),
            "ci_high": float(np.quantile(difference, 0.975)),
            "probability_primary_better": float((difference < 0).mean()),
        })
    return pd.DataFrame(rows)


def load_prediction(path, jobs):
    prediction = pd.read_parquet(path)[
        ["profile_index", "axis_index", "prediction"]
    ].sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    expected = jobs[["profile_index", "axis_index"]].sort_values(
        ["profile_index", "axis_index"]
    ).reset_index(drop=True)
    if not expected.equals(prediction[["profile_index", "axis_index"]]):
        raise ValueError(f"stored prediction panel mismatch: {path}")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--overlap-ledger", type=Path, required=True)
    parser.add_argument("--foundation-predictions", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, clean_jobs, clean, panel, axes = attach_clean_panel(
        data, args.audit_dir, args.overlap_ledger
    )
    test_rows, adaptation_rows = group_aware_split(clean)
    test_jobs = clean_jobs[clean_jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    if test_jobs.axis_index.nunique() != len(axes):
        raise ValueError("deduplicated fixed test lost axis coverage")
    split = split_frame(panel, clean, test_rows, adaptation_rows)
    split.to_csv(args.output_dir / "profile_split.csv", index=False)

    paths = {
        "V5_CCE": args.foundation_predictions,
        "KNN": args.baseline_dir / "knn_predictions.parquet",
        "RF": args.baseline_dir / "rf_predictions.parquet",
        "XGBoost": args.baseline_dir / "xgb_predictions.parquet",
    }
    predictions = {method: load_prediction(path, jobs) for method, path in paths.items()}
    metric_rows, axis_parts = [], []
    for panel_name, panel_jobs in (
        ("deduplicated_all", clean_jobs),
        ("fixed_test", test_jobs),
    ):
        scored = {}
        rows = set(panel_jobs.profile_index.unique())
        for method, prediction in predictions.items():
            part = prediction[prediction.profile_index.isin(rows)].reset_index(drop=True)
            metrics, per_axis, cells = score_profile_axis_macro(data, panel_jobs, part)
            metric_rows.append({"panel": panel_name, "method": method, **metrics})
            per_axis.insert(0, "method", method)
            per_axis.insert(0, "panel", panel_name)
            axis_parts.append(per_axis)
            scored[method] = cells.sort_values(
                ["profile_index", "axis_index"]
            ).reset_index(drop=True)
        paired_bootstrap(scored).assign(panel=panel_name).to_csv(
            args.output_dir / f"{panel_name}_paired_bootstrap.csv", index=False
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    manifest = {
        "status": "complete",
        "version": "taiwan_deduplicated_zero_shot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deduplication_scope": "all project train+validation source-native profiles",
        "deduplicated_profiles": int(len(clean)),
        "deduplicated_jobs": int(len(clean_jobs)),
        "fixed_test_profiles": int(len(test_rows)),
        "fixed_test_jobs": int(len(test_jobs)),
        "adaptation_pool_profiles": int(len(adaptation_rows)),
        "test_axes": int(len(axes)),
        "split_seed": SPLIT_SEED,
        "group_aware_split": True,
        "taiwan_training_rows_for_zero_shot": 0,
        "bootstrap_draws": 2_000,
        "bootstrap_unit": "Taiwan profile",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "overlap_ledger_sha256": digest(args.overlap_ledger),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "prediction_hashes": {method: digest(path) for method, path in paths.items()},
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

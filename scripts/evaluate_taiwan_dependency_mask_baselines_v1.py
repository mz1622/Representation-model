#!/usr/bin/env python3
"""External TFDA zero-shot comparison under dependency-mask v1."""
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
    attach_taiwan_panel, load_dependency_model,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    fit_predict_baseline, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261027


def bootstrap_matrix(scored, profiles, axes, error_column):
    profile_lookup = {value: i for i, value in enumerate(profiles)}
    axis_lookup = {value: i for i, value in enumerate(axes)}
    total = np.zeros((len(profiles), len(axes)), dtype=np.float64)
    count = np.zeros_like(total)
    rr = scored.profile_index.map(profile_lookup).to_numpy(dtype=np.int64)
    aa = scored.axis_index.map(axis_lookup).to_numpy(dtype=np.int64)
    np.add.at(total, (rr, aa), scored[error_column].to_numpy(dtype=float))
    np.add.at(count, (rr, aa), 1.0)
    return total, count


def paired_bootstrap(scored_by_method, draws=2_000):
    methods = sorted(scored_by_method)
    reference = scored_by_method[methods[0]][["profile_index", "axis_index"]]
    for method in methods[1:]:
        current = scored_by_method[method][["profile_index", "axis_index"]]
        if not reference.equals(current):
            raise ValueError("paired bootstrap cells differ across methods")
    profiles = np.sort(reference.profile_index.unique())
    axes = np.sort(reference.axis_index.unique())
    matrices = {
        method: bootstrap_matrix(frame, profiles, axes, "scaled_log_abs_error")
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
        estimates[method] = np.nanmean(
            np.divide(
                numerator, denominator,
                out=np.full_like(numerator, np.nan), where=denominator > 0,
            ), axis=1,
        )
    rows = []
    primary = "V5_CCE"
    for comparator in methods:
        if comparator == primary:
            continue
        difference = estimates[primary] - estimates[comparator]
        rows.append({
            "method": primary, "comparator": comparator, "draws": draws,
            "mean_difference": float(difference.mean()),
            "ci_low": float(np.quantile(difference, 0.025)),
            "ci_high": float(np.quantile(difference, 0.975)),
            "probability_primary_better": float((difference < 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--foundation-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, novel_rows, panel, axes = attach_taiwan_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    _, checkpoint, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, torch.device("cpu")
    )
    foundation = pd.read_parquet(args.foundation_predictions)[
        ["profile_index", "axis_index", "prediction"]
    ].sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    if len(foundation) != len(jobs):
        raise ValueError("foundation prediction panel differs from TFDA jobs")

    predictions = {"V5_CCE": foundation}
    support = []
    for method in ("knn", "rf", "xgb"):
        print(f"fitting Taiwan zero-shot baseline {method}", flush=True)
        prediction, method_support = fit_predict_baseline(
            data, policy, method, jobs, config
        )
        label = {"knn": "KNN", "rf": "RF", "xgb": "XGBoost"}[method]
        prediction = prediction.sort_values(
            ["profile_index", "axis_index"]
        ).reset_index(drop=True)
        predictions[label] = prediction
        prediction.to_parquet(args.output_dir / f"{method}_predictions.parquet", index=False)
        method_support.insert(0, "method", label)
        support.append(method_support)
    pd.concat(support, ignore_index=True).to_csv(
        args.output_dir / "baseline_training_support.csv", index=False
    )

    all_metrics = []
    all_axes = []
    for panel_name, panel_jobs in (
        ("strict_all", jobs),
        ("name_novel", jobs[jobs.profile_index.isin(novel_rows)]),
    ):
        scored = {}
        for method, prediction in predictions.items():
            prediction = prediction[prediction.profile_index.isin(
                panel_jobs.profile_index.unique()
            )]
            metrics, per_axis, cells = score_profile_axis_macro(
                data, panel_jobs, prediction
            )
            all_metrics.append({"panel": panel_name, "method": method, **metrics})
            per_axis.insert(0, "method", method)
            per_axis.insert(0, "panel", panel_name)
            all_axes.append(per_axis)
            scored[method] = cells.sort_values(
                ["profile_index", "axis_index"]
            ).reset_index(drop=True)
        paired_bootstrap(scored).assign(panel=panel_name).to_csv(
            args.output_dir / f"{panel_name}_paired_bootstrap.csv", index=False
        )
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.concat(all_axes, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    manifest = {
        "status": "complete", "version": "taiwan_dependency_mask_baselines_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "audit_sha256": digest(args.audit_dir / "audit_summary.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "foundation_predictions_sha256": digest(args.foundation_predictions),
        "profiles": len(panel), "jobs": len(jobs), "axes": len(axes),
        "name_novel_profiles": len(novel_rows),
        "taiwan_training_rows": 0, "taiwan_model_selection": False,
        "legacy_family_column_used": False,
        "bootstrap_draws": 2_000, "bootstrap_unit": "Taiwan profile",
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

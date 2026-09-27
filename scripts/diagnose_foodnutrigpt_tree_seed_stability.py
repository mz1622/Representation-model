"""Expose axis/source seed dispersion that a stable macro average can hide."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_confirmation import SEEDS
from foodcomp.research_r0 import digest, write_json


def summarize(frame, keys, support):
    metrics = ["scaled_log_mae", "log_mae"]
    if set(frame.seed) != set(SEEDS) or frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Require exactly the three distinct registered seeds per stratum.")
    if not np.isfinite(frame[metrics].to_numpy()).all() or (frame[metrics].to_numpy() < 0).any():
        raise ValueError("Invalid observed errors.")
    grouped = frame.groupby(keys, dropna=False)
    if not grouped.size().eq(3).all() or not grouped[support].nunique().eq(1).all().all():
        raise ValueError("Seed coverage or stratum support differs.")
    result = grouped[support].first()
    for metric in metrics:
        aggregate = grouped[metric].agg(["mean", "std", "min", "max"])
        aggregate.columns = [f"{metric}_{name}" for name in ["mean", "seed_sample_std", "min", "max"]]
        result = result.join(aggregate)
    return result.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    receipt = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        saved = json.loads((args.summary_dir / "summary.json").read_text())
        if saved["status"] != "complete" or saved["complete_test_opened"] or not saved["all_training_counts_verified"]:
            raise ValueError("Require the completed rescored full-row seed summary.")
        axis_file = args.summary_dir / "all_seed_axis_metrics.csv"
        source_file = args.summary_dir / "all_seed_source_metrics.csv"
        axes = summarize(pd.read_csv(axis_file), ["axis_index", "canonical_name", "loss_group"], ["candidate_support"])
        sources = summarize(pd.read_csv(source_file), ["source"], ["supported_nutrition_axes", "candidate_groups"])
        axes["sparse_candidate_support_below_30"] = axes.candidate_support < 30
        for group in ["nutrition", "food_metabolome", "all"]:
            part = axes if group == "all" else axes[axes.loss_group.eq(group)]
            if len(part) != saved["group_summaries"][group]["axes"]:
                raise ValueError("Registered axis coverage changed.")
            for metric in ["scaled_log_mae", "log_mae"]:
                np.testing.assert_allclose(part[f"{metric}_mean"].mean(),
                    saved["group_summaries"][group]["metrics"][metric]["mean"], rtol=1e-12)
        axes.to_csv(args.output_dir / "axis_seed_stability.csv", index=False)
        sources.to_csv(args.output_dir / "source_seed_stability.csv", index=False)
        nutrition = axes[axes.loss_group.eq("nutrition")]
        receipt.update(status="complete", method=saved["method"], seeds=list(SEEDS),
            rescored_summary_sha256=digest(args.summary_dir / "summary.json"),
            input_hashes={axis_file.name: digest(axis_file), source_file.name: digest(source_file)},
            nutrition_axes=len(nutrition), sparse_nutrition_axes=int(nutrition.sparse_candidate_support_below_30.sum()),
            largest_nutrition_axis_seed_sd=nutrition.nlargest(10, "scaled_log_mae_seed_sample_std").to_dict("records"),
            all_axis_count=len(axes), source_count=len(sources),
            scope="Descriptive sample SD of fixed-configuration errors across three seeds, not a confidence interval or independent observations per seed-cell. Macro stability does not imply per-axis stability. Source averages have different axis coverage; support does not establish a causal reason for dispersion.")
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / "summary.json", receipt)
        raise
    write_json(args.output_dir / "summary.json", receipt)
    print({k: receipt[k] for k in ["status", "method", "nutrition_axes", "sparse_nutrition_axes", "source_count"]})


if __name__ == "__main__":
    main()

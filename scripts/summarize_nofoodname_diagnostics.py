#!/usr/bin/env python3
"""Supplementary validation diagnostics using the frozen candidate/source grouping."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402


def diagnostics(data, prediction):
    keys = ["profile_index", "axis_index"]
    jobs = data.jobs[keys + ["target"]]
    if prediction.duplicated(keys).any():
        raise ValueError("duplicate prediction keys")
    merged = jobs.merge(prediction, on=keys, how="outer", validate="one_to_one",
                        indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("prediction does not exactly cover validation jobs")
    if not np.isfinite(merged.prediction).all() or merged.prediction.lt(0).any():
        raise ValueError("invalid predictions")
    merged = merged.merge(data.profiles[["profile_index", "exact_name_group_id",
                                         "source_key"]], on="profile_index",
                          validate="many_to_one")
    scale = data.scale[merged.axis_index.to_numpy(dtype=np.int64)]
    merged["log_abs"] = np.abs(np.log1p(merged.prediction / scale) -
                               np.log1p(merged.target / scale))
    zero = merged.assign(zero=merged.target.eq(0)).groupby(
        ["axis_index", "zero"], as_index=False).agg(
            job_count=("log_abs", "size"), job_log_mae=("log_abs", "mean"))
    source = merged.groupby(["exact_name_group_id", "axis_index", "source_key"],
                            as_index=False).agg(target=("target", "median"),
                                                 prediction=("prediction", "median"))
    scale = data.scale[source.axis_index.to_numpy(dtype=np.int64)]
    source["log_sq"] = (np.log1p(source.prediction / scale) -
                        np.log1p(source.target / scale)) ** 2
    candidate = source.groupby(["exact_name_group_id", "axis_index"],
                               as_index=False).agg(log_sq=("log_sq", "mean"))
    rmse = candidate.groupby("axis_index", as_index=False).agg(
        candidate_log_mse=("log_sq", "mean"), candidate_count=("log_sq", "size"))
    rmse["scaled_log_rmse"] = np.sqrt(rmse.candidate_log_mse)
    result = rmse.drop(columns="candidate_log_mse")
    for is_zero, label in ((True, "zero"), (False, "positive")):
        part = zero[zero.zero.eq(is_zero)][["axis_index", "job_count", "job_log_mae"]]
        result = result.merge(part.rename(columns={"job_count": f"{label}_jobs",
                                                   "job_log_mae": f"{label}_job_log_mae"}),
                              on="axis_index", how="left", validate="one_to_one")
    axes = np.asarray(data.targets, dtype=np.int64)
    observed = data.observed[data.train][:, axes]
    explicit_zero = (data.raw[data.train][:, axes] == 0) & observed
    train = pd.DataFrame({"axis_index": axes, "train_support": observed.sum(axis=0),
                          "train_zero_rate": explicit_zero.sum(axis=0) /
                          observed.sum(axis=0)})
    result = result.merge(train, on="axis_index", validate="one_to_one")
    result = result.merge(data.axes[["axis_index", "canonical_name", "loss_group"]],
                          on="axis_index", validate="one_to_one")
    summary = {}
    for group in ("nutrition", "food_metabolome", "all"):
        part = result if group == "all" else result[result.loss_group.eq(group)]
        summary[group] = {
            "axes": len(part),
            "macro_axis_scaled_log_rmse": float(part.scaled_log_rmse.mean()),
            "macro_axis_zero_job_log_mae": float(part.zero_job_log_mae.mean()),
            "macro_axis_positive_job_log_mae": float(part.positive_job_log_mae.mean()),
        }
    return summary, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True,
                        help="NAME=path/to/validation_predictions.parquet")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    args.output_dir.mkdir(parents=True)
    output = {"data_sha256": digest(args.data_dir / "manifest.json"),
              "code_sha256": digest(Path(__file__)), "complete_test_opened": False,
              "methods": {}}
    for item in args.prediction:
        name, separator, path_text = item.partition("=")
        if not separator or not name or name in output["methods"]:
            raise ValueError("prediction must be a distinct NAME=path")
        path = Path(path_text)
        summary, axis = diagnostics(data, pd.read_parquet(path))
        axis.to_csv(args.output_dir / f"{name}_axis_diagnostics.csv", index=False)
        output["methods"][name] = {"prediction_sha256": digest(path),
                                    "summary": summary}
        print(name, summary["all"]["macro_axis_scaled_log_rmse"])
    write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()

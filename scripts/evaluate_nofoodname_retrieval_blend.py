#!/usr/bin/env python3
"""Fixed equal-weight log-scale fusion of numeric Transformer and KNN retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view, score_subset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--knn-predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    neural = pd.read_parquet(args.neural_predictions)
    retrieved = pd.read_parquet(args.knn_predictions)
    keys = ["profile_index", "axis_index"]
    if neural.duplicated(keys).any() or retrieved.duplicated(keys).any():
        raise ValueError("duplicate prediction keys")
    paired = neural.merge(retrieved, on=keys, how="outer", indicator=True,
                          validate="one_to_one", suffixes=("_neural", "_knn"))
    if not paired._merge.eq("both").all():
        raise ValueError("neural and retrieval predictions must cover identical jobs")
    scale = data.scale[paired.axis_index.to_numpy(dtype=np.int64)]
    raw = paired[["prediction_neural", "prediction_knn"]].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("invalid nonnegative raw predictions")
    transformed = np.log1p(raw / scale[:, None]).mean(axis=1)
    prediction = paired[keys].copy()
    prediction["prediction"] = scale * np.expm1(transformed)
    axes = list(map(int, data.targets))
    metrics, per_axis = score_subset(data, prediction, axes,
                                     validation_rows=data.validation)
    args.output_dir.mkdir(parents=True)
    prediction.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "manifest.json", {
        "status": "complete", "version": "no_foodname_numeric_retrieval_blend",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "neural_prediction_sha256": digest(args.neural_predictions),
        "knn_prediction_sha256": digest(args.knn_predictions),
        "code_sha256": digest(Path(__file__)), "metrics": metrics,
        "blend": "fixed 0.5 + 0.5 arithmetic mean in train-scaled log1p space",
        "weight_selected_on_validation": False,
        "food_name_model_input": False, "complete_test_opened": False,
    })
    print(f"numeric retrieval blend: {metrics['all']['scaled_log_mae']:.6f}")


if __name__ == "__main__":
    main()

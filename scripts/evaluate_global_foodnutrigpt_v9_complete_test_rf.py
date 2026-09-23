#!/usr/bin/env python3
"""Evaluate the source-free V9 Random Forest baseline on the complete test panel.

The RF configuration, visible inputs, chemical-family masking, and source-free
candidate-cell aggregation must match the validation RF run.  This script is
intentionally separate from validation model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_global_foodnutrigpt_v9_rf_baseline as rf_code  # noqa: E402
import train_global_foodnutrigpt_v9_source_calibrated as v9  # noqa: E402


DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
TRANSFORMER_OUTPUT = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_validation"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_rf20_max5000_complete_test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--transformer-output", type=Path, default=TRANSFORMER_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--max-train-rows-per-axis", type=int, default=5_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir, split_dir = args.data_dir.resolve(), args.split_dir.resolve()
    transformer_output, output_dir = args.transformer_output.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    corpus = v9.SourceEqualizedCorpus(data_dir, split_dir)
    config_values = json.loads((transformer_output / "config.json").read_text())
    config_values.pop("data_dir", None)
    config_values.pop("split_dir", None)
    config = v9.Config(**config_values)
    train_rows = np.asarray(corpus.positions("train"), dtype=np.int64)
    test_rows = np.asarray(corpus.positions("test_complete_axis_panel"), dtype=np.int64)
    text = rf_code.load_cached_embeddings(corpus, transformer_output, include_complete_test=True)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=rf_code.SEED)
    text_pca = np.zeros((len(corpus.profiles), 32), dtype=np.float32)
    text_pca[train_rows] = pca.fit_transform(text[train_rows]).astype(np.float32)
    text_pca[test_rows] = pca.transform(text[test_rows]).astype(np.float32)
    values, observed = rf_code.build_dense_profile_cells(corpus, data_dir)
    base = np.concatenate([text_pca, values, observed], axis=1).astype(np.float32)
    jobs, hidden_by_axis = rf_code.complete_partition_jobs(corpus, config, "test_complete_axis_panel")
    loss_axes = corpus.axes[corpus.axes["loss_eligible"]].copy()
    if set(loss_axes["axis_index"]) != set(jobs):
        raise ValueError("RF loss axes do not match the V9 complete test panel.")

    rows: list[pd.DataFrame] = []
    for completed, axis_row in enumerate(loss_axes.itertuples(index=False), start=1):
        axis = int(axis_row.axis_index)
        labels = train_rows[observed[train_rows, axis].astype(bool)]
        if len(labels) < 20:
            raise ValueError(f"RF axis has fewer than 20 observed train labels: {axis_row.canonical_name}")
        if len(labels) > args.max_train_rows_per_axis:
            labels = np.sort(rf_code.stable_rng(axis).choice(labels, size=args.max_train_rows_per_axis, replace=False))
        hidden = hidden_by_axis[axis]
        x_train = base[labels].copy()
        x_train[:, 32 + hidden] = 0.0
        x_train[:, 32 + len(corpus.axes) + hidden] = 0.0
        estimator = RandomForestRegressor(
            n_estimators=args.trees, max_depth=12, max_features=0.35,
            min_samples_leaf=5, n_jobs=args.n_jobs, random_state=rf_code.SEED + axis,
        )
        estimator.fit(x_train, values[labels, axis], sample_weight=rf_code.train_axis_sample_weights(corpus, labels, axis))
        panel_rows = jobs[axis]
        if not observed[panel_rows, axis].all():
            raise AssertionError(f"An RF test job lacks an observed target: {axis_row.canonical_name}")
        x_test = base[panel_rows].copy()
        x_test[:, 32 + hidden] = 0.0
        x_test[:, 32 + len(corpus.axes) + hidden] = 0.0
        prediction_z = estimator.predict(x_test).astype(np.float32)
        target_raw = np.expm1(np.maximum(values[panel_rows, axis] * axis_row.scale_log1p + axis_row.median_log1p, 0.0))
        prediction_raw = np.expm1(np.maximum(prediction_z * axis_row.scale_log1p + axis_row.median_log1p, 0.0))
        rows.append(pd.DataFrame({
            "profile_id": corpus.profiles.iloc[panel_rows]["profile_id"].tolist(),
            "target_axis_id": axis_row.target_axis_id,
            "canonical_name": axis_row.canonical_name,
            "loss_group": axis_row.loss_group,
            "target_g_per_100g": target_raw,
            "prediction_g_per_100g": prediction_raw,
            "positive_probability": np.nan,
            "target_positive": (target_raw > 0).astype(np.int64),
        }))
        print(f"RF test axis {completed:03d}/{len(loss_axes)}: {axis_row.canonical_name} ({len(labels):,} train; {len(panel_rows):,} test)")

    rf_profile = pd.concat(rows, ignore_index=True)
    rf_cells, rf_axis, rf_metrics = v9.summarize_source_free_cells(rf_profile, corpus)
    transformer_profile = pd.read_csv(transformer_output / "complete_test_source_free_profile_predictions.csv")
    transformer_cells, transformer_axis, transformer_metrics = v9.summarize_source_free_cells(transformer_profile, corpus)
    rf_keys = set(zip(rf_profile["profile_id"], rf_profile["target_axis_id"]))
    transformer_keys = set(zip(transformer_profile["profile_id"], transformer_profile["target_axis_id"]))
    if rf_keys != transformer_keys:
        raise AssertionError(f"RF and Transformer do not score identical V9 test profile-axis cells: {len(rf_keys)} vs {len(transformer_keys)}")
    comparison = pd.DataFrame([
        {"method": "FoodNutriGPT_v9_source_free_base", **transformer_metrics},
        {"method": "RandomForest_source_free", **rf_metrics},
    ])
    rf_profile.to_csv(output_dir / "rf_complete_test_source_free_profile_predictions.csv", index=False)
    rf_cells.to_csv(output_dir / "rf_complete_test_source_free_candidate_cells.csv", index=False)
    rf_axis.to_csv(output_dir / "rf_complete_test_source_free_axis_metrics.csv", index=False)
    transformer_cells.to_csv(output_dir / "foodnutrigpt_complete_test_source_free_candidate_cells.csv", index=False)
    transformer_axis.to_csv(output_dir / "foodnutrigpt_complete_test_source_free_axis_metrics.csv", index=False)
    comparison.to_csv(output_dir / "complete_test_comparison.csv", index=False)
    manifest = {
        "dataset_version": json.loads((data_dir / "build_manifest.json").read_text())["dataset_version"],
        "protocol": "same V9 complete-test profile-axis jobs; source-free encoder/features; equal-source candidate-cell scoring",
        "source_policy": "No source feature in RF or Transformer. Source is only an offline equal-weight label aggregation unit.",
        "rf_features": "32 train-fitted PCA components of frozen MiniLM text, visible normalized values, and observedness indicators",
        "mask_policy": "all axes in the target chemical family hidden before RF fitting and inference",
        "rf_hyperparameters": {
            "n_estimators": args.trees, "max_depth": 12, "max_features": 0.35,
            "min_samples_leaf": 5, "max_train_rows_per_axis": args.max_train_rows_per_axis,
        },
        "metrics": comparison.to_dict(orient="records"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

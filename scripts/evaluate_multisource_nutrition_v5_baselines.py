#!/usr/bin/env python3
"""Evaluate leakage-safe v5 baselines on the fixed sequential test masks.

The baselines use exactly the FoodNutriGPT v5 data, train/validation/test split,
train-only normalization and fixed test masks.  For a target, all axes in the
same aggregate/component family are hidden from its input.  Test predictions
are generated in the fixed sequential order; an earlier prediction becomes
visible context for the later one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import evaluate_multisource_prediction_quality as quality
import train_multisource_nutrition_v5 as v5


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
DEFAULT_OUTPUT_DIR = ROOT / "output/multisource_nutrition_v5_baselines_v1"
METHODS = ("text_knn", "semantic_nn", "rf", "xgb")


def dense_profiles(bundle: dict[str, object], food_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized values and observedness indicators without zero-imputation."""
    axis_count = len(bundle["axes"])
    values = np.zeros((len(food_ids), axis_count), dtype=np.float32)
    observed = np.zeros((len(food_ids), axis_count), dtype=np.float32)
    for row, food_id in enumerate(food_ids):
        item = bundle["examples"][food_id]
        values[row, item["axes"]] = item["values"]
        observed[row, item["axes"]] = 1.0
    return values, observed


def feature_matrix(text: np.ndarray, values: np.ndarray, observed: np.ndarray, hidden_columns: np.ndarray) -> np.ndarray:
    values = values.copy()
    observed = observed.copy()
    values[:, hidden_columns] = 0.0
    observed[:, hidden_columns] = 0.0
    return np.concatenate([text, values, observed], axis=1)


def build_model(method: str, n_jobs: int):
    if method == "rf":
        return RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            max_features=0.35,
            min_samples_leaf=5,
            n_jobs=n_jobs,
            random_state=20260813,
        )
    if method == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=80,
            max_depth=5,
            learning_rate=0.08,
            min_child_weight=5,
            subsample=0.75,
            colsample_bytree=0.50,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            random_state=20260813,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported tabular method: {method}")


def text_predictions(
    method: str,
    train_text: np.ndarray,
    train_values: np.ndarray,
    train_observed: np.ndarray,
    test_text: np.ndarray,
    target_index: int,
) -> np.ndarray:
    candidate = train_observed[:, target_index].astype(bool)
    if not candidate.any():
        raise ValueError(f"No observed train labels for axis index {target_index}.")
    train_text = train_text[candidate]
    targets = train_values[candidate, target_index]
    similarity = test_text @ train_text.T
    if method == "semantic_nn":
        return targets[similarity.argmax(axis=1)]
    if method == "text_knn":
        count = min(5, len(targets))
        nearest = np.argpartition(-similarity, kth=count - 1, axis=1)[:, :count]
        weights = np.maximum(similarity[np.arange(len(test_text))[:, None], nearest], 0.0) + 1e-4
        return (targets[nearest] * weights).sum(axis=1) / weights.sum(axis=1)
    raise ValueError(f"Unsupported text method: {method}")


def prediction_log(normalized: np.ndarray, axis_index: int, axes: pd.DataFrame, normalization: pd.DataFrame) -> np.ndarray:
    axis_id = axes.loc[axes["axis_index"].eq(axis_index), "axis_id"].iloc[0]
    row = normalization.loc[normalization["axis_id"].eq(axis_id)].iloc[0]
    return normalized * float(row["baseline_log_rmse"]) + float(row["median"])


def evaluate_method(method: str, bundle: dict[str, object], masks: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    food_index = bundle["food_index"]
    axes = bundle["axes"]
    axis_by_id = dict(zip(axes["axis_id"], axes["axis_index"]))
    family_columns = {
        family: group["axis_index"].to_numpy(dtype=np.int64)
        for family, group in axes.groupby("mask_family", sort=False)
    }
    train_ids = list(bundle["splits"]["train"])
    test_ids = list(bundle["splits"]["test"])
    train_values, train_observed = dense_profiles(bundle, train_ids)
    test_values, test_observed = dense_profiles(bundle, test_ids)
    train_text = bundle["embeddings"][[food_index[food_id] for food_id in train_ids]].astype(np.float32)
    test_text = bundle["embeddings"][[food_index[food_id] for food_id in test_ids]].astype(np.float32)
    test_row = {food_id: index for index, food_id in enumerate(test_ids)}

    # Hide every requested test family before the first sequential prediction.
    requested_families = masks.groupby("canonical_food_id", sort=False)["mask_family"].unique()
    for food_id, families in requested_families.items():
        row = test_row[food_id]
        columns = np.concatenate([family_columns[family] for family in families])
        test_values[row, columns] = 0.0
        test_observed[row, columns] = 0.0

    fitted: dict[int, object] = {}
    rows: list[dict[str, object]] = []
    for order in range(1, int(masks["prediction_order"].max()) + 1):
        current = masks[masks["prediction_order"].eq(order)]
        pending: list[dict[str, object]] = []
        for axis_id, group in current.groupby("axis_id", sort=True):
            target_index = axis_by_id[axis_id]
            row_indices = np.array([test_row[food_id] for food_id in group["canonical_food_id"]], dtype=np.int64)
            if method in {"text_knn", "semantic_nn"}:
                predicted = text_predictions(method, train_text, train_values, train_observed, test_text[row_indices], target_index)
            else:
                if target_index not in fitted:
                    family = axes.loc[axes["axis_index"].eq(target_index), "mask_family"].iloc[0]
                    hidden = family_columns[family]
                    labels = train_observed[:, target_index].astype(bool)
                    features = feature_matrix(train_text[labels], train_values[labels], train_observed[labels], hidden)
                    estimator = build_model(method, n_jobs=n_jobs)
                    estimator.fit(features, train_values[labels, target_index])
                    fitted[target_index] = (estimator, hidden)
                estimator, hidden = fitted[target_index]
                predicted = estimator.predict(feature_matrix(test_text[row_indices], test_values[row_indices], test_observed[row_indices], hidden))
            predicted = np.asarray(predicted, dtype=np.float32)
            targets = np.array([bundle["examples"][food_id]["values"][np.where(bundle["examples"][food_id]["axes"] == target_index)[0][0]] for food_id in group["canonical_food_id"]], dtype=np.float32)
            logs = prediction_log(np.asarray(predicted, dtype=np.float32), target_index, axes, bundle["normalization"])
            for food_id, target, prediction, log_prediction, row_index in zip(group["canonical_food_id"], targets, predicted, logs, row_indices):
                pending.append({
                    "canonical_food_id": food_id,
                    "axis_id": axis_id,
                    "prediction_order": order,
                    "target_normalized": float(target),
                    "prediction_normalized": float(prediction),
                    "prediction_log1p_g_per_100g": float(log_prediction),
                    "row_index": int(row_index),
                    "target_index": int(target_index),
                })
        # Predictions tied at the same order share exactly the same context.
        # Only a completed order becomes visible to the following order.
        for item in pending:
            test_values[item["row_index"], item["target_index"]] = item["prediction_normalized"]
            test_observed[item["row_index"], item["target_index"]] = 1.0
            item.pop("row_index")
            item.pop("target_index")
            rows.append(item)
        print(f"{method}: sequential order {order} complete ({len(rows):,} predictions)")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker count for RF/XGBoost; use 1 to avoid nested BLAS/joblib contention.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir, split_dir = args.data_dir.resolve(), args.split_dir.resolve()
    config = v5.Config()
    v5.set_seed(config.seed)
    bundle = v5.load_bundle(config, output_dir, data_dir, split_dir)
    masks = pd.read_csv(split_dir / "sequential_masks.csv")
    masks = masks[masks["split"].eq("test")].copy()
    axes = bundle["axes"]
    normalization = bundle["normalization"]
    summary: dict[str, dict[str, object]] = {}
    for method in args.methods:
        frame = evaluate_method(method, bundle, masks, n_jobs=args.n_jobs)
        frame.to_csv(output_dir / f"{method}_sequential_test_predictions.csv", index=False)
        report_sections = []
        method_summary: dict[str, object] = {}
        for kind, label in (("core_nutrition", "Core Nutrition"), ("nutrient_chemical_form", "Nutrient Chemical Forms")):
            kind_axis_ids = set(axes.loc[axes["target_kind"].eq(kind), "axis_id"])
            kind_frame = frame[frame["axis_id"].isin(kind_axis_ids)].copy()
            axis_metrics, metrics = quality.analyze_predictions(kind_frame, axes, normalization, kind)
            axis_metrics.to_csv(output_dir / f"{method}_{kind}_axis_metrics.csv", index=False)
            report_sections.append(quality.render_summary(label, metrics, axis_metrics))
            method_summary[kind] = metrics
        (output_dir / f"{method}_quality_report.md").write_text(
            f"# {method} v5 baseline quality\n\n" + "\n\n".join(report_sections) + "\n", encoding="utf-8"
        )
        summary[method] = method_summary
    with (output_dir / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

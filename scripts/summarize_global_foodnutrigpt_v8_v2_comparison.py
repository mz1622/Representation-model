#!/usr/bin/env python3
"""Compare FoodNutriGPT variants and RF on identical complete-test cells.

All methods are evaluated once per ``(profile_id, target_axis_id)``.  Source-
native Transformer predictions are reduced by their within-cell median, and the
RF output supplies the shared median-log profile-axis target grid.

Colab:
    python scripts/summarize_global_foodnutrigpt_v8_v2_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUPED_OUTPUT = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_complete_test"
ALL_AXIS_OUTPUT = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_all_axis_loss"
RF_OUTPUT = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_rf_baseline_v2"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_method_comparison"


KEYS = ["profile_id", "target_axis_id", "canonical_name", "loss_group"]


def transformer_predictions(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=KEYS + ["prediction_g_per_100g"])
    return (
        frame.groupby(KEYS, as_index=False)["prediction_g_per_100g"].median()
        .assign(method=method)
    )


def rf_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = set(KEYS + ["target_g_per_100g", "prediction_g_per_100g"])
    if missing := required - set(frame.columns):
        raise ValueError(f"RF output lacks columns: {sorted(missing)}")
    return (
        frame[KEYS + ["prediction_g_per_100g"]].assign(method="RandomForest"),
        frame[KEYS + ["target_g_per_100g"]],
    )


def per_axis_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["raw_absolute_error"] = np.abs(
        predictions["prediction_g_per_100g"] - predictions["target_g_per_100g"]
    )
    predictions["log_absolute_error"] = np.abs(
        np.log1p(predictions["prediction_g_per_100g"])
        - np.log1p(predictions["target_g_per_100g"])
    )
    return predictions.groupby(
        ["method", "target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        profile_axis_cells=("profile_id", "size"),
        log_mae=("log_absolute_error", "mean"),
        log_mse=("log_absolute_error", lambda values: float(np.mean(np.square(values)))),
        raw_mae_g_per_100g=("raw_absolute_error", "mean"),
    )


def summary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, method_frame in metrics.groupby("method", sort=False):
        for scope, scope_frame in [("all_187_axes", method_frame)] + [
            (group, method_frame[method_frame["loss_group"].eq(group)])
            for group in sorted(method_frame["loss_group"].unique())
        ]:
            if scope_frame.empty:
                continue
            rows.append({
                "method": method,
                "scope": scope,
                "axes_scored": int(scope_frame["target_axis_id"].nunique()),
                "profile_axis_cells": int(scope_frame["profile_axis_cells"].sum()),
                "macro_axis_log_mae": float(scope_frame["log_mae"].mean()),
                "macro_axis_log_rmse": float(math.sqrt(scope_frame["log_mse"].mean())),
                "macro_axis_raw_mae_g_per_100g": float(scope_frame["raw_mae_g_per_100g"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-output", type=Path, default=GROUPED_OUTPUT)
    parser.add_argument("--all-axis-output", type=Path, default=ALL_AXIS_OUTPUT)
    parser.add_argument("--rf-output", type=Path, default=RF_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=None,
        help="Optional final selected Transformer evaluation output to add to the comparison.",
    )
    parser.add_argument(
        "--candidate-name",
        type=str,
        default="FoodNutriGPT_selected_candidate",
        help="Method label used when --candidate-output is supplied.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = transformer_predictions(
        args.grouped_output / "complete_axis_panel_source_unknown_predictions.csv",
        "FoodNutriGPT_group_balanced",
    )
    all_axis = transformer_predictions(
        args.all_axis_output / "complete_axis_panel_source_unknown_predictions.csv",
        "FoodNutriGPT_all_axis",
    )
    candidates: list[pd.DataFrame] = []
    if args.candidate_output is not None:
        candidates.append(
            transformer_predictions(
                args.candidate_output / "complete_axis_panel_source_unknown_predictions.csv",
                args.candidate_name,
            )
        )
    rf, targets = rf_predictions(args.rf_output / "rf_complete_axis_panel_predictions.csv")
    expected_keys = set(map(tuple, targets[KEYS].to_numpy()))
    method_frames = [
        ("FoodNutriGPT_group_balanced", grouped),
        ("FoodNutriGPT_all_axis", all_axis),
        *[(args.candidate_name, frame) for frame in candidates],
        ("RandomForest", rf),
    ]
    for method, frame in method_frames:
        keys = set(map(tuple, frame[KEYS].to_numpy()))
        if keys != expected_keys:
            raise AssertionError(f"{method} does not use the common profile-axis panel: {len(keys)} vs {len(expected_keys)}")

    predictions = pd.concat([grouped, all_axis, *candidates, rf], ignore_index=True).merge(
        targets, on=KEYS, how="inner", validate="many_to_one",
    )
    if len(predictions) != len(method_frames) * len(targets):
        raise AssertionError("Comparison lost or duplicated profile-axis prediction cells.")
    axis_metrics = per_axis_metrics(predictions)
    summary = summary_rows(axis_metrics)
    summary.to_csv(args.output_dir / "method_comparison_summary.csv", index=False)
    axis_metrics.to_csv(args.output_dir / "method_comparison_per_axis.csv", index=False)
    with (args.output_dir / "method_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "test_panel": "v2 complete 187-axis leave-one-mask-family-out profile-axis cells",
            "target_cell_policy": "median normalized log1p value per profile-axis, then inverse-transformed to g/100 g",
            "methods": summary.to_dict(orient="records"),
        }, handle, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

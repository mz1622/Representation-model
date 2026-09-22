#!/usr/bin/env python3
"""Report relative, scale-aware quality metrics for sequential predictions.

The primary metric compares each axis's log-space MSE with the train-median
baseline on the same masked test cells. This avoids treating a tiny IQR as a
universal unit of difficulty while retaining a separate high-content tail
diagnostic. Missing labels are never imputed or evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_mass_v4"
DEFAULT_OUTPUT_DIR = ROOT / "output/multisource_mass_v4_foodnutrigpt_v1"
PRIMARY_MIN_TEST_CELLS = 20


def metric_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "count": int(len(target)),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target, ddof=1)) if len(target) > 1 else 0.0,
        "target_q05": float(np.quantile(target, 0.05)),
        "target_q50": float(np.quantile(target, 0.50)),
        "target_q95": float(np.quantile(target, 0.95)),
    }


def safe_spearman(target: pd.Series, prediction: pd.Series) -> tuple[float, float]:
    if len(target) < 3 or target.nunique() < 2 or prediction.nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(target.to_numpy(), prediction.to_numpy())
    return float(result.statistic), float(result.pvalue)


def analyze_predictions(
    predictions: pd.DataFrame,
    axes: pd.DataFrame,
    normalization: pd.DataFrame,
    expected_kind: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scale_column = "baseline_log_rmse" if "baseline_log_rmse" in normalization.columns else "iqr_scale"
    q95_column = "train_log_q95" if "train_log_q95" in normalization.columns else None
    normalizer_columns = ["axis_id", "median", scale_column] + ([q95_column] if q95_column else [])
    frame = predictions.merge(
        axes[["axis_id", "axis_name", "target_kind", "axis_class", "mask_policy"]],
        on="axis_id",
        how="left",
        validate="many_to_one",
    ).merge(
        normalization[normalizer_columns],
        on="axis_id",
        how="left",
        validate="many_to_one",
    )
    if frame[["axis_name", "median", scale_column]].isna().any().any():
        raise ValueError("Prediction rows contain axis IDs without registry or normalization metadata.")
    kinds = set(frame["target_kind"])
    if kinds != {expected_kind}:
        raise ValueError(f"Expected only {expected_kind} rows, found target kinds: {sorted(kinds)}")

    for column in ("target_normalized", "prediction_normalized"):
        frame[f"{column}_log1p"] = frame[column] * frame[scale_column] + frame["median"]
        frame[f"{column}_g_per_100g"] = np.maximum(np.expm1(frame[f"{column}_log1p"]), 0.0)

    rows: list[dict[str, object]] = []
    for axis_id, group in frame.groupby("axis_id", sort=True):
        scaled = metric_summary(
            group["target_normalized"].to_numpy(), group["prediction_normalized"].to_numpy()
        )
        log1p = metric_summary(
            group["target_normalized_log1p"].to_numpy(), group["prediction_normalized_log1p"].to_numpy()
        )
        raw = metric_summary(
            group["target_normalized_g_per_100g"].to_numpy(),
            group["prediction_normalized_g_per_100g"].to_numpy(),
        )
        baseline_log_mse = float(np.mean((group["target_normalized_log1p"] - group["median"]) ** 2))
        relative_log_mse = log1p["mse"] / baseline_log_mse if baseline_log_mse > 1e-12 else float("nan")
        if q95_column:
            tail = group[group["target_normalized_log1p"] > group[q95_column]]
            tail_log_mse = float(np.mean((tail["prediction_normalized_log1p"] - tail["target_normalized_log1p"]) ** 2)) if len(tail) else float("nan")
        else:
            tail_log_mse = float("nan")
        rho, pvalue = safe_spearman(group["target_normalized_log1p"], group["prediction_normalized_log1p"])
        rows.append(
            {
                "axis_id": axis_id,
                "axis_name": group["axis_name"].iloc[0],
                "axis_class": group["axis_class"].iloc[0],
                "mask_policy": group["mask_policy"].iloc[0],
                "test_count": scaled["count"],
                "baseline_scaled_mse": scaled["mse"],
                "baseline_scaled_mae": scaled["mae"],
                "baseline_scaled_rmse": scaled["rmse"],
                "baseline_scaled_target_std": scaled["target_std"],
                "baseline_scaled_target_min": scaled["target_min"],
                "baseline_scaled_target_max": scaled["target_max"],
                "log1p_mse": log1p["mse"],
                "log1p_mae": log1p["mae"],
                "train_median_baseline_log_mse": baseline_log_mse,
                "relative_log_mse": relative_log_mse,
                "relative_log_skill": 1.0 - relative_log_mse if np.isfinite(relative_log_mse) else float("nan"),
                "tail_test_count_above_train_q95": int(len(tail)) if q95_column else 0,
                "tail_log1p_mse_above_train_q95": tail_log_mse,
                "log1p_target_std": log1p["target_std"],
                "log1p_target_min": log1p["target_min"],
                "log1p_target_max": log1p["target_max"],
                "raw_target_q05_g_per_100g": raw["target_q05"],
                "raw_target_q50_g_per_100g": raw["target_q50"],
                "raw_target_q95_g_per_100g": raw["target_q95"],
                "raw_target_min_g_per_100g": raw["target_min"],
                "raw_target_max_g_per_100g": raw["target_max"],
                "raw_target_std_g_per_100g": raw["target_std"],
                "raw_mae_g_per_100g": raw["mae"],
                "raw_rmse_g_per_100g": raw["rmse"],
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "negative_raw_prediction_fraction": float(
                    (group["prediction_normalized_g_per_100g"] < 0).mean()
                ),
            }
        )
    axis_metrics = pd.DataFrame(rows).sort_values("relative_log_mse", ascending=False, kind="stable")

    scaled_summary = metric_summary(
        frame["target_normalized"].to_numpy(), frame["prediction_normalized"].to_numpy()
    )
    log1p_summary = metric_summary(
        frame["target_normalized_log1p"].to_numpy(), frame["prediction_normalized_log1p"].to_numpy()
    )
    raw_summary = metric_summary(
        frame["target_normalized_g_per_100g"].to_numpy(),
        frame["prediction_normalized_g_per_100g"].to_numpy(),
    )
    overall_rho, overall_pvalue = safe_spearman(frame["target_normalized_log1p"], frame["prediction_normalized_log1p"])
    valid_axis_rho = axis_metrics.loc[axis_metrics["test_count"] >= 10, "spearman_rho"].dropna()
    weighted_axis_rho = axis_metrics.loc[axis_metrics["test_count"] >= 10, ["test_count", "spearman_rho"]].dropna()
    primary_axes = axis_metrics[axis_metrics["test_count"] >= PRIMARY_MIN_TEST_CELLS].copy()
    summary: dict[str, object] = {
        "baseline_scaled": scaled_summary,
        "log1p": log1p_summary,
        "raw_g_per_100g": raw_summary,
        "primary_min_test_cells": PRIMARY_MIN_TEST_CELLS,
        "primary_eligible_axes": int(len(primary_axes)),
        "macro_relative_log_mse": float(primary_axes["relative_log_mse"].mean()),
        "macro_relative_log_skill": float(primary_axes["relative_log_skill"].mean()),
        "median_relative_log_mse": float(primary_axes["relative_log_mse"].median()),
        "macro_train_scaled_mse": float(primary_axes["baseline_scaled_mse"].mean()),
        "median_train_scaled_mse": float(primary_axes["baseline_scaled_mse"].median()),
        "axes_better_than_train_median": int((primary_axes["relative_log_mse"] < 1.0).sum()),
        "tail_log1p_mse_above_train_q95": float(
            np.mean((frame.loc[frame["target_normalized_log1p"] > frame[q95_column], "prediction_normalized_log1p"] - frame.loc[frame["target_normalized_log1p"] > frame[q95_column], "target_normalized_log1p"]) ** 2)
        )
        if q95_column and (frame["target_normalized_log1p"] > frame[q95_column]).any()
        else float("nan"),
        "tail_count_above_train_q95": int((frame["target_normalized_log1p"] > frame[q95_column]).sum()) if q95_column else 0,
        "pooled_spearman_rho": overall_rho,
        "pooled_spearman_pvalue": overall_pvalue,
        "axis_spearman_eligible_axes": int(len(valid_axis_rho)),
        "axis_spearman_mean": float(valid_axis_rho.mean()) if len(valid_axis_rho) else float("nan"),
        "axis_spearman_median": float(valid_axis_rho.median()) if len(valid_axis_rho) else float("nan"),
        "axis_spearman_weighted_mean": float(
            np.average(weighted_axis_rho["spearman_rho"], weights=weighted_axis_rho["test_count"])
        )
        if len(weighted_axis_rho)
        else float("nan"),
        "negative_raw_prediction_fraction": float((frame["prediction_normalized_g_per_100g"] < 0).mean()),
    }
    return axis_metrics, summary


def render_summary(kind: str, summary: dict[str, object], axis_metrics: pd.DataFrame) -> str:
    scaled = summary["baseline_scaled"]
    log1p = summary["log1p"]
    raw = summary["raw_g_per_100g"]
    worst = axis_metrics.head(10)[
        ["axis_name", "test_count", "relative_log_mse", "baseline_scaled_mse", "baseline_scaled_mae", "spearman_rho"]
    ].copy()
    worst.columns = ["Axis", "n", "Relative log-MSE", "Scaled MSE", "Scaled MAE", "Spearman rho"]
    table_lines = ["| " + " | ".join(worst.columns) + " |", "|" + "|".join(["---"] * len(worst.columns)) + "|"]
    for row in worst.itertuples(index=False, name=None):
        cells = [str(row[0]), str(int(row[1]))]
        cells.extend("NA" if pd.isna(value) else f"{float(value):.4f}" for value in row[2:])
        table_lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(table_lines)
    return f"""## {kind.title()} Sequential Test

Primary metric: median per-axis relative log-MSE versus the train-median baseline, calculated only for axes with at least {summary['primary_min_test_cells']} masked test cells. A value below 1 is better than predicting the training median for every food. The macro mean is retained as a sensitivity diagnostic because an axis with unusually low test variance can make its relative score unstable.

| Quantity | Value |
|---|---:|
| Masked cells | {scaled['count']:,} |
| Median per-axis relative log-MSE (primary) | {summary['median_relative_log_mse']:.4f} |
| Macro mean relative log-MSE (sensitivity) | {summary['macro_relative_log_mse']:.4f} |
| Macro mean relative log skill (sensitivity) | {summary['macro_relative_log_skill']:.4f} |
| Primary eligible axes | {summary['primary_eligible_axes']:,} |
| Eligible axes better than train-median baseline | {summary['axes_better_than_train_median']:,} |
| Macro train-scaled MSE | {summary['macro_train_scaled_mse']:.4f} |
| Median train-scaled MSE | {summary['median_train_scaled_mse']:.4f} |
| Baseline-scaled micro MSE | {scaled['mse']:.4f} |
| Baseline-scaled micro MAE | {scaled['mae']:.4f} |
| Tail log-MSE (test values above train P95) | {summary['tail_log1p_mse_above_train_q95']:.4f} ({summary['tail_count_above_train_q95']:,} cells) |
| `log1p(g/100g)` target SD | {log1p['target_std']:.4f} |
| `log1p(g/100g)` target range | [{log1p['target_min']:.4f}, {log1p['target_max']:.4f}] |
| Raw target range, g/100g | [{raw['target_min']:.6g}, {raw['target_max']:.6g}] |
| Raw target 5th/50th/95th percentile, g/100g | {raw['target_q05']:.6g} / {raw['target_q50']:.6g} / {raw['target_q95']:.6g} |
| Pooled Spearman rho | {summary['pooled_spearman_rho']:.4f} |
| Per-axis Spearman median (axes with >=10 test cells) | {summary['axis_spearman_median']:.4f} |
| Per-axis Spearman mean (unweighted) | {summary['axis_spearman_mean']:.4f} |
| Per-axis Spearman mean (test-count weighted) | {summary['axis_spearman_weighted_mean']:.4f} |
| Axes eligible for per-axis Spearman | {summary['axis_spearman_eligible_axes']:,} |
| Negative raw predictions | {summary['negative_raw_prediction_fraction']:.2%} |

The relative score uses the train-median baseline on the same test cells, so it is comparable across axes without using test information in preprocessing. The pooled Spearman is secondary; use the per-axis Spearman values to assess within-axis ranking. Raw-scale pooled errors are not a primary metric because `g/100g` values of different analytes have different scales.

### Axes with the largest relative log-MSE

{table}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    axes = pd.read_csv(args.data_dir / "axis_registry.csv")
    normalization = pd.read_csv(args.data_dir / "train_only_axis_normalization.csv")
    analyses = []
    for stage, kind in (("stage1", "nutrient"), ("stage2", "compound")):
        predictions = pd.read_csv(args.output_dir / f"{stage}_sequential_test_predictions.csv")
        axis_metrics, summary = analyze_predictions(predictions, axes, normalization, kind)
        axis_metrics.to_csv(args.output_dir / f"{stage}_sequential_axis_metrics.csv", index=False)
        analyses.append(render_summary(kind, summary, axis_metrics))

    report = "# FoodNutriGPT v4 Prediction Quality\n\n" + "\n\n".join(analyses)
    report += "\n\n## Interpretation\n\n"
    report += (
        "MSE is sensitive to rare target values far from an axis's training median. "
        "The robust IQR normalization protects typical values but can make a legitimate high-value food a large normalized outlier when an axis has a small IQR. "
        "Therefore report normalized MSE/MAE with target SD and the per-axis table, rather than interpreting a single pooled raw mass error across chemicals as a universal accuracy measure.\n"
    )
    (args.output_dir / "prediction_quality_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {args.output_dir / 'prediction_quality_report.md'}")


if __name__ == "__main__":
    main()

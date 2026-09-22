#!/usr/bin/env python3
"""Report relative scale-aware metrics for core nutrition and chemical forms."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evaluate_multisource_prediction_quality as quality


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
DEFAULT_OUTPUT_DIR = ROOT / "output/multisource_nutrition_v5_foodnutrigpt_v1_finalrun"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    axes = pd.read_csv(args.data_dir / "axis_registry.csv")
    normalization = pd.read_csv(args.data_dir / "train_only_axis_normalization.csv")
    sections = []
    for stage, kind, label in (
        ("stage1", "core_nutrition", "Core Nutrition"),
        ("stage2", "nutrient_chemical_form", "Nutrient Chemical Forms"),
    ):
        predictions = pd.read_csv(args.output_dir / f"{stage}_sequential_test_predictions.csv")
        predictions = predictions[predictions["target_kind"].eq(kind)].copy()
        predictions = predictions.drop(columns="target_kind")
        axis_metrics, summary = quality.analyze_predictions(predictions, axes, normalization, kind)
        axis_metrics.to_csv(args.output_dir / f"{stage}_sequential_axis_metrics.csv", index=False)
        sections.append(quality.render_summary(label, summary, axis_metrics))
    report = "# FoodNutriGPT Prediction Quality\n\n" + "\n\n".join(sections)
    report += "\n\n## Interpretation\n\n"
    report += (
        "The primary score is median per-axis relative log-MSE against a train-median baseline. Macro mean relative scores, train-scaled MSE, and high-content tail MSE remain diagnostics, rather than being treated as a universal accuracy measure. "
        "Raw mass values are clamped to zero only when reporting physically meaningful raw-unit errors; the standardized regression score itself is not clipped.\n"
    )
    target = args.output_dir / "prediction_quality_report.md"
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()

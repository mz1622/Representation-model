#!/usr/bin/env python3
"""Build the non-VMH-limited scientific food-composition target panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.final_prediction_axes import build_final_prediction_axis_panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-registry", type=Path,
        default=ROOT / "data/processed/vmh_aligned_prediction_axis_design_v1/prediction_axis_registry.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/final_scientific_prediction_axis_panel_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/final_scientific_prediction_axis_panel_v1",
    )
    args = parser.parse_args()
    manifest = build_final_prediction_axis_panel(
        draft_registry_path=args.draft_registry,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print("Scientific prediction-axis panel built:")
    for key, value in manifest.items():
        if key != "source_authorities":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

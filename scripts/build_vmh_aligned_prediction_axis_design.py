#!/usr/bin/env python3
"""Build a VMH-aligned 200+ food-composition target panel.

The command downloads the public VMH nutrient catalogue only when no cached
snapshot is present, unless --refresh-vmh is supplied. It does not make a
training matrix or merge numerical values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.prediction_axis_design import build_prediction_axis_design  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-dir", type=Path,
        default=ROOT / "data/processed/global_food_metabolome_axis_atlas_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/vmh_aligned_prediction_axis_design_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/vmh_aligned_prediction_axis_design_v1",
    )
    parser.add_argument(
        "--vmh-catalog", type=Path,
        default=ROOT / "data/raw/reference_catalogues/vmh_nutrients_2026_09_11.json",
    )
    parser.add_argument(
        "--refresh-vmh", action="store_true",
        help="Replace the cached public VMH nutrient catalogue snapshot.",
    )
    args = parser.parse_args()
    result = build_prediction_axis_design(
        atlas_dir=args.atlas_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        vmh_catalog_path=args.vmh_catalog,
        refresh_vmh=args.refresh_vmh,
    )
    print("VMH-aligned prediction-axis design built:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

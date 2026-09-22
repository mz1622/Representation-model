#!/usr/bin/env python3
"""Compare every global-atlas source axis with the final 369-target panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.semantic_axis_coverage import audit_semantic_axis_coverage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-registry", type=Path,
        default=ROOT / "data/processed/final_scientific_prediction_axis_panel_v1/final_prediction_axis_registry.csv",
    )
    parser.add_argument(
        "--atlas-dir", type=Path,
        default=ROOT / "data/processed/global_food_metabolome_axis_atlas_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/final_prediction_axis_semantic_coverage_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/final_prediction_axis_semantic_coverage_v1",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data/cache/semantic_axis_coverage_v1",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    kwargs = {
        "target_registry_path": args.target_registry,
        "atlas_dir": args.atlas_dir,
        "output_dir": args.output_dir,
        "report_dir": args.report_dir,
        "cache_dir": args.cache_dir,
    }
    if args.model:
        kwargs["model_name"] = args.model
    manifest = audit_semantic_axis_coverage(**kwargs)
    print("Semantic axis coverage audit built:")
    for key, value in manifest.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the current target panel and audit source foods without value pooling.

Colab:
    python scripts/freeze_prediction_axes_and_audit_foods.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.frozen_food_audit import audit_frozen_prediction_panel_foods  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-panel",
        type=Path,
        default=ROOT / "data/processed/final_scientific_prediction_axis_panel_v1/final_prediction_axis_registry.csv",
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "data/processed/frozen_prediction_axis_panel_v1",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "data/processed/scientific_food_composition_v3/staging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/frozen_prediction_panel_food_audit_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports/frozen_prediction_panel_food_audit_v1",
    )
    args = parser.parse_args()
    manifest = audit_frozen_prediction_panel_foods(
        source_panel_path=args.source_panel,
        freeze_dir=args.freeze_dir,
        staging_dir=args.staging_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )
    print("Frozen-panel food audit completed.")
    for key in (
        "food_observations",
        "exact_name_food_groups",
        "preserved_target_measurement_records",
        "observed_frozen_target_axes",
        "cross_source_axis_conflicts_preserved",
    ):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()

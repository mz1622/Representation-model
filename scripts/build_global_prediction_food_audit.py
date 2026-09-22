#!/usr/bin/env python3
"""Integrate global source axes before exact food-name audit.

Colab:
    python scripts/build_global_prediction_food_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.global_prediction_food_audit import run_global_prediction_food_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/global_frozen_prediction_panel_food_audit_v8",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/global_frozen_prediction_panel_food_audit_v8",
    )
    args = parser.parse_args()
    manifest = run_global_prediction_food_audit(root=ROOT, output_dir=args.output_dir, report_dir=args.report_dir)
    print("Global prediction-axis food audit completed.")
    for key in (
        "registered_sources", "numerically_integrated_sources", "food_observations",
        "exact_name_food_groups", "mapped_numeric_measurements", "direct_mass_label_candidates",
        "observed_frozen_targets", "cross_source_axis_conflicts_preserved",
    ):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()

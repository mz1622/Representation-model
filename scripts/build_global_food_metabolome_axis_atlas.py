#!/usr/bin/env python3
"""Build the global source-native food-metabolome axis atlas.

This script does not create a numerical training matrix.  It is safe to run
before deciding any model targets because it only inventories axes and their
scientific meanings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.axis_atlas import build_axis_atlas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/global_food_metabolome_axis_atlas_v1")
    parser.add_argument("--report-dir", type=Path,
                        default=ROOT / "reports/global_food_metabolome_axis_atlas_v1")
    args = parser.parse_args()
    result = build_axis_atlas(ROOT, args.output_dir, args.report_dir)
    print("Global food-metabolome axis atlas built:")
    for key, value in result.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()

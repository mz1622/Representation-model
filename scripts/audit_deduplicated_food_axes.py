#!/usr/bin/env python3
"""Audit deduplicated measurement-expression axes from the global atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.axis_audit import build_axis_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-dir", type=Path,
        default=ROOT / "data/processed/global_food_metabolome_axis_atlas_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/global_food_metabolome_axis_audit_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports/global_food_metabolome_axis_audit_v1",
    )
    args = parser.parse_args()
    result = build_axis_audit(args.atlas_dir, args.output_dir, args.report_dir)
    print("Global food-metabolome axis audit built:")
    for key, value in result.items():
        print(f"  {key}: {value:,}")


if __name__ == "__main__":
    main()

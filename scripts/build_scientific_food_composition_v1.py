#!/usr/bin/env python3
"""Build source staging tables and the harmonized scientific candidate release."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.harmonize import run_harmonization
from foodcomp.sources import write_source_staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sources", nargs="*", help="Optional source keys to restage.")
    parser.add_argument("--skip-staging", action="store_true")
    parser.add_argument("--skip-harmonization", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    staging = root / "data/processed/scientific_food_composition_v1/staging"
    if not args.skip_staging:
        write_source_staging(root, staging, args.sources)
    if not args.skip_harmonization:
        run_harmonization(
            root,
            staging,
            root / "data/processed/scientific_food_composition_v1/release",
            root / "data/audits/scientific_food_composition_v1/dataset",
        )


if __name__ == "__main__":
    main()

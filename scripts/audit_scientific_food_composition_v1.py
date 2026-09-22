#!/usr/bin/env python3
"""Run all automated release-gate checks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.audit import main


if __name__ == "__main__":
    main()

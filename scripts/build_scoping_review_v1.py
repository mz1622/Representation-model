#!/usr/bin/env python3
"""Build the reproducible scoping-review corpus and source registry."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.review import main


if __name__ == "__main__":
    main()

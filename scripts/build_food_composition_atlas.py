#!/usr/bin/env python3
"""Build the read-only Food Composition Atlas and its review queue CSV files."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.atlas import main  # noqa: E402


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate bilingual reports, data cards and the candidate release manifest."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.report import run


if __name__ == "__main__":
    run(ROOT)

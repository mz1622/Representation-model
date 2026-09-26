"""Build a read-only-derived R0 audit and common benchmark; no frozen test evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from foodcomp.research_r0 import build_view, VERSION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"data/processed"/VERSION)
    args = parser.parse_args()
    print(json.dumps(build_view(ROOT, args.output_dir.resolve()), ensure_ascii=False, indent=2))

"""Audit and document the priority-selected, single-record v6.1 release."""

import argparse

from finalize_scientific_food_composition_v6 import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    main(version="scientific_food_composition_v6_1", report_only=parser.parse_args().report_only)

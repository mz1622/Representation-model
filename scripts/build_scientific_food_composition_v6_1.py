"""Rebuild v6.1 with one priority-selected source record per eligible cell."""

from build_scientific_food_composition_v5 import main


if __name__ == "__main__":
    main(version="scientific_food_composition_v6_1", revision=True, single_record=True)

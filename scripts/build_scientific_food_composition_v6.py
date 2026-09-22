"""Rebuild v6 from the same immutable raw-staged inputs; preserve v5.1.

Colab: pip install -r requirements-colab.txt
       python scripts/build_scientific_food_composition_v6.py
"""

from build_scientific_food_composition_v5 import main


if __name__ == "__main__":
    main(version="scientific_food_composition_v6", revision=True)

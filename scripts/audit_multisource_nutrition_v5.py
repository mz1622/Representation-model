#!/usr/bin/env python3
"""Independently validate v5 data integrity and masking semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
AUDIT_DIR = ROOT / "data/audits/multisource_nutrition_v5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    foods = pd.read_csv(DATA_DIR / "food_entities.csv")
    values = pd.read_csv(DATA_DIR / "observed_axis_values.csv")
    axes = pd.read_csv(DATA_DIR / "axis_registry.csv")
    normalizers = pd.read_csv(DATA_DIR / "train_only_axis_normalization.csv")
    members = pd.read_csv(DATA_DIR / "duplicate_group_members.csv")
    masks = pd.read_csv(SPLIT_DIR / "sequential_masks.csv")
    with (SPLIT_DIR / "splits.json").open() as handle:
        split_payload = json.load(handle)
    splits = {name: split_payload[name] for name in ("train", "validation", "test")}

    require(values["value_g_per_100g"].notna().all(), "Missing values must be absent rows, never NaN labels.")
    require(np.isfinite(values["value_g_per_100g"]).all(), "All retained values must be finite.")
    require(values["value_g_per_100g"].between(0, 100).all(), "Mass-scale values must be within [0, 100] g/100g.")
    require(values["axis_id"].isin(axes["axis_id"]).all(), "Observed values contain an axis absent from the registry.")
    require(values["canonical_food_id"].isin(foods["canonical_food_id"]).all(), "Observed values contain an unknown food.")
    require(axes["axis_id"].is_unique and axes["axis_index"].is_unique, "Axis registry IDs and indices must be unique.")
    require(set(axes["target_kind"]) == {"core_nutrition", "nutrient_chemical_form"}, "v5 training matrix contains a non-nutrition target kind.")
    require(not axes["axis_name"].str.fullmatch("Ash|Moisture|Alcohol", case=False, na=False).any(), "Structural axes must not enter v5.")
    require(normalizers["axis_id"].is_unique and set(normalizers["axis_id"]) == set(axes["axis_id"]), "Each retained axis requires exactly one train-only normalizer.")
    require((normalizers["iqr_scale"] > 0).all(), "Normalizer scales must be positive.")

    all_split_ids = [food for ids in splits.values() for food in ids]
    require(len(all_split_ids) == len(set(all_split_ids)), "Food split overlap detected.")
    require(set(all_split_ids) == set(foods["canonical_food_id"]), "Split does not cover exactly the retained foods.")
    duplicate_split_counts = members.merge(foods[["canonical_food_id", "split"]], on="canonical_food_id", how="inner").groupby("duplicate_component_root")["split"].nunique()
    require((duplicate_split_counts <= 1).all(), "Duplicate-connected foods cross a split boundary.")

    family_counts = masks.groupby(["split", "canonical_food_id", "target_kind", "mask_family"]).size()
    require((family_counts <= 1).all(), "A mask contains multiple targets from the same aggregate/component family.")
    masked_lookup = masks.merge(axes[["axis_id", "target_kind", "mask_policy"]], on=["axis_id", "target_kind"], how="left", validate="many_to_one")
    require(masked_lookup["mask_policy"].eq("maskable_target").all(), "Mask includes a context-only axis.")

    result = {
        "foods": int(len(foods)), "observed_values": int(len(values)), "retained_axes": int(len(axes)),
        "core_axes": int(axes["target_kind"].eq("core_nutrition").sum()),
        "chemical_form_axes": int(axes["target_kind"].eq("nutrient_chemical_form").sum()),
        "explicit_zero_values": int((values["value_g_per_100g"] == 0).sum()),
        "duplicate_components": int(members["duplicate_component_root"].nunique()),
        "test_masks": int(masks["split"].eq("test").sum()),
        "validation_masks": int(masks["split"].eq("validation").sum()),
        "status": "passed",
    }
    (AUDIT_DIR / "independent_v5_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

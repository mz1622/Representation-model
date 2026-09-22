#!/usr/bin/env python3
"""Independently validate the final multisource_mass_v4 pretraining corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/multisource_mass_v4"
SPLIT_DIR = ROOT / "data/splits/multisource_mass_v4"
AUDIT_DIR = ROOT / "data/audits/multisource_mass_v4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    foods = pd.read_csv(DATA_DIR / "food_entities.csv")
    values = pd.read_csv(DATA_DIR / "observed_axis_values.csv")
    axes = pd.read_csv(DATA_DIR / "axis_registry.csv")
    normalization = pd.read_csv(DATA_DIR / "train_only_axis_normalization.csv")
    masks = pd.read_csv(SPLIT_DIR / "sequential_masks.csv")
    members = pd.read_csv(DATA_DIR / "duplicate_group_members.csv")
    with (SPLIT_DIR / "splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)

    require(set(foods["canonical_food_id"]) == set(values["canonical_food_id"]), "Food entities and observed values do not reference the same food set.")
    require(not values.duplicated(["canonical_food_id", "axis_id"]).any(), "Duplicate final food-axis cells found.")
    require(np.isfinite(values["value_g_per_100g"]).all(), "Non-finite final values found.")
    require(values["value_g_per_100g"].between(0, 100).all(), "Final values are outside [0, 100] g/100g.")
    require(axes["model_unit"].eq("g/100g").all(), "A non-mass model unit remains.")
    require(set(values["axis_id"]) == set(axes["axis_id"]), "Axis registry and values do not agree.")

    support = values.groupby("axis_id").agg(
        observed_foods=("canonical_food_id", "nunique"),
        positive_foods=("value_g_per_100g", lambda series: int((series > 0).sum())),
        explicit_zero_foods=("value_g_per_100g", lambda series: int((series == 0).sum())),
    )
    require((support["observed_foods"] >= 20).all(), "An axis with fewer than 20 observed foods remains.")
    merged_support = axes.set_index("axis_id")[["observed_foods", "positive_foods", "explicit_zero_foods", "mask_policy"]].join(support, rsuffix="_recomputed")
    require((merged_support["observed_foods"] == merged_support["observed_foods_recomputed"]).all(), "Stored axis support differs from recomputed support.")
    require((merged_support["positive_foods"] == merged_support["positive_foods_recomputed"]).all(), "Stored positive support differs from recomputed support.")
    expected_maskable = (support["observed_foods"] >= 50) & (support["positive_foods"] >= 20)
    actual_maskable = axes.set_index("axis_id")["mask_policy"].eq("maskable_target")
    require(expected_maskable.equals(actual_maskable.reindex(expected_maskable.index)), "Maskable-axis policy is inconsistent with support rules.")

    total_per_food = values.groupby("canonical_food_id")["axis_id"].nunique()
    nutrient_axes = set(axes.loc[axes["target_kind"].eq("nutrient"), "axis_id"])
    nutrient_per_food = values[values["axis_id"].isin(nutrient_axes)].groupby("canonical_food_id")["axis_id"].nunique()
    require((total_per_food >= 3).all(), "A final food has fewer than three observed axes.")
    require((nutrient_per_food.reindex(foods["canonical_food_id"], fill_value=0) >= 3).all(), "A final food has fewer than three macro/micro nutrients.")

    split_sets = {name: set(ids) for name, ids in (("train", splits["train"]), ("validation", splits["validation"]), ("test", splits["test"]))}
    require(set.union(*split_sets.values()) == set(foods["canonical_food_id"]), "Splits do not cover exactly the final food IDs.")
    require(not (split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]), "Food IDs leak across splits.")
    require(members.groupby("canonical_food_id").size().index.isin(foods["canonical_food_id"]).all(), "Duplicate member references a non-final food.")
    require(set(normalization["axis_id"]) == set(axes["axis_id"]), "Normalization is missing an axis or contains an extra axis.")
    require((normalization["iqr_scale"] > 0).all(), "Non-positive normalization scale found.")

    mask_targets = masks.merge(axes[["axis_id", "target_kind", "mask_policy"]], on=["axis_id", "target_kind"], how="left", validate="many_to_one")
    require(mask_targets["mask_policy"].eq("maskable_target").all(), "A context-only axis was placed in the mask set.")
    mask_values = masks.merge(values[["canonical_food_id", "axis_id", "value_g_per_100g"]], on=["canonical_food_id", "axis_id"], how="left", validate="one_to_one")
    require(mask_values["value_g_per_100g"].gt(0).all(), "A mask target is not an observed positive value.")
    require(not masks.duplicated(["split", "canonical_food_id", "axis_id"]).any(), "Duplicate mask target found.")
    for (_, food_id, target_kind), group in masks.groupby(["split", "canonical_food_id", "target_kind"]):
        expected = list(range(1, len(group) + 1))
        require(sorted(group["prediction_order"].tolist()) == expected, f"Invalid sequential prediction order for {food_id} {target_kind}.")

    report = {
        "foods": int(len(foods)),
        "axes": int(len(axes)),
        "observed_values": int(len(values)),
        "duplicate_final_food_axis_cells": int(values.duplicated(["canonical_food_id", "axis_id"]).sum()),
        "axis_support_minimum": int(support["observed_foods"].min()),
        "food_observed_axis_minimum": int(total_per_food.min()),
        "food_macro_micro_axis_minimum": int(nutrient_per_food.min()),
        "maskable_axes": int(actual_maskable.sum()),
        "context_only_axes": int((~actual_maskable).sum()),
        "validation_mask_cells": int((masks["split"] == "validation").sum()),
        "test_mask_cells": int((masks["split"] == "test").sum()),
        "max_sequential_prediction_order": int(masks["prediction_order"].max()),
        "sources": foods["primary_source"].value_counts().to_dict(),
        "axis_classes": axes["axis_class"].value_counts().to_dict(),
        "split_counts": foods["split"].value_counts().to_dict(),
    }
    with (AUDIT_DIR / "final_protocol_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the v6 relative-scale protocol from the audited v5 matrix.

The food rows, observed values, axis registry, splits, and fixed masks are
copied unchanged. Only the train-only normalization changes: every positive
axis value is log-transformed, centered at its training median, and scaled by
the training RMSE of the median predictor. This makes one standardized unit
equal to the train-only median-baseline RMSE for that axis.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
SOURCE_SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_nutrition_v6_relative"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v6_relative"
DEFAULT_AUDIT_DIR = ROOT / "data/audits/multisource_nutrition_v6_relative"


def fit_relative_normalization(values: pd.DataFrame, train_food_ids: set[str], axis_ids: set[str]) -> pd.DataFrame:
    train_values = values[
        values["canonical_food_id"].isin(train_food_ids)
        & values["axis_id"].isin(axis_ids)
        & values["value_g_per_100g"].gt(0)
    ].copy()
    train_values["log_value"] = np.log1p(train_values["value_g_per_100g"].to_numpy(dtype=float))
    rows: list[dict[str, object]] = []
    positive_axis_ids: set[str] = set()
    for axis_id, group in train_values.groupby("axis_id", sort=True):
        positive_axis_ids.add(axis_id)
        log_values = group["log_value"].to_numpy(dtype=float)
        median = float(np.median(log_values))
        baseline_rmse = float(np.sqrt(np.mean((log_values - median) ** 2)))
        degenerate = baseline_rmse < 1e-12
        rows.append(
            {
                "axis_id": axis_id,
                "transform": "log1p",
                "median": median,
                "baseline_log_rmse": 1.0 if degenerate else baseline_rmse,
                "train_log_q05": float(np.quantile(log_values, 0.05)),
                "train_log_q25": float(np.quantile(log_values, 0.25)),
                "train_log_q75": float(np.quantile(log_values, 0.75)),
                "train_log_q95": float(np.quantile(log_values, 0.95)),
                "positive_train_foods": int(group["canonical_food_id"].nunique()),
                "positive_train_records": int(len(group)),
                "degenerate_train_axis": bool(degenerate),
                "zero_only_train_axis": False,
                "scale_definition": "train_log_rmse_of_median_baseline",
            }
        )
    for axis_id in sorted(axis_ids - positive_axis_ids):
        rows.append(
            {
                "axis_id": axis_id,
                "transform": "log1p",
                "median": 0.0,
                "baseline_log_rmse": 1.0,
                "train_log_q05": 0.0,
                "train_log_q25": 0.0,
                "train_log_q75": 0.0,
                "train_log_q95": 0.0,
                "positive_train_foods": 0,
                "positive_train_records": 0,
                "degenerate_train_axis": True,
                "zero_only_train_axis": True,
                "scale_definition": "zero_only_context_axis",
            }
        )
    result = pd.DataFrame(rows)
    missing = axis_ids - set(result["axis_id"])
    if missing:
        raise AssertionError(f"Normalization rows are missing retained axes: {sorted(missing)[:10]}")
    return result.sort_values("axis_id", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    data_dir, split_dir, audit_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.audit_dir.resolve()
    for path in (data_dir, split_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    for name in ("food_entities.csv", "observed_axis_values.csv", "axis_registry.csv", "duplicate_group_members.csv"):
        shutil.copy2(SOURCE_DATA_DIR / name, data_dir / name)
    for name in ("loss_axis_ids.csv", "sequential_masks.csv"):
        shutil.copy2(SOURCE_SPLIT_DIR / name, split_dir / name)

    foods = pd.read_csv(data_dir / "food_entities.csv", dtype={"canonical_food_id": str})
    values = pd.read_csv(data_dir / "observed_axis_values.csv", dtype={"canonical_food_id": str, "axis_id": str})
    axes = pd.read_csv(data_dir / "axis_registry.csv", dtype={"axis_id": str})
    with (SOURCE_SPLIT_DIR / "splits.json").open(encoding="utf-8") as handle:
        source_split_payload = json.load(handle)
    splits = {name: source_split_payload[name] for name in ("train", "validation", "test")}
    normalization = fit_relative_normalization(values, set(splits["train"]), set(axes["axis_id"]))
    normalization.to_csv(data_dir / "train_only_axis_normalization.csv", index=False)

    split_payload = {
        **source_split_payload,
        "protocol_version": "multisource_nutrition_v6_relative",
        "parent_protocol": "multisource_nutrition_v5",
        "normalization": "z=(log1p(g_per_100g)-train_median)/train_median_baseline_log_rmse",
    }
    (split_dir / "splits.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    audit = {
        "protocol_version": "multisource_nutrition_v6_relative",
        "parent_protocol": "multisource_nutrition_v5",
        "foods": int(len(foods)),
        "observed_values": int(len(values)),
        "axes": int(len(axes)),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "test_masks": int((pd.read_csv(split_dir / "sequential_masks.csv")["split"] == "test").sum()),
        "normalization": split_payload["normalization"],
        "degenerate_axes": int(normalization["degenerate_train_axis"].sum()),
    }
    (audit_dir / "protocol_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

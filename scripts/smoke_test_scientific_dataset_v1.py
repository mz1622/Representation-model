#!/usr/bin/env python3
"""Exercise the model data contract without tuning or opening validation."""

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dataset import FoodCompositionDataset, masked_mse


def main() -> None:
    release = ROOT / "data/processed/scientific_food_composition_v1/release"
    dataset = FoodCompositionDataset(release, partition="train")
    checked = 0
    target_cells = 0
    for index in range(min(len(dataset), 256)):
        item = dataset[index]
        assert item["context_values"].shape == (len(dataset.component_ids),)
        assert not np.any((item["context_mask"] > 0) & (item["target_mask"] > 0))
        assert np.all(np.isfinite(item["context_values"]))
        target_cells += int(item["target_mask"].sum())
        checked += 1
    zero_prediction = np.zeros(len(dataset.component_ids), dtype=np.float32)
    example = next((dataset[index] for index in range(len(dataset)) if dataset[index]["target_mask"].sum() > 0), None)
    if example is None:
        raise RuntimeError("No train example has an eligible masked target family.")
    loss = masked_mse(zero_prediction, example["target_values"], example["target_mask"])
    summary = {
        "partition": "train", "foods": len(dataset), "components": len(dataset.component_ids),
        "maskable_components": len(dataset.target_columns), "examples_checked": checked,
        "target_cells_checked": target_cells, "finite_example_loss": bool(np.isfinite(loss)),
        "validation_opened": False,
    }
    output = ROOT / "output/scientific_food_composition_v1_interface_smoke"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

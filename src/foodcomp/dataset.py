"""Minimal Colab-friendly data interface for future model experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import RANDOM_SEED
from .util import read_component_csv


class FoodCompositionDataset:
    """Read the fixed release matrix without changing its partition semantics."""

    stage = None

    def __init__(self, release_dir: str | Path, partition: str = "train", seed: int = RANDOM_SEED,
                 *, stage: int | None = None):
        release_dir = Path(release_dir)
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be 'train' or 'validation'")
        if stage not in {None, 1, 2}:
            raise ValueError("stage must be None, 1 or 2")
        archive = np.load(release_dir / "canonical_profile_matrix.npz", allow_pickle=False)
        all_values = archive["values"].astype(np.float32)
        all_food_ids = archive["food_ids"].astype(str)
        self.component_ids = archive["component_ids"].astype(str)
        foods = pd.read_csv(release_dir / "ml_partition.csv").set_index("food_concept_id").loc[all_food_ids]
        components = read_component_csv(release_dir / "component_concept.csv.gz").set_index("component_concept_id").loc[self.component_ids]
        keep = foods["partition"].eq(partition).to_numpy()
        if partition == "validation" and "benchmark_eligible" in foods:
            eligible = foods["benchmark_eligible"]
            if eligible.dtype != bool:
                eligible = eligible.astype(str).str.casefold().eq("true")
            keep &= eligible.to_numpy()
        self.values = all_values[keep]
        self.foods = foods.iloc[np.flatnonzero(keep)].reset_index()
        self.components = components.reset_index()
        self.target_columns = np.flatnonzero(self.components["training_role"].eq("maskable_target").to_numpy())
        self.context_columns = np.flatnonzero(self.components["training_role"].ne("excluded").to_numpy())
        self.family_by_column = self.components["component_family"].fillna(self.components["component_concept_id"]).to_numpy(str)
        self.seed = seed
        self.stage = stage
        if stage is not None and "prediction_stage" not in self.components:
            raise ValueError("Staged loading requires the annotated v6 release; run its finalizer.")

    def __len__(self) -> int:
        return len(self.foods)

    def __getitem__(self, index: int) -> dict[str, Any]:
        values = self.values[index].copy()
        observed = np.isfinite(values)
        eligible_targets = [column for column in self.target_columns if observed[column]]
        families = sorted({self.family_by_column[column] for column in eligible_targets})
        if not families:
            target_mask = np.zeros_like(observed)
            hidden_family = np.zeros_like(observed)
        else:
            key = f"{self.seed}:{self.foods.iloc[index].food_concept_id}"
            if self.stage is None:
                hidden = [min(families, key=lambda item: hashlib.sha256(f"{key}:{item}".encode()).hexdigest())]
            else:
                # Both stages hide the same families, including future Stage 2
                # labels, so stage-specific loss cannot expose the other task.
                hidden = []
                for training_stage in (1, 2):
                    candidates = sorted({self.family_by_column[c] for c in eligible_targets
                                         if self.components.iloc[c].prediction_stage == training_stage})
                    if candidates:
                        hidden.append(min(candidates, key=lambda item: hashlib.sha256(f"{key}:{item}".encode()).hexdigest()))
            target_mask = observed & np.isin(self.family_by_column, hidden) & np.isin(np.arange(len(values)), self.target_columns)
            hidden_family = observed & np.isin(self.family_by_column, hidden)
        jointly_hidden_targets = target_mask.copy()
        if self.stage is not None:
            target_mask &= self.components.prediction_stage.eq(self.stage).to_numpy()
        context_mask = observed & ~hidden_family
        context_values = np.nan_to_num(values, nan=0.0)
        context_values[~context_mask] = 0.0
        return {
            "food_concept_id": self.foods.iloc[index].food_concept_id,
            "text": self.food_text(index),
            "context_values": context_values,
            "context_mask": context_mask.astype(np.float32),
            "target_values": np.nan_to_num(values, nan=0.0),
            "target_mask": target_mask.astype(np.float32),
            "jointly_hidden_target_mask": jointly_hidden_targets.astype(np.float32),
        }

    def food_text(self, index: int) -> str:
        row = self.foods.iloc[index]
        pieces = [f"name: {row.canonical_name}"]
        for label in ("scientific_name", "food_group", "food_subgroup", "food_type", "part", "processing"):
            value = row.get(label)
            if pd.notna(value) and str(value).strip():
                pieces.append(f"{label.replace('_', ' ')}: {value}")
        return ". ".join(pieces)


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"Shape mismatch: prediction={prediction.shape}, target={target.shape}, mask={mask.shape}")
    return float((((prediction - target) ** 2) * mask).sum() / (mask.sum() + eps))

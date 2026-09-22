#!/usr/bin/env python3
"""Train FoodNutriGPT with a single joint masked-reconstruction objective.

Values are represented as train-median-centred ``log1p(g/100g)`` scores.
Their scale is each axis's train-only median-baseline RMSE, which makes a
zero prediction a meaningful, axis-specific initial estimate. Aggregate and
component families are co-masked to prevent direct target leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import train_multisource_foodnutrigpt_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
DEFAULT_OUTPUT_ROOT = ROOT / "output/multisource_nutrition_v5_foodnutrigpt_joint"


@dataclass
class Config:
    seed: int = 20260812
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 48
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    feedforward_dim: int = 1024
    dropout: float = 0.15
    axis_residual_rank: int = 16
    mask_ratio: float = 0.30
    epochs: int = 120
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    lambda_nutrition: float = 1.0
    lambda_metabolome: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bundle(config: Config, output_dir: Path, data_dir: Path = DATA_DIR, split_dir: Path = SPLIT_DIR) -> dict[str, object]:
    foods = pd.read_csv(data_dir / "food_entities.csv").sort_values("canonical_food_id", kind="stable").reset_index(drop=True)
    axes = pd.read_csv(data_dir / "axis_registry.csv").sort_values("axis_index", kind="stable").reset_index(drop=True)
    values = pd.read_csv(data_dir / "observed_axis_values.csv")
    normalization = pd.read_csv(data_dir / "train_only_axis_normalization.csv").set_index("axis_id")
    with (split_dir / "splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    embeddings = v4.load_text_embeddings(foods, config, output_dir / "text_cache")
    food_index = {food_id: index for index, food_id in enumerate(foods["canonical_food_id"])}
    axis_index = dict(zip(axes["axis_id"], axes["axis_index"]))
    values = values[values["canonical_food_id"].isin(food_index) & values["axis_id"].isin(axis_index)].copy()
    values["axis_index"] = values["axis_id"].map(axis_index).astype(int)
    values["log_value"] = np.log1p(values["value_g_per_100g"].to_numpy(dtype=np.float32))
    values["median"] = values["axis_id"].map(normalization["median"]).to_numpy(dtype=np.float32)
    values["scale"] = values["axis_id"].map(normalization["baseline_log_rmse"]).to_numpy(dtype=np.float32)
    values["normalized_value"] = (values["log_value"] - values["median"]) / values["scale"]
    values = values.merge(axes[["axis_index", "target_kind", "mask_policy", "mask_family"]], on="axis_index", how="left", validate="many_to_one")
    examples: dict[str, dict[str, np.ndarray]] = {}
    for food_id, group in values.groupby("canonical_food_id", sort=False):
        ordered = group.sort_values("axis_index", kind="stable")
        examples[food_id] = {
            "axes": ordered["axis_index"].to_numpy(dtype=np.int64),
            "values": ordered["normalized_value"].to_numpy(dtype=np.float32),
            "positive": ordered["value_g_per_100g"].gt(0).to_numpy(dtype=bool),
            "is_core": ordered["target_kind"].eq("core_nutrition").to_numpy(dtype=bool),
            "is_form": ordered["target_kind"].eq("nutrient_chemical_form").to_numpy(dtype=bool),
            "is_maskable": ordered["mask_policy"].eq("maskable_target").to_numpy(dtype=bool),
            "families": ordered["mask_family"].astype(str).to_numpy(dtype=object),
            "medians": ordered["median"].to_numpy(dtype=np.float32),
            "scales": ordered["scale"].to_numpy(dtype=np.float32),
        }
    return {
        "foods": foods, "axes": axes, "values": values, "splits": splits, "embeddings": embeddings,
        "food_index": food_index, "examples": examples, "normalization": normalization.reset_index(),
    }


def choose_families(candidate: np.ndarray, families: np.ndarray, rng: np.random.Generator, ratio: float) -> set[str]:
    unique = np.array(sorted(set(families[candidate].tolist())))
    if not len(unique):
        return set()
    count = max(1, math.ceil(len(unique) * ratio))
    return set(rng.choice(unique, size=count, replace=False).tolist())


class SparseFoodDataset(Dataset):
    def __init__(self, food_ids: list[str], bundle: dict[str, object], seed: int, corrupt: bool, mask_ratio: float):
        self.food_ids, self.bundle, self.seed = food_ids, bundle, seed
        self.corrupt, self.mask_ratio, self.epoch = corrupt, mask_ratio, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.food_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        food_id = self.food_ids[index]
        item = self.bundle["examples"][food_id]
        axes, values = item["axes"].copy(), item["values"].copy()
        masked = np.zeros(len(axes), dtype=bool)
        if self.corrupt:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            for candidate in (
                item["is_core"] & item["is_maskable"] & item["positive"],
                item["is_form"] & item["is_maskable"] & item["positive"],
            ):
                selected_families = choose_families(candidate, item["families"], rng, self.mask_ratio)
                if selected_families:
                    masked |= np.isin(item["families"], list(selected_families))
        nutrition_weight = (masked & item["is_core"]).astype(np.float32)
        metabolome_weight = (masked & item["is_form"]).astype(np.float32)
        return {
            "food_id": food_id, "axes": axes, "values": values, "masked": masked,
            "nutrition_loss_weight": nutrition_weight, "metabolome_loss_weight": metabolome_weight,
            "medians": item["medians"], "scales": item["scales"],
        }


def collate(batch: list[dict[str, object]], bundle: dict[str, object]) -> dict[str, torch.Tensor | list[str]]:
    length, size = max(len(item["axes"]) for item in batch), len(batch)
    axis = torch.zeros((size, length), dtype=torch.long)
    value, masked, valid = torch.zeros((size, length)), torch.zeros((size, length), dtype=torch.bool), torch.zeros((size, length), dtype=torch.bool)
    nutrition_weight = torch.zeros((size, length))
    metabolome_weight = torch.zeros((size, length))
    median, scale = torch.zeros((size, length)), torch.ones((size, length))
    text_rows, food_ids = [], []
    for row, item in enumerate(batch):
        count = len(item["axes"])
        axis[row, :count] = torch.tensor(item["axes"].tolist(), dtype=torch.long)
        value[row, :count] = torch.tensor(item["values"].tolist(), dtype=torch.float32)
        masked[row, :count] = torch.tensor(item["masked"].tolist(), dtype=torch.bool)
        valid[row, :count] = True
        nutrition_weight[row, :count] = torch.tensor(item["nutrition_loss_weight"].tolist(), dtype=torch.float32)
        metabolome_weight[row, :count] = torch.tensor(item["metabolome_loss_weight"].tolist(), dtype=torch.float32)
        median[row, :count] = torch.tensor(item["medians"].tolist(), dtype=torch.float32)
        scale[row, :count] = torch.tensor(item["scales"].tolist(), dtype=torch.float32)
        food_ids.append(item["food_id"])
        text_rows.append(bundle["embeddings"][bundle["food_index"][item["food_id"]]])
    return {
        "axis": axis, "value": value, "masked": masked, "valid": valid,
        "nutrition_loss_weight": nutrition_weight, "metabolome_loss_weight": metabolome_weight,
        "median": median, "scale": scale, "text": torch.tensor(np.stack(text_rows), dtype=torch.float32),
        "food_ids": food_ids,
    }


class NormalizedAxisHead(nn.Module):
    def __init__(self, d_model: int, axis_count: int, rank: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.projection = nn.Linear(d_model, rank, bias=False)
        self.axis_residual = nn.Embedding(axis_count, rank)
        self.axis_bias = nn.Embedding(axis_count, 1)
        nn.init.zeros_(self.axis_residual.weight)
        nn.init.zeros_(self.axis_bias.weight)
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)

    def forward(self, hidden: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        return self.shared(hidden).squeeze(-1) + (self.projection(hidden) * self.axis_residual(axis)).sum(dim=-1) + self.axis_bias(axis).squeeze(-1)


class FoodNutriGPTV5(nn.Module):
    def __init__(self, text_dim: int, axis_count: int, config: Config):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(nn.Linear(1, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(config.d_model))
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model))
        self.head = NormalizedAxisHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        value_token = self.value_encoder(batch["value"].unsqueeze(-1))
        value_token = torch.where(batch["masked"].unsqueeze(-1), self.mask_value.view(1, 1, -1), value_token)
        axis_token = self.axis_embedding(batch["axis"]) + value_token
        sequence = torch.cat([self.cls.expand(len(batch["axis"]), -1, -1), self.text_projection(batch["text"]).unsqueeze(1), axis_token], dim=1)
        padding = torch.cat([torch.zeros((len(batch["axis"]), 2), dtype=torch.bool, device=batch["axis"].device), ~batch["valid"]], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.head(encoded[:, 2:], batch["axis"])


def macro_axis_mse(prediction: torch.Tensor, target: torch.Tensor, axis: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average masked squared error within axes before averaging across axes."""
    active = weight.gt(0)
    if not active.any():
        return prediction.sum() * 0.0, torch.zeros((), device=prediction.device)
    axis_ids = axis[active]
    errors = (prediction[active] - target[active]).square()
    target_weights = weight[active]
    unique_axes, inverse = torch.unique(axis_ids, sorted=True, return_inverse=True)
    error_sum = torch.zeros(len(unique_axes), device=prediction.device).scatter_add_(0, inverse, errors)
    count = torch.zeros(len(unique_axes), device=prediction.device).scatter_add_(0, inverse, torch.ones_like(errors))
    axis_mse = error_sum / count.clamp_min(1.0)
    axis_weight_sum = torch.zeros(len(unique_axes), device=prediction.device).scatter_add_(0, inverse, target_weights)
    axis_weight = axis_weight_sum / count.clamp_min(1.0)
    weight_mass = axis_weight.sum().clamp_min(1.0)
    return (axis_mse * axis_weight).sum() / weight_mass, weight_mass


def run_epoch(model: nn.Module, loader: DataLoader, config: Config, optimizer: AdamW | None, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    totals = {"joint": 0.0, "nutrition": 0.0, "metabolome": 0.0}
    count = 0
    model.train(training)
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.set_grad_enabled(training):
            prediction = model(tensors)
            nutrition_loss, _ = macro_axis_mse(
                prediction, tensors["value"], tensors["axis"], tensors["nutrition_loss_weight"]
            )
            metabolome_loss, _ = macro_axis_mse(
                prediction, tensors["value"], tensors["axis"], tensors["metabolome_loss_weight"]
            )
            loss = (
                config.lambda_nutrition * nutrition_loss
                + config.lambda_metabolome * metabolome_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        totals["joint"] += float(loss.detach().cpu())
        totals["nutrition"] += float(nutrition_loss.detach().cpu())
        totals["metabolome"] += float(metabolome_loss.detach().cpu())
        count += 1
    return {name: value / max(count, 1) for name, value in totals.items()}


def train_joint(model: nn.Module, bundle: dict[str, object], config: Config, output_dir: Path, device: torch.device) -> Path:
    splits = bundle["splits"]
    train_set = SparseFoodDataset(splits["train"], bundle, config.seed, True, config.mask_ratio)
    valid_set = SparseFoodDataset(splits["validation"], bundle, config.seed + 99, True, config.mask_ratio)
    loader = lambda dataset, shuffle: DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda rows: collate(rows, bundle))
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    best, stale, history = math.inf, 0, []
    checkpoint = output_dir / "joint_best.pt"
    for epoch in range(1, config.epochs + 1):
        train_set.set_epoch(epoch)
        valid_set.set_epoch(0)
        train_metrics = run_epoch(model, loader(train_set, True), config, optimizer, device)
        validation_metrics = run_epoch(model, loader(valid_set, False), config, None, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch, "learning_rate": current_lr,
            **{f"train_{name}_macro_mse": value for name, value in train_metrics.items()},
            **{f"validation_{name}_macro_mse": value for name, value in validation_metrics.items()},
        })
        print(
            f"joint epoch {epoch:02d}: train={train_metrics['joint']:.5f} "
            f"validation={validation_metrics['joint']:.5f} "
            f"(nutrition={validation_metrics['nutrition']:.5f}, "
            f"metabolome={validation_metrics['metabolome']:.5f})"
        )
        if validation_metrics["joint"] < best:
            best, stale = validation_metrics["joint"], 0
            torch.save({
                "model_state": model.state_dict(), "config": asdict(config),
                "training_mode": "single_stage_joint", "validation_joint_macro_mse": best,
            }, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
        scheduler.step()
    pd.DataFrame(history).to_csv(output_dir / "joint_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--lambda-nutrition", type=float, default=Config.lambda_nutrition)
    parser.add_argument("--lambda-metabolome", type=float, default=Config.lambda_metabolome)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args()
    config = Config(
        epochs=args.epochs, learning_rate=args.learning_rate, patience=args.patience,
        lambda_nutrition=args.lambda_nutrition, lambda_metabolome=args.lambda_metabolome,
    )
    output_dir = args.output_dir.resolve()
    data_dir, split_dir = args.data_dir.resolve(), args.split_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    bundle = load_bundle(config, output_dir, data_dir, split_dir)
    device = v4.device_for_training()
    print(f"device={device}; foods={len(bundle['foods'])}; axes={len(bundle['axes'])}")
    model = FoodNutriGPTV5(bundle["embeddings"].shape[1], len(bundle["axes"]), config).to(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            **asdict(config), "data_dir": str(data_dir), "split_dir": str(split_dir),
            "training_mode": "single_stage_joint", "normalization": "train_median_baseline_log_rmse",
            "test_policy": "This training entry point never reads or evaluates the test split.",
        }, handle, indent=2)
    checkpoint = train_joint(model, bundle, config, output_dir, device)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "checkpoint": str(checkpoint), "training_mode": "single_stage_joint",
            "loss": "lambda_nutrition * macro_axis_mse(nutrition) + lambda_metabolome * macro_axis_mse(food_metabolome)",
            "test_evaluated": False,
        }, handle, indent=2)


if __name__ == "__main__":
    main()

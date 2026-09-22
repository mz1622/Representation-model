#!/usr/bin/env python3
"""Two-stage FoodNutriGPT training on the audited multisource_mass_v4 corpus.

Stage 1 reconstructs macro/micro nutrients. Compound axes remain visible
context, except that maskable compound values receive input-only corruption.
Stage 2 reconstructs compounds and replays nutrient reconstruction at a lower
weight to reduce nutrient forgetting. Missing axes never become tokens.
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
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/multisource_mass_v4"
SPLIT_DIR = ROOT / "data/splits/multisource_mass_v4"
DEFAULT_OUTPUT_ROOT = ROOT / "output/multisource_mass_v4_foodnutrigpt"


@dataclass
class Config:
    seed: int = 20260808
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 48
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    feedforward_dim: int = 1024
    dropout: float = 0.15
    axis_residual_rank: int = 16
    mask_ratio: float = 0.30
    stage1_epochs: int = 20
    stage2_epochs: int = 20
    stage1_lr: float = 1e-4
    stage2_lr: float = 5e-5
    weight_decay: float = 1e-4
    patience: int = 4
    nutrient_replay_weight: float = 0.25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def food_text(row: pd.Series) -> str:
    name = str(row.food_name).strip()
    description = str(row.food_description).strip()
    return name if not description or description == name else f"{name}. {description}"


def load_text_embeddings(foods: pd.DataFrame, config: Config, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "text_embeddings.npy"
    ids_path = cache_dir / "text_embedding_food_ids.csv"
    expected_ids = foods["canonical_food_id"].tolist()
    if cache.exists() and ids_path.exists():
        cached_ids = pd.read_csv(ids_path)["canonical_food_id"].tolist()
        embeddings = np.load(cache)
        if cached_ids == expected_ids and len(embeddings) == len(foods):
            return embeddings.astype(np.float32)
    # Load only the native PyTorch encoder files.  SentenceTransformer 2.2
    # snapshots optional ONNX/OpenVINO variants as well, which makes a simple
    # frozen text-context cache unnecessarily large and slow.
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    encoder = AutoModel.from_pretrained(config.text_model).to(device_for_training()).eval()
    encoder_device = next(encoder.parameters()).device
    text = foods.apply(food_text, axis=1).tolist()
    pieces = []
    for start in range(0, len(text), 64):
        encoded = tokenizer(text[start : start + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {name: tensor.to(encoder_device) for name, tensor in encoded.items()}
        with torch.no_grad():
            token_embeddings = encoder(**encoded).last_hidden_state
        weights = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pieces.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        if start % 640 == 0:
            print(f"Encoded text rows {min(start + 64, len(text)):,}/{len(text):,}")
    embeddings = np.concatenate(pieces, axis=0).astype(np.float32)
    np.save(cache, embeddings)
    pd.DataFrame({"canonical_food_id": expected_ids}).to_csv(ids_path, index=False)
    return embeddings


def load_bundle(config: Config, output_dir: Path) -> dict[str, object]:
    foods = pd.read_csv(DATA_DIR / "food_entities.csv").sort_values("canonical_food_id", kind="stable").reset_index(drop=True)
    axes = pd.read_csv(DATA_DIR / "axis_registry.csv").sort_values("axis_index", kind="stable").reset_index(drop=True)
    values = pd.read_csv(DATA_DIR / "observed_axis_values.csv")
    normalization = pd.read_csv(DATA_DIR / "train_only_axis_normalization.csv")
    with (SPLIT_DIR / "splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    embeddings = load_text_embeddings(foods, config, output_dir / "text_cache")
    food_index = {food_id: index for index, food_id in enumerate(foods["canonical_food_id"])}
    axis_index = dict(zip(axes["axis_id"], axes["axis_index"]))
    normalizer = normalization.set_index("axis_id")
    values = values[values["canonical_food_id"].isin(food_index) & values["axis_id"].isin(axis_index)].copy()
    values["axis_index"] = values["axis_id"].map(axis_index).astype(int)
    values["normalized_value"] = (
        np.log1p(values["value_g_per_100g"].to_numpy(dtype=np.float32))
        - values["axis_id"].map(normalizer["median"]).to_numpy(dtype=np.float32)
    ) / values["axis_id"].map(normalizer["iqr_scale"]).to_numpy(dtype=np.float32)
    values = values.merge(axes[["axis_index", "target_kind", "mask_policy"]], on="axis_index", how="left", validate="many_to_one")
    examples: dict[str, dict[str, np.ndarray]] = {}
    for food_id, group in values.groupby("canonical_food_id", sort=False):
        ordered = group.sort_values("axis_index", kind="stable")
        examples[food_id] = {
            "axes": ordered["axis_index"].to_numpy(dtype=np.int64),
            "values": ordered["normalized_value"].to_numpy(dtype=np.float32),
            "positive": ordered["value_g_per_100g"].gt(0).to_numpy(dtype=bool),
            "is_nutrient": ordered["target_kind"].eq("nutrient").to_numpy(dtype=bool),
            "is_maskable": ordered["mask_policy"].eq("maskable_target").to_numpy(dtype=bool),
        }
    return {"foods": foods, "axes": axes, "values": values, "splits": splits, "embeddings": embeddings, "food_index": food_index, "examples": examples, "normalization": normalization}


class SparseFoodDataset(Dataset):
    def __init__(self, food_ids: list[str], bundle: dict[str, object], stage: str, seed: int, corrupt: bool):
        self.food_ids, self.bundle, self.stage, self.seed, self.corrupt, self.epoch = food_ids, bundle, stage, seed, corrupt, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.food_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        food_id = self.food_ids[index]
        item = self.bundle["examples"][food_id]
        axes, values = item["axes"].copy(), item["values"].copy()
        mask = np.zeros(len(axes), dtype=bool)
        if self.corrupt:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            nutrient = item["is_nutrient"] & item["is_maskable"] & item["positive"]
            compound = ~item["is_nutrient"] & item["is_maskable"] & item["positive"]
            # Stage 1 masks compound values in the input but gives them no
            # loss, preventing the first stage from relying on every future
            # compound target as visible context.
            for candidate in (nutrient, compound):
                positions = np.flatnonzero(candidate)
                if len(positions):
                    count = max(1, math.ceil(len(positions) * 0.30))
                    mask[rng.choice(positions, size=count, replace=False)] = True
        loss_weight = np.zeros(len(axes), dtype=np.float32)
        if self.stage == "stage1":
            loss_weight[mask & item["is_nutrient"]] = 1.0
        elif self.stage == "stage2":
            loss_weight[mask & ~item["is_nutrient"]] = 1.0
            loss_weight[mask & item["is_nutrient"]] = 0.25
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        return {"food_id": food_id, "axes": axes, "values": values, "masked": mask, "loss_weight": loss_weight}


def collate(batch: list[dict[str, object]], bundle: dict[str, object]) -> dict[str, torch.Tensor | list[str]]:
    length = max(len(item["axes"]) for item in batch)
    size = len(batch)
    axis = torch.zeros((size, length), dtype=torch.long)
    value = torch.zeros((size, length), dtype=torch.float32)
    masked = torch.zeros((size, length), dtype=torch.bool)
    valid = torch.zeros((size, length), dtype=torch.bool)
    weight = torch.zeros((size, length), dtype=torch.float32)
    text_rows, food_ids = [], []
    indexes = bundle["food_index"]
    embeddings = bundle["embeddings"]
    for row, item in enumerate(batch):
        count = len(item["axes"])
        # ``torch.from_numpy`` is not reliable with some local NumPy 2 /
        # PyTorch combinations.  Explicit construction keeps the Colab and
        # local paths behaviourally identical.
        axis[row, :count] = torch.tensor(item["axes"].tolist(), dtype=torch.long)
        value[row, :count] = torch.tensor(item["values"].tolist(), dtype=torch.float32)
        masked[row, :count] = torch.tensor(item["masked"].tolist(), dtype=torch.bool)
        valid[row, :count] = True
        weight[row, :count] = torch.tensor(item["loss_weight"].tolist(), dtype=torch.float32)
        food_ids.append(item["food_id"])
        text_rows.append(embeddings[indexes[item["food_id"]]])
    return {"axis": axis, "value": value, "masked": masked, "valid": valid, "loss_weight": weight, "text": torch.tensor(np.stack(text_rows), dtype=torch.float32), "food_ids": food_ids}


class AxisPredictionHead(nn.Module):
    def __init__(self, d_model: int, axis_count: int, rank: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.projection = nn.Linear(d_model, rank, bias=False)
        self.axis_residual = nn.Embedding(axis_count, rank)
        self.axis_bias = nn.Embedding(axis_count, 1)
        nn.init.zeros_(self.axis_residual.weight)
        nn.init.zeros_(self.axis_bias.weight)

    def forward(self, hidden: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        return self.shared(hidden).squeeze(-1) + (self.projection(hidden) * self.axis_residual(axis)).sum(dim=-1) + self.axis_bias(axis).squeeze(-1)


class FoodNutriGPT(nn.Module):
    def __init__(self, text_dim: int, axis_count: int, config: Config):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.text_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(nn.Linear(1, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(config.d_model))
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, config.feedforward_dim, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model))
        self.head = AxisPredictionHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        axis, value, masked, valid, text = batch["axis"], batch["value"], batch["masked"], batch["valid"], batch["text"]
        value_token = self.value_encoder(value.unsqueeze(-1))
        value_token = torch.where(masked.unsqueeze(-1), self.mask_value.view(1, 1, -1), value_token)
        axis_token = self.axis_embedding(axis) + value_token
        sequence = torch.cat([self.cls.expand(len(axis), -1, -1), self.text_projection(text).unsqueeze(1), axis_token], dim=1)
        padding = torch.cat([torch.zeros((len(axis), 2), dtype=torch.bool, device=axis.device), ~valid], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.head(encoded[:, 2:], axis)


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (((prediction - target) ** 2) * weight).sum() / weight.sum().clamp_min(1.0)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: AdamW | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0.0
    for batch in loader:
        tensor_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.set_grad_enabled(training):
            prediction = model(tensor_batch)
            loss = weighted_mse(prediction, tensor_batch["value"], tensor_batch["loss_weight"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        mass = float(tensor_batch["loss_weight"].sum().item())
        total += float(loss.detach().cpu()) * mass
        count += mass
    return total / max(count, 1.0)


def train_stage(model: nn.Module, bundle: dict[str, object], config: Config, stage: str, epochs: int, lr: float, output_dir: Path, device: torch.device) -> Path:
    splits = bundle["splits"]
    train_set = SparseFoodDataset(splits["train"], bundle, stage, config.seed, corrupt=True)
    validation_set = SparseFoodDataset(splits["validation"], bundle, stage, config.seed + 99, corrupt=True)
    make_loader = lambda dataset, shuffle: DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda items: collate(items, bundle))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    best, stale, history = math.inf, 0, []
    checkpoint = output_dir / f"{stage}_best.pt"
    for epoch in range(1, epochs + 1):
        train_set.set_epoch(epoch)
        validation_set.set_epoch(0)
        train_loss = run_epoch(model, make_loader(train_set, True), optimizer, device)
        validation_loss = run_epoch(model, make_loader(validation_set, False), None, device)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
        print(f"{stage} epoch {epoch:02d}: train={train_loss:.5f} validation={validation_loss:.5f}")
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "stage": stage, "validation_mse": best}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
    pd.DataFrame(history).to_csv(output_dir / f"{stage}_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    return checkpoint


@torch.no_grad()
def sequential_evaluate(model: nn.Module, bundle: dict[str, object], stage: str, device: torch.device, output_dir: Path) -> dict[str, float]:
    masks = pd.read_csv(SPLIT_DIR / "sequential_masks.csv")
    target_kind = "nutrient" if stage == "stage1" else "compound"
    masks = masks[(masks["split"] == "test") & (masks["target_kind"] == target_kind)].copy()
    axis_index = dict(zip(bundle["axes"]["axis_id"], bundle["axes"]["axis_index"]))
    rows = []
    model.eval()
    for food_id, food_masks in masks.groupby("canonical_food_id", sort=False):
        item = bundle["examples"][food_id]
        axes, values = item["axes"], item["values"].copy()
        positions = {axis: index for index, axis in enumerate(axes)}
        ordered = food_masks.sort_values("prediction_order", kind="stable")
        target_positions = [positions[axis_index[axis_id]] for axis_id in ordered["axis_id"]]
        masked = np.zeros(len(axes), dtype=bool)
        masked[target_positions] = True
        for row in ordered.itertuples(index=False):
            position = positions[axis_index[row.axis_id]]
            batch = {
                "axis": torch.tensor(axes[None, :], dtype=torch.long, device=device),
                "value": torch.tensor(values[None, :], dtype=torch.float32, device=device),
                "masked": torch.tensor(masked[None, :], dtype=torch.bool, device=device),
                "valid": torch.ones((1, len(axes)), dtype=torch.bool, device=device),
                "text": torch.tensor(bundle["embeddings"][bundle["food_index"][food_id]][None, :], dtype=torch.float32, device=device),
            }
            predicted = float(model(batch)[0, position].cpu())
            target = float(item["values"][position])
            rows.append({"canonical_food_id": food_id, "axis_id": row.axis_id, "prediction_order": row.prediction_order, "target_normalized": target, "prediction_normalized": predicted})
            values[position] = predicted
            masked[position] = False
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{stage}_sequential_test_predictions.csv", index=False)
    error = frame["prediction_normalized"] - frame["target_normalized"]
    metrics = {"count": int(len(frame)), "mse": float((error ** 2).mean()), "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error ** 2).mean()))}
    with (output_dir / f"{stage}_sequential_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"{stage} sequential test: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-epochs", type=int, default=Config.stage1_epochs)
    parser.add_argument("--stage2-epochs", type=int, default=Config.stage2_epochs)
    args = parser.parse_args()
    config = Config(stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    bundle = load_bundle(config, output_dir)
    device = device_for_training()
    print(f"device={device}; foods={len(bundle['foods'])}; axes={len(bundle['axes'])}")
    model = FoodNutriGPT(bundle["embeddings"].shape[1], len(bundle["axes"]), config).to(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
    stage1_checkpoint = train_stage(model, bundle, config, "stage1", config.stage1_epochs, config.stage1_lr, output_dir, device)
    nutrient_metrics = sequential_evaluate(model, bundle, "stage1", device, output_dir)
    stage2_checkpoint = train_stage(model, bundle, config, "stage2", config.stage2_epochs, config.stage2_lr, output_dir, device)
    compound_metrics = sequential_evaluate(model, bundle, "stage2", device, output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"stage1_checkpoint": str(stage1_checkpoint), "stage2_checkpoint": str(stage2_checkpoint), "nutrient_sequential_test": nutrient_metrics, "compound_sequential_test": compound_metrics}, handle, indent=2)


if __name__ == "__main__":
    main()

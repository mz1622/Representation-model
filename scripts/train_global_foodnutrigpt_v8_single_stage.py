#!/usr/bin/env python3
"""Train and evaluate one source-aware, single-stage FoodNutriGPT v8 model.

The model receives a profile-level source token during most training examples,
but source dropout teaches the same model to operate with SOURCE_UNKNOWN.  The
complete held-out panel remains outside model selection. Training experiments
may use ``--skip-test`` so candidate selection opens only the validation set.

Colab:
    python scripts/train_global_foodnutrigpt_v8_single_stage.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_complete_test"


@dataclass
class Config:
    seed: int = 20260920
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_batch_size: int = 64
    batch_size: int = 32
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 3
    feedforward_dim: int = 768
    dropout: float = 0.15
    axis_residual_rank: int = 16
    max_tokens: int = 256
    mask_ratio: float = 0.30
    zero_only_family_mask_ratio: float = 0.10
    text_only_probability: float = 0.15
    source_dropout: float = 0.30
    epochs: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    lambda_nutrition: float = 1.0
    lambda_metabolome: float = 1.0
    amount_loss_weight: float = 1.0
    loss_mode: str = "group_balanced"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode_texts(profiles: pd.DataFrame, config: Config, cache_dir: Path, cache_name: str) -> tuple[list[str], np.ndarray]:
    """Encode only the supplied partition; cache IDs guard against stale reuse."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{cache_name}_embeddings.npy"
    ids_path = cache_dir / f"{cache_name}_profile_ids.csv"
    profile_ids = profiles["profile_id"].tolist()
    if matrix_path.exists() and ids_path.exists():
        cached_ids = pd.read_csv(ids_path)["profile_id"].tolist()
        matrix = np.load(matrix_path)
        if cached_ids == profile_ids and len(matrix) == len(profile_ids):
            return profile_ids, matrix.astype(np.float32)

    device = device_for_training()
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    encoder = AutoModel.from_pretrained(config.text_model).to(device).eval()
    rows: list[np.ndarray] = []
    texts = profiles["food_text"].fillna("").astype(str).tolist()
    for start in range(0, len(texts), config.text_batch_size):
        encoded = tokenizer(
            texts[start : start + config.text_batch_size], padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            hidden = encoder(**encoded).last_hidden_state
        attention = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1.0)
        rows.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
        if start % (config.text_batch_size * 20) == 0:
            print(f"{cache_name} text embeddings: {min(start + config.text_batch_size, len(texts)):,}/{len(texts):,}")
    matrix = np.concatenate(rows, axis=0).astype(np.float32)
    np.save(matrix_path, matrix)
    pd.DataFrame({"profile_id": profile_ids}).to_csv(ids_path, index=False)
    return profile_ids, matrix


class SourceNativeCorpus:
    def __init__(self, data_dir: Path, split_dir: Path):
        self.profiles = pd.read_csv(data_dir / "food_profiles.csv.gz", low_memory=False).sort_values("profile_id", kind="stable").reset_index(drop=True)
        self.axes = pd.read_csv(data_dir / "axis_registry.csv", low_memory=False).sort_values("axis_index", kind="stable").reset_index(drop=True)
        tokens = pd.read_csv(data_dir / "source_native_axis_tokens.csv.gz", low_memory=False)
        with (split_dir / "splits.json").open(encoding="utf-8") as handle:
            self.splits = json.load(handle)

        self.profile_index = {profile_id: index for index, profile_id in enumerate(self.profiles["profile_id"])}
        tokens = tokens[tokens["profile_id"].isin(self.profile_index)].copy()
        tokens["profile_index"] = tokens["profile_id"].map(self.profile_index).astype(np.int64)
        tokens = tokens.sort_values(["profile_index", "axis_index", "measurement_id"], kind="stable").reset_index(drop=True)
        starts = tokens.groupby("profile_index", sort=False).size().cumsum().to_numpy(dtype=np.int64)
        self.starts = np.zeros(len(self.profiles), dtype=np.int64)
        self.ends = np.zeros(len(self.profiles), dtype=np.int64)
        cursor = 0
        for profile_index, end in zip(tokens.groupby("profile_index", sort=False).size().index.to_numpy(dtype=np.int64), starts):
            self.starts[profile_index] = cursor
            self.ends[profile_index] = end
            cursor = end

        family_codes, _ = pd.factorize(tokens["mask_family"].astype(str), sort=True)
        self.axis = tokens["axis_index"].to_numpy(dtype=np.int64)
        self.value = tokens["normalized_log1p_value"].to_numpy(dtype=np.float32)
        self.raw_value = tokens["normalized_value_g_per_100g"].to_numpy(dtype=np.float32)
        self.positive = tokens["is_positive"].to_numpy(dtype=bool)
        self.loss_eligible = tokens["loss_eligible"].astype(bool).to_numpy()
        self.loss_group = tokens["loss_group"].eq("food_metabolome").to_numpy(dtype=np.int64)
        self.family = family_codes.astype(np.int64)
        self.cell_weight = tokens["within_profile_axis_loss_weight"].to_numpy(dtype=np.float32)
        self.axis_median = self.axes["median_log1p"].to_numpy(dtype=np.float32)
        self.axis_scale = self.axes["scale_log1p"].to_numpy(dtype=np.float32)
        self.source = self.profiles["source_index"].to_numpy(dtype=np.int64)
        self.test_axis = np.isin(
            self.axis,
            self.axes.loc[
                self.axes["target_axis_id"].isin(self.splits["test_axis_ids"]), "axis_index"
            ].to_numpy(dtype=np.int64),
        )
        if not np.all(self.ends > self.starts):
            empty = self.profiles.loc[~(self.ends > self.starts), "profile_id"].tolist()[:5]
            raise ValueError(f"Profiles without tokens entered the corpus: {empty}")

    def positions(self, split: str) -> list[int]:
        ids = self.splits[split]
        return [self.profile_index[profile_id] for profile_id in ids]


def _select_families(candidate: np.ndarray, families: np.ndarray, ratio: float, rng: np.random.Generator) -> set[int]:
    unique = np.unique(families[candidate])
    if not len(unique):
        return set()
    count = max(1, math.ceil(len(unique) * ratio))
    return set(rng.choice(unique, size=count, replace=False).tolist())


class MaskedProfileDataset(Dataset):
    def __init__(
        self, corpus: SourceNativeCorpus, positions: list[int], config: Config, *, training: bool,
        evaluation_axes: np.ndarray | None = None, force_unknown_source: bool = False,
        cover_all_evaluation_families: bool = False,
    ):
        self.corpus, self.positions, self.config = corpus, positions, config
        self.training, self.evaluation_axes, self.force_unknown_source = training, evaluation_axes, force_unknown_source
        self.cover_all_evaluation_families = cover_all_evaluation_families
        self.epoch = 0
        self.sample_specs: list[tuple[int, int | None]] = [(position, None) for position in positions]
        if cover_all_evaluation_families:
            if training or evaluation_axes is None:
                raise ValueError("Complete-family evaluation requires non-training evaluation axes.")
            self.sample_specs = []
            for profile_index in positions:
                indexes = self._token_indexes(profile_index)
                candidate = (
                    self.corpus.loss_eligible[indexes]
                    & np.isin(self.corpus.axis[indexes], evaluation_axes)
                )
                for family_id in np.unique(self.corpus.family[indexes][candidate]).tolist():
                    self.sample_specs.append((profile_index, int(family_id)))
            if not self.sample_specs:
                raise ValueError("The complete test panel has no evaluable mask families.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_specs)

    def _token_indexes(self, profile_index: int) -> np.ndarray:
        start, end = self.corpus.starts[profile_index], self.corpus.ends[profile_index]
        indexes = np.arange(start, end, dtype=np.int64)
        if len(indexes) > self.config.max_tokens:
            # Only a small number of profiles exceed this cap.  The stable seed
            # makes the same source-native subset visible on every evaluation.
            rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003 + profile_index)
            indexes = np.sort(rng.choice(indexes, size=self.config.max_tokens, replace=False))
        return indexes

    def __getitem__(self, item_index: int) -> dict[str, np.ndarray | int]:
        profile_index, forced_family = self.sample_specs[item_index]
        indexes = self._token_indexes(profile_index)
        rng = np.random.default_rng(self.config.seed + self.epoch * 1_000_003 + profile_index)

        axis = self.corpus.axis[indexes]
        value = self.corpus.value[indexes]
        raw_value = self.corpus.raw_value[indexes]
        positive = self.corpus.positive[indexes]
        family = self.corpus.family[indexes]
        loss_eligible = self.corpus.loss_eligible[indexes]
        loss_group = self.corpus.loss_group[indexes]
        cell_weight = self.corpus.cell_weight[indexes]
        candidate = loss_eligible.copy()
        if self.evaluation_axes is not None:
            candidate &= np.isin(axis, self.evaluation_axes)

        if forced_family is not None:
            # Test every eligible chemical family once per food.  This both
            # blocks aggregate/form leakage and guarantees all 187 axes receive
            # a real masked prediction rather than merely occurring in test.
            masked = family == forced_family
        else:
            masked_families: set[int] = set()
            for group in (0, 1):
                group_candidate = candidate & (loss_group == group)
                positive_candidate = group_candidate & positive
                selected_positive = _select_families(positive_candidate, family, self.config.mask_ratio, rng)
                masked_families.update(selected_positive)
                selected_positive_array = np.isin(family, list(selected_positive)) if selected_positive else np.zeros(len(axis), dtype=bool)
                zero_only = group_candidate & ~positive & ~selected_positive_array
                masked_families.update(_select_families(zero_only, family, self.config.zero_only_family_mask_ratio, rng))
            masked = np.isin(family, list(masked_families)) if masked_families else np.zeros(len(axis), dtype=bool)
        target = masked & candidate
        if not target.any() and candidate.any():
            chosen = int(rng.choice(np.flatnonzero(candidate)))
            masked |= family == family[chosen]
            target = masked & candidate

        source = int(self.corpus.source[profile_index])
        text_only = self.training and rng.random() < self.config.text_only_probability
        if text_only and target.any():
            # Keep only explicit requested axes; their values are all masked.
            keep = target
            axis, value, raw_value = axis[keep], value[keep], raw_value[keep]
            positive, family = positive[keep], family[keep]
            loss_eligible, loss_group = loss_eligible[keep], loss_group[keep]
            cell_weight, masked, target = cell_weight[keep], np.ones(keep.sum(), dtype=bool), np.ones(keep.sum(), dtype=bool)
            source = 0
        elif self.training and rng.random() < self.config.source_dropout:
            source = 0
        if self.force_unknown_source:
            source = 0

        return {
            "profile_index": profile_index, "source": source, "axis": axis, "value": value,
            "raw_value": raw_value, "positive": positive, "masked": masked, "target": target,
            "loss_group": loss_group, "cell_weight": cell_weight,
        }


def collate(rows: list[dict[str, np.ndarray | int]], text_embeddings: np.ndarray) -> dict[str, torch.Tensor]:
    batch_size, length = len(rows), max(len(row["axis"]) for row in rows)
    axis = torch.zeros((batch_size, length), dtype=torch.long)
    value = torch.zeros((batch_size, length), dtype=torch.float32)
    raw_value = torch.zeros((batch_size, length), dtype=torch.float32)
    positive = torch.zeros((batch_size, length), dtype=torch.bool)
    masked = torch.zeros((batch_size, length), dtype=torch.bool)
    target = torch.zeros((batch_size, length), dtype=torch.bool)
    loss_group = torch.zeros((batch_size, length), dtype=torch.long)
    cell_weight = torch.zeros((batch_size, length), dtype=torch.float32)
    valid = torch.zeros((batch_size, length), dtype=torch.bool)
    profile_indices, sources, text = [], [], []
    for row_index, row in enumerate(rows):
        count = len(row["axis"])
        axis[row_index, :count] = torch.tensor(row["axis"].tolist(), dtype=torch.long)
        value[row_index, :count] = torch.tensor(row["value"].tolist(), dtype=torch.float32)
        raw_value[row_index, :count] = torch.tensor(row["raw_value"].tolist(), dtype=torch.float32)
        positive[row_index, :count] = torch.tensor(row["positive"].tolist(), dtype=torch.bool)
        masked[row_index, :count] = torch.tensor(row["masked"].tolist(), dtype=torch.bool)
        target[row_index, :count] = torch.tensor(row["target"].tolist(), dtype=torch.bool)
        loss_group[row_index, :count] = torch.tensor(row["loss_group"].tolist(), dtype=torch.long)
        cell_weight[row_index, :count] = torch.tensor(row["cell_weight"].tolist(), dtype=torch.float32)
        valid[row_index, :count] = True
        profile_indices.append(int(row["profile_index"]))
        sources.append(int(row["source"]))
        text.append(text_embeddings[int(row["profile_index"])])
    return {
        "axis": axis, "value": value, "raw_value": raw_value, "positive": positive,
        "masked": masked, "target": target, "loss_group": loss_group, "cell_weight": cell_weight,
        "valid": valid, "source": torch.tensor(sources, dtype=torch.long),
        "profile_index": torch.tensor(profile_indices, dtype=torch.long),
        "text": torch.tensor(np.stack(text), dtype=torch.float32),
    }


class AxisCalibratedHead(nn.Module):
    def __init__(self, d_model: int, axis_count: int, rank: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self.projection = nn.Linear(d_model, rank, bias=False)
        self.axis_residual = nn.Embedding(axis_count, rank)
        self.axis_bias = nn.Embedding(axis_count, 1)
        nn.init.zeros_(self.axis_residual.weight)
        nn.init.zeros_(self.axis_bias.weight)

    def forward(self, hidden: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        return (
            self.shared(hidden).squeeze(-1)
            + (self.projection(hidden) * self.axis_residual(axis)).sum(dim=-1)
            + self.axis_bias(axis).squeeze(-1)
        )


class SourceAwareFoodNutriGPT(nn.Module):
    def __init__(self, text_dim: int, axis_count: int, source_count: int, config: Config):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.source_embedding = nn.Embedding(source_count, config.d_model)
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model),
        )
        self.mask_value = nn.Parameter(torch.zeros(config.d_model))
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.feedforward_dim, config.dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, norm=nn.LayerNorm(config.d_model))
        self.presence_head = AxisCalibratedHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        self.amount_head = AxisCalibratedHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value_token = self.value_encoder(batch["value"].unsqueeze(-1))
        value_token = torch.where(batch["masked"].unsqueeze(-1), self.mask_value.view(1, 1, -1), value_token)
        axis_token = self.axis_embedding(batch["axis"]) + value_token
        sequence = torch.cat([
            self.cls.expand(len(batch["axis"]), -1, -1),
            self.text_projection(batch["text"]).unsqueeze(1),
            self.source_embedding(batch["source"]).unsqueeze(1),
            axis_token,
        ], dim=1)
        padding = torch.cat([
            torch.zeros((len(batch["axis"]), 3), dtype=torch.bool, device=batch["axis"].device),
            ~batch["valid"],
        ], dim=1)
        hidden = self.encoder(sequence, src_key_padding_mask=padding)[:, 3:]
        return {
            "positive_logit": self.presence_head(hidden, batch["axis"]),
            "amount_normalized": self.amount_head(hidden, batch["axis"]),
        }


def macro_hurdle_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], group: int | None, amount_loss_weight: float,
) -> tuple[torch.Tensor, int]:
    active = batch["target"]
    if group is not None:
        active = active & batch["loss_group"].eq(group)
    if not active.any():
        return outputs["positive_logit"].sum() * 0.0, 0
    positive_target = batch["positive"].float()
    presence = F.binary_cross_entropy_with_logits(outputs["positive_logit"], positive_target, reduction="none")
    amount = F.smooth_l1_loss(outputs["amount_normalized"], batch["value"], beta=1.0, reduction="none")
    per_token = presence + amount_loss_weight * amount * positive_target
    axis = batch["axis"][active]
    token_weight = batch["cell_weight"][active]
    weighted_error = per_token[active] * token_weight
    unique_axis, inverse = torch.unique(axis, sorted=True, return_inverse=True)
    error_sum = torch.zeros(len(unique_axis), device=axis.device).scatter_add_(0, inverse, weighted_error)
    weight_sum = torch.zeros(len(unique_axis), device=axis.device).scatter_add_(0, inverse, token_weight)
    return (error_sum / weight_sum.clamp_min(1e-8)).mean(), int(len(unique_axis))


def joint_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: Config) -> tuple[torch.Tensor, dict[str, float]]:
    nutrition, nutrition_axes = macro_hurdle_loss(outputs, batch, 0, config.amount_loss_weight)
    metabolome, metabolome_axes = macro_hurdle_loss(outputs, batch, 1, config.amount_loss_weight)
    if config.loss_mode == "all_axis":
        joint, joint_axes = macro_hurdle_loss(outputs, batch, None, config.amount_loss_weight)
        if not joint_axes:
            return outputs["positive_logit"].sum() * 0.0, {"joint": 0.0, "nutrition": 0.0, "metabolome": 0.0}
        return joint, {
            "joint": float(joint.detach().cpu()),
            "nutrition": float(nutrition.detach().cpu()) if nutrition_axes else float("nan"),
            "metabolome": float(metabolome.detach().cpu()) if metabolome_axes else float("nan"),
        }
    if config.loss_mode != "group_balanced":
        raise ValueError(f"Unsupported loss mode: {config.loss_mode}")
    weighted: list[torch.Tensor] = []
    if nutrition_axes:
        weighted.append(config.lambda_nutrition * nutrition)
    if metabolome_axes:
        weighted.append(config.lambda_metabolome * metabolome)
    if not weighted:
        return outputs["positive_logit"].sum() * 0.0, {"joint": 0.0, "nutrition": 0.0, "metabolome": 0.0}
    joint = torch.stack(weighted).mean()
    return joint, {
        "joint": float(joint.detach().cpu()),
        "nutrition": float(nutrition.detach().cpu()) if nutrition_axes else float("nan"),
        "metabolome": float(metabolome.detach().cpu()) if metabolome_axes else float("nan"),
    }


def run_epoch(
    model: nn.Module, loader: DataLoader, config: Config, device: torch.device, optimizer: AdamW | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = {"joint": 0.0, "nutrition": 0.0, "metabolome": 0.0}
    counts = {"joint": 0, "nutrition": 0, "metabolome": 0}
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            outputs = model(tensors)
            loss, metrics = joint_loss(outputs, tensors, config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for name, value in metrics.items():
            if np.isfinite(value):
                total[name] += value
                counts[name] += 1
    return {name: total[name] / max(counts[name], 1) for name in total}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, corpus: SourceNativeCorpus, device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[pd.DataFrame] = []
    axis_median = torch.as_tensor(corpus.axis_median, dtype=torch.float32, device=device)
    axis_scale = torch.as_tensor(corpus.axis_scale, dtype=torch.float32, device=device)
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items()}
        outputs = model(tensors)
        active = tensors["target"].bool()
        axis = tensors["axis"][active]
        profile_index = tensors["profile_index"].unsqueeze(1).expand_as(active)[active]
        positive_probability = torch.sigmoid(outputs["positive_logit"])[active]
        amount_normalized = outputs["amount_normalized"][active]
        target_raw = tensors["raw_value"][active]
        target_positive = tensors["positive"][active]
        predicted_log_amount = torch.clamp(
            amount_normalized * axis_scale[axis] + axis_median[axis], min=0.0,
        )
        predicted_conditional_amount = torch.expm1(predicted_log_amount)
        predicted_raw = positive_probability * predicted_conditional_amount
        # Convert through Python scalars rather than torch's NumPy bridge.  That
        # bridge is inconsistent on some Apple/MPS installations during inference.
        rows.append(pd.DataFrame({
            "profile_id": corpus.profiles.iloc[profile_index.detach().cpu().tolist()]["profile_id"].tolist(),
            "axis_index": axis.detach().cpu().tolist(),
            "target_g_per_100g": target_raw.detach().cpu().tolist(),
            "target_positive": target_positive.detach().cpu().to(torch.int64).tolist(),
            "positive_probability": positive_probability.detach().cpu().tolist(),
            "conditional_positive_g_per_100g": predicted_conditional_amount.detach().cpu().tolist(),
            "prediction_g_per_100g": predicted_raw.detach().cpu().tolist(),
        }))
    predictions = pd.concat(rows, ignore_index=True)
    predictions = predictions.merge(
        corpus.axes[["axis_index", "target_axis_id", "canonical_name", "loss_group"]],
        on="axis_index", validate="many_to_one",
    )
    predictions["absolute_error_g_per_100g"] = np.abs(predictions["prediction_g_per_100g"] - predictions["target_g_per_100g"])
    predictions["absolute_log1p_error"] = np.abs(
        np.log1p(predictions["prediction_g_per_100g"]) - np.log1p(predictions["target_g_per_100g"])
    )
    axis_metrics = predictions.groupby("target_axis_id", as_index=False).agg(
        log_mae=("absolute_log1p_error", "mean"),
        log_mse=("absolute_log1p_error", lambda x: float((x * x).mean())),
        raw_mae_g_per_100g=("absolute_error_g_per_100g", "mean"),
        labels=("target_g_per_100g", "size"),
    )
    summary = {
        "prediction_records": int(len(predictions)),
        "axes_scored": int(axis_metrics["target_axis_id"].nunique()),
        "macro_axis_log_mae": float(axis_metrics["log_mae"].mean()),
        "macro_axis_log_rmse": float(math.sqrt(axis_metrics["log_mse"].mean())),
        "macro_axis_raw_mae_g_per_100g": float(axis_metrics["raw_mae_g_per_100g"].mean()),
        "positive_classification_accuracy": float(
            ((predictions["positive_probability"] >= 0.5).astype(int) == predictions["target_positive"]).mean()
        ),
    }
    if predictions["target_positive"].nunique() == 2:
        summary["positive_classification_auroc"] = float(
            roc_auc_score(predictions["target_positive"], predictions["positive_probability"])
        )
    return predictions, summary


def make_loader(dataset: Dataset, embeddings: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=lambda rows: collate(rows, embeddings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--loss-mode", choices=("group_balanced", "all_axis"), default=Config.loss_mode)
    parser.add_argument("--text-only-probability", type=float, default=Config.text_only_probability)
    parser.add_argument("--source-dropout", type=float, default=Config.source_dropout)
    parser.add_argument("--amount-loss-weight", type=float, default=Config.amount_loss_weight)
    parser.add_argument("--axis-residual-rank", type=int, default=Config.axis_residual_rank)
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Write validation-only artifacts and leave the complete held-out panel unopened.",
    )
    args = parser.parse_args()
    data_dir, split_dir, output_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    config = Config(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        patience=args.patience, loss_mode=args.loss_mode,
        text_only_probability=args.text_only_probability, source_dropout=args.source_dropout,
        amount_loss_weight=args.amount_loss_weight, axis_residual_rank=args.axis_residual_rank,
    )
    if not 0.0 <= config.text_only_probability <= 1.0:
        raise ValueError("--text-only-probability must be in [0, 1].")
    if not 0.0 <= config.source_dropout <= 1.0:
        raise ValueError("--source-dropout must be in [0, 1].")
    if config.amount_loss_weight < 0.0:
        raise ValueError("--amount-loss-weight must be non-negative.")
    if config.axis_residual_rank < 1:
        raise ValueError("--axis-residual-rank must be at least 1.")
    set_seed(config.seed)
    corpus = SourceNativeCorpus(data_dir, split_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_positions, valid_positions, test_positions = (
        corpus.positions("train"), corpus.positions("validation"), corpus.positions("test_complete_axis_panel")
    )
    train_valid_profiles = corpus.profiles.iloc[sorted(train_positions + valid_positions)].copy()
    ids, train_valid_embeddings = _encode_texts(train_valid_profiles, config, output_dir / "text_cache", "train_validation")
    embedding_by_id = dict(zip(ids, train_valid_embeddings))
    text_embeddings = np.zeros((len(corpus.profiles), train_valid_embeddings.shape[1]), dtype=np.float32)
    for profile_index in train_positions + valid_positions:
        text_embeddings[profile_index] = embedding_by_id[corpus.profiles.iloc[profile_index].profile_id]

    device = device_for_training()
    source_count = int(corpus.profiles["source_index"].max()) + 1
    model = SourceAwareFoodNutriGPT(text_embeddings.shape[1], len(corpus.axes), source_count, config).to(device)
    train_dataset = MaskedProfileDataset(corpus, train_positions, config, training=True)
    valid_dataset = MaskedProfileDataset(corpus, valid_positions, config, training=False, force_unknown_source=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    history: list[dict[str, float | int]] = []
    best_loss, stale = math.inf, 0
    checkpoint = output_dir / "validation_selected_joint_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_dataset.set_epoch(epoch)
        valid_dataset.set_epoch(0)
        train_metrics = run_epoch(model, make_loader(train_dataset, text_embeddings, config.batch_size, True), config, device, optimizer)
        valid_metrics = run_epoch(model, make_loader(valid_dataset, text_embeddings, config.batch_size, False), config, device)
        history.append({
            "epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in valid_metrics.items()},
        })
        print(
            f"epoch {epoch:02d}: train={train_metrics['joint']:.5f}; "
            f"validation={valid_metrics['joint']:.5f} "
            f"(nutrition={valid_metrics['nutrition']:.5f}, metabolome={valid_metrics['metabolome']:.5f})"
        )
        if valid_metrics["joint"] < best_loss:
            best_loss, stale = valid_metrics["joint"], 0
            torch.save({
                "model_state": model.state_dict(), "config": asdict(config),
                "validation_joint_hurdle_loss": best_loss,
                "training_mode": f"single_stage_source_aware_{config.loss_mode}_hurdle",
            }, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
        scheduler.step()
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    valid_dataset.set_epoch(0)
    validation_predictions, validation_metrics = evaluate(
        model, make_loader(valid_dataset, text_embeddings, config.batch_size, False), corpus, device,
    )
    validation_predictions.to_csv(output_dir / "validation_source_unknown_predictions.csv", index=False)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**asdict(config), "data_dir": str(data_dir), "split_dir": str(split_dir)}, handle, indent=2)
    summary = {
        "training_mode": f"single_stage_source_aware_{config.loss_mode}_hurdle",
        "checkpoint": str(checkpoint),
        "best_validation_joint_hurdle_loss": best_loss,
        "validation_source_policy": "SOURCE_UNKNOWN for every validation profile.",
        "validation_metrics": validation_metrics,
        "complete_test_opened": not args.skip_test,
    }
    if not args.skip_test:
        # Complete-test text is encoded only after checkpoint selection and is
        # never used in fitting. It remains unopened during ablations.
        test_profiles = corpus.profiles.iloc[test_positions].copy()
        ids, test_embeddings = _encode_texts(test_profiles, config, output_dir / "text_cache", "complete_test_after_selection")
        test_embedding_by_id = dict(zip(ids, test_embeddings))
        for profile_index in test_positions:
            text_embeddings[profile_index] = test_embedding_by_id[corpus.profiles.iloc[profile_index].profile_id]
        evaluation_axes = corpus.axes.loc[
            corpus.axes["target_axis_id"].isin(corpus.splits["test_axis_ids"]), "axis_index"
        ].to_numpy(dtype=np.int64)
        test_dataset = MaskedProfileDataset(
            corpus, test_positions, config, training=False, evaluation_axes=evaluation_axes,
            force_unknown_source=True, cover_all_evaluation_families=True,
        )
        test_dataset.set_epoch(0)
        predictions, test_metrics = evaluate(
            model, make_loader(test_dataset, text_embeddings, config.batch_size, False), corpus, device,
        )
        predictions.to_csv(output_dir / "complete_axis_panel_source_unknown_predictions.csv", index=False)
        summary.update({
            "test_source_policy": "SOURCE_UNKNOWN for every complete-test profile; Foundation source embedding never appears in training.",
            "test_protocol": "Leave-one-mask-family-out across the complete 187-axis panel.",
            "test_metrics": test_metrics,
        })
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

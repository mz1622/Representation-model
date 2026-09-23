#!/usr/bin/env python3
"""Train FoodNutriGPT v9 with source-calibrated supervision and source-free inference.

V9 intentionally reuses the immutable V8 source-native corpus and grouped
splits.  A source identifier is *never* inserted into the Transformer input.
During training only, a zero-centred source-by-axis residual may explain
source-specific measurement conventions.  The deployable prediction is always
the source-free base head.  Each exact-name-candidate / axis / source receives
equal total reconstruction weight, so duplicated observations cannot multiply
supervision.

The default command evaluates the grouped validation split only.  It does not
open the frozen complete test panel:

    python scripts/train_global_foodnutrigpt_v9_source_calibrated.py --skip-test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_global_foodnutrigpt_v8_single_stage as v8  # noqa: E402


DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_validation"


@dataclass
class Config:
    seed: int = 20260922
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_batch_size: int = 64
    batch_size: int = 64
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 3
    feedforward_dim: int = 768
    dropout: float = 0.15
    axis_residual_rank: int = 16
    max_tokens: int = 256
    mask_ratio: float = 0.30
    zero_only_family_mask_ratio: float = 0.10
    text_only_probability: float = 0.0
    source_dropout: float = 0.0
    epochs: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    amount_loss_weight: float = 1.0
    source_free_loss_weight: float = 1.0
    source_calibrated_loss_weight: float = 1.0
    source_residual_l2: float = 1e-4


class SourceEqualizedCorpus(v8.SourceNativeCorpus):
    """V8 corpus with equal source support per exact-name candidate cell."""

    def __init__(self, data_dir: Path, split_dir: Path):
        super().__init__(data_dir, split_dir)
        tokens = pd.read_csv(data_dir / "source_native_axis_tokens.csv.gz", low_memory=False)
        tokens = tokens[tokens["profile_id"].isin(self.profile_index)].copy()
        tokens["profile_index"] = tokens["profile_id"].map(self.profile_index).astype(np.int64)
        tokens = tokens.sort_values(["profile_index", "axis_index", "measurement_id"], kind="stable").reset_index(drop=True)
        if not np.array_equal(tokens["axis_index"].to_numpy(dtype=np.int64), self.axis):
            raise AssertionError("V9 token alignment differs from the immutable V8 corpus.")

        group_axis = ["exact_name_group_id", "axis_index"]
        source_count = tokens.groupby(group_axis)["source_key"].transform("nunique").to_numpy(dtype=np.float32)
        profiles_per_source = tokens.groupby(group_axis + ["source_key"])["profile_id"].transform("nunique").to_numpy(dtype=np.float32)
        observations_per_profile = tokens["within_profile_axis_observation_count"].to_numpy(dtype=np.float32)
        if np.any(source_count < 1) or np.any(profiles_per_source < 1) or np.any(observations_per_profile < 1):
            raise AssertionError("V9 found an invalid source-equalization denominator.")
        self.cell_weight = 1.0 / (source_count * profiles_per_source * observations_per_profile)
        self.equalization_summary = {
            "policy": "equal total weight per exact-name-candidate / axis / independent source",
            "token_weight": "1 / (source_count_in_candidate_axis * profiles_in_candidate_axis_source * observations_in_profile_axis)",
            "source_native_tokens": int(len(tokens)),
            "cross_source_candidate_axis_cells": int((source_count > 1).sum()),
        }
        train_sources = sorted({int(self.source[position]) for position in self.positions("train")})
        if not train_sources or 0 in train_sources:
            raise AssertionError("Train source IDs must be non-empty and exclude SOURCE_UNKNOWN (0).")
        self.train_source_indices = np.asarray(train_sources, dtype=np.int64)


class SourceCalibratedFoodNutriGPT(nn.Module):
    """Set Transformer whose encoder is invariant to database source.

    The two residual tables are not encoder inputs.  They are consulted only
    when calculating the training-only calibrated objective.
    """

    def __init__(self, text_dim: int, axis_count: int, source_count: int, train_source_indices: np.ndarray, config: Config):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
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
        self.presence_head = v8.AxisCalibratedHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        self.amount_head = v8.AxisCalibratedHead(config.d_model, axis_count, config.axis_residual_rank, config.dropout)
        self.source_presence_residual = nn.Embedding(source_count, axis_count)
        self.source_amount_residual = nn.Embedding(source_count, axis_count)
        self.register_buffer("train_source_indices", torch.as_tensor(train_source_indices, dtype=torch.long))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.zeros_(self.source_presence_residual.weight)
        nn.init.zeros_(self.source_amount_residual.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        value_token = self.value_encoder(batch["value"].unsqueeze(-1))
        value_token = torch.where(batch["masked"].unsqueeze(-1), self.mask_value.view(1, 1, -1), value_token)
        axis_token = self.axis_embedding(batch["axis"]) + value_token
        sequence = torch.cat([
            self.cls.expand(len(batch["axis"]), -1, -1),
            self.text_projection(batch["text"]).unsqueeze(1),
            axis_token,
        ], dim=1)
        padding = torch.cat([
            torch.zeros((len(batch["axis"]), 2), dtype=torch.bool, device=batch["axis"].device),
            ~batch["valid"],
        ], dim=1)
        hidden = self.encoder(sequence, src_key_padding_mask=padding)[:, 2:]
        return {
            "positive_logit": self.presence_head(hidden, batch["axis"]),
            "amount_normalized": self.amount_head(hidden, batch["axis"]),
        }

    def _centred_source_offsets(self, table: nn.Embedding) -> torch.Tensor:
        """Return offsets centred over sources available in the training split.

        SOURCE_UNKNOWN and sources held out of train receive exactly zero.  The
        source-free head is therefore the central prediction, not a hidden
        source-specific value.
        """
        offsets = torch.zeros_like(table.weight)
        learned = table.weight[self.train_source_indices]
        centred = learned - learned.mean(dim=0, keepdim=True)
        return offsets.index_copy(0, self.train_source_indices, centred)

    def calibrated_outputs(self, base: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = batch["source"].unsqueeze(1).expand_as(batch["axis"])
        axis = batch["axis"]
        return {
            "positive_logit": base["positive_logit"] + self._centred_source_offsets(self.source_presence_residual)[source, axis],
            "amount_normalized": base["amount_normalized"] + self._centred_source_offsets(self.source_amount_residual)[source, axis],
        }

    def source_residual_penalty(self) -> torch.Tensor:
        presence = self.source_presence_residual.weight[self.train_source_indices]
        amount = self.source_amount_residual.weight[self.train_source_indices]
        return presence.square().mean() + amount.square().mean()


def loss_for_all_axes(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], amount_loss_weight: float) -> tuple[torch.Tensor, int]:
    return v8.macro_hurdle_loss(outputs, batch, None, amount_loss_weight)


def joint_loss(
    model: SourceCalibratedFoodNutriGPT,
    batch: dict[str, torch.Tensor],
    config: Config,
    *,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = model(batch)
    source_free, source_free_axes = loss_for_all_axes(base, batch, config.amount_loss_weight)
    if not source_free_axes:
        return base["positive_logit"].sum() * 0.0, {"joint": 0.0, "source_free": 0.0, "source_calibrated": float("nan")}
    if not training:
        return source_free, {
            "joint": float(source_free.detach().cpu()),
            "source_free": float(source_free.detach().cpu()),
            "source_calibrated": float("nan"),
        }
    calibrated, calibrated_axes = loss_for_all_axes(
        model.calibrated_outputs(base, batch), batch, config.amount_loss_weight,
    )
    if calibrated_axes != source_free_axes:
        raise AssertionError("Source-free and calibrated losses must cover identical axes.")
    joint = (
        config.source_free_loss_weight * source_free
        + config.source_calibrated_loss_weight * calibrated
        + config.source_residual_l2 * model.source_residual_penalty()
    )
    return joint, {
        "joint": float(joint.detach().cpu()),
        "source_free": float(source_free.detach().cpu()),
        "source_calibrated": float(calibrated.detach().cpu()),
    }


def run_epoch(
    model: SourceCalibratedFoodNutriGPT,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = {"joint": 0.0, "source_free": 0.0, "source_calibrated": 0.0}
    counts = {key: 0 for key in total}
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training):
            loss, metrics = joint_loss(model, tensors, config, training=training)
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
def evaluate_profile_axis(
    model: SourceCalibratedFoodNutriGPT,
    loader: DataLoader,
    corpus: SourceEqualizedCorpus,
    device: torch.device,
) -> pd.DataFrame:
    """Predict with the source-free base head only."""
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
        predicted_log_amount = torch.clamp(amount_normalized * axis_scale[axis] + axis_median[axis], min=0.0)
        predicted_raw = positive_probability * torch.expm1(predicted_log_amount)
        rows.append(pd.DataFrame({
            "profile_id": corpus.profiles.iloc[profile_index.detach().cpu().tolist()]["profile_id"].tolist(),
            "axis_index": axis.detach().cpu().tolist(),
            "target_g_per_100g": tensors["raw_value"][active].detach().cpu().tolist(),
            "target_positive": tensors["positive"][active].detach().cpu().to(torch.int64).tolist(),
            "positive_probability": positive_probability.detach().cpu().tolist(),
            "prediction_g_per_100g": predicted_raw.detach().cpu().tolist(),
        }))
    predictions = pd.concat(rows, ignore_index=True)
    return predictions.merge(
        corpus.axes[["axis_index", "target_axis_id", "canonical_name", "loss_group"]],
        on="axis_index", validate="many_to_one",
    )


def summarize_source_free_cells(
    predictions: pd.DataFrame,
    corpus: SourceEqualizedCorpus,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Score each exact-name candidate cell with equal source influence.

    Source is used only for offline label aggregation.  It is omitted from the
    released candidate-cell prediction table and never reaches a model input.
    """
    identity = corpus.profiles[["profile_id", "exact_name_group_id", "source_key"]]
    profile_axis = predictions.groupby(
        ["profile_id", "target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        target_g_per_100g=("target_g_per_100g", "median"),
        prediction_g_per_100g=("prediction_g_per_100g", "median"),
        positive_probability=("positive_probability", "median"),
        target_positive=("target_positive", "median"),
    ).merge(identity, on="profile_id", validate="many_to_one")
    source_cells = profile_axis.groupby(
        ["exact_name_group_id", "source_key", "target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        target_g_per_100g=("target_g_per_100g", "median"),
        prediction_g_per_100g=("prediction_g_per_100g", "median"),
        positive_probability=("positive_probability", "median"),
        target_positive=("target_positive", "median"),
        profiles_in_source=("profile_id", "nunique"),
    )
    source_cells["absolute_error_g_per_100g"] = np.abs(
        source_cells["prediction_g_per_100g"] - source_cells["target_g_per_100g"]
    )
    source_cells["absolute_log1p_error"] = np.abs(
        np.log1p(source_cells["prediction_g_per_100g"]) - np.log1p(source_cells["target_g_per_100g"])
    )
    source_cells["squared_log1p_error"] = source_cells["absolute_log1p_error"] ** 2

    candidate_cells = source_cells.groupby(
        ["exact_name_group_id", "target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        target_g_per_100g=("target_g_per_100g", "median"),
        prediction_g_per_100g=("prediction_g_per_100g", "median"),
        positive_probability=("positive_probability", "median"),
        target_positive=("target_positive", "median"),
        source_count=("source_key", "nunique"),
        source_equal_log_mae=("absolute_log1p_error", "mean"),
        source_equal_log_mse=("squared_log1p_error", "mean"),
        source_equal_raw_mae_g_per_100g=("absolute_error_g_per_100g", "mean"),
    )
    # Do not expose source keys in the source-free release table.
    per_axis = candidate_cells.groupby(
        ["target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        candidate_food_cells=("exact_name_group_id", "nunique"),
        log_mae=("source_equal_log_mae", "mean"),
        log_mse=("source_equal_log_mse", "mean"),
        raw_mae_g_per_100g=("source_equal_raw_mae_g_per_100g", "mean"),
    )
    summary = {
        "source_free_candidate_cells": int(len(candidate_cells)),
        "prediction_records_before_candidate_aggregation": int(len(predictions)),
        "axes_scored": int(per_axis["target_axis_id"].nunique()),
        "macro_axis_log_mae": float(per_axis["log_mae"].mean()),
        "macro_axis_log_rmse": float(math.sqrt(per_axis["log_mse"].mean())),
        "macro_axis_raw_mae_g_per_100g": float(per_axis["raw_mae_g_per_100g"].mean()),
    }
    return candidate_cells, per_axis, summary


def make_loader(dataset: Dataset, embeddings: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=lambda rows: v8.collate(rows, embeddings),
    )


def complete_panel_dataset(corpus: SourceEqualizedCorpus, positions: list[int], config: Config) -> v8.MaskedProfileDataset:
    evaluation_axes = corpus.axes.loc[corpus.axes["loss_eligible"], "axis_index"].to_numpy(dtype=np.int64)
    dataset = v8.MaskedProfileDataset(
        corpus, positions, config, training=False, evaluation_axes=evaluation_axes,
        force_unknown_source=True, cover_all_evaluation_families=True,
    )
    dataset.set_epoch(0)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--text-only-probability", type=float, default=Config.text_only_probability)
    parser.add_argument("--mask-ratio", type=float, default=Config.mask_ratio)
    parser.add_argument("--zero-only-family-mask-ratio", type=float, default=Config.zero_only_family_mask_ratio)
    parser.add_argument("--amount-loss-weight", type=float, default=Config.amount_loss_weight)
    parser.add_argument("--source-calibrated-loss-weight", type=float, default=Config.source_calibrated_loss_weight)
    parser.add_argument("--source-residual-l2", type=float, default=Config.source_residual_l2)
    parser.add_argument("--axis-residual-rank", type=int, default=Config.axis_residual_rank)
    parser.add_argument("--d-model", type=int, default=Config.d_model)
    parser.add_argument("--n-heads", type=int, default=Config.n_heads)
    parser.add_argument("--n-layers", type=int, default=Config.n_layers)
    parser.add_argument("--feedforward-dim", type=int, default=Config.feedforward_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--skip-test", action="store_true", help="Do not open the frozen V8 complete test panel.")
    args = parser.parse_args()

    data_dir, split_dir, output_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        # A caller may seed only the immutable train/validation MiniLM cache
        # from a prior run over the identical profiles.  Any model artifact
        # means this is a real existing experiment and must not be overwritten.
        allowed = {
            output_dir / "text_cache" / "train_validation_embeddings.npy",
            output_dir / "text_cache" / "train_validation_profile_ids.csv",
        }
        existing = {path for path in output_dir.rglob("*") if path.is_file()}
        if not existing.issubset(allowed):
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    config = Config(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        patience=args.patience, text_only_probability=args.text_only_probability,
        mask_ratio=args.mask_ratio, zero_only_family_mask_ratio=args.zero_only_family_mask_ratio,
        amount_loss_weight=args.amount_loss_weight,
        source_calibrated_loss_weight=args.source_calibrated_loss_weight,
        source_residual_l2=args.source_residual_l2, axis_residual_rank=args.axis_residual_rank,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        feedforward_dim=args.feedforward_dim, dropout=args.dropout,
    )
    for option, value in {
        "--text-only-probability": config.text_only_probability,
        "--mask-ratio": config.mask_ratio,
        "--zero-only-family-mask-ratio": config.zero_only_family_mask_ratio,
        "--dropout": config.dropout,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{option} must be in [0, 1].")
    if config.amount_loss_weight < 0.0:
        raise ValueError("--amount-loss-weight must be non-negative.")
    if config.source_calibrated_loss_weight < 0.0 or config.source_residual_l2 < 0.0:
        raise ValueError("Source calibration weights must be non-negative.")
    if config.axis_residual_rank < 1 or config.n_layers < 1 or config.feedforward_dim < 1:
        raise ValueError("Model rank, layer count, and feed-forward dimension must be positive.")
    if config.d_model < 1 or config.n_heads < 1 or config.d_model % config.n_heads:
        raise ValueError("--d-model must be positive and divisible by --n-heads.")

    v8.set_seed(config.seed)
    corpus = SourceEqualizedCorpus(data_dir, split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_positions, valid_positions, test_positions = (
        corpus.positions("train"), corpus.positions("validation"), corpus.positions("test_complete_axis_panel"),
    )
    train_valid_profiles = corpus.profiles.iloc[sorted(train_positions + valid_positions)].copy()
    ids, train_valid_embeddings = v8._encode_texts(
        train_valid_profiles, config, output_dir / "text_cache", "train_validation",
    )
    embedding_by_id = dict(zip(ids, train_valid_embeddings))
    text_embeddings = np.zeros((len(corpus.profiles), train_valid_embeddings.shape[1]), dtype=np.float32)
    for profile_index in train_positions + valid_positions:
        text_embeddings[profile_index] = embedding_by_id[corpus.profiles.iloc[profile_index].profile_id]

    device = v8.device_for_training()
    source_count = int(corpus.profiles["source_index"].max()) + 1
    model = SourceCalibratedFoodNutriGPT(
        text_embeddings.shape[1], len(corpus.axes), source_count, corpus.train_source_indices, config,
    ).to(device)
    train_dataset = v8.MaskedProfileDataset(corpus, train_positions, config, training=True)
    valid_dataset = v8.MaskedProfileDataset(corpus, valid_positions, config, training=False, force_unknown_source=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    history: list[dict[str, float | int]] = []
    best_loss, stale = math.inf, 0
    checkpoint = output_dir / "validation_selected_source_free_base_model.pt"
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
        print(f"epoch {epoch:02d}: train={train_metrics['joint']:.5f}; validation source-free={valid_metrics['joint']:.5f}")
        if valid_metrics["joint"] < best_loss:
            best_loss, stale = valid_metrics["joint"], 0
            torch.save({
                "model_state": model.state_dict(), "config": asdict(config),
                "validation_source_free_hurdle_loss": best_loss,
                "training_mode": "single_stage_source_calibrated_encoder_source_free",
            }, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
        scheduler.step()
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])

    validation_complete = complete_panel_dataset(corpus, valid_positions, config)
    validation_profile_predictions = evaluate_profile_axis(
        model, make_loader(validation_complete, text_embeddings, config.batch_size, False), corpus, device,
    )
    validation_cells, validation_axis, validation_metrics = summarize_source_free_cells(validation_profile_predictions, corpus)
    validation_profile_predictions.to_csv(output_dir / "validation_complete_axis_panel_source_free_profile_predictions.csv", index=False)
    validation_cells.to_csv(output_dir / "validation_complete_axis_panel_source_free_candidate_cells.csv", index=False)
    validation_axis.to_csv(output_dir / "validation_complete_axis_panel_source_free_axis_metrics.csv", index=False)

    summary: dict[str, object] = {
        "training_mode": "single_stage_source_calibrated_encoder_source_free",
        "checkpoint": str(checkpoint),
        "best_validation_source_free_hurdle_loss": best_loss,
        "encoder_source_policy": "No source token or source embedding is present in the Transformer input.",
        "training_source_policy": "Source is used only in zero-centred output residuals and never in the source-free base prediction.",
        "evaluation_source_policy": "The validation complete panel uses the source-free base head. Source enters only offline equal-source label aggregation.",
        "equalization": corpus.equalization_summary,
        "validation_complete_panel_protocol": "Leave-one-mask-family-out across every validation loss axis; candidate-food axis scoring gives each observed source equal weight.",
        "validation_metrics": validation_metrics,
        "complete_test_opened": not args.skip_test,
    }
    if not args.skip_test:
        test_profiles = corpus.profiles.iloc[test_positions].copy()
        ids, test_embeddings = v8._encode_texts(test_profiles, config, output_dir / "text_cache", "complete_test_after_selection")
        test_embedding_by_id = dict(zip(ids, test_embeddings))
        for profile_index in test_positions:
            text_embeddings[profile_index] = test_embedding_by_id[corpus.profiles.iloc[profile_index].profile_id]
        test_complete = complete_panel_dataset(corpus, test_positions, config)
        test_profile_predictions = evaluate_profile_axis(
            model, make_loader(test_complete, text_embeddings, config.batch_size, False), corpus, device,
        )
        test_cells, test_axis, test_metrics = summarize_source_free_cells(test_profile_predictions, corpus)
        test_profile_predictions.to_csv(output_dir / "complete_test_source_free_profile_predictions.csv", index=False)
        test_cells.to_csv(output_dir / "complete_test_source_free_candidate_cells.csv", index=False)
        test_axis.to_csv(output_dir / "complete_test_source_free_axis_metrics.csv", index=False)
        summary["test_metrics"] = test_metrics
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**asdict(config), "data_dir": str(data_dir), "split_dir": str(split_dir)}, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Composition-only, source-native masked-value experiment.

Food names are used by the frozen data protocol for grouping and scoring only.
Neither this model nor the three baseline feature builders can receive them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from torch import nn
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Config:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20261005
    rf_trees: int = 200
    rf_leaf: int = 2
    xgb_trees: int = 300
    xgb_depth: int = 6
    knn_neighbors: int = 16
    n_jobs: int = 4


def numeric_only_training_view(data, train_rows=None):
    """Replace legacy name-group-weighted scaling and loss weights.

    The frozen split may still use names to prevent duplicate leakage, and the
    evaluator may use names for aggregation. Pretraining uses only numeric
    observations from the training partition to construct its scale and loss.
    """
    train_rows = data.train if train_rows is None else np.asarray(train_rows, dtype=np.int64)
    if not len(train_rows) or not np.isin(train_rows, data.train).all():
        raise ValueError("normalization rows must be nonempty training profiles")
    scale = np.ones(len(data.axes), dtype=np.float64)
    for axis in range(len(data.axes)):
        positive = data.raw[train_rows, axis]
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if len(positive):
            scale[axis] = float(np.median(positive))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid numeric-only training scale")
    data.scale = scale
    data.values = np.where(data.observed, np.log1p(data.raw / scale), 0).astype(np.float32)
    data.weights = np.zeros_like(data.values, dtype=np.float32)
    data.weights[train_rows] = data.observed[train_rows].astype(np.float32)
    return data


class CompositionSetTransformer(nn.Module):
    """Position-free set encoder over observed feature/value pairs plus [FOOD]."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.axis_count = axis_count
        self.food_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.feedforward_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layers, norm=nn.LayerNorm(config.d_model),
        )
        self.head = nn.Linear(config.d_model, axis_count)
        nn.init.normal_(self.food_token, std=0.02)

    def forward(self, axis: torch.Tensor, value: torch.Tensor,
                padding: torch.Tensor) -> torch.Tensor:
        if axis.shape != value.shape or axis.shape != padding.shape or axis.ndim != 2:
            raise ValueError("axis, value and padding must have equal [batch, tokens] shape")
        if axis.dtype != torch.long or padding.dtype != torch.bool:
            raise TypeError("axis must be long and padding must be bool")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite composition input")
        observed = self.axis_embedding(axis) + self.value_encoder(value.unsqueeze(-1))
        sequence = torch.cat((self.food_token.expand(len(axis), -1, -1), observed), dim=1)
        mask = torch.cat((torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device), padding), dim=1)
        food = self.encoder(sequence, src_key_padding_mask=mask)[:, 0]
        return self.head(food)


def family_tasks(data, train_rows=None):
    """Every observed training target occurs in exactly one family task."""
    eligible = np.zeros(len(data.axes), dtype=bool)
    eligible[data.targets] = True
    train_rows = data.train if train_rows is None else np.asarray(train_rows, dtype=np.int64)
    rows, families = [], []
    for family in np.unique(data.families[data.targets]):
        selected = train_rows[(data.observed[train_rows] & eligible &
                               (data.families == family)).any(axis=1)]
        rows.append(selected)
        families.append(np.repeat(str(family), len(selected)))
    return np.concatenate(rows), np.concatenate(families)


def visible_context(data, rows, family):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) != len(family):
        raise ValueError("rows/family mismatch")
    family = np.asarray(family, dtype=str)
    if not np.isin(family, np.unique(data.families)).all():
        raise ValueError("unknown mask family")
    return data.observed[rows] & (data.families[None, :] != family[:, None])


def set_batch(data, rows, family, device, *, with_targets=False):
    """Pack only visible observations. Missing and held-out cells have no token."""
    visible = visible_context(data, rows, family)
    lengths = visible.sum(axis=1)
    width = max(1, int(lengths.max(initial=0)))
    axis = np.zeros((len(rows), width), dtype=np.int64)
    value = np.zeros((len(rows), width), dtype=np.float32)
    padding = np.ones((len(rows), width), dtype=bool)
    for i in range(len(rows)):
        found = np.flatnonzero(visible[i])
        n = len(found)
        axis[i, :n] = found
        value[i, :n] = data.values[rows[i], found]
        padding[i, :n] = False
    result = (torch.from_numpy(axis).to(device), torch.from_numpy(value).to(device),
              torch.from_numpy(padding).to(device))
    if not with_targets:
        return result
    eligible = np.zeros(len(data.axes), dtype=bool)
    eligible[data.targets] = True
    target = (data.observed[rows] & eligible[None, :] &
              (data.families[None, :] == np.asarray(family)[:, None]))
    return result + (torch.from_numpy(data.values[rows]).to(device),
                     torch.from_numpy(target).to(device),
                     torch.from_numpy(data.weights[rows]).to(device))


def dense_features(data, rows, family):
    """The same visible cells as set_batch, in a fixed axis/value + mask grid."""
    visible = visible_context(data, rows, family)
    x = np.concatenate((np.where(visible, data.values[rows], 0),
                        visible.astype(np.float32)), axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite baseline input")
    return x


def inverse_values(transformed, scale):
    transformed = np.asarray(transformed, dtype=np.float64)
    if not np.isfinite(transformed).all() or not np.isfinite(scale).all():
        raise FloatingPointError("nonfinite prediction or scale")
    with np.errstate(over="raise", invalid="raise"):
        raw = np.expm1(np.maximum(transformed, 0)) * scale
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise FloatingPointError("invalid inverse-transformed prediction")
    return raw


def make_baseline(method: str, axis: int, config: Config):
    seed = config.seed + int(axis)
    if method == "rf":
        return RandomForestRegressor(
            n_estimators=config.rf_trees, min_samples_leaf=config.rf_leaf,
            max_features=0.5, criterion="squared_error", bootstrap=True,
            random_state=seed, n_jobs=config.n_jobs,
        )
    if method == "xgb":
        return XGBRegressor(
            n_estimators=config.xgb_trees, max_depth=config.xgb_depth,
            learning_rate=0.05, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            objective="reg:squarederror", random_state=seed, n_jobs=config.n_jobs,
        )
    if method == "knn":
        return NearestNeighbors(n_neighbors=config.knn_neighbors, metric="euclidean",
                                algorithm="auto", n_jobs=config.n_jobs)
    raise ValueError(method)


def baseline_predictions(data, method, axes, config, *, train_rows=None, validation_rows=None):
    train_rows = data.train if train_rows is None else np.asarray(train_rows, dtype=np.int64)
    jobs = data.jobs[data.jobs.axis_index.isin(axes)]
    if validation_rows is not None:
        jobs = jobs[jobs.profile_index.isin(validation_rows)]
    pieces, fits = [], []
    for axis in axes:
        train = train_rows[data.observed[train_rows, axis]]
        subset = jobs[jobs.axis_index.eq(axis)]
        rows = subset.profile_index.to_numpy(dtype=np.int64)
        if not len(train) or not len(rows):
            raise ValueError(f"Axis {axis} lacks training or validation support")
        family = str(data.families[axis])
        x_train = dense_features(data, train, np.repeat(family, len(train)))
        x_valid = dense_features(data, rows, np.repeat(family, len(rows)))
        y = data.values[train, axis]
        weight = data.weights[train, axis]
        model = make_baseline(method, axis, config)
        if method == "knn":
            model.set_params(n_neighbors=min(config.knn_neighbors, len(train)))
            model.fit(x_train)
            distance, neighbor = model.kneighbors(x_valid)
            importance = weight[neighbor] / (distance + 1e-3)
            predicted = (importance * y[neighbor]).sum(axis=1) / importance.sum(axis=1)
        else:
            model.fit(x_train, y, sample_weight=weight / weight.mean())
            if method == "rf":
                model.n_jobs = 1  # deterministic tree accumulation at inference
            predicted = model.predict(x_valid)
        raw = inverse_values(predicted, data.scale[axis])
        pieces.append(pd.DataFrame({"profile_index": rows, "axis_index": int(axis), "prediction": raw}))
        fits.append({"axis_index": int(axis), "train_profiles": len(train),
                     "validation_jobs": len(rows), "feature_count": x_train.shape[1]})
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(fits)


def score_subset(data, predictions, axes, *, validation_rows=None):
    """Frozen evaluator's candidate/source/axis aggregation with numeric-only scale."""
    truth = data.jobs[data.jobs.axis_index.isin(axes)][["profile_index", "axis_index", "target"]]
    if validation_rows is not None:
        truth = truth[truth.profile_index.isin(validation_rows)]
    if predictions.duplicated(["profile_index", "axis_index"]).any():
        raise ValueError("duplicate predictions")
    scored = truth.merge(predictions, on=["profile_index", "axis_index"],
                         how="outer", validate="one_to_one", indicator=True)
    if not scored._merge.eq("both").all() or not np.isfinite(scored.prediction).all():
        raise ValueError("predictions do not exactly cover selected validation jobs")
    scored = scored.merge(data.profiles[["profile_index", "exact_name_group_id", "source_key"]],
                          on="profile_index", validate="many_to_one")
    source = scored.groupby(["exact_name_group_id", "axis_index", "source_key"],
                            as_index=False).agg(target=("target", "median"),
                                                 prediction=("prediction", "median"))
    scale = data.scale[source.axis_index.to_numpy()]
    source["scaled_log_mae"] = np.abs(np.log1p(source.prediction / scale) -
                                      np.log1p(source.target / scale))
    source["raw_mae"] = np.abs(source.prediction - source.target)
    candidate = source.groupby(["exact_name_group_id", "axis_index"],
                               as_index=False)[["scaled_log_mae", "raw_mae"]].mean()
    per_axis = candidate.groupby("axis_index", as_index=False)[["scaled_log_mae", "raw_mae"]].mean()
    per_axis = per_axis.merge(data.axes[["axis_index", "canonical_name", "loss_group"]],
                              on="axis_index", validate="one_to_one")
    metrics = {}
    for group in ("nutrition", "food_metabolome", "all"):
        part = per_axis if group == "all" else per_axis[per_axis.loss_group.eq(group)]
        metrics[group] = {"axes": len(part), "scaled_log_mae": float(part.scaled_log_mae.mean())
                          if len(part) else None, "raw_mae": float(part.raw_mae.mean()) if len(part) else None}
    return metrics, per_axis

"""Shared training and evaluation primitives for dependency-mask benchmark v1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy
from foodcomp.research_no_foodname_v1 import Config, inverse_values
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer
from foodcomp.research_r0 import canonical_cells


def dependency_tasks(data, rows, targets=None):
    """Return one task for every observed row/target cell."""
    rows = np.asarray(rows, dtype=np.int64)
    targets = np.asarray(data.targets if targets is None else targets, dtype=np.int64)
    local_row, local_axis = np.where(data.observed[np.ix_(rows, targets)])
    return rows[local_row], targets[local_axis]


def visible_context(data, rows, target_axes, policy: DependencyMaskPolicy):
    rows = np.asarray(rows, dtype=np.int64)
    target_axes = np.asarray(target_axes, dtype=np.int64)
    if rows.ndim != 1 or target_axes.ndim != 1 or len(rows) != len(target_axes):
        raise ValueError("rows and target_axes must be aligned vectors")
    visible = data.observed[rows].copy()
    visible &= ~policy.mask_matrix(target_axes)
    return visible


def dense_features(data, rows, target_axes, policy):
    visible = visible_context(data, rows, target_axes, policy)
    x = np.concatenate(
        (np.where(visible, data.values[rows], 0.0), visible.astype(np.float32)), axis=1
    ).astype(np.float32)
    if not np.isfinite(x).all():
        raise FloatingPointError("nonfinite baseline feature")
    return x


def masked_axis_batch(data, rows, target_axes, policy, device, *, with_targets=False):
    """Pack target-specific context and exactly one masked query per task."""
    rows = np.asarray(rows, dtype=np.int64)
    target_axes = np.asarray(target_axes, dtype=np.int64)
    if not len(rows):
        raise ValueError("empty batch")
    visible = visible_context(data, rows, target_axes, policy)
    width = max(1, int(visible.sum(axis=1).max(initial=0)))
    axis = np.zeros((len(rows), width), dtype=np.int64)
    value = np.zeros((len(rows), width), dtype=np.float32)
    padding = np.ones((len(rows), width), dtype=bool)
    packed_row, packed_axis = np.nonzero(visible)
    packed_position = (np.cumsum(visible, axis=1) - 1)[packed_row, packed_axis]
    axis[packed_row, packed_position] = packed_axis
    value[packed_row, packed_position] = data.values[rows[packed_row], packed_axis]
    padding[packed_row, packed_position] = False
    target_axis = target_axes[:, None]
    target_padding = np.zeros((len(rows), 1), dtype=bool)
    result = tuple(torch.from_numpy(x).to(device) for x in (
        axis, value, padding, target_axis, target_padding
    ))
    if not with_targets:
        return result
    label = data.values[rows, target_axes][:, None].astype(np.float32)
    observed = data.observed[rows, target_axes][:, None]
    weight = data.weights[rows, target_axes][:, None].astype(np.float32)
    if not observed.all():
        raise ValueError("training task contains an unobserved label")
    return result + tuple(torch.from_numpy(x).to(device) for x in (label, observed, weight))


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
            objective="reg:squarederror", random_state=seed,
            n_jobs=config.n_jobs,
        )
    if method == "knn":
        return NearestNeighbors(
            n_neighbors=config.knn_neighbors, metric="euclidean",
            algorithm="auto", n_jobs=config.n_jobs,
        )
    raise ValueError(method)


def fit_predict_baseline(data, policy, method, jobs, config, train_rows=None):
    """Fit per-axis estimators and predict an arbitrary profile-axis job table."""
    train_rows = np.asarray(data.train if train_rows is None else train_rows, dtype=np.int64)
    axes = sorted(map(int, jobs.axis_index.unique()))
    pieces, support = [], []
    for number, axis in enumerate(axes, start=1):
        train = train_rows[data.observed[train_rows, axis]]
        part = jobs[jobs.axis_index.eq(axis)]
        rows = part.profile_index.to_numpy(dtype=np.int64)
        if not len(train) or not len(rows):
            raise ValueError(f"axis {axis} lacks train or evaluation support")
        train_axis = np.full(len(train), axis, dtype=np.int64)
        eval_axis = np.full(len(rows), axis, dtype=np.int64)
        x_train = dense_features(data, train, train_axis, policy)
        x_eval = dense_features(data, rows, eval_axis, policy)
        y = data.values[train, axis]
        weight = data.weights[train, axis]
        model = make_baseline(method, axis, config)
        if method == "knn":
            model.set_params(n_neighbors=min(config.knn_neighbors, len(train)))
            model.fit(x_train)
            distance, neighbor = model.kneighbors(x_eval)
            importance = weight[neighbor] / (distance + 1e-3)
            prediction = (importance * y[neighbor]).sum(1) / importance.sum(1)
        else:
            model.fit(x_train, y, sample_weight=weight / weight.mean())
            if method == "rf":
                model.n_jobs = 1
            prediction = model.predict(x_eval)
        pieces.append(pd.DataFrame({
            "profile_index": rows,
            "axis_index": axis,
            "prediction": inverse_values(prediction, data.scale[axis]),
        }))
        support.append({
            "axis_index": axis, "train_profiles": len(train),
            "evaluation_jobs": len(rows), "feature_count": x_train.shape[1],
            "mask_size": len(policy.mask_indices(axis)),
        })
        print(f"{method} axis {number}/{len(axes)} complete", flush=True)
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(support)


def score_profile_axis_macro(data, jobs, predictions):
    """Direct profile-axis errors, then equal-weight mean over target axes."""
    truth = jobs[["profile_index", "axis_index", "target"]].copy()
    if predictions.duplicated(["profile_index", "axis_index"]).any():
        raise ValueError("duplicate predictions")
    scored = truth.merge(
        predictions, on=["profile_index", "axis_index"], how="outer",
        validate="one_to_one", indicator=True,
    )
    if not scored._merge.eq("both").all() or not np.isfinite(scored.prediction).all():
        raise ValueError("predictions do not exactly cover evaluation jobs")
    scale = data.scale[scored.axis_index.to_numpy(dtype=np.int64)]
    delta = (
        np.log1p(np.maximum(scored.prediction.to_numpy(float), 0) / scale)
        - np.log1p(scored.target.to_numpy(float) / scale)
    )
    scored["scaled_log_abs_error"] = np.abs(delta)
    scored["scaled_log_squared_error"] = np.square(delta)
    scored["raw_abs_error"] = np.abs(scored.prediction - scored.target)
    per_axis = scored.groupby("axis_index", as_index=False).agg(
        jobs=("target", "size"),
        scaled_log_mae=("scaled_log_abs_error", "mean"),
        scaled_log_mse=("scaled_log_squared_error", "mean"),
        raw_mae=("raw_abs_error", "mean"),
    )
    per_axis["scaled_log_rmse"] = np.sqrt(per_axis.pop("scaled_log_mse"))
    per_axis = per_axis.merge(
        data.axes[["axis_index", "canonical_name"]], on="axis_index",
        validate="one_to_one",
    )
    metrics = {
        "axes": int(len(per_axis)),
        "jobs": int(len(scored)),
        "macro_scaled_log_mae": float(per_axis.scaled_log_mae.mean()),
        "macro_scaled_log_rmse": float(per_axis.scaled_log_rmse.mean()),
        "macro_raw_mae": float(per_axis.raw_mae.mean()),
    }
    return metrics, per_axis, scored.drop(columns="_merge")


class FoodPromptCceTransformer(AxisPairBiasMaskedAxisTransformer):
    """V4 core with V5's detached two-pass prompt; CCE adds no parameters."""

    def forward_details_with_food_prompt(
        self, axis, value, padding, target_axis, target_padding, food_prompt
    ):
        if food_prompt.shape != (len(axis), self.context_token.shape[-1]):
            raise ValueError("invalid FOOD prompt shape")
        visible = self.axis_embedding(axis) + self.value_encoder(value.unsqueeze(-1))
        target = self.axis_embedding(target_axis) + self.mask_value
        tokens = torch.cat((food_prompt.unsqueeze(1), visible, target), dim=1)
        sequence_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding, target_padding,
        ), dim=1)
        sequence_axis = torch.cat((
            torch.full((len(axis), 1), self.axis_count, dtype=torch.long, device=axis.device),
            axis, target_axis,
        ), dim=1)
        for block in self.blocks:
            tokens = block(tokens, sequence_padding, sequence_axis)
        tokens = self.final_norm(tokens)
        hidden = tokens[:, 1 + axis.shape[1]:]
        prediction = self.value_decoder(hidden).squeeze(-1).masked_fill(target_padding, 0.0)
        return prediction, tokens[:, 0], hidden

    def forward_prompt_details(self, axis, value, padding, target_axis, target_padding):
        first, food, first_hidden = self.forward_details(
            axis, value, padding, target_axis, target_padding
        )
        second, second_food, second_hidden = self.forward_details_with_food_prompt(
            axis, value, padding, target_axis, target_padding, food.detach()
        )
        return first, second, food, second_food, first_hidden, second_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_prompt_details(
            axis, value, padding, target_axis, target_padding
        )[1]


def multi_positive_cce_loss(first, second, profile_index, temperature=0.5):
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2:
        raise ValueError("CCE views must be aligned [batch, hidden] with batch >= 2")
    if temperature <= 0:
        raise ValueError("CCE temperature must be positive")
    first = torch.nn.functional.normalize(first, dim=1)
    second = torch.nn.functional.normalize(second, dim=1)
    logits = first @ second.T / temperature
    positive = profile_index[:, None].eq(profile_index[None, :])
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def axis_balanced_loss(
    prediction, label, axis, totals, axis_count, task_count, batch_size
):
    """Mini-batch estimator of mean per-axis normalized absolute error."""
    weights = 1.0 / totals[axis]
    multiplier = task_count / (batch_size * axis_count)
    return (weights * (prediction - label).abs()).sum() * multiplier


def attach_complete_test(data, frozen_data_dir: Path, split_path: Path):
    """Append the frozen 659-profile complete-test panel without using it in fit."""
    frozen_data_dir, split_path = Path(frozen_data_dir), Path(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split["test_complete_axis_panel"])
    if len(test_ids) != 659:
        raise ValueError(f"expected 659 complete-test profiles, found {len(test_ids)}")
    expected_ids = set(data.axes.loc[data.axes.loss_eligible.astype(bool), "target_axis_id"])
    if set(split["test_axis_ids"]) != expected_ids:
        raise ValueError("complete-test axes differ from active targets")
    profiles = pd.read_csv(frozen_data_dir / "food_profiles.csv.gz", low_memory=False)
    profiles = profiles[profiles.profile_id.isin(test_ids)].sort_values("profile_id").reset_index(drop=True)
    if len(profiles) != 659 or not profiles.partition.eq("test_complete_axis_panel").all():
        raise ValueError("invalid complete-test profile table")
    if set(profiles.exact_name_group_id) & set(data.profiles.exact_name_group_id):
        raise ValueError("complete-test group overlaps development data")
    offset = len(data.profiles)
    profiles["profile_index"] = offset + np.arange(len(profiles), dtype=np.int64)
    columns = [
        "profile_id", "axis_index", "measurement_id", "component_observation_id",
        "raw_unit", "raw_basis", "source_value_origin", "value_status",
        "normalized_value_g_per_100g", "mask_family",
    ]
    parts = []
    for chunk in pd.read_csv(
        frozen_data_dir / "source_native_axis_tokens.csv.gz", usecols=columns,
        chunksize=250_000, low_memory=False,
    ):
        selected = chunk[chunk.profile_id.isin(test_ids)]
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError("complete-test profiles contain no tokens")
    tokens = pd.concat(parts, ignore_index=True)
    if not tokens.value_status.isin(["observed", "explicit_zero"]).all():
        raise ValueError("invalid complete-test value status")
    cells = canonical_cells(tokens, profiles)
    primary = cells[~cells.quarantined]
    raw = np.full((len(profiles), len(data.axes)), np.nan, dtype=np.float64)
    local = primary.profile_index.to_numpy(dtype=np.int64) - offset
    raw[local, primary.axis_index.to_numpy(dtype=np.int64)] = primary.value
    observed = np.isfinite(raw)
    values = np.where(observed, np.log1p(raw / data.scale), 0).astype(np.float32)
    data.raw = np.concatenate((data.raw, raw))
    data.observed = np.concatenate((data.observed, observed))
    data.values = np.concatenate((data.values, values))
    data.weights = np.concatenate((data.weights, np.zeros_like(values, dtype=np.float32)))
    data.profiles = pd.concat((data.profiles, profiles), ignore_index=True, sort=False)
    target_flag = np.zeros(len(data.axes), dtype=bool)
    target_flag[np.asarray(data.targets, dtype=np.int64)] = True
    row, axis = np.where(observed & target_flag[None, :])
    jobs = pd.DataFrame({
        "profile_index": offset + row,
        "axis_index": axis,
        "target": raw[row, axis],
    })
    if len(jobs) != 19_690 or jobs.axis_index.nunique() != 187:
        raise ValueError("unexpected complete-test job panel")
    audit = {
        "profiles": 659, "jobs": int(len(jobs)), "axes": int(jobs.axis_index.nunique()),
        "minimum_axis_support": int(jobs.groupby("axis_index").size().min()),
        "maximum_axis_support": int(jobs.groupby("axis_index").size().max()),
        "quarantined_cells": int(cells.quarantined.sum()),
    }
    return profiles.profile_index.to_numpy(dtype=np.int64), jobs, audit


@torch.no_grad()
def predict_transformer(model, data, policy, jobs, device, batch_size, two_pass):
    model.eval()
    parts = []
    rows = jobs.profile_index.to_numpy(dtype=np.int64)
    axes = jobs.axis_index.to_numpy(dtype=np.int64)
    for start in range(0, len(jobs), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_axes = axes[start:start + batch_size]
        packed = masked_axis_batch(data, batch_rows, batch_axes, policy, device)
        if two_pass:
            prediction = model.forward_prompt_details(*packed)[1]
        else:
            prediction = model(*packed)
        raw = inverse_values(
            prediction[:, 0].cpu().numpy(), data.scale[batch_axes]
        )
        parts.append(pd.DataFrame({
            "profile_index": batch_rows,
            "axis_index": batch_axes,
            "prediction": raw,
        }))
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "dependency_tasks", "visible_context", "dense_features", "masked_axis_batch",
    "fit_predict_baseline", "score_profile_axis_macro", "FoodPromptCceTransformer",
    "multi_positive_cce_loss", "axis_balanced_loss", "predict_transformer",
    "attach_complete_test",
]

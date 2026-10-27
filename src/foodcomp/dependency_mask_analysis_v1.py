"""Analysis helpers for the frozen dependency-mask v1 checkpoints.

The functions in this module keep the evaluation mask identical to the mask
used in training: for every target, the target plus its registered ancestors
and descendants are removed before optional analysis-specific thinning or
perturbation is applied.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy, VERSION as MASK_VERSION
from foodcomp.research_dependency_mask_v1 import FoodPromptCceTransformer
from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_r0 import digest


def load_dependency_model(
    checkpoint_path: Path,
    data,
    policy_manifest: Path,
    device: torch.device,
):
    """Load one V4/V5/CCE checkpoint and enforce its frozen input policy."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("mask_policy_version") != MASK_VERSION:
        raise ValueError("checkpoint dependency-mask version mismatch")
    if checkpoint.get("mask_policy_sha256") != digest(Path(policy_manifest)):
        raise ValueError("checkpoint dependency-mask manifest mismatch")
    if checkpoint.get("data_sha256") != digest(data.root / "manifest.json"):
        raise ValueError("checkpoint data manifest mismatch")
    for key in ("food_name_model_input", "source_model_input", "family_label_model_input"):
        if checkpoint.get(key) is not False:
            raise ValueError(f"unexpected model input policy: {key}")
    if checkpoint.get("legacy_mask_family_used_for_masking") is not False:
        raise ValueError("checkpoint used the legacy family mask")
    config = Config(**checkpoint["config"])
    model = FoodPromptCceTransformer(len(data.axes), config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint, config


def pack_visible(
    data,
    rows: np.ndarray,
    target_axes: np.ndarray,
    visible: np.ndarray,
    device: torch.device,
    normalized_values: np.ndarray | None = None,
):
    """Pack an explicit visible mask and optional normalized value override."""
    rows = np.asarray(rows, dtype=np.int64)
    target_axes = np.asarray(target_axes, dtype=np.int64)
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != (len(rows), len(data.axes)):
        raise ValueError("visible mask has the wrong shape")
    if np.any(visible & ~data.observed[rows]):
        raise ValueError("analysis made an unobserved value visible")
    values = data.values[rows] if normalized_values is None else np.asarray(normalized_values)
    if values.shape != visible.shape:
        raise ValueError("normalized value override has the wrong shape")
    width = max(1, int(visible.sum(axis=1).max(initial=0)))
    axis = np.zeros((len(rows), width), dtype=np.int64)
    value = np.zeros((len(rows), width), dtype=np.float32)
    padding = np.ones((len(rows), width), dtype=bool)
    rr, aa = np.nonzero(visible)
    position = (np.cumsum(visible, axis=1) - 1)[rr, aa]
    axis[rr, position] = aa
    value[rr, position] = values[rr, aa]
    padding[rr, position] = False
    target_axis = target_axes[:, None]
    target_padding = np.zeros((len(rows), 1), dtype=bool)
    return tuple(torch.from_numpy(x).to(device) for x in (
        axis, value, padding, target_axis, target_padding,
    ))


@torch.inference_mode()
def predict_normalized(
    model,
    data,
    rows: np.ndarray,
    target_axes: np.ndarray,
    visible: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    two_pass: bool,
    normalized_values: np.ndarray | None = None,
):
    """Predict normalized log values for aligned profile-target jobs."""
    model.eval()
    pieces = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        packed = pack_visible(
            data,
            rows[start:stop],
            target_axes[start:stop],
            visible[start:stop],
            device,
            None if normalized_values is None else normalized_values[start:stop],
        )
        if two_pass:
            prediction = model.forward_prompt_details(*packed)[1]
        else:
            prediction = model.forward_details(*packed)[0]
        pieces.append(prediction[:, 0].detach().cpu().numpy())
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite model prediction")
    return result


@torch.inference_mode()
def extract_food_embeddings(
    model,
    data,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    visible_override: np.ndarray | None = None,
):
    """Extract the first-pass FOOD state with an empty target query grid."""
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    pieces = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        observed = (
            data.observed[batch_rows]
            if visible_override is None
            else np.asarray(visible_override[start:start + batch_size], dtype=bool)
        )
        width = max(1, int(observed.sum(axis=1).max(initial=0)))
        axis = np.zeros((len(batch_rows), width), dtype=np.int64)
        value = np.zeros((len(batch_rows), width), dtype=np.float32)
        padding = np.ones((len(batch_rows), width), dtype=bool)
        rr, aa = np.nonzero(observed)
        position = (np.cumsum(observed, axis=1) - 1)[rr, aa]
        axis[rr, position] = aa
        value[rr, position] = data.values[batch_rows[rr], aa]
        padding[rr, position] = False
        tensors = tuple(torch.from_numpy(x).to(device) for x in (axis, value, padding))
        target = torch.empty((len(batch_rows), 0), dtype=torch.long, device=device)
        target_padding = torch.empty((len(batch_rows), 0), dtype=torch.bool, device=device)
        food = model.forward_details(*tensors, target, target_padding)[1]
        pieces.append(food.detach().cpu().numpy())
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite FOOD representation")
    return result


def dependency_visible(data, policy: DependencyMaskPolicy, rows, target_axes):
    rows = np.asarray(rows, dtype=np.int64)
    target_axes = np.asarray(target_axes, dtype=np.int64)
    return data.observed[rows] & ~policy.mask_matrix(target_axes)


def attach_taiwan_panel(data, audit_dir: Path):
    """Append the frozen TFDA numerical panel and return aligned target jobs."""
    audit_dir = Path(audit_dir)
    foods = pd.read_parquet(audit_dir / "foods.parquet")
    mapped = pd.read_parquet(audit_dir / "mapped_values.parquet")
    coverage = pd.read_csv(audit_dir / "axis_coverage.csv")
    axes = np.sort(coverage.loc[
        coverage.strict_sample_backed_foods.ge(20), "axis_index"
    ].to_numpy(dtype=np.int64))
    if len(axes) != 80 or not set(axes).issubset(set(map(int, data.targets))):
        raise ValueError("unexpected TFDA axis panel")
    eligible = mapped[mapped.numeric_eligible.astype(bool)].copy()
    strict = mapped[
        mapped.strict_sample_backed_target.astype(bool)
        & mapped.axis_index.isin(axes)
    ].copy()
    panel = foods[
        foods.food_id.isin(strict.food_id) & foods.mapped_eligible_axes.ge(10)
    ].sort_values("food_id").reset_index(drop=True)
    if len(panel) != 2_212:
        raise ValueError("unexpected TFDA profile count")
    eligible = eligible[eligible.food_id.isin(panel.food_id)]
    strict = strict[strict.food_id.isin(panel.food_id)]
    if len(strict) != 128_470 or strict.axis_index.nunique() != 80:
        raise ValueError("unexpected TFDA job panel")
    if eligible.duplicated(["food_id", "axis_index"]).any():
        raise ValueError("duplicate mapped TFDA cell")
    offset = len(data.profiles)
    panel["profile_index"] = offset + np.arange(len(panel), dtype=np.int64)
    lookup = panel.set_index("food_id").profile_index
    raw = np.full((len(panel), len(data.axes)), np.nan, dtype=np.float64)
    rr = eligible.food_id.map(lookup).to_numpy(dtype=np.int64) - offset
    aa = eligible.axis_index.to_numpy(dtype=np.int64)
    raw[rr, aa] = eligible.value_g_per_100g.to_numpy(dtype=np.float64)
    observed = np.isfinite(raw)
    transformed = np.where(observed, np.log1p(raw / data.scale), 0).astype(np.float32)
    data.raw = np.concatenate((data.raw, raw))
    data.observed = np.concatenate((data.observed, observed))
    data.values = np.concatenate((data.values, transformed))
    data.weights = np.concatenate((data.weights, np.zeros_like(transformed)))
    jobs = strict[["food_id", "axis_index", "value_g_per_100g"]].copy()
    jobs["profile_index"] = jobs.food_id.map(lookup).to_numpy(dtype=np.int64)
    jobs = jobs.rename(columns={"value_g_per_100g": "target"})[
        ["profile_index", "axis_index", "target"]
    ].sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    novel_rows = panel.loc[
        ~panel.english_name_seen_in_pretraining.astype(bool), "profile_index"
    ].to_numpy(dtype=np.int64)
    if len(novel_rows) != 1_789:
        raise ValueError("unexpected name-novel TFDA profile count")
    return jobs, np.sort(novel_rows), panel, axes


def checkpoint_metadata(checkpoint_path: Path, checkpoint: dict):
    return {
        "path": str(Path(checkpoint_path)),
        "sha256": digest(Path(checkpoint_path)),
        "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("completed_epoch"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "config": checkpoint.get("config"),
    }


__all__ = [
    "load_dependency_model", "pack_visible", "predict_normalized",
    "extract_food_embeddings", "dependency_visible", "attach_taiwan_panel",
    "checkpoint_metadata",
]

"""Single-model distillation utilities for masked-axis completion.

Teachers are used only to create dense soft targets for training rows.  The
exported student is one gated masked-axis Transformer and has no dependency on
teacher models at inference time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from foodcomp.research_no_foodname_v3 import family_target_axes, masked_axis_batch


def distillation_axis_totals(data, families) -> np.ndarray:
    """Count how many dense teacher targets each supervised axis receives."""
    families = np.asarray(families, dtype=str)
    if families.ndim != 1 or not len(families):
        raise ValueError("distillation families must be a nonempty vector")
    totals = np.zeros(len(data.axes), dtype=np.int64)
    for family, count in zip(*np.unique(families, return_counts=True)):
        totals[family_target_axes(data, family)] = int(count)
    if not totals.any():
        raise ValueError("no supervised axis has distillation tasks")
    return totals


def macro_axis_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_axis: torch.Tensor,
    target_padding: torch.Tensor,
    axis_totals: torch.Tensor,
    *,
    task_count: int,
    active_axis_count: int,
) -> torch.Tensor:
    """Macro-axis MAE for a randomly sampled task batch.

    Each axis contributes equally over a complete epoch, even though mask
    families and observation counts differ substantially.
    """
    if prediction.shape != target.shape or prediction.shape != target_axis.shape:
        raise ValueError("prediction, target and target_axis shapes differ")
    if prediction.shape != target_padding.shape or prediction.ndim != 2:
        raise ValueError("target padding must match a two-dimensional prediction")
    if target_axis.dtype != torch.long or target_padding.dtype != torch.bool:
        raise TypeError("target_axis must be long and target_padding must be bool")
    if task_count < len(prediction) or active_axis_count < 1:
        raise ValueError("invalid task or active-axis count")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite distillation value")
    denominator = axis_totals[target_axis].clamp_min(1)
    weight = (~target_padding).to(prediction.dtype) / denominator
    multiplier = task_count / (len(prediction) * active_axis_count)
    return (weight * (prediction - target).abs()).sum() * multiplier


@torch.no_grad()
def mean_teacher_prediction(teachers, packed) -> torch.Tensor:
    """Average frozen teacher functions in the scaled-log training space."""
    if not teachers:
        raise ValueError("at least one teacher is required")
    predictions = [teacher(*packed) for teacher in teachers]
    if any(item.shape != predictions[0].shape for item in predictions[1:]):
        raise ValueError("teacher prediction shapes differ")
    result = torch.stack(predictions).mean(dim=0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite teacher prediction")
    return result


@torch.no_grad()
def write_teacher_cache(
    path: Path,
    teachers,
    data,
    tasks,
    families,
    device,
    batch_size: int,
) -> tuple[int, int]:
    """Write deterministic dense teacher targets in family-task order."""
    tasks = np.asarray(tasks, dtype=np.int64)
    families = np.asarray(families, dtype=str)
    if tasks.ndim != 1 or tasks.shape != families.shape or not len(tasks):
        raise ValueError("teacher-cache task vectors are invalid")
    if batch_size < 1:
        raise ValueError("teacher-cache batch size must be positive")
    target_width = max(
        len(family_target_axes(data, family)) for family in np.unique(families)
    )
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(tasks), target_width)
    )
    cache[:] = 0.0
    for start in range(0, len(tasks), batch_size):
        stop = min(start + batch_size, len(tasks))
        packed = masked_axis_batch(
            data, tasks[start:stop], families[start:stop], device
        )
        prediction = mean_teacher_prediction(teachers, packed)
        width = prediction.shape[1]
        cache[start:stop, :width] = prediction.cpu().numpy()
    cache.flush()
    return cache.shape

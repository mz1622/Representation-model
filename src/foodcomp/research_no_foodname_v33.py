"""Train-only empirical shrinkage for a masked-axis Transformer decoder."""
from __future__ import annotations

import numpy as np
import torch

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


def weighted_median(values, weights) -> float:
    """Return the lower weighted median of finite positive-weight samples."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or not len(values):
        raise ValueError("weighted median inputs must be equal nonempty vectors")
    if (not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()):
        raise ValueError("weighted median inputs are invalid")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="left"))
    return float(ordered_values[position])


def fit_axis_retention(prediction, target, weight, axis, axis_prior, axis_count):
    """Fit bounded per-axis L1 shrinkage using training predictions only.

    For a fixed axis prior ``m`` and base prediction ``p``, this minimizes
    ``sum w * |m + alpha * (p - m) - y|`` over ``0 <= alpha <= 1``. The exact
    solution is a weighted median of ``(y-m)/(p-m)`` with weights
    ``w * |p-m|``.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.int64)
    prior = np.asarray(axis_prior, dtype=np.float64)
    if not (prediction.shape == target.shape == weight.shape == axis.shape):
        raise ValueError("calibration observations must have equal shapes")
    if prediction.ndim != 1 or prior.shape != (axis_count,):
        raise ValueError("calibration observations or priors have invalid shape")
    if (not np.isfinite(prediction).all() or not np.isfinite(target).all()
            or not np.isfinite(weight).all() or (weight < 0).any()
            or (axis < 0).any() or (axis >= axis_count).any()):
        raise ValueError("calibration observations are invalid")

    retention = np.ones(axis_count, dtype=np.float32)
    support = np.zeros(axis_count, dtype=np.int64)
    for axis_index in np.unique(axis):
        chosen = axis == axis_index
        support[axis_index] = int(chosen.sum())
        delta = prediction[chosen] - prior[axis_index]
        informative = (np.abs(delta) > 1e-8) & (weight[chosen] > 0)
        if not informative.any():
            continue
        ratio = (
            (target[chosen][informative] - prior[axis_index])
            / delta[informative]
        )
        ratio_weight = weight[chosen][informative] * np.abs(delta[informative])
        retention[axis_index] = np.clip(
            weighted_median(ratio, ratio_weight), 0.0, 1.0
        )
    return retention, support


class TrainCalibratedShrinkageTransformer(AxisPairBiasMaskedAxisTransformer):
    """Apply a train-only per-axis shrinkage layer to the shared decoder."""

    def __init__(
        self, axis_count: int, config: Config, axis_prior, axis_retention
    ):
        super().__init__(axis_count, config)
        prior = np.asarray(axis_prior, dtype=np.float32)
        retention = np.asarray(axis_retention, dtype=np.float32)
        if (prior.shape != (axis_count,) or retention.shape != (axis_count,)
                or not np.isfinite(prior).all()
                or not np.isfinite(retention).all()
                or (retention < 0).any() or (retention > 1).any()):
            raise ValueError("axis calibration arrays are invalid")
        self.register_buffer("axis_prior", torch.from_numpy(prior))
        self.register_buffer("axis_retention", torch.from_numpy(retention))

    def forward_details(
        self, axis: torch.Tensor, value: torch.Tensor,
        padding: torch.Tensor, target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        base, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding
        )
        prior = self.axis_prior[target_axis]
        prediction = prior + self.axis_retention[target_axis] * (base - prior)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

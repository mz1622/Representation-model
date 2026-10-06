"""Axis-prior shrinkage decoder for masked-axis completion."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class AxisPriorShrinkageTransformer(AxisPairBiasMaskedAxisTransformer):
    """Shrink each prediction toward its train-only axis median."""

    def __init__(
        self, axis_count: int, config: Config, axis_prior,
        initial_retention: float = 0.95,
    ):
        super().__init__(axis_count, config)
        prior = np.asarray(axis_prior, dtype=np.float32)
        if prior.shape != (axis_count,) or not np.isfinite(prior).all():
            raise ValueError("axis_prior must be one finite value per axis")
        if not 0 < initial_retention < 1:
            raise ValueError("initial_retention must be strictly between zero and one")
        self.register_buffer(
            "axis_prior", torch.from_numpy(prior), persistent=True
        )
        initial_logit = math.log(initial_retention / (1 - initial_retention))
        self.axis_retention_logit = nn.Parameter(
            torch.full((axis_count,), initial_logit)
        )

    def forward_details(
        self, axis: torch.Tensor, value: torch.Tensor,
        padding: torch.Tensor, target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        base, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding
        )
        prior = self.axis_prior[target_axis]
        retention = torch.sigmoid(self.axis_retention_logit[target_axis])
        prediction = prior + retention * (base - prior)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

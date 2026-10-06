"""Target-specific layer routing with train-only prior residual decoding."""
from __future__ import annotations

import numpy as np
import torch

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v23 import TargetLayerRoutingTransformer


class RoutedPriorResidualTransformer(TargetLayerRoutingTransformer):
    """Combine depth routing with residual prediction around axis medians."""

    def __init__(self, axis_count: int, config: Config, axis_prior):
        super().__init__(axis_count, config)
        prior = np.asarray(axis_prior, dtype=np.float32)
        if prior.shape != (axis_count,) or not np.isfinite(prior).all():
            raise ValueError("axis_prior must be one finite value per axis")
        self.register_buffer(
            "axis_prior", torch.from_numpy(prior), persistent=True
        )

    def forward_details(
        self,
        axis: torch.Tensor,
        value: torch.Tensor,
        padding: torch.Tensor,
        target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        residual, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding
        )
        prediction = residual + self.axis_prior[target_axis]
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

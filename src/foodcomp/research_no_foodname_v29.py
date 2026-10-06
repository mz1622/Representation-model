"""Axis-calibrated decoder for masked-axis composition completion."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class AxisCalibratedTransformer(AxisPairBiasMaskedAxisTransformer):
    """Add a trainable affine output calibration for every target axis."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        # Direct zero tensors preserve every inherited v18 initialization.
        # exp(0)=1 and bias=0 make the initial function exactly the parent.
        self.axis_log_scale = nn.Parameter(torch.zeros(axis_count))
        self.axis_bias = nn.Parameter(torch.zeros(axis_count))

    def forward_details(
        self, axis: torch.Tensor, value: torch.Tensor,
        padding: torch.Tensor, target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        base, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding
        )
        scale = self.axis_log_scale[target_axis].exp()
        prediction = base * scale + self.axis_bias[target_axis]
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

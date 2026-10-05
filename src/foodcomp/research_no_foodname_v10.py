"""ReGLU masked-axis completion with an axis-specific output residual."""
from __future__ import annotations

import math

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v7 import GatedMaskedAxisTransformer


class AxisSpecificResidualDecoder(nn.Module):
    """Add a target-axis-specific linear residual to the shared scGPT decoder."""

    def __init__(self, axis_count: int, d_model: int):
        super().__init__()
        self.weight = nn.Embedding(axis_count, d_model)
        self.bias = nn.Embedding(axis_count, 1)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.bias.weight)
        self.scale = math.sqrt(d_model)

    def forward(self, hidden: torch.Tensor,
                target_axis: torch.Tensor) -> torch.Tensor:
        residual = (hidden * self.weight(target_axis)).sum(dim=-1) / self.scale
        return residual + self.bias(target_axis).squeeze(-1)


class AxisSpecificGatedTransformer(GatedMaskedAxisTransformer):
    """Preserve the shared decoder and learn a small per-axis residual head."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.axis_residual_decoder = AxisSpecificResidualDecoder(
            axis_count, config.d_model)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        padding: torch.Tensor, target_axis: torch.Tensor,
                        target_padding: torch.Tensor):
        shared, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding)
        prediction = shared + self.axis_residual_decoder(
            target_hidden, target_axis)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden, shared

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding)[0]

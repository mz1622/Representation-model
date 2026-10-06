"""Axis-pair Transformer with a direct quantitative relation residual."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class DirectRelationResidualTransformer(AxisPairBiasMaskedAxisTransformer):
    """Add a learned axis-to-axis linear path from visible values to targets.

    Attention can represent these dependencies indirectly. The residual gives
    direct quantitative relations a short path while the Transformer retains
    responsibility for nonlinear and higher-order context. The coefficient
    table starts at zero, so the initial function exactly matches v18.
    """

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.direct_value_relation = nn.Parameter(
            torch.zeros(axis_count, axis_count)
        )

    def forward_details(
        self,
        axis: torch.Tensor,
        value: torch.Tensor,
        padding: torch.Tensor,
        target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        prediction, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding
        )
        relation = self.direct_value_relation[
            target_axis.unsqueeze(-1), axis.unsqueeze(1)
        ]
        valid = (~padding).unsqueeze(1) & (~target_padding).unsqueeze(-1)
        visible_count = valid.sum(dim=-1).clamp_min(1).to(value.dtype)
        residual = (
            relation * value.unsqueeze(1) * valid.to(value.dtype)
        ).sum(dim=-1) / visible_count.sqrt()
        prediction = (prediction + residual).masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

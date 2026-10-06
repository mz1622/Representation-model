"""Axis-pair Transformer with axis-conditioned continuous-value FiLM."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class AxisConditionedValueTransformer(AxisPairBiasMaskedAxisTransformer):
    """Let every axis modulate the shared numeric embedding.

    The parent model uses the same numeric MLP for every composition axis. This
    variant adds a visible-token-only feature-wise affine modulation generated
    from the axis identity. Both tables start at zero, so the initial function
    and every inherited parameter exactly match the parent model.
    """

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.value_scale = nn.Embedding(axis_count, config.d_model)
        self.value_shift = nn.Embedding(axis_count, config.d_model)
        nn.init.zeros_(self.value_scale.weight)
        nn.init.zeros_(self.value_shift.weight)

    def forward_details(
        self,
        axis: torch.Tensor,
        value: torch.Tensor,
        padding: torch.Tensor,
        target_axis: torch.Tensor,
        target_padding: torch.Tensor,
    ):
        if axis.shape != value.shape or axis.shape != padding.shape or axis.ndim != 2:
            raise ValueError(
                "visible axis, value and padding must have equal [batch, tokens] shape"
            )
        if target_axis.shape != target_padding.shape or target_axis.ndim != 2:
            raise ValueError(
                "target axis and padding must have equal [batch, targets] shape"
            )
        if len(axis) != len(target_axis):
            raise ValueError("visible and target batches must have equal size")
        if axis.dtype != torch.long or target_axis.dtype != torch.long:
            raise TypeError("axis tensors must be long")
        if padding.dtype != torch.bool or target_padding.dtype != torch.bool:
            raise TypeError("padding tensors must be bool")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite composition input")

        numeric = self.value_encoder(value.unsqueeze(-1))
        numeric = numeric * (1.0 + torch.tanh(self.value_scale(axis)))
        numeric = numeric + self.value_shift(axis)
        visible = self.axis_embedding(axis) + numeric
        target = self.axis_embedding(target_axis) + self.mask_value
        tokens = torch.cat((
            self.context_token.expand(len(axis), -1, -1), visible, target,
        ), dim=1)
        sequence_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding,
            target_padding,
        ), dim=1)
        sequence_axis = torch.cat((
            torch.full(
                (len(axis), 1), self.axis_count,
                dtype=torch.long, device=axis.device,
            ),
            axis,
            target_axis,
        ), dim=1)
        for block in self.blocks:
            tokens = block(tokens, sequence_padding, sequence_axis)
        tokens = self.final_norm(tokens)
        target_hidden = tokens[:, 1 + axis.shape[1]:]
        prediction = self.value_decoder(target_hidden).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, tokens[:, 0], target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

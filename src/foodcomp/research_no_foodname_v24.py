"""Axis-pair Transformer with multiplicative axis/value token interaction."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class MultiplicativeAxisValueTransformer(AxisPairBiasMaskedAxisTransformer):
    """Add a shared gated Hadamard interaction to every visible token.

    The parent combines axis identity and numeric content only by addition.
    This model also exposes their elementwise product, allowing the same scaled
    value to change meaning with the axis before attention. A zero-initialized
    global gate preserves the exact v18 initial function and avoids per-axis
    numeric parameters.
    """

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.axis_value_gate = nn.Parameter(torch.zeros(config.d_model))

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

        identity = self.axis_embedding(axis)
        numeric = self.value_encoder(value.unsqueeze(-1))
        interaction = identity * numeric * torch.tanh(self.axis_value_gate)
        visible = identity + numeric + interaction
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

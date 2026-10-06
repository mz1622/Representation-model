"""Axis-pair Transformer with target-specific scalar layer routing."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import AxisPairBiasMaskedAxisTransformer


class TargetLayerRoutingTransformer(AxisPairBiasMaskedAxisTransformer):
    """Route lower block states into each target with two learned scalars.

    Every target axis can choose a different effective representation depth.
    Zero-initialized tanh gates make the initial function exactly v18 while
    adding only ``axis_count * (n_layers - 1)`` parameters.
    """

    def __init__(self, axis_count: int, config: Config):
        if config.n_layers < 2:
            raise ValueError("target layer routing requires at least two blocks")
        super().__init__(axis_count, config)
        self.layer_count = config.n_layers
        self.target_layer_gate = nn.Parameter(
            torch.zeros(axis_count, config.n_layers - 1)
        )

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

        visible = self.axis_embedding(axis) + self.value_encoder(
            value.unsqueeze(-1)
        )
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
        target_layers = []
        target_start = 1 + axis.shape[1]
        for block in self.blocks:
            tokens = block(tokens, sequence_padding, sequence_axis)
            target_layers.append(tokens[:, target_start:])
        routed_target = target_layers[-1]
        gate = torch.tanh(self.target_layer_gate[target_axis])
        for layer_index, lower_state in enumerate(target_layers[:-1]):
            routed_target = (
                routed_target
                + gate[:, :, layer_index:layer_index + 1] * lower_state
            )
        target_hidden = self.final_norm(routed_target)
        context = self.final_norm(tokens[:, 0])
        prediction = self.value_decoder(target_hidden).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

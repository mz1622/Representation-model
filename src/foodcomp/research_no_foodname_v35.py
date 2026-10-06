"""V18 with disentangled content-axis attention score terms."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v16 import (
    AxisPairBiasMaskedAxisTransformer,
    AxisPairBiasSelfAttention,
)


class DisentangledAxisContentAttention(AxisPairBiasSelfAttention):
    """Add content-to-axis and axis-to-content score terms to V18."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.axis_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.axis_key = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.zeros_(self.axis_query.weight)
        nn.init.zeros_(self.axis_key.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        padding: torch.Tensor,
        sequence_axis: torch.Tensor,
        axis_features: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or padding.shape != tokens.shape[:2]:
            raise ValueError("disentangled attention token/padding shapes differ")
        if sequence_axis.shape != padding.shape:
            raise ValueError("disentangled sequence axes do not match tokens")
        if axis_features.shape != tokens.shape:
            raise ValueError("axis features must match token shape")
        if padding.dtype != torch.bool or sequence_axis.dtype != torch.long:
            raise TypeError("padding must be bool and sequence axes must be long")
        if (sequence_axis < 0).any() or (sequence_axis >= self.identity_count).any():
            raise ValueError("disentangled sequence axis is out of range")

        projected = F.linear(
            tokens,
            self.projections.in_proj_weight,
            self.projections.in_proj_bias,
        )
        content_query, content_key, value = projected.chunk(3, dim=-1)
        axis_query = self.axis_query(axis_features)
        axis_key = self.axis_key(axis_features)
        batch, length, _ = tokens.shape

        def split_heads(item: torch.Tensor) -> torch.Tensor:
            return item.view(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        content_query, content_key, value, axis_query, axis_key = map(
            split_heads,
            (content_query, content_key, value, axis_query, axis_key),
        )
        logits = (
            torch.matmul(content_query, content_key.transpose(-2, -1))
            + torch.matmul(content_query, axis_key.transpose(-2, -1))
            + torch.matmul(axis_query, content_key.transpose(-2, -1))
        ) / math.sqrt(self.head_dim)
        pair_index = (
            sequence_axis[:, :, None] * self.identity_count
            + sequence_axis[:, None, :]
        )
        bias = F.embedding(
            pair_index, self.pair_bias
        ).permute(0, 3, 1, 2)
        logits = logits + bias
        logits = logits.masked_fill(padding[:, None, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )
        return F.linear(
            output,
            self.projections.out_proj.weight,
            self.projections.out_proj.bias,
        )


class DisentangledAxisContentTransformer(AxisPairBiasMaskedAxisTransformer):
    """V18 with explicit content-axis cross terms in every attention layer."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        rng_state = torch.get_rng_state()
        try:
            for block in self.blocks:
                inherited = block.attention
                replacement = DisentangledAxisContentAttention(axis_count, config)
                replacement.projections.load_state_dict(
                    inherited.projections.state_dict()
                )
                replacement.pair_bias.data.copy_(inherited.pair_bias.data)
                block.attention = replacement
        finally:
            torch.set_rng_state(rng_state)

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

        visible_axis = self.axis_embedding(axis)
        target_axis_features = self.axis_embedding(target_axis)
        visible = visible_axis + self.value_encoder(value.unsqueeze(-1))
        target = target_axis_features + self.mask_value
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
        axis_features = torch.cat((
            torch.zeros(
                (len(axis), 1, self.axis_embedding.embedding_dim),
                dtype=visible.dtype, device=axis.device,
            ),
            visible_axis,
            target_axis_features,
        ), dim=1)
        for block in self.blocks:
            normalized = block.attention_norm(tokens)
            update = block.attention(
                normalized, sequence_padding, sequence_axis, axis_features
            )
            tokens = tokens + block.attention_dropout(update)
            tokens = tokens + block.ffn(block.ffn_norm(tokens))
        tokens = self.final_norm(tokens)
        target_hidden = tokens[:, 1 + axis.shape[1]:]
        prediction = self.value_decoder(target_hidden).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, tokens[:, 0], target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

    def mechanism_diagnostics(self) -> dict[str, float]:
        projections = []
        with torch.no_grad():
            axis_features = self.axis_embedding.weight
            for block in self.blocks:
                projections.extend((
                    block.attention.axis_query(axis_features).flatten(),
                    block.attention.axis_key(axis_features).flatten(),
                ))
        values = torch.cat(projections)
        return {
            "axis_cross_projection_rms": float(values.square().mean().sqrt())
        }

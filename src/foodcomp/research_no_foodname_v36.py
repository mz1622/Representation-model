"""V18 with factorized ordered-axis relation vectors in attention values."""
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


class RelationValueAxisPairAttention(AxisPairBiasSelfAttention):
    """Add an ordered axis relation to each attended value message."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        shape = (self.identity_count, self.n_heads, self.head_dim)
        self.relation_query = nn.Parameter(torch.zeros(shape))
        self.relation_value = nn.Parameter(torch.zeros(shape))

    def forward(
        self,
        tokens: torch.Tensor,
        padding: torch.Tensor,
        sequence_axis: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or padding.shape != tokens.shape[:2]:
            raise ValueError("relation-value attention token/padding shapes differ")
        if sequence_axis.shape != padding.shape:
            raise ValueError("relation-value sequence axes do not match tokens")
        if padding.dtype != torch.bool or sequence_axis.dtype != torch.long:
            raise TypeError("padding must be bool and sequence axes must be long")
        if (sequence_axis < 0).any() or (sequence_axis >= self.identity_count).any():
            raise ValueError("relation-value sequence axis is out of range")

        projected = F.linear(
            tokens,
            self.projections.in_proj_weight,
            self.projections.in_proj_bias,
        )
        query, key, value = projected.chunk(3, dim=-1)
        batch, length, _ = tokens.shape

        def split_heads(item: torch.Tensor) -> torch.Tensor:
            return item.view(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        logits = torch.matmul(
            query, key.transpose(-2, -1)
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

        relation_query = self.relation_query[sequence_axis].permute(0, 2, 1, 3)
        relation_value = self.relation_value[sequence_axis].permute(0, 2, 1, 3)
        content_output = torch.matmul(attention, value)
        relation_output = (1.0 + relation_query) * torch.matmul(
            attention, relation_value
        )
        output = content_output + relation_output
        output = output.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )
        return F.linear(
            output,
            self.projections.out_proj.weight,
            self.projections.out_proj.bias,
        )


class RelationValueAxisPairTransformer(AxisPairBiasMaskedAxisTransformer):
    """V18 with factorized query-axis/key-axis value relations."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        rng_state = torch.get_rng_state()
        try:
            for block in self.blocks:
                inherited = block.attention
                replacement = RelationValueAxisPairAttention(axis_count, config)
                replacement.projections.load_state_dict(
                    inherited.projections.state_dict()
                )
                replacement.pair_bias.data.copy_(inherited.pair_bias.data)
                block.attention = replacement
        finally:
            torch.set_rng_state(rng_state)

    def mechanism_diagnostics(self) -> dict[str, float]:
        effective_squares = []
        query_squares = []
        value_squares = []
        for block in self.blocks:
            query = block.attention.relation_query.detach()
            value = block.attention.relation_value.detach()
            query_factor = (1.0 + query).square().mean(dim=0)
            value_factor = value.square().mean(dim=0)
            effective_squares.append((query_factor * value_factor).mean())
            query_squares.append(query.square().mean())
            value_squares.append(value.square().mean())
        return {
            "relation_value_effective_rms": float(
                torch.stack(effective_squares).mean().sqrt()
            ),
            "relation_query_rms": float(
                torch.stack(query_squares).mean().sqrt()
            ),
            "relation_value_rms": float(
                torch.stack(value_squares).mean().sqrt()
            ),
        }

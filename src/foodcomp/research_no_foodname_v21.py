"""ReGLU masked-axis completion with low-rank axis-pair attention bias."""
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


class LowRankAxisPairBiasSelfAttention(AxisPairBiasSelfAttention):
    """Factorize each head's ordered axis-pair bias through a shared rank."""

    def __init__(self, axis_count: int, config: Config, rank: int = 16):
        if rank < 1:
            raise ValueError("rank must be positive")
        super().__init__(axis_count, config)
        del self.pair_bias
        self.rank = rank
        self.query_factor = nn.Parameter(
            torch.empty(self.identity_count, config.n_heads, rank)
        )
        self.key_factor = nn.Parameter(
            torch.zeros(self.identity_count, config.n_heads, rank)
        )
        nn.init.normal_(self.query_factor, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        padding: torch.Tensor,
        sequence_axis: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or padding.shape != tokens.shape[:2]:
            raise ValueError("pair-bias attention token/padding shapes differ")
        if sequence_axis.shape != padding.shape:
            raise ValueError("pair-bias sequence axes do not match tokens")
        if padding.dtype != torch.bool or sequence_axis.dtype != torch.long:
            raise TypeError("pair-bias padding must be bool and axes must be long")
        if (sequence_axis < 0).any() or (sequence_axis >= self.identity_count).any():
            raise ValueError("pair-bias sequence axis is out of range")

        projected = F.linear(
            tokens,
            self.projections.in_proj_weight,
            self.projections.in_proj_bias,
        )
        query, key, value = projected.chunk(3, dim=-1)
        batch, length, _ = tokens.shape

        def split_heads(item):
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
        bias_table = torch.einsum(
            "ahr,bhr->abh", self.query_factor, self.key_factor
        ) / math.sqrt(self.rank)
        bias = F.embedding(
            pair_index, bias_table.reshape(-1, self.n_heads)
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

    def materialized_bias(self) -> torch.Tensor:
        """Return [axis-or-context, axis-or-context, head] bias values."""
        return torch.einsum(
            "ahr,bhr->abh", self.query_factor, self.key_factor
        ) / math.sqrt(self.rank)


class LowRankAxisPairBiasTransformer(AxisPairBiasMaskedAxisTransformer):
    """Replace v18's free pair tables with rank-16 relation factors."""

    def __init__(self, axis_count: int, config: Config, rank: int = 16):
        super().__init__(axis_count, config)
        for block in self.blocks:
            inherited = block.attention
            factorized = LowRankAxisPairBiasSelfAttention(
                axis_count, config, rank=rank
            )
            factorized.projections.load_state_dict(
                inherited.projections.state_dict()
            )
            block.attention = factorized

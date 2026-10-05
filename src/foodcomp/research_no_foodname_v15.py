"""ReGLU masked-axis completion with query-key normalized attention."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v7 import ReGLU


class QKNormSelfAttention(nn.Module):
    """Multi-head self-attention with unit-norm Q/K and learned temperature."""

    def __init__(self, config: Config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        # Retain PyTorch's projection parameters and initialization exactly.
        self.projections = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout,
            batch_first=True,
        )
        initial_temperature = math.sqrt(self.head_dim)
        self.log_temperature = nn.Parameter(
            torch.full((config.n_heads,), math.log(initial_temperature))
        )

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or padding.shape != tokens.shape[:2]:
            raise ValueError("QK attention token/padding shapes differ")
        if padding.dtype != torch.bool:
            raise TypeError("QK attention padding must be bool")
        projected = F.linear(
            tokens,
            self.projections.in_proj_weight,
            self.projections.in_proj_bias,
        )
        query, key, value = projected.chunk(3, dim=-1)
        batch, length, _ = tokens.shape

        def split_heads(item):
            return item.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query = F.normalize(split_heads(query), p=2, dim=-1, eps=1e-6)
        key = F.normalize(split_heads(key), p=2, dim=-1, eps=1e-6)
        value = split_heads(value)
        temperature = self.log_temperature.clamp(max=math.log(100.0)).exp()
        logits = torch.matmul(query, key.transpose(-2, -1))
        logits = logits * temperature.view(1, self.n_heads, 1, 1)
        logits = logits.masked_fill(padding[:, None, None, :], -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return F.linear(
            output,
            self.projections.out_proj.weight,
            self.projections.out_proj.bias,
        )


class QKNormReGLUBlock(nn.Module):
    """The v7 pre-normalized ReGLU block with QKNorm attention only."""

    def __init__(self, config: Config):
        super().__init__()
        gated_hidden = max(1, round(config.feedforward_dim * 2 / 3))
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = QKNormSelfAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 2 * gated_hidden),
            ReGLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gated_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        update = self.attention(self.attention_norm(tokens), padding)
        tokens = tokens + self.attention_dropout(update)
        return tokens + self.ffn(self.ffn_norm(tokens))


class QKNormMaskedAxisTransformer(nn.Module):
    """The v7 target-token model with QK-normalized attention."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__()
        self.axis_count = axis_count
        self.context_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList(
            QKNormReGLUBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.context_token, std=0.02)
        nn.init.normal_(self.mask_value, std=0.02)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        padding: torch.Tensor, target_axis: torch.Tensor,
                        target_padding: torch.Tensor):
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

        visible = self.axis_embedding(axis) + self.value_encoder(value.unsqueeze(-1))
        target = self.axis_embedding(target_axis) + self.mask_value
        tokens = torch.cat((
            self.context_token.expand(len(axis), -1, -1), visible, target,
        ), dim=1)
        sequence_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding, target_padding,
        ), dim=1)
        for block in self.blocks:
            tokens = block(tokens, sequence_padding)
        tokens = self.final_norm(tokens)
        target_hidden = tokens[:, 1 + axis.shape[1]:]
        prediction = self.value_decoder(target_hidden).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, tokens[:, 0], target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding
        )[0]

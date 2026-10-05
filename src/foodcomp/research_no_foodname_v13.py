"""Masked-axis completion with parameter-matched SwiGLU FFN blocks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from foodcomp.research_no_foodname_v1 import Config


class SwiGLU(nn.Module):
    """Split the last dimension and gate one half with SiLU of the other."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = value.chunk(2, dim=-1)
        return left * F.silu(gate)


class SwiGLUEncoderBlock(nn.Module):
    """Pre-normalized self-attention block with a matched-width SwiGLU FFN."""

    def __init__(self, config: Config):
        super().__init__()
        gated_hidden = max(1, round(config.feedforward_dim * 2 / 3))
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 2 * gated_hidden),
            SwiGLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gated_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        update, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=padding, need_weights=False,
        )
        tokens = tokens + self.attention_dropout(update)
        return tokens + self.ffn(self.ffn_norm(tokens))


class SwiGLUMaskedAxisTransformer(nn.Module):
    """The v3 target-token completion layout with SwiGLU FFN blocks."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.axis_count = axis_count
        self.context_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList(
            SwiGLUEncoderBlock(config) for _ in range(config.n_layers)
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

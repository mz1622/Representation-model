"""One-factor architecture variants for composition-only pretraining.

The only runtime inputs are visible axis IDs, their numeric values, and padding.
Food names and source identifiers are absent from every model interface.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from foodcomp.research_no_foodname_v1 import Config


VARIANTS = ("target_query", "piecewise_numeric", "mean_pool", "hurdle")


def training_knots(data, train_rows, *, bins=8):
    """Fit per-axis quantile knots from numeric observations in train only."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if not len(train_rows) or not np.isin(train_rows, data.train).all():
        raise ValueError("knots require nonempty training rows")
    knots = np.empty((len(data.axes), bins + 1), dtype=np.float32)
    for axis in range(len(data.axes)):
        values = data.values[train_rows, axis]
        values = values[data.observed[train_rows, axis]]
        if len(values):
            row = np.quantile(values, np.linspace(0, 1, bins + 1)).astype(np.float32)
        else:
            row = np.linspace(0, 1, bins + 1, dtype=np.float32)
        for i in range(1, len(row)):
            row[i] = max(row[i], row[i - 1] + 1e-5)
        knots[axis] = row
    if not np.isfinite(knots).all() or not np.all(np.diff(knots, axis=1) > 0):
        raise ValueError("invalid training knots")
    return knots


class VariantSetTransformer(nn.Module):
    """Exact v1 backbone, with one selected input, pooling, or output change."""

    def __init__(self, axis_count: int, config: Config, variant: str,
                 knots: np.ndarray | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown architecture variant: {variant}")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.axis_count = axis_count
        self.variant = variant
        self.food_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        if variant == "piecewise_numeric":
            if knots is None or knots.shape[0] != axis_count or knots.shape[1] < 3:
                raise ValueError("piecewise variant requires train-fitted knots")
            self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float32))
            self.piecewise_embedding = nn.Parameter(
                torch.empty(axis_count, knots.shape[1], config.d_model))
            nn.init.normal_(self.piecewise_embedding, std=0.02)
        else:
            self.value_encoder = nn.Sequential(
                nn.Linear(1, config.d_model), nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.feedforward_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layers, norm=nn.LayerNorm(config.d_model),
        )
        if variant == "target_query":
            self.target_query = nn.Embedding(axis_count, config.d_model)
            self.query_norm = nn.LayerNorm(config.d_model)
            self.query_head = nn.Linear(config.d_model, 1)
        elif variant == "hurdle":
            self.positive_head = nn.Linear(config.d_model, axis_count)
            self.amount_head = nn.Linear(config.d_model, axis_count)
        else:
            self.head = nn.Linear(config.d_model, axis_count)
        nn.init.normal_(self.food_token, std=0.02)

    def _numeric_embedding(self, axis, value):
        if self.variant != "piecewise_numeric":
            return self.value_encoder(value.unsqueeze(-1))
        knots = self.knots[axis]
        width = (knots[..., 1:] - knots[..., :-1]).clamp_min(1e-5)
        ramps = ((value.unsqueeze(-1) - knots[..., :-1]) / width).clamp(0, 1)
        # The first term keeps a linear tail beyond the largest train value.
        basis = torch.cat((value.unsqueeze(-1), ramps), dim=-1)
        return (basis.unsqueeze(-1) * self.piecewise_embedding[axis]).sum(dim=-2)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        padding: torch.Tensor):
        if axis.shape != value.shape or axis.shape != padding.shape or axis.ndim != 2:
            raise ValueError("axis, value and padding must have equal [batch, tokens] shape")
        if axis.dtype != torch.long or padding.dtype != torch.bool:
            raise TypeError("axis must be long and padding must be bool")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite composition input")
        observed = self.axis_embedding(axis) + self._numeric_embedding(axis, value)
        sequence = torch.cat((self.food_token.expand(len(axis), -1, -1), observed), dim=1)
        mask = torch.cat((torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
                          padding), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=mask)
        if self.variant == "target_query":
            query = self.target_query.weight.unsqueeze(0).expand(len(axis), -1, -1)
            attention = torch.matmul(query, encoded.transpose(1, 2)) / math.sqrt(query.shape[-1])
            attention = attention.masked_fill(mask[:, None, :], -torch.inf).softmax(dim=-1)
            context = torch.matmul(attention, encoded)
            prediction = self.query_head(self.query_norm(query + context)).squeeze(-1)
            return prediction, None, None
        if self.variant == "mean_pool":
            keep = (~padding).unsqueeze(-1)
            count = keep.sum(dim=1)
            mean = (encoded[:, 1:] * keep).sum(dim=1) / count.clamp_min(1)
            pooled = torch.where(count > 0, mean, encoded[:, 0])
        else:
            pooled = encoded[:, 0]
        if self.variant == "hurdle":
            logits = self.positive_head(pooled)
            amount = F.softplus(self.amount_head(pooled))
            return torch.sigmoid(logits) * amount, logits, amount
        return self.head(pooled), None, None

    def forward(self, axis, value, padding):
        return self.forward_details(axis, value, padding)[0]

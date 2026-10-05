"""Evidence-led architecture candidates built on masked-axis completion."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config


def training_ple_bins(data, train_rows, *, bins=8):
    """Fit deduplicated per-axis quantile bins using training observations only."""
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if not len(train_rows) or not np.isin(train_rows, data.train).all():
        raise ValueError("PLE bins require nonempty training rows")
    if bins < 2:
        raise ValueError("PLE requires at least two requested bins")
    left = np.zeros((len(data.axes), bins), dtype=np.float32)
    right = np.ones((len(data.axes), bins), dtype=np.float32)
    mask = np.zeros((len(data.axes), bins), dtype=bool)
    edges_by_axis = []
    for axis in range(len(data.axes)):
        values = data.values[train_rows, axis]
        values = values[data.observed[train_rows, axis]]
        if len(values):
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        else:
            edges = np.array([0.0, 1.0])
        if len(edges) == 1:
            radius = max(0.5, abs(float(edges[0])) * 0.01)
            edges = np.array([edges[0] - radius, edges[0] + radius])
        edges = edges.astype(np.float32)
        if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError(f"invalid PLE bin edges for axis {axis}")
        count = len(edges) - 1
        left[axis, :count] = edges[:-1]
        right[axis, :count] = edges[1:]
        mask[axis, :count] = True
        edges_by_axis.append(edges)
    return left, right, mask, edges_by_axis


class PiecewiseMaskedAxisTransformer(nn.Module):
    """Masked-axis model with train-quantile piecewise-linear value tokens."""

    def __init__(self, axis_count: int, config: Config, ple_bins):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        left, right, bin_mask = ple_bins[:3]
        if left.shape != right.shape or left.shape != bin_mask.shape:
            raise ValueError("PLE tensors must have equal [axes, bins] shape")
        if left.shape[0] != axis_count or left.shape[1] < 1:
            raise ValueError("PLE axis/bin shape mismatch")
        if not np.all(right[bin_mask] > left[bin_mask]):
            raise ValueError("PLE bins must have positive width")
        self.axis_count = axis_count
        self.food_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.piecewise_weight = nn.Parameter(
            torch.empty(axis_count, left.shape[1], config.d_model))
        self.register_buffer("ple_left", torch.as_tensor(left, dtype=torch.float32))
        self.register_buffer("ple_right", torch.as_tensor(right, dtype=torch.float32))
        self.register_buffer("ple_bin_mask", torch.as_tensor(bin_mask, dtype=torch.bool))
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.feedforward_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.n_layers, norm=nn.LayerNorm(config.d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.food_token, std=0.02)
        nn.init.normal_(self.mask_value, std=0.02)
        nn.init.normal_(self.piecewise_weight, std=0.02)

    def value_embedding(self, axis: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        left = self.ple_left[axis]
        right = self.ple_right[axis]
        valid = self.ple_bin_mask[axis]
        encoding = ((value.unsqueeze(-1) - left) /
                    (right - left).clamp_min(1e-12)).clamp(0, 1)
        encoding = encoding * valid
        return (encoding.unsqueeze(-1) * self.piecewise_weight[axis]).sum(dim=-2)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        padding: torch.Tensor, target_axis: torch.Tensor,
                        target_padding: torch.Tensor):
        if axis.shape != value.shape or axis.shape != padding.shape or axis.ndim != 2:
            raise ValueError("visible axis, value and padding must have equal [batch, tokens] shape")
        if target_axis.shape != target_padding.shape or target_axis.ndim != 2:
            raise ValueError("target axis and padding must have equal [batch, targets] shape")
        if len(axis) != len(target_axis):
            raise ValueError("visible and target batches must have equal size")
        if axis.dtype != torch.long or target_axis.dtype != torch.long:
            raise TypeError("axis tensors must be long")
        if padding.dtype != torch.bool or target_padding.dtype != torch.bool:
            raise TypeError("padding tensors must be bool")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite composition input")
        observed = self.axis_embedding(axis) + self.value_embedding(axis, value)
        masked = self.axis_embedding(target_axis) + self.mask_value
        sequence = torch.cat((
            self.food_token.expand(len(axis), -1, -1), observed, masked,
        ), dim=1)
        sequence_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding, target_padding,
        ), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=sequence_padding)
        target_hidden = encoded[:, 1 + axis.shape[1]:]
        prediction = self.decoder(target_hidden).squeeze(-1)
        return prediction.masked_fill(target_padding, 0.0), encoded[:, 0]

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(axis, value, padding, target_axis, target_padding)[0]

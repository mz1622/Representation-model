"""Missingness-aware axis tokens for masked-axis value completion."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v3 import family_target_axes


def missing_aware_axis_batch(data, rows, family, device, *, with_targets=False):
    """Pack every non-target-family axis as observed-value or missing token."""
    rows = np.asarray(rows, dtype=np.int64)
    family = np.asarray(family, dtype=str)
    if rows.ndim != 1 or family.ndim != 1 or len(rows) != len(family):
        raise ValueError("rows/family mismatch")
    if not len(rows):
        raise ValueError("missing-aware batch must be nonempty")

    visible_sets = [np.flatnonzero(data.families != item) for item in family]
    visible_width = max(map(len, visible_sets))
    axis = np.zeros((len(rows), visible_width), dtype=np.int64)
    value = np.zeros((len(rows), visible_width), dtype=np.float32)
    observed = np.zeros((len(rows), visible_width), dtype=bool)
    padding = np.ones((len(rows), visible_width), dtype=bool)

    query_sets = [family_target_axes(data, item) for item in family]
    target_width = max(map(len, query_sets))
    target_axis = np.zeros((len(rows), target_width), dtype=np.int64)
    target_padding = np.ones((len(rows), target_width), dtype=bool)
    target_value = np.zeros((len(rows), target_width), dtype=np.float32)
    target_observed = np.zeros((len(rows), target_width), dtype=bool)
    target_weight = np.zeros((len(rows), target_width), dtype=np.float32)

    for i, (row, visible_axes, queries) in enumerate(
            zip(rows, visible_sets, query_sets)):
        count = len(visible_axes)
        axis[i, :count] = visible_axes
        value[i, :count] = data.values[row, visible_axes]
        observed[i, :count] = data.observed[row, visible_axes]
        padding[i, :count] = False

        target_count = len(queries)
        target_axis[i, :target_count] = queries
        target_padding[i, :target_count] = False
        if with_targets:
            target_value[i, :target_count] = data.values[row, queries]
            target_observed[i, :target_count] = data.observed[row, queries]
            target_weight[i, :target_count] = data.weights[row, queries]

    result = (
        torch.from_numpy(axis).to(device),
        torch.from_numpy(value).to(device),
        torch.from_numpy(observed).to(device),
        torch.from_numpy(padding).to(device),
        torch.from_numpy(target_axis).to(device),
        torch.from_numpy(target_padding).to(device),
    )
    if not with_targets:
        return result
    return result + (
        torch.from_numpy(target_value).to(device),
        torch.from_numpy(target_observed).to(device),
        torch.from_numpy(target_weight).to(device),
    )


class MissingAwareMaskedAxisTransformer(nn.Module):
    """Represent natural missingness explicitly while decoding target axes."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.axis_count = axis_count
        self.context_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.missing_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
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
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.context_token, std=0.02)
        nn.init.normal_(self.missing_value, std=0.02)
        nn.init.normal_(self.mask_value, std=0.02)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        observed: torch.Tensor, padding: torch.Tensor,
                        target_axis: torch.Tensor, target_padding: torch.Tensor):
        if (axis.shape != value.shape or axis.shape != observed.shape
                or axis.shape != padding.shape or axis.ndim != 2):
            raise ValueError("visible tensors must have equal [batch, tokens] shape")
        if target_axis.shape != target_padding.shape or target_axis.ndim != 2:
            raise ValueError("target axis and padding must have equal [batch, targets] shape")
        if len(axis) != len(target_axis):
            raise ValueError("visible and target batches must have equal size")
        if axis.dtype != torch.long or target_axis.dtype != torch.long:
            raise TypeError("axis tensors must be long")
        if observed.dtype != torch.bool or padding.dtype != torch.bool:
            raise TypeError("observed and padding tensors must be bool")
        if target_padding.dtype != torch.bool:
            raise TypeError("target padding must be bool")
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite composition input")

        numeric = self.value_encoder(value.unsqueeze(-1))
        value_token = torch.where(
            observed.unsqueeze(-1), numeric,
            self.missing_value.expand(len(axis), axis.shape[1], -1),
        )
        visible = self.axis_embedding(axis) + value_token
        target = self.axis_embedding(target_axis) + self.mask_value
        sequence = torch.cat((
            self.context_token.expand(len(axis), -1, -1), visible, target,
        ), dim=1)
        sequence_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding, target_padding,
        ), dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=sequence_padding)
        target_hidden = encoded[:, 1 + axis.shape[1]:]
        prediction = self.value_decoder(target_hidden).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, encoded[:, 0], target_hidden

    def forward(self, axis, value, observed, padding,
                target_axis, target_padding):
        return self.forward_details(
            axis, value, observed, padding, target_axis, target_padding)[0]

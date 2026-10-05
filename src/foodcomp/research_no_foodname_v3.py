"""scGPT-aligned masked-axis completion without food-name input.

Visible composition values and masked target axes share one Transformer.  The
target axis identity stays in the sequence while its value is replaced by one
learned mask embedding.  Predictions are decoded from target-axis positions;
the [FOOD] position is retained only as a sample representation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config, visible_context


def family_target_axes(data, family: str) -> np.ndarray:
    """Return the fixed supervised-axis query set for one existing mask family."""
    targets = np.asarray(data.targets, dtype=np.int64)
    selected = targets[np.asarray(data.families)[targets] == str(family)]
    if not len(selected):
        raise ValueError(f"mask family has no supervised target axes: {family}")
    return selected


def masked_axis_batch(data, rows, family, device, *, with_targets=False):
    """Pack visible values plus a label-independent masked-axis query grid."""
    rows = np.asarray(rows, dtype=np.int64)
    family = np.asarray(family, dtype=str)
    if rows.ndim != 1 or family.ndim != 1 or len(rows) != len(family):
        raise ValueError("rows/family mismatch")
    if not len(rows):
        raise ValueError("masked-axis batch must be nonempty")

    visible = visible_context(data, rows, family)
    visible_width = max(1, int(visible.sum(axis=1).max(initial=0)))
    axis = np.zeros((len(rows), visible_width), dtype=np.int64)
    value = np.zeros((len(rows), visible_width), dtype=np.float32)
    padding = np.ones((len(rows), visible_width), dtype=bool)

    query_sets = [family_target_axes(data, item) for item in family]
    target_width = max(map(len, query_sets))
    target_axis = np.zeros((len(rows), target_width), dtype=np.int64)
    target_padding = np.ones((len(rows), target_width), dtype=bool)
    target_value = np.zeros((len(rows), target_width), dtype=np.float32)
    target_observed = np.zeros((len(rows), target_width), dtype=bool)
    target_weight = np.zeros((len(rows), target_width), dtype=np.float32)

    for i, (row, queries) in enumerate(zip(rows, query_sets)):
        found = np.flatnonzero(visible[i])
        axis[i, :len(found)] = found
        value[i, :len(found)] = data.values[row, found]
        padding[i, :len(found)] = False

        count = len(queries)
        target_axis[i, :count] = queries
        target_padding[i, :count] = False
        if with_targets:
            target_value[i, :count] = data.values[row, queries]
            target_observed[i, :count] = data.observed[row, queries]
            target_weight[i, :count] = data.weights[row, queries]

    result = (
        torch.from_numpy(axis).to(device),
        torch.from_numpy(value).to(device),
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


class MaskedAxisTokenTransformer(nn.Module):
    """Predict masked values at target-axis tokens, following scGPT's GEP role."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.axis_count = axis_count
        self.food_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
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
        # Match scGPT's shared per-token expression decoder shape and activation.
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.food_token, std=0.02)
        nn.init.normal_(self.mask_value, std=0.02)

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

        observed = self.axis_embedding(axis) + self.value_encoder(value.unsqueeze(-1))
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
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, encoded[:, 0]

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(axis, value, padding, target_axis, target_padding)[0]

"""Learned latent set bottleneck for target-axis completion."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v5 import TargetCrossAttentionBlock


class LatentInducingTransformer(nn.Module):
    """Compress the visible set into latents, then decode target-axis queries."""

    def __init__(self, axis_count: int, config: Config, latent_count: int = 16):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if latent_count < 1:
            raise ValueError("latent_count must be positive")
        self.axis_count = axis_count
        self.latent_count = latent_count
        self.context_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_value = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.latent = nn.Parameter(torch.zeros(1, latent_count, config.d_model))
        self.axis_embedding = nn.Embedding(axis_count, config.d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.latent_reader = TargetCrossAttentionBlock(config)
        latent_layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_heads, config.feedforward_dim,
            config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.latent_encoder = nn.TransformerEncoder(
            latent_layer, config.n_layers, norm=nn.LayerNorm(config.d_model),
        )
        self.target_reader = TargetCrossAttentionBlock(config)
        self.target_norm = nn.LayerNorm(config.d_model)
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, config.d_model), nn.LeakyReLU(),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.context_token, std=0.02)
        nn.init.normal_(self.mask_value, std=0.02)
        nn.init.normal_(self.latent, std=0.02)

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

        visible = self.axis_embedding(axis) + self.value_encoder(value.unsqueeze(-1))
        visible = torch.cat((
            self.context_token.expand(len(axis), -1, -1), visible,
        ), dim=1)
        visible_padding = torch.cat((
            torch.zeros((len(axis), 1), dtype=torch.bool, device=axis.device),
            padding,
        ), dim=1)
        latent = self.latent.expand(len(axis), -1, -1)
        latent = self.latent_reader(latent, visible, visible_padding)
        latent = self.latent_encoder(latent)

        query = self.axis_embedding(target_axis) + self.mask_value
        latent_padding = torch.zeros(
            (len(axis), self.latent_count), dtype=torch.bool, device=axis.device)
        query = self.target_reader(query, latent, latent_padding)
        query = self.target_norm(query)
        prediction = self.value_decoder(query).squeeze(-1)
        prediction = prediction.masked_fill(target_padding, 0.0)
        return prediction, latent, query

    def forward(self, axis, value, padding, target_axis, target_padding):
        return self.forward_details(
            axis, value, padding, target_axis, target_padding)[0]

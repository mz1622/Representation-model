"""ReGLU masked-axis completion with an scGPT-style MVC auxiliary loss."""
from __future__ import annotations

import torch
from torch import nn

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v7 import GatedMaskedAxisTransformer


class MVCInnerProductDecoder(nn.Module):
    """scGPT inner-product MVC decoder for a sample embedding and axis IDs."""

    def __init__(self, d_model: int):
        super().__init__()
        self.axis_to_query = nn.Linear(d_model, d_model)
        self.query_activation = nn.Sigmoid()
        self.sample_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, sample: torch.Tensor,
                axis_embedding: torch.Tensor) -> torch.Tensor:
        query = self.query_activation(self.axis_to_query(axis_embedding))
        projected = self.sample_projection(query)
        return torch.bmm(projected, sample.unsqueeze(2)).squeeze(2)


class GatedMvcMaskedAxisTransformer(GatedMaskedAxisTransformer):
    """Use MVC only as training regularization; primary output stays axis-token based."""

    def __init__(self, axis_count: int, config: Config):
        super().__init__(axis_count, config)
        self.mvc_decoder = MVCInnerProductDecoder(config.d_model)

    def forward_details(self, axis: torch.Tensor, value: torch.Tensor,
                        padding: torch.Tensor, target_axis: torch.Tensor,
                        target_padding: torch.Tensor):
        primary, context, target_hidden = super().forward_details(
            axis, value, padding, target_axis, target_padding)
        mvc = self.mvc_decoder(context, self.axis_embedding(target_axis))
        mvc = mvc.masked_fill(target_padding, 0.0)
        return primary, mvc, context, target_hidden

    def forward(self, axis, value, padding, target_axis, target_padding):
        # Evaluation and exported predictions always use target-token completion.
        return self.forward_details(
            axis, value, padding, target_axis, target_padding)[0]

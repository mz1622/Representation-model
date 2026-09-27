"""Optional nonlinear axis-query readout; not yet a registered training candidate.

The caller retains its original linear head and adds this residual. Context is
encoded once, independently of the requested query set. No target value,
availability, source or food-name metadata enters the query path.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AxisQueryResidual(nn.Module):
    def __init__(self, context_dim, axis_count, query_dim=32, hidden_dim=128):
        super().__init__()
        if any(not isinstance(x, int) or x < 1 for x in [context_dim, axis_count, query_dim, hidden_dim]):
            raise ValueError("Positive integer query-head dimensions required.")
        self.axis_count = axis_count
        self.context_dim = context_dim
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.axis_embedding = nn.Embedding(axis_count, query_dim)
        self.joint = nn.Sequential(nn.Linear(hidden_dim + query_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        # Exact zero contribution at initialization preserves the parent's function.
        # Earlier residual layers first receive nonzero gradients after this layer moves.
        nn.init.zeros_(self.joint[-1].weight)
        nn.init.zeros_(self.joint[-1].bias)

    def forward(self, context, query_axes=None):
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("Expected batch by context-dimension representations.")
        if query_axes is None:
            query_axes = torch.arange(self.axis_count, device=context.device)
        if query_axes.ndim != 1 or query_axes.dtype != torch.long or query_axes.device != context.device:
            raise ValueError("Query axes must be a one-dimensional long tensor on the context device.")
        if (query_axes < 0).any() or (query_axes >= self.axis_count).any():
            raise ValueError("Unknown requested axis.")
        queries = self.axis_embedding(query_axes)
        hidden = self.context_projection(context)
        # Factor the first joint linear map: its context term is identical for all
        # requested axes and need not be multiplied once per batch/axis pair.
        linear = self.joint[0]
        context_term = F.linear(hidden, linear.weight[:, :hidden.shape[1]], linear.bias)
        query_term = F.linear(queries, linear.weight[:, hidden.shape[1]:])
        joint_hidden = self.joint[1](context_term[:, None] + query_term[None])
        return self.joint[-1](joint_hidden).squeeze(-1)

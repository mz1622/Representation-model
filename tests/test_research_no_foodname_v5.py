"""Tests for target-axis cross-attention completion."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v5 import TargetQueryTransformer  # noqa: E402


def test_prediction_is_decoded_from_each_target_axis_query():
    model = TargetQueryTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    axis = torch.tensor([[0, 1]], dtype=torch.long)
    value = torch.tensor([[0.3, 0.7]])
    padding = torch.tensor([[False, False]])
    target_axis = torch.tensor([[2, 3]], dtype=torch.long)
    target_padding = torch.tensor([[False, False]])
    with torch.no_grad():
        prediction, context, query = model.forward_details(
            axis, value, padding, target_axis, target_padding)
    assert prediction.shape == (1, 2)
    assert context.shape == (1, 8)
    assert query.shape == (1, 2, 8)
    assert torch.isfinite(prediction).all()


def test_empty_visible_set_is_supported_by_context_memory_only():
    model = TargetQueryTransformer(
        3, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor([[0]], dtype=torch.long), torch.tensor([[0.0]]),
            torch.tensor([[True]]), torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[False]]),
        )
    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all()


def test_padded_target_query_is_zeroed():
    model = TargetQueryTransformer(
        3, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor([[0]], dtype=torch.long), torch.tensor([[1.0]]),
            torch.tensor([[False]]), torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[False, True]]),
        )
    assert prediction[0, 1].item() == 0.0

"""Tests for the target-axis-specific residual decoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v10 import (  # noqa: E402
    AxisSpecificGatedTransformer, AxisSpecificResidualDecoder,
)


def test_axis_residual_starts_at_zero_and_is_axis_specific():
    decoder = AxisSpecificResidualDecoder(3, 4)
    hidden = torch.ones(1, 2, 4)
    axes = torch.tensor([[0, 1]])
    torch.testing.assert_close(decoder(hidden, axes), torch.zeros(1, 2))
    with torch.no_grad():
        decoder.bias.weight[1] = 2.0
    torch.testing.assert_close(decoder(hidden, axes), torch.tensor([[0.0, 2.0]]))


def test_model_prediction_remains_target_token_based():
    model = AxisSpecificGatedTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        prediction, context, target_hidden, shared = model.forward_details(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
            torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
            torch.tensor([[False, False]]),
        )
    torch.testing.assert_close(prediction, shared)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)

"""Tests for target-isolated encoder-decoder attention."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v28 import (  # noqa: E402
    TargetIsolatedTransformer,
)


def test_other_target_tokens_do_not_change_a_target_prediction_or_context():
    model = TargetIsolatedTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=2,
               feedforward_dim=16, dropout=0),
    ).eval()
    axis = torch.tensor([[0, 1]])
    value = torch.tensor([[0.2, 0.8]])
    padding = torch.tensor([[False, False]])
    with torch.no_grad():
        single, single_context, _ = model.forward_details(
            axis, value, padding,
            torch.tensor([[2]]), torch.tensor([[False]]),
        )
        multiple, multiple_context, _ = model.forward_details(
            axis, value, padding,
            torch.tensor([[2, 3]]), torch.tensor([[False, False]]),
        )
    torch.testing.assert_close(single[:, 0], multiple[:, 0])
    torch.testing.assert_close(single_context, multiple_context)


def test_visible_token_permutation_does_not_change_predictions():
    model = TargetIsolatedTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=2,
               feedforward_dim=16, dropout=0),
    ).eval()
    axis = torch.tensor([[0, 1]])
    value = torch.tensor([[0.2, 0.8]])
    padding = torch.tensor([[False, False]])
    target = torch.tensor([[2, 3]])
    target_padding = torch.tensor([[False, False]])
    with torch.no_grad():
        first = model(axis, value, padding, target, target_padding)
        second = model(
            axis[:, [1, 0]], value[:, [1, 0]], padding[:, [1, 0]],
            target, target_padding,
        )
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

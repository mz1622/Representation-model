"""Tests for learned latent set completion."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v6 import LatentInducingTransformer  # noqa: E402


def model():
    return LatentInducingTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0), latent_count=3,
    ).eval()


def test_targets_are_decoded_from_axis_queries_after_latent_bottleneck():
    network = model()
    with torch.no_grad():
        prediction, latent, query = network.forward_details(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
            torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
            torch.tensor([[False, False]]),
        )
    assert prediction.shape == (1, 2)
    assert latent.shape == (1, 3, 8)
    assert query.shape == (1, 2, 8)
    assert torch.isfinite(prediction).all()


def test_visible_set_order_does_not_change_prediction():
    network = model()
    target_axis = torch.tensor([[2]])
    target_padding = torch.tensor([[False]])
    with torch.no_grad():
        first = network(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
            torch.tensor([[False, False]]), target_axis, target_padding,
        )
        second = network(
            torch.tensor([[1, 0]]), torch.tensor([[0.8, 0.2]]),
            torch.tensor([[False, False]]), target_axis, target_padding,
        )
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

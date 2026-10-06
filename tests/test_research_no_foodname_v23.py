"""Tests for target-specific scalar layer routing."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v23 import (  # noqa: E402
    TargetLayerRoutingTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_zero_layer_routing_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(53)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(53)
    candidate = TargetLayerRoutingTransformer(4, config).eval()
    assert torch.count_nonzero(candidate.target_layer_gate) == 0
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_layer_routing_is_target_specific():
    config = Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(59)
    model = TargetLayerRoutingTransformer(4, config).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.target_layer_gate[2, 0] = 1.0
        changed = model(*_inputs())
    assert not torch.allclose(baseline[:, 0], changed[:, 0])
    torch.testing.assert_close(baseline[:, 1], changed[:, 1])

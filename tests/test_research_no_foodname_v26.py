"""Tests for routed prior-residual decoding."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v23 import (  # noqa: E402
    TargetLayerRoutingTransformer,
)
from foodcomp.research_no_foodname_v26 import (  # noqa: E402
    RoutedPriorResidualTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_combined_model_starts_as_routing_plus_prior():
    config = Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )
    prior = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    torch.manual_seed(73)
    parent = TargetLayerRoutingTransformer(4, config).eval()
    torch.manual_seed(73)
    candidate = RoutedPriorResidualTransformer(4, config, prior).eval()
    with torch.no_grad():
        expected = parent(*_inputs()) + torch.tensor([[0.2, 0.3]])
        torch.testing.assert_close(expected, candidate(*_inputs()))
    assert torch.count_nonzero(candidate.target_layer_gate) == 0

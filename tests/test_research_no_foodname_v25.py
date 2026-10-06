"""Tests for train-only axis-prior residual decoding."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v25 import (  # noqa: E402
    PriorResidualTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_prior_residual_adds_target_axis_prior():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    prior = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    torch.manual_seed(71)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(71)
    candidate = PriorResidualTransformer(4, config, prior).eval()
    with torch.no_grad():
        expected = parent(*_inputs()) + torch.tensor([[0.2, 0.3]])
        torch.testing.assert_close(expected, candidate(*_inputs()))


def test_prior_is_a_persistent_nontrainable_buffer():
    model = PriorResidualTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=1,
               feedforward_dim=16, dropout=0),
        np.arange(4, dtype=np.float32),
    )
    assert "axis_prior" in model.state_dict()
    assert "axis_prior" not in dict(model.named_parameters())

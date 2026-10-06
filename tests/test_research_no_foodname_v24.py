"""Tests for multiplicative axis/value token interaction."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v24 import (  # noqa: E402
    MultiplicativeAxisValueTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_zero_axis_value_gate_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(61)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(61)
    candidate = MultiplicativeAxisValueTransformer(4, config).eval()
    assert torch.count_nonzero(candidate.axis_value_gate) == 0
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_multiplicative_gate_changes_visible_token_semantics():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(67)
    model = MultiplicativeAxisValueTransformer(4, config).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.axis_value_gate.fill_(1.0)
        changed = model(*_inputs())
    assert not torch.allclose(baseline, changed)

"""Tests for the direct quantitative relation residual."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v20 import (  # noqa: E402
    DirectRelationResidualTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_zero_relation_residual_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(29)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(29)
    candidate = DirectRelationResidualTransformer(4, config).eval()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name])
    assert torch.count_nonzero(candidate.direct_value_relation) == 0
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_relation_residual_uses_visible_value_and_target_axis():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(31)
    model = DirectRelationResidualTransformer(4, config).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.direct_value_relation[2, 0] = 1.0
        changed = model(*_inputs())
    expected_delta = torch.tensor([[0.2 / (2 ** 0.5), 0.0]])
    torch.testing.assert_close(changed - baseline, expected_delta)

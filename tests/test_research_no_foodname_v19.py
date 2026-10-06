"""Tests for axis-conditioned continuous-value FiLM."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v19 import (  # noqa: E402
    AxisConditionedValueTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_zero_initialized_film_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(19)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(19)
    candidate = AxisConditionedValueTransformer(4, config).eval()

    parent_state = parent.state_dict()
    candidate_state = candidate.state_dict()
    for name, value in parent_state.items():
        torch.testing.assert_close(value, candidate_state[name])
    assert torch.count_nonzero(candidate.value_scale.weight) == 0
    assert torch.count_nonzero(candidate.value_shift.weight) == 0

    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_axis_conditioning_changes_visible_numeric_semantics():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(23)
    model = AxisConditionedValueTransformer(4, config).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.value_scale.weight[0].fill_(0.5)
        model.value_shift.weight[1].fill_(0.25)
        changed = model(*_inputs())
    assert not torch.allclose(baseline, changed)

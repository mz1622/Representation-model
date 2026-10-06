"""Tests for the per-axis affine output decoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v29 import (  # noqa: E402
    AxisCalibratedTransformer,
)


def test_identity_calibration_exactly_matches_parent_initialization():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(19)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(19)
    candidate = AxisCalibratedTransformer(4, config).eval()
    axis = torch.tensor([[0, 1]])
    value = torch.tensor([[0.2, 0.8]])
    padding = torch.tensor([[False, False]])
    target_axis = torch.tensor([[2, 3]])
    target_padding = torch.tensor([[False, False]])
    with torch.no_grad():
        expected = parent(
            axis, value, padding, target_axis, target_padding
        )
        actual = candidate(
            axis, value, padding, target_axis, target_padding
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(candidate.axis_log_scale) == 0
    assert torch.count_nonzero(candidate.axis_bias) == 0


def test_calibration_is_axis_specific_and_preserves_padding():
    model = AxisCalibratedTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=1,
               feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        model.axis_log_scale[2] = torch.log(torch.tensor(2.0))
        model.axis_bias[2] = 1.0
    axis = torch.tensor([[0, 1]])
    value = torch.tensor([[0.2, 0.8]])
    padding = torch.tensor([[False, False]])
    target_axis = torch.tensor([[2, 3]])
    target_padding = torch.tensor([[False, True]])
    with torch.no_grad():
        base, _, _ = AxisPairBiasMaskedAxisTransformer.forward_details(
            model, axis, value, padding, target_axis, target_padding
        )
        actual = model(
            axis, value, padding, target_axis, target_padding
        )
    torch.testing.assert_close(actual[:, 0], base[:, 0] * 2 + 1)
    assert actual[:, 1].item() == 0

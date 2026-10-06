"""Tests for the V34-V36 V18 attention descendants."""
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v34 import (  # noqa: E402
    TargetQueryResidualTransformer,
)
from foodcomp.research_no_foodname_v35 import (  # noqa: E402
    DisentangledAxisContentTransformer,
)
from foodcomp.research_no_foodname_v36 import (  # noqa: E402
    RelationValueAxisPairTransformer,
)


VARIANTS = (
    TargetQueryResidualTransformer,
    DisentangledAxisContentTransformer,
    RelationValueAxisPairTransformer,
)


def config():
    return Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )


def inputs():
    return (
        torch.tensor([[0, 1, 4], [3, 2, 0]]),
        torch.tensor([[0.2, 0.8, 99.0], [0.5, 88.0, 77.0]]),
        torch.tensor([[False, False, True], [False, True, True]]),
        torch.tensor([[2, 3], [1, 4]]),
        torch.tensor([[False, False], [False, True]]),
    )


@pytest.mark.parametrize("model_class", VARIANTS)
def test_zero_initialized_variant_matches_v18(model_class):
    torch.manual_seed(17)
    parent = AxisPairBiasMaskedAxisTransformer(5, config()).eval()
    parent_rng_state = torch.get_rng_state()
    torch.manual_seed(17)
    candidate = model_class(5, config()).eval()
    candidate_rng_state = torch.get_rng_state()
    torch.testing.assert_close(candidate_rng_state, parent_rng_state)
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    with torch.no_grad():
        expected = parent(*inputs())
        actual = candidate(*inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", VARIANTS)
def test_visible_permutation_and_padding_do_not_change_predictions(model_class):
    torch.manual_seed(31)
    model = model_class(5, config()).eval()
    axis, value, padding, target_axis, target_padding = inputs()
    permutation = torch.tensor([1, 0, 2])
    altered_axis = axis.clone()
    altered_value = value.clone()
    altered_axis[padding] = 0
    altered_value[padding] = -1234.0
    with torch.no_grad():
        original = model(axis, value, padding, target_axis, target_padding)
        permuted = model(
            axis[:, permutation], value[:, permutation], padding[:, permutation],
            target_axis, target_padding,
        )
        altered_padding = model(
            altered_axis, altered_value, padding, target_axis, target_padding
        )
    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(original, altered_padding, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(original[target_padding]) == 0


@pytest.mark.parametrize("model_class", VARIANTS)
def test_checkpoint_round_trip_preserves_predictions(model_class):
    torch.manual_seed(37)
    model = model_class(5, config()).eval()
    with torch.no_grad():
        expected = model(*inputs())
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = model_class(5, config()).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        actual = restored(*inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_new_parameter_counts_match_predeclared_attention_changes():
    parent = AxisPairBiasMaskedAxisTransformer(5, config())
    parent_count = sum(parameter.numel() for parameter in parent.parameters())
    expected = {
        TargetQueryResidualTransformer: 2 * (8 * 8 + 8),
        DisentangledAxisContentTransformer: 2 * 2 * 8 * 8,
        RelationValueAxisPairTransformer: 2 * 2 * (5 + 1) * 8,
    }
    for model_class, increment in expected.items():
        model = model_class(5, config())
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count - parent_count == increment


def test_zero_start_mechanisms_receive_gradients():
    for model_class, parameter_name in (
        (TargetQueryResidualTransformer, "target_query.weight"),
        (DisentangledAxisContentTransformer, "axis_query.weight"),
        (RelationValueAxisPairTransformer, "relation_value"),
    ):
        torch.manual_seed(41)
        model = model_class(5, config())
        model(*inputs()).sum().backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.endswith(parameter_name)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)

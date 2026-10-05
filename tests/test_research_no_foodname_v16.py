"""Tests for learned axis-pair attention bias."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v7 import GatedMaskedAxisTransformer  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer, AxisPairBiasSelfAttention,
)


def test_zero_pair_bias_matches_standard_attention():
    torch.manual_seed(5)
    module = AxisPairBiasSelfAttention(
        4, Config(d_model=8, n_heads=2, dropout=0)
    ).eval()
    tokens = torch.randn(2, 3, 8)
    padding = torch.tensor([[False, False, True], [False, True, True]])
    axes = torch.tensor([[0, 1, 4], [2, 4, 4]])
    with torch.no_grad():
        expected, _ = module.projections(
            tokens, tokens, tokens, key_padding_mask=padding,
            need_weights=False,
        )
        actual = module(tokens, padding, axes)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_pair_bias_model_is_permutation_invariant_and_decodes_targets():
    torch.manual_seed(9)
    model = AxisPairBiasMaskedAxisTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    axis = torch.tensor([[0, 1]])
    value = torch.tensor([[0.2, 0.8]])
    padding = torch.tensor([[False, False]])
    target_axis = torch.tensor([[2, 3]])
    target_padding = torch.tensor([[False, False]])
    with torch.no_grad():
        first, context, target_hidden = model.forward_details(
            axis, value, padding, target_axis, target_padding
        )
        second = model(
            axis[:, [1, 0]], value[:, [1, 0]], padding[:, [1, 0]],
            target_axis, target_padding,
        )
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    assert first.shape == (1, 2)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)


def test_zero_bias_addition_preserves_v7_parameter_initialization():
    config = Config(d_model=8, n_heads=2, n_layers=1,
                    feedforward_dim=16, dropout=0)
    torch.manual_seed(11)
    parent = GatedMaskedAxisTransformer(4, config)
    torch.manual_seed(11)
    candidate = AxisPairBiasMaskedAxisTransformer(4, config)
    torch.testing.assert_close(parent.axis_embedding.weight,
                               candidate.axis_embedding.weight)
    for left, right in zip(parent.value_encoder.parameters(),
                           candidate.value_encoder.parameters()):
        torch.testing.assert_close(left, right)
    for left, right in zip(parent.blocks[0].ffn.parameters(),
                           candidate.blocks[0].ffn.parameters()):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(
        parent.blocks[0].attention.in_proj_weight,
        candidate.blocks[0].attention.projections.in_proj_weight,
    )
    assert torch.count_nonzero(candidate.blocks[0].attention.pair_bias) == 0

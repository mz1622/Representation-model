"""Tests for Talking-Heads ReGLU completion."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v7 import GatedMaskedAxisTransformer  # noqa: E402
from foodcomp.research_no_foodname_v17 import (  # noqa: E402
    TalkingHeadsMaskedAxisTransformer, TalkingHeadsSelfAttention,
)


def test_identity_head_mixing_matches_standard_attention():
    torch.manual_seed(5)
    module = TalkingHeadsSelfAttention(
        Config(d_model=8, n_heads=2, dropout=0)
    ).eval()
    tokens = torch.randn(2, 3, 8)
    padding = torch.tensor([[False, False, True], [False, True, True]])
    with torch.no_grad():
        expected, _ = module.projections(
            tokens, tokens, tokens, key_padding_mask=padding,
            need_weights=False,
        )
        actual = module(tokens, padding)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_talking_heads_preserves_v7_initialization_and_decodes_targets():
    config = Config(d_model=8, n_heads=2, n_layers=1,
                    feedforward_dim=16, dropout=0)
    torch.manual_seed(11)
    parent = GatedMaskedAxisTransformer(4, config)
    torch.manual_seed(11)
    model = TalkingHeadsMaskedAxisTransformer(4, config).eval()
    torch.testing.assert_close(parent.axis_embedding.weight,
                               model.axis_embedding.weight)
    for left, right in zip(parent.blocks[0].ffn.parameters(),
                           model.blocks[0].ffn.parameters()):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(
        parent.blocks[0].attention.in_proj_weight,
        model.blocks[0].attention.projections.in_proj_weight,
    )
    with torch.no_grad():
        prediction, context, target_hidden = model.forward_details(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
            torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
            torch.tensor([[False, False]]),
        )
    assert prediction.shape == (1, 2)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)

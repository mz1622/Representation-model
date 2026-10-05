"""Tests for QK-normalized ReGLU completion."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v15 import (  # noqa: E402
    QKNormMaskedAxisTransformer, QKNormSelfAttention,
)


def test_qk_attention_ignores_padded_keys_and_is_permutation_equivariant():
    torch.manual_seed(7)
    attention = QKNormSelfAttention(
        Config(d_model=8, n_heads=2, dropout=0)
    ).eval()
    tokens = torch.randn(1, 3, 8)
    padding = torch.tensor([[False, False, True]])
    altered = tokens.clone()
    altered[:, 2] = 1e3
    with torch.no_grad():
        first = attention(tokens, padding)
        second = attention(altered, padding)
        permuted = attention(tokens[:, [1, 0, 2]], padding[:, [1, 0, 2]])
    torch.testing.assert_close(first[:, :2], second[:, :2])
    torch.testing.assert_close(first[:, [1, 0, 2]], permuted)


def test_qk_model_decodes_target_axis_states():
    model = QKNormMaskedAxisTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        prediction, context, target_hidden = model.forward_details(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
            torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
            torch.tensor([[False, False]]),
        )
    assert prediction.shape == (1, 2)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)
    assert torch.isfinite(prediction).all()

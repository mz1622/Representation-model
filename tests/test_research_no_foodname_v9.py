"""Tests for scGPT MVC auxiliary training."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v9 import (  # noqa: E402
    GatedMvcMaskedAxisTransformer, MVCInnerProductDecoder,
)


def test_mvc_inner_product_matches_explicit_product():
    decoder = MVCInnerProductDecoder(4).eval()
    sample = torch.randn(2, 4)
    axes = torch.randn(2, 3, 4)
    with torch.no_grad():
        actual = decoder(sample, axes)
        query = torch.sigmoid(decoder.axis_to_query(axes))
        expected = (decoder.sample_projection(query) * sample[:, None]).sum(-1)
    torch.testing.assert_close(actual, expected)


def test_exported_forward_is_primary_axis_token_prediction():
    model = GatedMvcMaskedAxisTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    inputs = (
        torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )
    with torch.no_grad():
        primary, mvc, context, target_hidden = model.forward_details(*inputs)
        exported = model(*inputs)
    torch.testing.assert_close(exported, primary)
    assert mvc.shape == primary.shape == (1, 2)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)

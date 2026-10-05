"""Tests for SwiGLU masked-axis completion."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v13 import (  # noqa: E402
    SwiGLU, SwiGLUMaskedAxisTransformer,
)


def test_swiglu_shape_and_values():
    value = torch.tensor([[[2.0, -3.0, 1.0, -2.0]]])
    expected = value[..., :2] * F.silu(value[..., 2:])
    torch.testing.assert_close(SwiGLU()(value), expected)


def test_swiglu_model_decodes_target_axis_states():
    model = SwiGLUMaskedAxisTransformer(
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

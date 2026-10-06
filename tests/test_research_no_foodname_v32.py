"""Tests for train-only axis-prior shrinkage."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v32 import (  # noqa: E402
    AxisPriorShrinkageTransformer,
)


def test_initial_decoder_is_exact_axis_prior_shrinkage():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    prior = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    torch.manual_seed(29)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(29)
    candidate = AxisPriorShrinkageTransformer(
        4, config, prior, initial_retention=0.95
    ).eval()
    inputs = (
        torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )
    with torch.no_grad():
        base = parent(*inputs)
        actual = candidate(*inputs)
    expected_prior = torch.tensor([[0.3, 0.4]])
    torch.testing.assert_close(
        actual, expected_prior + 0.95 * (base - expected_prior)
    )


def test_padding_remains_zero_after_shrinkage():
    model = AxisPriorShrinkageTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=1,
               feedforward_dim=16, dropout=0),
        np.arange(4, dtype=np.float32),
    ).eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor([[0]]), torch.tensor([[0.2]]),
            torch.tensor([[False]]), torch.tensor([[2, 3]]),
            torch.tensor([[False, True]]),
        )
    assert prediction[:, 1].item() == 0

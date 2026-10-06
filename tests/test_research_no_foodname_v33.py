"""Tests for train-only analytical axis shrinkage."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v33 import (  # noqa: E402
    TrainCalibratedShrinkageTransformer, fit_axis_retention, weighted_median,
)


def test_weighted_median_and_exact_l1_axis_retention():
    assert weighted_median([1, 2, 9], [1, 4, 1]) == 2
    retention, support = fit_axis_retention(
        prediction=np.array([2.0, 4.0, 2.0, 4.0]),
        target=np.array([1.0, 2.0, 0.5, 1.0]),
        weight=np.ones(4),
        axis=np.array([0, 0, 1, 1]),
        axis_prior=np.zeros(3),
        axis_count=3,
    )
    assert retention[0] == pytest.approx(0.5)
    assert retention[1] == pytest.approx(0.25)
    assert retention[2] == 1
    np.testing.assert_array_equal(support, [2, 2, 0])


def test_model_applies_axis_shrinkage_and_preserves_padding():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    prior = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    retention = np.array([1.0, 1.0, 0.25, 0.75], dtype=np.float32)
    torch.manual_seed(31)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(31)
    model = TrainCalibratedShrinkageTransformer(
        4, config, prior, retention
    ).eval()
    inputs = (
        torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]), torch.tensor([[2, 3]]),
        torch.tensor([[False, True]]),
    )
    with torch.no_grad():
        base = parent(*inputs)
        actual = model(*inputs)
    torch.testing.assert_close(
        actual[:, 0], torch.tensor(0.3) + 0.25 * (base[:, 0] - 0.3)
    )
    assert actual[:, 1].item() == 0

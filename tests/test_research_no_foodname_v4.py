"""Tests for evidence-led no-name architecture candidates."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v4 import (  # noqa: E402
    PiecewiseMaskedAxisTransformer, training_ple_bins,
)


def example():
    values = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    return SimpleNamespace(
        axes=pd.DataFrame({"axis_index": [0, 1]}),
        values=values,
        observed=np.ones_like(values, dtype=bool),
        train=np.arange(4),
    )


def test_ple_bins_are_train_only_deduplicated_and_continuous():
    bins = training_ple_bins(example(), np.arange(4), bins=8)
    left, right, mask, edges = bins
    assert len(edges[0]) < 9  # repeated zero quantiles were removed
    assert np.all(right[mask] > left[mask])
    model = PiecewiseMaskedAxisTransformer(
        2, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0), bins).eval()
    axis = torch.tensor([[0]], dtype=torch.long)
    below = model.value_embedding(axis, torch.tensor([[0.4999]]))
    above = model.value_embedding(axis, torch.tensor([[0.5001]]))
    assert torch.max(torch.abs(above - below)).item() < 0.01


def test_piecewise_model_predicts_at_target_axis_positions():
    bins = training_ple_bins(example(), np.arange(4), bins=8)
    model = PiecewiseMaskedAxisTransformer(
        2, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0), bins).eval()
    axis = torch.tensor([[0]], dtype=torch.long)
    value = torch.tensor([[1.0]])
    padding = torch.tensor([[False]])
    target_axis = torch.tensor([[1]], dtype=torch.long)
    target_padding = torch.tensor([[False]])
    with torch.no_grad():
        prediction, food = model.forward_details(
            axis, value, padding, target_axis, target_padding)
    assert prediction.shape == (1, 1)
    assert food.shape == (1, 8)
    assert torch.isfinite(prediction).all()

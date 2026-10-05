"""Invariants for scGPT-aligned masked-axis completion."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v3 import (  # noqa: E402
    MaskedAxisTokenTransformer, masked_axis_batch,
)


def example():
    raw = np.array([[1.0, 0.0, np.nan, 3.0], [np.nan, 2.0, 4.0, 0.0]])
    observed = np.isfinite(raw)
    return SimpleNamespace(
        axes=pd.DataFrame({"axis_index": range(4)}),
        profiles=pd.DataFrame({"original_name": ["apple", "pear"]}),
        observed=observed,
        values=np.where(observed, raw, 0).astype(np.float32),
        raw=raw,
        train=np.array([0]),
        weights=observed.astype(np.float32),
        targets=np.array([0, 1, 2, 3]),
        families=np.array(["sugar", "sugar", "mineral", "mineral"]),
    )


def test_masked_query_grid_ignores_names_labels_and_missingness():
    data = example()
    rows = np.array([0, 1])
    family = np.array(["mineral", "mineral"])
    first = masked_axis_batch(data, rows, family, "cpu", with_targets=True)
    assert first[3].tolist() == [[2, 3], [2, 3]]
    assert not first[4].any()
    assert first[6].tolist() == [[False, True], [True, True]]

    data.profiles["original_name"] = ["changed", "also changed"]
    data.values[:, 2:] = 999.0
    second = masked_axis_batch(data, rows, family, "cpu", with_targets=False)
    for left, right in zip(first[:5], second):
        torch.testing.assert_close(left, right)


def test_predictions_belong_to_target_tokens_and_are_permutation_equivariant():
    torch.manual_seed(9)
    config = Config(d_model=16, n_heads=4, n_layers=1,
                    feedforward_dim=32, dropout=0)
    model = MaskedAxisTokenTransformer(4, config).eval()
    axis = torch.tensor([[0, 1]], dtype=torch.long)
    value = torch.tensor([[0.2, 0.7]])
    padding = torch.zeros((1, 2), dtype=torch.bool)
    target_axis = torch.tensor([[2, 3]], dtype=torch.long)
    target_padding = torch.zeros((1, 2), dtype=torch.bool)
    with torch.no_grad():
        prediction, food = model.forward_details(
            axis, value, padding, target_axis, target_padding)
        visible_permutation = torch.tensor([1, 0])
        target_permutation = torch.tensor([1, 0])
        reordered, reordered_food = model.forward_details(
            axis[:, visible_permutation], value[:, visible_permutation],
            padding[:, visible_permutation], target_axis[:, target_permutation],
            target_padding[:, target_permutation])
        empty_prediction = model(
            torch.zeros_like(axis), torch.zeros_like(value),
            torch.ones_like(padding), target_axis, target_padding)
    assert prediction.shape == (1, 2)
    assert food.shape == (1, config.d_model)
    torch.testing.assert_close(prediction[:, target_permutation], reordered,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(food, reordered_food, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(empty_prediction).all()

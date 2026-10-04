import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config, CompositionSetTransformer, dense_features, numeric_only_training_view, set_batch,
)


def example():
    raw = np.array([[1.0, 0.0, np.nan, 3.0], [np.nan, 2.0, 4.0, 0.0]])
    observed = np.isfinite(raw)
    return SimpleNamespace(
        axes=pd.DataFrame({"axis_index": range(4)}),
        profiles=pd.DataFrame({"original_name": ["apple", "pear"]}),
        observed=observed, values=np.where(observed, raw, 0).astype(np.float32),
        raw=raw, train=np.array([0]),
        weights=np.ones((2, 4), dtype=np.float32), targets=np.array([0, 1, 2, 3]),
        families=np.array(["sugar", "sugar", "mineral", "mineral"]),
    )


def test_names_never_enter_model_or_baseline_inputs():
    data = example()
    family = np.array(["sugar", "mineral"])
    original_set = set_batch(data, np.array([0, 1]), family, "cpu")
    original_dense = dense_features(data, np.array([0, 1]), family)
    data.profiles["original_name"] = ["random changed name", "another name"]
    for left, right in zip(original_set, set_batch(data, np.array([0, 1]), family, "cpu")):
        torch.testing.assert_close(left, right)
    np.testing.assert_array_equal(original_dense, dense_features(data, np.array([0, 1]), family))
    # An explicit zero remains a visible value token; a missing cell has no token.
    assert original_set[2][0].logical_not().sum().item() == 1
    assert original_set[1][0, 0].item() == 3.0


def test_model_is_permutation_invariant_in_eval_mode():
    torch.manual_seed(4)
    model = CompositionSetTransformer(4, Config(d_model=16, n_heads=4, n_layers=1,
                                                 feedforward_dim=32, dropout=0)).eval()
    axis = torch.tensor([[0, 1, 3]])
    value = torch.tensor([[1.0, 0.0, 3.0]])
    padding = torch.tensor([[False, False, False]])
    with torch.no_grad():
        first = model(axis, value, padding)
        permutation = torch.tensor([2, 0, 1])
        second = model(axis[:, permutation], value[:, permutation], padding[:, permutation])
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_numeric_training_scale_and_weights_ignore_names():
    first = numeric_only_training_view(example())
    second = example()
    second.profiles["original_name"] = ["changed", "names"]
    second = numeric_only_training_view(second)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.weights[0, 1] == 1  # explicit zero is supervised
    assert first.weights[0, 2] == 0  # missing is never supervised

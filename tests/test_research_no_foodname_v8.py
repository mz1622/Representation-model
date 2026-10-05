"""Tests for missingness-aware masked-axis completion."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v8 import (  # noqa: E402
    MissingAwareMaskedAxisTransformer, missing_aware_axis_batch,
)


def test_batch_keeps_natural_missing_axes_but_excludes_target_family():
    data = SimpleNamespace(
        axes=pd.DataFrame({"axis_index": [0, 1, 2]}),
        families=np.array(["a", "b", "b"]),
        targets=np.array([0, 1, 2]),
        values=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        observed=np.array([[True, False, True]]),
        weights=np.ones((1, 3), dtype=np.float32),
    )
    packed = missing_aware_axis_batch(
        data, np.array([0]), np.array(["a"]), torch.device("cpu"),
        with_targets=True,
    )
    axis, value, observed, padding, target_axis, target_padding = packed[:6]
    assert axis.tolist() == [[1, 2]]
    assert value.tolist() == [[0.0, 0.0]]
    assert observed.tolist() == [[False, True]]
    assert padding.tolist() == [[False, False]]
    assert target_axis.tolist() == [[0]]
    assert target_padding.tolist() == [[False]]


def test_observed_zero_and_natural_missing_use_distinct_value_tokens():
    model = MissingAwareMaskedAxisTransformer(
        3, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    axis = torch.tensor([[0, 0]])
    value = torch.tensor([[0.0, 0.0]])
    observed = torch.tensor([[True, False]])
    numeric = model.value_encoder(value.unsqueeze(-1))
    tokens = torch.where(
        observed.unsqueeze(-1), numeric,
        model.missing_value.expand(1, 2, -1),
    )
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_prediction_is_still_decoded_from_target_axis_state():
    model = MissingAwareMaskedAxisTransformer(
        4, Config(d_model=8, n_heads=2, n_layers=1,
                  feedforward_dim=16, dropout=0),
    ).eval()
    with torch.no_grad():
        prediction, context, target_hidden = model.forward_details(
            torch.tensor([[0, 1]]), torch.tensor([[0.2, 0.0]]),
            torch.tensor([[True, False]]), torch.tensor([[False, False]]),
            torch.tensor([[2, 3]]), torch.tensor([[False, False]]),
        )
    assert prediction.shape == (1, 2)
    assert context.shape == (1, 8)
    assert target_hidden.shape == (1, 2, 8)
    assert torch.isfinite(prediction).all()

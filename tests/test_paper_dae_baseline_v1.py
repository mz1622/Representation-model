import numpy as np
import pandas as pd
import torch

from foodcomp.paper_dae_baseline_v1 import (
    PaperDenoisingAutoencoder,
    build_paper_axis_crosswalk,
    paper_external_features,
)


def test_paper_dae_shape_and_eval_determinism():
    model = PaperDenoisingAutoencoder(198)
    value = torch.randn(4, 198)
    model.eval()
    first = model(value)
    second = model(value)
    assert first.shape == value.shape
    torch.testing.assert_close(first, second)


def test_crosswalk_is_strict_and_converts_units():
    nutrient = pd.DataFrame({
        "id": [1003, 1087, 1008],
        "name": ["Protein", "Calcium, Ca", "Energy"],
        "unit_name": ["G", "MG", "KCAL"],
    })
    axes = pd.DataFrame({
        "axis_index": [0, 1],
        "canonical_name": ["Protein, total", "Calcium"],
        "loss_eligible": [True, True],
    })
    links, audit = build_paper_axis_crosswalk(["1003", "1087", "1008"], nutrient, axes)
    assert [link.axis_index for link in links] == [0, 1]
    assert [link.grams_to_paper for link in links] == [1.0, 1000.0]
    assert audit.status.tolist()[-1] == "no_strict_current_axis_equivalent"


def test_external_features_apply_dependency_mask():
    nutrient = pd.DataFrame({
        "id": [1003, 1087], "name": ["Protein", "Calcium, Ca"],
        "unit_name": ["G", "MG"],
    })
    axes = pd.DataFrame({
        "axis_index": [0, 1], "canonical_name": ["Protein, total", "Calcium"],
        "loss_eligible": [True, True],
    })
    links, _ = build_paper_axis_crosswalk(["1003", "1087"], nutrient, axes)

    class Data:
        observed = np.ones((1, 2), dtype=bool)
        raw = np.asarray([[12.0, 0.02]])

    class Policy:
        @staticmethod
        def mask_matrix(target):
            result = np.zeros((len(target), 2), dtype=bool)
            result[:, 1] = True
            return result

    result = paper_external_features(
        Data(), np.asarray([0]), np.asarray([1]), Policy(), links,
        np.asarray([3.0, 7.0]),
    )
    assert result.tolist() == [[12.0, 7.0]]

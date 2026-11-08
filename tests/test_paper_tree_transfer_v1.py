import numpy as np

from foodcomp.paper_dae_baseline_v1 import PaperAxisLink
from foodcomp.paper_tree_transfer_v1 import (
    build_external_feature_view,
    build_paper_training_view,
)


def _links():
    return [
        PaperAxisLink(0, 1003, "Protein", "G", "Protein, total", 0, True, 1.0, "test"),
        PaperAxisLink(1, 1087, "Calcium", "MG", "Calcium", 2, True, 1000.0, "test"),
    ]


def test_paper_view_converts_units_and_fits_training_only_scale():
    value = np.asarray([[10.0, 100.0], [20.0, 300.0], [1000.0, 900.0]])
    view = build_paper_training_view(value, _links(), 4, np.asarray([0, 1]))
    assert view.raw[:, 0].tolist() == [10.0, 20.0, 1000.0]
    assert np.allclose(view.raw[:, 2], [0.1, 0.3, 0.9])
    assert view.scale[0] == 15.0
    assert np.isclose(view.scale[2], 0.2)
    assert not view.observed[:, 1].any()


def test_external_view_hides_unmapped_axes_and_uses_paper_scale():
    class Current:
        raw = np.asarray([[30.0, 7.0, 0.4, 9.0]])
        observed = np.ones((1, 4), dtype=bool)

    scale = np.asarray([15.0, 1.0, 0.2, 1.0])
    view = build_external_feature_view(Current(), scale, [0, 2])
    assert view.observed.tolist() == [[True, False, True, False]]
    assert np.isclose(view.values[0, 0], np.log1p(2.0))
    assert np.isclose(view.values[0, 2], np.log1p(2.0))
    assert view.values[0, 1] == 0

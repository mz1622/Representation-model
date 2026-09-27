import numpy as np
import pytest
from foodcomp.research_translation import translation_summary


def test_large_common_offset_changes_unit_metric_but_not_predictions_or_residual_geometry():
    a = np.array([[1., 0., 100.], [-1., 0., 100.], [0., 1., 100.], [0., -1., 100.]])
    b = a[[2, 3, 1, 0]]
    w = np.array([[.7, -.2, 1.3], [.1, .8, -.4]])
    bias = np.array([1., -2.])
    result = translation_summary(a, b, np.ones(4), 4, 2, w, bias)
    original, centred, enlarged = result["coordinate_cases"]
    assert centred["geometry"]["axis_weighted_mean_view_distance"] == pytest.approx(1.)
    assert original["geometry"]["axis_weighted_mean_view_distance"] < .0002
    assert enlarged["geometry"]["axis_weighted_mean_view_distance"] < original["geometry"]["axis_weighted_mean_view_distance"] / 100
    assert result["pooled_centred_second_moment"] == pytest.approx(1.)
    assert result["weighted_pair_distance_over_centred_second_moment"] == pytest.approx(1.)
    for case in result["coordinate_cases"]:
        assert case["max_abs_prediction_error_A"] < 1e-10
        assert case["max_abs_prediction_error_B"] < 1e-10
        assert case["max_abs_pair_vector_error"] == 0
    shifted = translation_summary(a + 13., b + 13., np.ones(4), 4, 2, w, bias - w @ np.full(3, 13.))
    assert shifted["weighted_pair_distance_over_centred_second_moment"] == pytest.approx(result["weighted_pair_distance_over_centred_second_moment"])


def test_nonuniform_weights_and_inputs_are_not_mutated():
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=(2, 16, 5)); originals = a.copy(), b.copy()
    q = np.arange(1, 17)
    result = translation_summary(a, b, q, 32, 3, rng.normal(size=(7, 5)), np.zeros(7))
    expected = np.sum(q * .5*np.sum((a-b)**2, axis=1)) / q.sum()
    assert result["weighted_half_squared_pair_distance"] == pytest.approx(expected)
    assert np.array_equal(a, originals[0]) and np.array_equal(b, originals[1])


def test_zero_variance_is_explicitly_undefined_and_nonfinite_fails():
    a = np.ones((3, 2)); w = np.ones((1, 2))
    result = translation_summary(a, a, np.ones(3), 3, 1, w, np.zeros(1))
    assert result["variance_ratio_status"] == "undefined_zero_variance"
    assert result["weighted_pair_distance_over_centred_second_moment"] is None
    with pytest.raises(FloatingPointError):
        translation_summary(a, a, np.ones(3), 3, 1, w * np.nan, np.zeros(1))
    with pytest.raises(ValueError):
        translation_summary(a, a, np.ones(3), 3, 1, np.ones((1, 3)), np.zeros(1))

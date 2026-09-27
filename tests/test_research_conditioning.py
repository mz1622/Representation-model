from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from foodcomp.research_conditioning import fit_training_name_statistics


def fixture():
    data = SimpleNamespace(train=np.array([0, 1, 2]), profiles=pd.DataFrame({
        "partition": ["train", "train", "train", "validation"],
        "original_name": ["a", "a", "b", "held out"],
    }))
    return data, np.array([[1., 2.], [1., 2.], [3., 6.], [900., -999.]], np.float32)


def test_statistics_use_unique_train_names_and_ignore_held_out_vectors():
    data, text = fixture()
    mean, std, meta = fit_training_name_statistics(data, text)
    np.testing.assert_array_equal(mean, [2., 4.])
    np.testing.assert_array_equal(std, [1., 2.])
    assert meta["train_unique_names"] == 2
    text[-1] = np.nan
    after = fit_training_name_statistics(data, text)
    np.testing.assert_array_equal(mean, after[0])
    np.testing.assert_array_equal(std, after[1])
    assert meta == after[2]


def test_conditioning_rejects_held_out_fit_and_inconsistent_duplicates():
    data, text = fixture()
    data.train = np.array([0, 2, 3])
    with pytest.raises(ValueError, match="Held-out"):
        fit_training_name_statistics(data, text)
    data.train = np.array([0, 1, 2])
    text[1, 0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        fit_training_name_statistics(data, text)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_features_fail(bad):
    data, text = fixture()
    text[0, 0] = bad
    with pytest.raises(FloatingPointError):
        fit_training_name_statistics(data, text)


def test_constant_coordinate_is_not_silently_normalized():
    data, text = fixture()
    text[:3, 0] = 1
    with pytest.raises(ValueError, match="Degenerate"):
        fit_training_name_statistics(data, text)

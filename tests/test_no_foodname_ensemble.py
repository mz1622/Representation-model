"""Tests for scaled-log Transformer ensembling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.no_foodname_ensemble import equal_log_ensemble  # noqa: E402


def frame(values):
    return pd.DataFrame({
        "profile_index": [0, 1], "axis_index": [0, 1],
        "prediction": values,
    })


def test_equal_log_ensemble_uses_geometric_mean_in_scaled_space():
    result = equal_log_ensemble(
        [frame([0.0, 3.0]), frame([3.0, 0.0])], np.array([1.0, 1.0]))
    expected = np.expm1((np.log1p(3.0) + np.log1p(0.0)) / 2)
    np.testing.assert_allclose(result.prediction, [expected, expected])


def test_equal_log_ensemble_rejects_key_mismatch():
    other = frame([1.0, 2.0])
    other.loc[1, "axis_index"] = 0
    with pytest.raises(ValueError, match="keys differ"):
        equal_log_ensemble([frame([1.0, 2.0]), other], np.ones(2))

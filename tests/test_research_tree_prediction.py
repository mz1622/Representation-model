import json
import pickle
import numpy as np
import pytest
from foodcomp.research_tree_prediction import make_tree, deterministic_inference, predict_raw, parameter_record


@pytest.mark.parametrize('kind', ['rf', 'xgb'])
def test_fixed_query_order_and_reload_with_weighted_explicit_zero_targets(kind):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(128, 12)).astype(np.float32)
    y = np.maximum(x[:, 1] + 1, 0).astype(np.float32)
    weights = rng.uniform(.1, 3, len(x)).astype(np.float32)
    cfg = {'kind': kind, 'trees': 24, 'max_depth': 4, 'leaf_size': 1, 'max_features': .5}
    model = make_tree(cfg, seed=20260922, axis=4, n_jobs=2)
    json.dumps(parameter_record(model), allow_nan=False)
    model.fit(x, y, sample_weight=weights / weights.mean())
    deterministic_inference(model)
    query = np.concatenate([x[:17], x[:2]], axis=0)
    raw = predict_raw(model, query, .001)
    np.testing.assert_array_equal(predict_raw(model, query[::-1], .001)[::-1], raw)
    np.testing.assert_array_equal(np.concatenate([predict_raw(model, q[None], .001) for q in query]), raw)
    np.testing.assert_array_equal(predict_raw(pickle.loads(pickle.dumps(model)), query, .001), raw)
    assert np.isfinite(raw).all() and (raw >= 0).all() and (y == 0).any()


def test_parallel_forest_prediction_is_not_silently_accepted():
    model = make_tree({'kind': 'rf', 'trees': 2, 'leaf_size': 1, 'max_features': .5}, seed=1, axis=0, n_jobs=2)
    with pytest.raises(ValueError, match='serial'):
        predict_raw(model, np.zeros((1, 3)), 1.)


@pytest.mark.parametrize('value', [np.nan, np.inf, 10000.])
def test_bad_outputs_fail_instead_of_masking_or_clipping_high_predictions(value):
    class Invalid:
        def predict(self, features): return np.full(len(features), value)
    with pytest.raises(FloatingPointError):
        predict_raw(Invalid(), np.zeros((2, 3)), 1.)

"""Full-data tree recipe and deterministic, strict transformed-value predictions."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def make_tree(config, *, seed, axis, n_jobs=4):
    if config['kind'] == 'rf':
        return RandomForestRegressor(n_estimators=config['trees'], max_depth=None,
            min_samples_leaf=config['leaf_size'], max_features=config['max_features'],
            criterion='squared_error', random_state=seed + int(axis), n_jobs=n_jobs)
    if config['kind'] == 'xgb':
        return XGBRegressor(n_estimators=config['trees'], max_depth=config['max_depth'],
            learning_rate=.03, min_child_weight=5, subsample=.8, colsample_bytree=.8,
            reg_lambda=1., tree_method='hist', objective='reg:squarederror',
            random_state=seed + int(axis), n_jobs=n_jobs)
    raise ValueError('Only registered RF/XGB trees are supported.')


def deterministic_inference(model):
    # sklearn's parallel forest prediction sums trees in thread-completion order.
    # Keep the fitted trees and their order, but accumulate serially for all queries.
    if isinstance(model, RandomForestRegressor):
        model.n_jobs = 1
    return model


def parameter_record(model):
    parameters = model.get_params()
    for key, value in parameters.items():
        if isinstance(value, float) and not np.isfinite(value):
            if key != 'missing' or not np.isnan(value):
                raise ValueError(f'Unexpected nonfinite hyperparameter: {key}')
            parameters[key] = 'NaN (library missing-value marker; actual inputs must be finite)'
    return parameters


def predict_raw(model, features, scale):
    x = np.asarray(features)
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Expected finite two-dimensional features and positive axis scale.')
    if isinstance(model, RandomForestRegressor) and model.n_jobs != 1:
        raise ValueError('Forest inference must use fixed serial tree accumulation.')
    transformed = np.asarray(model.predict(x), dtype=np.float64)
    if transformed.shape != (len(x),) or not np.isfinite(transformed).all():
        raise FloatingPointError('Nonfinite or malformed tree predictions.')
    with np.errstate(over='raise', invalid='raise'):
        raw = scale * np.expm1(np.maximum(transformed, 0))
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise FloatingPointError('Invalid inverse-transformed tree predictions.')
    return raw

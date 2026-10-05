"""Prediction-space utilities for frozen no-food-name Transformer ensembles."""
from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["profile_index", "axis_index"]


def equal_log_ensemble(predictions, axis_scale):
    """Average raw predictions in the training target's scaled-log space."""
    predictions = list(predictions)
    if len(predictions) < 2:
        raise ValueError("ensemble requires at least two prediction sets")
    base = predictions[0].sort_values(KEYS).reset_index(drop=True)
    if base.duplicated(KEYS).any() or not np.isfinite(base.prediction).all():
        raise ValueError("invalid base predictions")
    axis_index = base.axis_index.to_numpy(dtype=np.int64)
    scale = np.asarray(axis_scale, dtype=np.float64)[axis_index]
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid axis scale")
    transformed = []
    for frame in predictions:
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        if not frame[KEYS].equals(base[KEYS]):
            raise ValueError("ensemble member prediction keys differ")
        raw = frame.prediction.to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("invalid ensemble member prediction")
        transformed.append(np.log1p(raw / scale))
    mean = np.stack(transformed).mean(axis=0)
    result = base[KEYS].copy()
    result["prediction"] = np.expm1(mean) * scale
    return result

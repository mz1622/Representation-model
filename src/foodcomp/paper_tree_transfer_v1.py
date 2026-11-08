"""Paper-data-only feature views for RF/XGBoost transfer evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from foodcomp.paper_dae_baseline_v1 import PaperAxisLink


VERSION = "gjorshoska_2022_paper_tree_external_v1"


@dataclass
class DenseTreeView:
    """Minimal matrix contract consumed by dependency-mask dense features."""

    raw: np.ndarray
    observed: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    scale: np.ndarray


def build_paper_training_view(
    paper_value: np.ndarray,
    links: Iterable[PaperAxisLink],
    axis_count: int,
    train_rows: np.ndarray,
) -> DenseTreeView:
    """Map paper nutrients into the current axis grid and fit paper-only scales."""
    paper_value = np.asarray(paper_value, dtype=np.float64)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    links = list(links)
    if paper_value.ndim != 2 or not np.isfinite(paper_value).all():
        raise ValueError("paper nutrient matrix must be finite and two-dimensional")
    if (paper_value < 0).any() or not len(train_rows):
        raise ValueError("paper values must be nonnegative and training rows nonempty")
    if len({link.axis_index for link in links}) != len(links):
        raise ValueError("paper links contain duplicate current axes")

    raw = np.full((len(paper_value), axis_count), np.nan, dtype=np.float64)
    for link in links:
        raw[:, link.axis_index] = (
            paper_value[:, link.paper_column] / link.grams_to_paper
        )
    observed = np.isfinite(raw)
    scale = np.ones(axis_count, dtype=np.float64)
    for link in links:
        positive = raw[train_rows, link.axis_index]
        positive = positive[positive > 0]
        if len(positive):
            scale[link.axis_index] = float(np.median(positive))
    values = np.where(observed, np.log1p(raw / scale), 0.0).astype(np.float32)
    weights = observed.astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(scale).all():
        raise FloatingPointError("invalid paper-normalized feature matrix")
    return DenseTreeView(raw, observed, values, weights, scale)


def build_external_feature_view(
    current_data,
    paper_scale: np.ndarray,
    mapped_axes: Iterable[int],
) -> DenseTreeView:
    """Normalize current inputs with paper-only scales and hide unmapped axes."""
    paper_scale = np.asarray(paper_scale, dtype=np.float64)
    if paper_scale.shape != (current_data.raw.shape[1],):
        raise ValueError("paper scale does not match current axis count")
    mapped = np.zeros(current_data.raw.shape[1], dtype=bool)
    mapped[np.asarray(list(mapped_axes), dtype=np.int64)] = True
    observed = current_data.observed & mapped[None, :]
    raw = np.where(observed, current_data.raw, np.nan)
    values = np.where(observed, np.log1p(raw / paper_scale), 0.0).astype(np.float32)
    weights = np.zeros_like(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("invalid external paper-normalized features")
    return DenseTreeView(raw, observed, values, weights, paper_scale.copy())


__all__ = [
    "VERSION",
    "DenseTreeView",
    "build_paper_training_view",
    "build_external_feature_view",
]

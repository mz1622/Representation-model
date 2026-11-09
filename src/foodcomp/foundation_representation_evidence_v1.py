"""Reusable metrics for the foundation-representation evidence audit.

The functions in this module operate only on already-computed representations
and evaluation metadata.  They deliberately avoid UMAP so that projection
choices cannot create the reported separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 values."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def balanced_panel_indices(
    labels: np.ndarray,
    eligible: np.ndarray,
    class_order: list[str] | tuple[str, ...],
    *,
    cap_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Select the same number of eligible rows from every declared class."""
    labels = np.asarray(labels, dtype=object)
    eligible = np.asarray(eligible, dtype=bool)
    if labels.shape != eligible.shape:
        raise ValueError("labels and eligible must have the same shape")
    candidates = {
        label: np.flatnonzero(eligible & (labels == label)) for label in class_order
    }
    if any(len(rows) == 0 for rows in candidates.values()):
        empty = [label for label, rows in candidates.items() if len(rows) == 0]
        raise ValueError(f"declared classes without eligible rows: {empty}")
    per_class = min(len(rows) for rows in candidates.values())
    if cap_per_class is not None:
        per_class = min(per_class, int(cap_per_class))
    if per_class < 2:
        raise ValueError("balanced panel requires at least two rows per class")
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(candidates[label], per_class, replace=False)
        for label in class_order
    ]
    return np.sort(np.concatenate(chosen)).astype(np.int64), per_class


def neighbor_purity(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    query_values: np.ndarray,
    query_labels: np.ndarray,
    *,
    k: int = 15,
) -> tuple[float, float]:
    """Compute micro- and class-equal cosine k-neighbour purity."""
    train_values = np.asarray(train_values)
    query_values = np.asarray(query_values)
    train_labels = np.asarray(train_labels, dtype=object)
    query_labels = np.asarray(query_labels, dtype=object)
    if len(train_values) != len(train_labels) or len(query_values) != len(query_labels):
        raise ValueError("representation and label lengths disagree")
    if not 1 <= k <= len(train_values):
        raise ValueError("k must be between one and the training-row count")
    indices = NearestNeighbors(
        n_neighbors=k, metric="cosine", n_jobs=4
    ).fit(train_values).kneighbors(query_values, return_distance=False)
    purity = (train_labels[indices] == query_labels[:, None]).mean(axis=1)
    macro = pd.Series(purity).groupby(query_labels, sort=False).mean().mean()
    return float(purity.mean()), float(macro)


def source_restricted_neighbor_purity(
    values: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    train_rows: np.ndarray,
    query_rows: np.ndarray,
    *,
    cross_source: bool,
    k: int = 15,
) -> tuple[float, float, int, int]:
    """Category purity with candidates restricted by database source.

    With ``cross_source=True``, every query retrieves only training profiles
    from other databases.  With ``False``, it retrieves only from the same
    database.  Metrics are pooled per query and then macro-averaged by category.
    """
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=object)
    source_ids = np.asarray(source_ids, dtype=object)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    purity_parts = []
    label_parts = []
    evaluated_sources = 0
    for source in sorted(pd.Series(source_ids[query_rows]).unique()):
        query = query_rows[source_ids[query_rows] == source]
        if cross_source:
            candidates = train_rows[source_ids[train_rows] != source]
        else:
            candidates = train_rows[source_ids[train_rows] == source]
        if len(candidates) < k or not len(query):
            continue
        indices = NearestNeighbors(
            n_neighbors=k, metric="cosine", n_jobs=4
        ).fit(values[candidates]).kneighbors(values[query], return_distance=False)
        neighbor_rows = candidates[indices]
        purity_parts.append((labels[neighbor_rows] == labels[query, None]).mean(axis=1))
        label_parts.append(labels[query])
        evaluated_sources += 1
    if not purity_parts:
        raise ValueError("no source has enough candidates for restricted retrieval")
    purity = np.concatenate(purity_parts)
    query_labels = np.concatenate(label_parts)
    macro = pd.Series(purity).groupby(query_labels, sort=False).mean().mean()
    return float(purity.mean()), float(macro), int(len(purity)), evaluated_sources


def multiclass_cosine_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute cosine silhouette after checking label support."""
    labels = np.asarray(labels, dtype=object)
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("silhouette requires at least two classes with two rows each")
    return float(silhouette_score(values, labels, metric="cosine"))


def one_vs_rest_silhouettes(
    values: np.ndarray,
    labels: np.ndarray,
    class_order: list[str] | tuple[str, ...],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Balanced one-vs-rest cosine silhouettes.

    Every repeat retains all positive examples and samples the same number of
    negatives, spread as evenly as possible over the remaining classes.
    """
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=object)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows: list[dict[str, object]] = []
    for class_number, positive_label in enumerate(class_order):
        positive = np.flatnonzero(labels == positive_label)
        negative_classes = [label for label in class_order if label != positive_label]
        if len(positive) < 2:
            raise ValueError(f"insufficient positive rows for {positive_label}")
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 10_000 * class_number + repeat)
            allocation = np.full(len(negative_classes), len(positive) // len(negative_classes))
            allocation[: len(positive) % len(negative_classes)] += 1
            rng.shuffle(allocation)
            negative_parts = []
            for label, count in zip(negative_classes, allocation):
                candidates = np.flatnonzero(labels == label)
                negative_parts.append(rng.choice(candidates, int(count), replace=False))
            negative = np.concatenate(negative_parts)
            selected = np.concatenate((positive, negative))
            binary = np.concatenate((
                np.ones(len(positive), dtype=np.int8),
                np.zeros(len(negative), dtype=np.int8),
            ))
            rows.append({
                "category": positive_label,
                "repeat": repeat,
                "positive_profiles": len(positive),
                "negative_profiles": len(negative),
                "silhouette_cosine": float(silhouette_score(
                    values[selected], binary, metric="cosine"
                )),
            })
    return pd.DataFrame(rows)


def pairwise_silhouettes(
    values: np.ndarray,
    labels: np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Compute each unordered two-class cosine silhouette."""
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=object)
    rows = []
    for left_number, left in enumerate(class_order):
        for right in class_order[left_number + 1:]:
            selected = np.flatnonzero((labels == left) | (labels == right))
            rows.append({
                "category_a": left,
                "category_b": right,
                "profiles": len(selected),
                "silhouette_cosine": multiclass_cosine_silhouette(
                    values[selected], labels[selected]
                ),
            })
    return pd.DataFrame(rows)


def cross_source_retrieval(
    values: np.ndarray,
    group_ids: np.ndarray,
    source_ids: np.ndarray,
    *,
    block_size: int = 256,
) -> dict[str, float | int]:
    """Retrieve the same exact-name candidate from a different database.

    The best positive is any row with the same candidate group and a different
    source.  Candidate groups remain a conservative name-based identity signal;
    the function does not assert biological identity or pool measurements.
    """
    values = l2_normalize(values)
    group_ids = np.asarray(group_ids, dtype=object)
    source_ids = np.asarray(source_ids, dtype=object)
    if len(values) != len(group_ids) or len(values) != len(source_ids):
        raise ValueError("representation and metadata lengths disagree")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    eligible = np.array([
        np.any((group_ids == group_ids[index]) & (source_ids != source_ids[index]))
        for index in range(len(values))
    ])
    query_rows = np.flatnonzero(eligible)
    reciprocal_ranks = []
    top1 = []
    top5 = []
    candidate_counts = []
    positive_counts = []
    for start in range(0, len(query_rows), block_size):
        query = query_rows[start:start + block_size]
        similarity = values[query] @ values.T
        for local, row in enumerate(query):
            valid = source_ids != source_ids[row]
            positive = valid & (group_ids == group_ids[row])
            scores = similarity[local]
            best_positive = float(scores[positive].max())
            rank = 1 + int(np.count_nonzero(valid & (scores > best_positive)))
            best = np.flatnonzero(valid)[np.argmax(scores[valid])]
            top = np.argpartition(
                np.where(valid, scores, -np.inf), -min(5, int(valid.sum()))
            )[-min(5, int(valid.sum())):]
            reciprocal_ranks.append(1.0 / rank)
            top1.append(group_ids[best] == group_ids[row])
            top5.append(bool(np.any(group_ids[top] == group_ids[row])))
            candidate_counts.append(int(valid.sum()))
            positive_counts.append(int(positive.sum()))
    chance_top1 = np.mean(
        np.asarray(positive_counts, dtype=np.float64)
        / np.asarray(candidate_counts, dtype=np.float64)
    )
    return {
        "queries": int(len(query_rows)),
        "candidate_groups": int(pd.Series(group_ids[query_rows]).nunique()),
        "sources": int(pd.Series(source_ids[query_rows]).nunique()),
        "top1_accuracy": float(np.mean(top1)),
        "top5_accuracy": float(np.mean(top5)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "mean_candidates": float(np.mean(candidate_counts)),
        "mean_positive_candidates": float(np.mean(positive_counts)),
        "chance_top1_from_pool": float(chance_top1),
    }

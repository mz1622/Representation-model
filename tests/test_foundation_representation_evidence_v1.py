import numpy as np

from foodcomp.foundation_representation_evidence_v1 import (
    balanced_panel_indices,
    cross_source_retrieval,
    l2_normalize,
    neighbor_purity,
    one_vs_rest_silhouettes,
    pairwise_silhouettes,
    source_restricted_neighbor_purity,
)


def test_balanced_panel_is_reproducible_and_equal():
    labels = np.array(list("AAAABBBBCCCC"), dtype=object)
    eligible = np.ones(len(labels), dtype=bool)
    left, count = balanced_panel_indices(
        labels, eligible, ["A", "B", "C"], cap_per_class=3, seed=7
    )
    right, _ = balanced_panel_indices(
        labels, eligible, ["A", "B", "C"], cap_per_class=3, seed=7
    )
    assert count == 3
    assert np.array_equal(left, right)
    assert np.array_equal(np.unique(labels[left], return_counts=True)[1], [3, 3, 3])


def test_neighbor_and_binary_geometry_metrics_separate_clusters():
    values = np.array([
        [1.0, 0.0], [0.9, 0.1], [1.0, 0.1],
        [0.0, 1.0], [0.1, 0.9], [0.1, 1.0],
    ])
    labels = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    micro, macro = neighbor_purity(
        values, labels, values, labels, k=2
    )
    assert micro == 1.0
    assert macro == 1.0
    ovr = one_vs_rest_silhouettes(
        values, labels, ["A", "B"], repeats=2, seed=5
    )
    assert len(ovr) == 4
    assert (ovr.silhouette_cosine > 0.8).all()
    pairwise = pairwise_silhouettes(values, labels, ["A", "B"])
    assert len(pairwise) == 1
    assert pairwise.iloc[0].silhouette_cosine > 0.8


def test_cross_source_retrieval_recovers_matching_groups():
    values = np.array([
        [1.0, 0.0], [0.99, 0.01],
        [0.0, 1.0], [0.01, 0.99],
    ])
    groups = np.array(["food_a", "food_a", "food_b", "food_b"], dtype=object)
    sources = np.array(["s1", "s2", "s1", "s2"], dtype=object)
    result = cross_source_retrieval(values, groups, sources, block_size=2)
    assert result["queries"] == 4
    assert result["top1_accuracy"] == 1.0
    assert result["top5_accuracy"] == 1.0
    assert result["mean_reciprocal_rank"] == 1.0


def test_l2_normalize_handles_zero_row():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(result[0], [0.6, 0.8])
    assert np.allclose(result[1], [0.0, 0.0])


def test_source_restricted_neighbor_purity_uses_requested_candidate_pool():
    values = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
        [1.0, 0.01], [0.01, 1.0],
    ])
    labels = np.array(["A", "A", "B", "B", "A", "B"], dtype=object)
    sources = np.array(["s1", "s2", "s1", "s2", "s3", "s3"], dtype=object)
    result = source_restricted_neighbor_purity(
        values, labels, sources,
        np.array([0, 1, 2, 3]), np.array([4, 5]),
        cross_source=True, k=1,
    )
    assert result[:2] == (1.0, 1.0)
    assert result[2:] == (2, 1)

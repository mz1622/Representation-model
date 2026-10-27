from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy, EdgeSpec
from foodcomp.research_dependency_mask_v1 import (
    FoodPromptCceTransformer, dense_features, visible_context,
)
from foodcomp.research_no_foodname_v1 import Config


def synthetic_policy():
    axes = pd.DataFrame({
        "axis_index": [0, 1, 2, 3, 4],
        "canonical_name": ["total", "subtotal", "child_a", "child_b", "unrelated"],
    })
    source = "frozen-test-authority"
    specs = (
        EdgeSpec("total", "subtotal", "explicit_subtotal", source, "test"),
        EdgeSpec("subtotal", "child_a", "explicit_sum_component", source, "test"),
        EdgeSpec("subtotal", "child_b", "explicit_sum_component", source, "test"),
    )
    return DependencyMaskPolicy(axes, specs)


def test_ancestor_descendant_mask_and_visible_sibling():
    policy = synthetic_policy()
    assert policy.mask_indices(0).tolist() == [0, 1, 2, 3]
    assert policy.mask_indices(1).tolist() == [0, 1, 2, 3]
    assert policy.mask_indices(2).tolist() == [0, 1, 2]
    assert policy.mask_indices(3).tolist() == [0, 1, 3]
    assert policy.mask_indices(4).tolist() == [4]
    assert 3 not in policy.mask_indices(2)  # sibling remains visible


def test_context_and_dense_baseline_use_identical_mask():
    policy = synthetic_policy()
    data = SimpleNamespace(
        observed=np.ones((2, 5), dtype=bool),
        values=np.arange(10, dtype=np.float32).reshape(2, 5),
    )
    rows = np.asarray([0, 1])
    targets = np.asarray([2, 4])
    visible = visible_context(data, rows, targets, policy)
    dense = dense_features(data, rows, targets, policy)
    np.testing.assert_array_equal(dense[:, 5:].astype(bool), visible)
    np.testing.assert_array_equal(dense[:, :5], np.where(visible, data.values, 0))
    np.testing.assert_array_equal(visible[0], [False, False, False, True, True])
    np.testing.assert_array_equal(visible[1], [True, True, True, True, False])


def test_cycles_and_missing_names_fail_closed():
    axes = pd.DataFrame({"axis_index": [0, 1], "canonical_name": ["a", "b"]})
    source = "test"
    try:
        DependencyMaskPolicy(axes, (
            EdgeSpec("a", "b", "x", source, "x"),
            EdgeSpec("b", "a", "x", source, "x"),
        ))
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle was accepted")
    try:
        DependencyMaskPolicy(axes, (EdgeSpec("a", "missing", "x", source, "x"),))
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("missing axis name was accepted")


def test_transformer_order_padding_and_checkpoint_roundtrip():
    torch.manual_seed(7)
    config = Config(
        d_model=24, n_heads=4, n_layers=2, feedforward_dim=32,
        dropout=0.0, batch_size=2, epochs=1,
    )
    model = FoodPromptCceTransformer(5, config).eval()
    axis = torch.tensor([[0, 2, 4], [1, 3, 0]])
    value = torch.tensor([[0.2, 0.4, 0.7], [0.1, 0.3, 0.0]])
    padding = torch.tensor([[False, False, False], [False, False, True]])
    target = torch.tensor([[1], [4]])
    target_padding = torch.zeros((2, 1), dtype=torch.bool)
    with torch.no_grad():
        reference = model(axis, value, padding, target, target_padding)
        order = torch.tensor([2, 0, 1])
        permuted = model(
            axis[:, order], value[:, order], padding[:, order], target, target_padding
        )
        padded = model(
            torch.cat((axis, torch.zeros((2, 2), dtype=torch.long)), dim=1),
            torch.cat((value, torch.zeros((2, 2))), dim=1),
            torch.cat((padding, torch.ones((2, 2), dtype=torch.bool)), dim=1),
            target, target_padding,
        )
    torch.testing.assert_close(reference, permuted, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(reference, padded, atol=2e-6, rtol=2e-6)
    clone = FoodPromptCceTransformer(5, config).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        reloaded = clone(axis, value, padding, target, target_padding)
    torch.testing.assert_close(reference, reloaded, atol=0, rtol=0)

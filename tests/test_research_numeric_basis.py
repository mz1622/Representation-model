import io

import pytest
import torch

from foodcomp.research_numeric_basis import FrozenNumericBasis


def test_known_ramps_duplicate_knots_and_padding():
    basis = FrozenNumericBasis([[0, 0, 1, 3], [2, 2], []], 3)
    x = torch.tensor([[.5, 2., 7.], [2., 5., 0.]])
    visible = torch.ones_like(x, dtype=torch.bool)
    out = basis(x, visible)
    assert torch.equal(out[:, :3], x)
    assert torch.equal(out[:, 3:6], visible.float())
    expected = torch.zeros(2, 3, 3)
    expected[0, 0] = torch.tensor([.5, 0., 0.])
    expected[1, 0] = torch.tensor([1., .5, 0.])
    assert torch.equal(out[:, 6:].reshape(2, 3, 3), expected)


def test_hidden_changes_including_nan_do_not_affect_features_or_gradients():
    basis = FrozenNumericBasis([[0, 1, 2], [1, 3]], 2)
    x = torch.tensor([[.5, 2.], [1.5, 0.]], requires_grad=True)
    visible = torch.tensor([[True, False], [False, True]])
    out = basis(x, visible)
    changed = x.detach().clone()
    changed[~visible] = float('nan')
    assert torch.equal(out, basis(changed, visible))
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.equal(x.grad[~visible], torch.zeros(2))
    assert not list(basis.parameters())


def test_zero_and_missing_distinct_but_name_only_all_zero():
    basis = FrozenNumericBasis([[.1, 1, 2]], 2)
    x = torch.tensor([[0.], [100.]])
    out = basis(x, torch.tensor([[True], [False]]))
    assert out[0, 0] == out[1, 0] == 0
    assert out[0, 1] == 1 and out[1, 1] == 0
    assert torch.equal(out[1], torch.zeros(basis.output_features))


def test_out_of_range_information_retained_and_tiny_width_finite():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
    basis = FrozenNumericBasis([[0, tiny]], 1)
    x = torch.tensor([[torch.finfo(torch.float32).max], [1.], [0.]])
    out = basis(x, torch.ones_like(x, dtype=torch.bool))
    assert torch.isfinite(out).all()
    assert torch.equal(out[:, :1], x)
    assert torch.equal(out[:, 2], torch.tensor([1., 1., 0.]))


def test_batch_permutation_serialization_and_rng_independence():
    state = torch.get_rng_state().clone()
    basis = FrozenNumericBasis([[0, 1, 4], [2]], 2)
    assert torch.equal(state, torch.get_rng_state())
    x = torch.arange(14, dtype=torch.float32).reshape(7, 2) / 3
    visible = x.remainder(2).ne(0)
    whole = basis(x, visible)
    order = torch.tensor([6, 4, 2, 0, 1, 3, 5])
    assert torch.equal(basis(x[order], visible[order]), whole[order])
    assert torch.equal(torch.cat([basis(x[:3], visible[:3]), basis(x[3:], visible[3:])]), whole)
    memory = io.BytesIO()
    torch.save(basis.state_dict(), memory)
    memory.seek(0)
    restored = FrozenNumericBasis([[], []], 2)
    restored.load_state_dict(torch.load(memory, weights_only=True))
    assert torch.equal(restored(x, visible), whole)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_invalid_visible_values_fail(value):
    basis = FrozenNumericBasis([[0, 1]], 1)
    with pytest.raises(FloatingPointError):
        basis(torch.tensor([[value]]), torch.tensor([[True]]))


@pytest.mark.parametrize('knots', [[[1, 0]], [[float('nan')]], [[-1, 0]], [[0, 1, 2]]])
def test_invalid_knots_fail(knots):
    with pytest.raises(ValueError):
        FrozenNumericBasis(knots, 1)


def test_invalid_loaded_width_fails():
    basis = FrozenNumericBasis([[0, 1]], 1)
    basis.width.zero_()
    with pytest.raises(ValueError, match='width'):
        basis(torch.tensor([[.5]]), torch.tensor([[True]]))

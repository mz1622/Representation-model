import copy
import pytest
import torch
from foodcomp.research_query import AxisQueryResidual
from foodcomp.research_neural import make_model, batch_from_arrays
from types import SimpleNamespace
import numpy as np


def test_zero_initial_residual_preserves_original_linear_predictions_and_gradient():
    torch.manual_seed(22)
    baseline = torch.nn.Linear(16, 7)
    readout = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    a = torch.randn(5, 16, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    target = torch.randn(5, 7)
    original = baseline(a)
    expanded = baseline(b) + readout(b)
    torch.testing.assert_close(original, expanded, rtol=0, atol=0)
    (original-target).square().sum().backward()
    (expanded-target).square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert readout.joint[-1].weight.grad.abs().sum() > 0
    assert readout.context_projection.weight.grad.abs().sum() == 0


def test_requested_axis_set_order_and_batchmates_do_not_change_a_prediction():
    torch.manual_seed(23)
    model = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    # Exercise nonzero outputs, avoiding a vacuous invariance test at zero initialization.
    torch.nn.init.normal_(model.joint[-1].weight, std=.1)
    context = torch.randn(5, 16)
    full = model(context)
    selected = torch.tensor([6, 1, 4, 1])
    torch.testing.assert_close(model(context, selected), full[:, selected], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model(context[:1], selected), full[:1, selected], rtol=1e-6, atol=1e-7)
    assert model(context, torch.tensor([], dtype=torch.long)).shape == (5, 0)


def test_observed_target_mask_controls_supervision_without_entering_query_head():
    torch.manual_seed(24)
    model = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    torch.nn.init.normal_(model.joint[-1].weight, std=.1)
    context = torch.randn(3, 16)
    output = model(context)
    output.retain_grad()
    observed = torch.zeros_like(output, dtype=torch.bool)
    observed[:, [1, 5]] = True
    target = torch.ones_like(output)
    target[:, 1] = 0.  # Explicit zero must still be supervised.
    (output-target).abs()[observed].sum().backward()
    assert output.grad[:, 1].abs().sum() > 0
    assert output.grad[~observed].abs().sum() == 0
    assert model.axis_embedding.weight.grad[[0, 2, 3, 4, 6]].abs().sum() == 0


def test_hidden_query_layers_train_after_zero_output_layer_moves_and_reload_matches(tmp_path):
    torch.manual_seed(25)
    model = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    context, target = torch.randn(5, 16), torch.randn(5, 7)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    before = copy.deepcopy(model.state_dict())
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        (model(context)-target).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert model.context_projection.weight.grad.abs().sum() > 0
    assert model.axis_embedding.weight.grad.abs().sum() > 0
    assert not torch.equal(before["context_projection.weight"], model.context_projection.weight)
    assert not torch.equal(before["axis_embedding.weight"], model.axis_embedding.weight)
    torch.save(model.state_dict(), tmp_path/"readout.pt")
    loaded = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    loaded.load_state_dict(torch.load(tmp_path/"readout.pt", weights_only=True))
    torch.testing.assert_close(model(context), loaded(context), rtol=0, atol=0)


@pytest.mark.parametrize("query", [torch.tensor([-1]), torch.tensor([7]), torch.tensor([1.]), torch.tensor([[1]])])
def test_invalid_query_axes_fail(query):
    model = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 16), query)


def test_factorized_joint_linear_matches_literal_concatenation_and_gradients():
    torch.manual_seed(26)
    model = AxisQueryResidual(16, 7, query_dim=4, hidden_dim=8).double()
    torch.nn.init.normal_(model.joint[-1].weight, std=.1)
    literal = copy.deepcopy(model)
    a = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    axes = torch.tensor([5, 1, 6])
    actual = model(a, axes)
    hidden, query = literal.context_projection(b), literal.axis_embedding(axes)
    joined = torch.cat([hidden[:, None].expand(-1, 3, -1), query[None].expand(3, -1, -1)], -1)
    expected = literal.joint(joined).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-14)
    for first, second in zip(model.parameters(), literal.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-12, atol=1e-14)


def test_composite_head_preserves_parent_initialization_rng_and_predictions():
    data = SimpleNamespace(axes=list(range(252)))
    torch.manual_seed(20260922)
    parent, _ = make_model(data, 32, "mlp", mlp_width=512)
    rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    candidate, _ = make_model(data, 32, "mlp", mlp_width=512, mlp_query_residual=True)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in candidate.parameters()) == 762365
    batch = batch_from_arrays(np.zeros((3, 252)), np.zeros((3, 252), bool), np.ones((3, 32)), "cpu")
    torch.testing.assert_close(parent(batch)["amount_normalized"], candidate(batch)["amount_normalized"], rtol=0, atol=0)


def test_nonzero_composite_query_head_keeps_hidden_values_and_source_out_of_inputs(tmp_path):
    data = SimpleNamespace(axes=list(range(4)))
    model, _ = make_model(data, 3, "mlp", mlp_query_residual=True)
    torch.nn.init.normal_(model.query_residual.joint[-1].weight, std=.1)
    batch = batch_from_arrays(np.array([[0., 3., 2., 5.]]), np.array([[True, False, True, False]]), np.ones((1, 3)), "cpu")
    reference = model(batch)["amount_normalized"]
    batch["value"][batch["masked"]] = 12345
    batch["source"] = torch.tensor([99])
    batch["target"] = torch.ones((1, 4), dtype=torch.bool)
    torch.testing.assert_close(reference, model(batch)["amount_normalized"], rtol=0, atol=0)
    torch.save(model.state_dict(), tmp_path/"model.pt")
    loaded, _ = make_model(data, 3, "mlp", mlp_query_residual=True)
    loaded.load_state_dict(torch.load(tmp_path/"model.pt", weights_only=True))
    torch.testing.assert_close(reference, loaded(batch)["amount_normalized"], rtol=0, atol=0)


def test_unsupported_query_combinations_fail():
    data = SimpleNamespace(axes=list(range(4)))
    for kwargs in [dict(kind="numeric_mlp"), dict(kind="v9"), dict(kind="mlp", mlp_task_heads="separate")]:
        with pytest.raises(ValueError, match="shared head"):
            make_model(data, 3, mlp_query_residual=True, **kwargs)

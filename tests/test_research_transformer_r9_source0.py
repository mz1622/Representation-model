"""Loss-scale and calibration-isolation checks for the source-weight intervention."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from foodcomp.research_transformer_r9 import make_transformer, transformer_loss
from foodcomp.research_transformer_r9_source0 import load_method
from test_research_transformer_r9 import example


@pytest.mark.parametrize('training', [False, True])
def test_zero_residual_shared_initial_loss_gradients_and_rng_match(training):
    data, spec, batch = example()
    torch.manual_seed(20260922)
    control, c1 = make_transformer(data, 128, spec)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(20260922)
    candidate, c0 = make_transformer(data, 128, {**spec, 'source_weight': 0.})
    torch.testing.assert_close(old_rng, torch.get_rng_state(), rtol=0, atol=0)
    for key, tensor in control.state_dict().items():
        torch.testing.assert_close(tensor, candidate.state_dict()[key], rtol=0, atol=0)
    control.train(training); candidate.train(training)
    rng = torch.get_rng_state()
    a = transformer_loss(control, batch, c1, 'mae')
    a.backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    b = transformer_loss(candidate, batch, c0, 'mae')
    b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
    for (name, left), (other, right) in zip(control.named_parameters(), candidate.named_parameters()):
        assert name == other
        if name == 'source_amount_residual.weight':
            assert left.grad.count_nonzero() > 0
            assert right.grad.count_nonzero() == 0
        elif left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0., 1.])
def test_mae_scale_and_gradients_against_independent_cell_formula(weight):
    p = torch.nn.Parameter(torch.tensor([[1., 4.], [3., 2.]], dtype=torch.float64))
    residual = torch.nn.Parameter(torch.tensor([[.5, -3.]], dtype=torch.float64))
    batch = {'value': torch.tensor([[0., 999.], [1., 0.]], dtype=torch.float64),
             'target': torch.tensor([[True, False], [True, True]]),
             'cell_weight': torch.tensor([[.25, 10.], [.75, .5]], dtype=torch.float64),
             'axis_total': torch.tensor([2., 4.], dtype=torch.float64), 'objective_multiplier': 3.}
    class Model:
        def __call__(self, batch):
            return {'amount_normalized': p}
        def calibrated_outputs(self, base, batch):
            return {'amount_normalized': p + residual}
        def source_residual_penalty(self):
            return residual.square().mean()
    cfg = SimpleNamespace(source_calibrated_loss_weight=weight, source_residual_l2=1e-4)
    actual = transformer_loss(Model(), batch, cfg, 'mae')
    expected = p.new_zeros(())
    gradient = torch.zeros_like(p)
    for i, a in [(0, 0), (1, 0), (1, 1)]:
        factor = batch['cell_weight'][i,a] / batch['axis_total'][a] * 3.
        error = p[i,a] - batch['value'][i,a]
        shifted = error + residual[0,a]
        expected = expected + factor * (error.abs() + weight*shifted.abs())/(1+weight)
        gradient[i,a] = factor * (error.sign() + weight*shifted.sign())/(1+weight)
    expected = expected + 1e-4*residual.square().mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    actual.backward()
    torch.testing.assert_close(p.grad, gradient, rtol=0, atol=1e-12)
    assert p.grad[0,1] == 0 and p.grad[0,0] != 0


def test_source_offsets_remain_zero_with_actual_optimizer():
    data, spec, batch = example()
    model, config = make_transformer(data, 128, {**spec, 'source_weight': 0.})
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        transformer_loss(model, batch, config, 'mae').backward()
        optimizer.step()
        assert model.source_amount_residual.weight.count_nonzero() == 0
    changed = deepcopy(batch)
    changed['source'][:] = 0
    torch.testing.assert_close(transformer_loss(model, batch, config, 'mae'),
                               transformer_loss(model, changed, config, 'mae'), rtol=0, atol=0)


@pytest.mark.parametrize('change', [{'source_weight': 1.}, {'objective': 'mse'},
                                    {'source_weight': 0., 'learning_rate': .001}])
def test_registration_rejects_undeclared_factor(tmp_path, change):
    plan = {'status':'registered', 'purpose':'r9_single_factor_source_weight_contrast',
            'changes':change, 'seed':20260922, 'candidate':'tf192_mae_source0_lr3e4_60',
            'candidate_index':5, 'max_candidates':12}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError, match='registered same-seed'):
        load_method(tmp_path,path,{},tmp_path/'unused_freeze.json')

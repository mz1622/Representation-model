"""Optimizer-only contrast: identical starting functions and a doubled cosine schedule."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from foodcomp.research_transformer_r9 import make_transformer, transformer_loss
from foodcomp.research_transformer_r9_lr6e4 import load_method
from test_research_transformer_r9 import example


@pytest.mark.parametrize('training', [False, True])
def test_learning_rate_does_not_change_initial_function_gradient_or_rng(training):
    data, spec, batch = example()
    torch.manual_seed(20260922)
    control, ca = make_transformer(data, 128, {**spec, 'learning_rate': .0003})
    constructor_rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    candidate, cb = make_transformer(data, 128, {**spec, 'learning_rate': .0006})
    torch.testing.assert_close(constructor_rng, torch.get_rng_state(), rtol=0, atol=0)
    control.train(training); candidate.train(training)
    rng = torch.get_rng_state()
    left = transformer_loss(control, batch, ca, 'mae')
    left.backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    right = transformer_loss(candidate, batch, cb, 'mae')
    right.backward()
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
    for (a, x), (b, y) in zip(control.named_parameters(), candidate.named_parameters()):
        assert a == b
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        if x.grad is None:
            assert y.grad is None
        else:
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


def test_adamw_first_update_and_entire_cosine_schedule_double():
    # Constant externally supplied gradients separate optimizer behavior from evolving model gradients.
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a = torch.optim.AdamW([p], lr=.0003, weight_decay=1e-4)
    b = torch.optim.AdamW([q], lr=.0006, weight_decay=1e-4)
    sa = torch.optim.lr_scheduler.CosineAnnealingLR(a, T_max=60, eta_min=.000003)
    sb = torch.optim.lr_scheduler.CosineAnnealingLR(b, T_max=60, eta_min=.000006)
    initial = p.detach().clone()
    for epoch in range(60):
        expected = .0003 * (.01 + .99 * (1 + np.cos(np.pi * epoch / 60)) / 2)
        assert a.param_groups[0]['lr'] == pytest.approx(expected, abs=1e-15)
        assert b.param_groups[0]['lr'] == pytest.approx(2 * expected, abs=1e-15)
        p.grad = torch.tensor([.25, -.5], dtype=torch.float64)
        q.grad = p.grad.clone()
        a.step(); b.step()
        if epoch == 0:
            torch.testing.assert_close(q - initial, 2 * (p - initial), rtol=0, atol=1e-15)
        sa.step(); sb.step()
    assert sa.last_epoch == sb.last_epoch == 60
    assert b.param_groups[0]['lr'] == pytest.approx(.000006)


def test_numeric_batch_loop_and_checkpoint_payload_unchanged():
    root = Path(__file__).resolve().parents[1]
    def part(name, cls, key, value):
        tree = ast.parse((root / 'scripts' / name).read_text(encoding='utf-8'))
        nodes = [n for n in ast.walk(tree) if isinstance(n, cls) and key(n) == value]
        assert len(nodes) == 1
        return ast.dump(nodes[0], include_attributes=False)
    old = 'train_foodnutrigpt_v9_r9_transformer.py'
    new = 'train_foodnutrigpt_v9_r9_lr6e4.py'
    for cls, key, value in [(ast.For, lambda n: getattr(n.target, 'id', None), 'offset'),
                            (ast.FunctionDef, lambda n: n.name, 'checkpoint')]:
        assert part(old, cls, key, value) == part(new, cls, key, value)


@pytest.mark.parametrize('changes', [{'learning_rate': .0003},
    {'learning_rate': .0006, 'source_weight': 0.}, {'objective': 'mse'}])
def test_registration_rejects_other_changes(tmp_path, changes):
    plan = {'status': 'registered', 'purpose': 'r9_single_factor_learning_rate_contrast',
            'changes': changes, 'seed': 20260922, 'candidate': 'tf192_mae_lr6e4_60',
            'candidate_index': 7, 'max_candidates': 12}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='registered same-seed'):
        load_method(tmp_path, path, {}, tmp_path / 'unused')

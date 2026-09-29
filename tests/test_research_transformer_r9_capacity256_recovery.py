"""Recovery preserves trajectory state; synthetic checks do not select methods."""
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from foodcomp.research_transformer_r9_capacity256_recovery import (
    assert_state_equal, require_finite, restore, validate_epoch_boundary)


def test_serialization_boundary_preserves_cpu_dropout_adamw_and_cosine_trajectory():
    torch.manual_seed(31)
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    def build():
        model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Dropout(.15), torch.nn.Linear(7, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=.000003)
        return model, optimizer, scheduler
    def step(model, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x)-y).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        return float(loss.detach())
    a, ao, ar = build()
    for _ in range(4):
        step(a, ao, ar)
    state = copy.deepcopy(dict(model_state=a.state_dict(), optimizer=ao.state_dict(),
        scheduler=ar.state_dict(), cpu_rng=torch.get_rng_state(), cuda_rng=[]))
    expected = [step(a, ao, ar) for _ in range(3)]
    expected_rng = torch.get_rng_state()
    b, bo, br = build()
    restore(b, bo, br, state, torch.device('cpu'))
    actual = [step(b, bo, br) for _ in range(3)]
    assert actual == expected
    assert_state_equal(a.state_dict(), b.state_dict())
    assert_state_equal(ao.state_dict(), bo.state_dict())
    assert_state_equal(ar.state_dict(), br.state_dict())
    assert_state_equal(expected_rng, torch.get_rng_state())


def boundary():
    spec = dict(seed=22, epochs=60, learning_rate=.0003, weight_decay=.0001, batch_size=4)
    h = pd.DataFrame({'epoch': np.arange(1, 49), 'validation_primary': 1./np.arange(1, 49),
        'elapsed_seconds': np.arange(1, 49, dtype=float), 'training_tasks': 8,
        'observed_target_cells': 16, 'training_order_sha256': ['order'+str(i) for i in range(48)],
        'learning_rate': [.0003*(.01+.99*(1+np.cos(np.pi*i/60))/2) for i in range(48)]})
    manifest = dict(status='running', epoch_completed=48, spec=spec, seed=22,
        data_hash='d', panel_hash='p', name_cache_hash='n', best_primary=1./48,
        elapsed_seconds=48.1, training_tasks=8, data_modified=False,
        baseline_refit=False, complete_test_opened=False)
    lr = .0003*(.01+.99*(1+np.cos(np.pi*48/60))/2)
    state = {k: manifest[k] for k in ['spec', 'seed', 'data_hash', 'panel_hash', 'name_cache_hash']}
    state.update(kind='transformer_direct', config={'width': 256}, text_dim=128,
        data_root='data', view='quarantined', name_cache='name', best_epoch=48,
        scheduler=dict(T_max=60, last_epoch=48, _step_count=49, _last_lr=[lr]),
        optimizer={'param_groups': [dict(lr=lr, initial_lr=.0003, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8)],
            'state': {0: dict(step=torch.tensor(96.), exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2))}},
        cpu_rng=torch.get_rng_state(), cuda_rng=[torch.tensor([1], dtype=torch.uint8)])
    return manifest, h, state, copy.deepcopy(state), h.copy()


def test_complete_consistent_saved_epoch_boundary_passes():
    validate_epoch_boundary(*boundary())


@pytest.mark.parametrize('change', ['epoch', 'history', 'spec', 'best', 'scheduler', 'lr', 'moments', 'steps', 'rng', 'scope'])
def test_partial_or_inconsistent_boundary_is_rejected(change):
    m, h, s, b, p = boundary()
    if change == 'epoch': s['best_epoch'] = 49
    elif change == 'history': h = h.iloc[:-1]
    elif change == 'spec': b['spec'] = dict(b['spec'], epochs=61)
    elif change == 'best': b['best_epoch'] = 47
    elif change == 'scheduler': s['scheduler']['last_epoch'] = 0
    elif change == 'lr': s['optimizer']['param_groups'][0]['lr'] = .0003
    elif change == 'moments': s['optimizer']['state'] = {}
    elif change == 'steps': s['optimizer']['state'][0]['step'] -= 1
    elif change == 'rng': s['cuda_rng'] = []
    elif change == 'scope': m['data_modified'] = True
    with pytest.raises((ValueError, AssertionError)):
        validate_epoch_boundary(m, h, s, b, p)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_nested_saved_state_is_fatal(value):
    with pytest.raises(FloatingPointError):
        require_finite({'optimizer': [{'moment': torch.tensor([value])}]})


def test_missing_state_key_cannot_be_called_exact_reload():
    with pytest.raises(ValueError):
        assert_state_equal({'step': 1, 'moment': torch.zeros(2)}, {'step': 1})

"""Operational checkpoint recovery; no change to the registered numerical recipe."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .research_r0 import digest
from .research_transformer_r9_capacity256 import verify_bindings


def assert_state_equal(left, right):
    """Compare serialized optimizer/model structures exactly across device placement."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise ValueError('Tensor state type changed')
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('State keys changed')
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('State sequence changed')
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    elif left != right:
        raise ValueError('Scalar state changed')


def require_finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite saved tensor')
    elif isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_finite(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError('Nonfinite saved scalar')


def validate_epoch_boundary(manifest, history, latest, best, parent_history):
    """Accept only the complete recorded 48-epoch boundary, not a partial epoch."""
    require_finite(latest)
    require_finite(best)
    spec = manifest['spec']
    if (manifest['status'] != 'running' or manifest['epoch_completed'] != 48
            or spec['epochs'] != 60 or latest['best_epoch'] != 48
            or not np.array_equal(history.epoch, np.arange(1, 49))
            or not np.isfinite(history.select_dtypes('number')).all().all()):
        raise ValueError('Complete finite epoch48 boundary required')
    if any(manifest[k] for k in ['data_modified', 'baseline_refit', 'complete_test_opened']):
        raise ValueError('Recovery cannot change experiment scope')
    for key in ['spec', 'seed', 'data_hash', 'panel_hash', 'name_cache_hash']:
        if latest[key] != manifest[key] or best[key] != manifest[key]:
            raise ValueError('Checkpoint identity differs: ' + key)
    for key in ['kind', 'config', 'text_dim', 'data_root', 'view', 'name_cache']:
        if latest[key] != best[key]:
            raise ValueError('Latest and best checkpoint structures differ: ' + key)
    if (manifest['best_primary'] != float(history.validation_primary.min())
            or best['best_epoch'] != int(history.loc[history.validation_primary.idxmin(), 'epoch'])
            or manifest['elapsed_seconds'] < float(history.elapsed_seconds.iloc[-1])):
        raise ValueError('Saved best/elapsed bookkeeping disagrees with trajectory')
    for key in ['epoch', 'training_tasks', 'observed_target_cells',
                'training_order_sha256', 'learning_rate']:
        np.testing.assert_array_equal(history[key], parent_history[key].iloc[:48])
    scheduler = latest['scheduler']
    expected_lr = spec['learning_rate'] * (.01 + .99 * (1 + np.cos(np.pi*48/60))/2)
    if (scheduler['T_max'] != 60 or scheduler['last_epoch'] != 48
            or scheduler['_step_count'] != 49 or len(latest['optimizer']['param_groups']) != 1):
        raise ValueError('The60-epoch scheduler position was not preserved')
    np.testing.assert_allclose(scheduler['_last_lr'], [expected_lr], atol=1e-15, rtol=1e-12)
    group = latest['optimizer']['param_groups'][0]
    if (group['weight_decay'] != spec['weight_decay'] or group['initial_lr'] != spec['learning_rate']
            or group['betas'] != (.9, .999) or group['eps'] != 1e-8):
        raise ValueError('AdamW settings changed')
    np.testing.assert_allclose(group['lr'], expected_lr, atol=1e-15, rtol=1e-12)
    if not latest['optimizer']['state']:
        raise ValueError('Optimizer moments missing')
    expected_steps = 48 * int(np.ceil(manifest['training_tasks']/spec['batch_size']))
    for state in latest['optimizer']['state'].values():
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != expected_steps:
            raise ValueError('Optimizer moments/step count incomplete')
    if (latest['cpu_rng'].dtype != torch.uint8 or latest['cpu_rng'].ndim != 1
            or len(latest['cuda_rng']) != 1 or latest['cuda_rng'][0].dtype != torch.uint8):
        raise ValueError('CPU/CUDA random state missing or malformed')


def restore(model, optimizer, scheduler, state, device):
    model.load_state_dict(state['model_state'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    assert_state_equal(state['model_state'], model.state_dict())
    assert_state_equal(state['optimizer'], optimizer.state_dict())
    assert_state_equal(state['scheduler'], scheduler.state_dict())
    torch.set_rng_state(state['cpu_rng'].cpu())
    if device.type == 'cuda':
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda_rng']])
        assert_state_equal(state['cuda_rng'], torch.cuda.get_rng_state_all())
    elif state['cuda_rng']:
        raise ValueError('Cannot silently move the GPU training run to CPU')
    assert_state_equal(state['cpu_rng'], torch.get_rng_state())


def load_recovery(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Recovery registration must stay inside repository')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan['status'] != 'registered_operational_recovery' or plan['source_epoch'] != 48
            or plan['resume_at_epoch'] != 49 or plan['end_epoch'] != 60
            or plan['candidate'] != 'tf256_mae_lr3e4_60'
            or plan['source_run'] != 'output/v9_r9_methods/tf256_mae_lr3e4_60'
            or plan['output_run'] != 'output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1'
            or plan['new_hyperparameter_candidate'] or plan['numerical_recipe_changed']):
        raise ValueError('Expected the registered continuation of candidate6')
    hashes = dict(plan['input_hashes'])
    hashes[path.relative_to(root).as_posix()] = digest(path)
    verify_bindings(root, hashes)
    source = root/plan['source_run']
    manifest = json.loads((source/'run_manifest.json').read_text(encoding='utf-8'))
    history = pd.read_csv(source/'history.csv', float_precision='round_trip')
    latest = torch.load(source/'latest_training_state.pt', map_location='cpu', weights_only=True)
    best = torch.load(source/'best_model.pt', map_location='cpu', weights_only=True)
    parent = pd.read_csv(root/'output/v9_r9/tf192_mae_lr3e4_60/history.csv', float_precision='round_trip')
    validate_epoch_boundary(manifest, history, latest, best, parent)
    verify_bindings(root, manifest['code_hashes'])
    verify_bindings(root, manifest['method_evidence_hashes'])
    return plan, manifest, history, latest, hashes

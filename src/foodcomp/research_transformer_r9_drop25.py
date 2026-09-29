"""Read an explicitly registered one-factor dropout contrast without changing old protocols."""
import copy
import json
from pathlib import Path

from .research_r0 import digest


def dropout_sites(model):
    """Return every configured dropout site, including attention probability dropout."""
    from torch import nn
    sites = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            sites[name + '.p'] = float(module.p)
        elif isinstance(module, nn.MultiheadAttention):
            sites[name + '.attention_dropout'] = float(module.dropout)
    if not sites:
        raise ValueError('Expected configured dropout sites')
    return sites


def verify_bindings(repo, hashes):
    root = Path(repo).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Registered method evidence changed: ' + name)


def load_method(repo, path, frozen, freeze_path):
    root = Path(repo).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Method registration must be inside repository')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan['status'] != 'registered' or plan['purpose'] != 'r9_single_factor_dropout_contrast'
            or plan['changes'] != {'dropout': 0.25} or plan['seed'] != 20260922
            or plan['candidate'] != 'tf192_mae_drop25_lr3e4_60'
            or plan['candidate_index'] != 9 or plan['max_candidates'] != 12
            or plan['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Expected the registered same-seed dropout.15-to.25 contrast')
    hashes = dict(plan['input_hashes'])
    hashes[path.relative_to(root).as_posix()] = digest(path)
    verify_bindings(root, hashes)
    for key, status in [('prior_decision', 'screen_complete_fixed_lr2e4_recipe_rejected'),
                        ('prior_review', 'reviewed_complete_lr2e4_stage_report')]:
        if plan[key] not in hashes:
            raise ValueError('Prior closure must be bound: ' + key)
        prior = json.loads((root / plan[key]).read_text(encoding='utf-8'))
        if prior['status'] != status:
            raise ValueError('Prior learning-rate interpretation must be complete.')
    parent_path = root / plan['parent_run'] / 'run_manifest.json'
    if parent_path.relative_to(root).as_posix() not in hashes:
        raise ValueError('Parent manifest must be bound')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    audit = json.loads((root / plan['parent_audit']).read_text(encoding='utf-8'))
    if (parent['status'] != 'complete' or parent['seed'] != plan['seed']
            or parent['spec']['objective'] != 'mae' or parent['spec']['source_weight'] != 1.0
            or parent['spec']['learning_rate'] != .0003 or parent['spec']['dropout'] != .15 or parent['epoch_completed'] != 60
            or audit['status'] != 'complete' or audit['manifest_sha256'] != digest(parent_path)
            or audit['checkpoint_sha256'] != parent['checkpoint_sha256']
            or not audit['both323809_predictions_public_loader_replayed_exact']):
        raise ValueError('Complete audited MAE parent required')
    for key in ['data_hash', 'panel_hash', 'name_cache_hash']:
        if parent[key] != frozen[key]:
            raise ValueError('Frozen parent mismatch: ' + key)
    for key in ['data_modified', 'baseline_refit', 'complete_test_opened']:
        if parent[key] or plan[key]:
            raise ValueError('Method-only scope violated')
    spec = copy.deepcopy(parent['spec'])
    spec.update(plan['changes'], name=plan['candidate'])
    difference = {key for key in spec if spec[key] != parent['spec'][key]}
    if difference != {'name', 'dropout'}:
        raise ValueError('More than the declared numerical factor changed')
    hashes.update(parent['code_hashes'])
    verify_bindings(root, hashes)
    return plan, parent, spec, hashes

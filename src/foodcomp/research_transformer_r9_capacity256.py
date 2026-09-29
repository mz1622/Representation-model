"""Read an explicitly registered PDF capacity-group contrast without changing old protocols."""
import copy
import json
from pathlib import Path

from .research_r0 import digest


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
    if (plan['status'] != 'registered' or plan['purpose'] != 'r9_capacity_group_256_8_1024_contrast'
            or plan['changes'] != {'d_model': 256, 'n_heads': 8, 'feedforward_dim': 1024} or plan['seed'] != 20260922
            or plan['candidate'] != 'tf256_mae_lr3e4_60'
            or plan['candidate_index'] != 6 or plan['max_candidates'] != 12
            or plan['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Expected the registered same-seed PDF capacity-group contrast')
    hashes = dict(plan['input_hashes'])
    hashes[path.relative_to(root).as_posix()] = digest(path)
    verify_bindings(root, hashes)
    for key, status in [('prior_decision', 'screen_complete_fixed_source0_recipe_rejected'),
                        ('prior_review', 'reviewed_complete_source0_stage_report')]:
        if plan[key] not in hashes:
            raise ValueError('Prior closure must be bound: ' + key)
        prior = json.loads((root / plan[key]).read_text(encoding='utf-8'))
        if prior['status'] != status:
            raise ValueError('Prior source-calibration interpretation must be complete.')
    parent_path = root / plan['parent_run'] / 'run_manifest.json'
    if parent_path.relative_to(root).as_posix() not in hashes:
        raise ValueError('Parent manifest must be bound')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    audit = json.loads((root / plan['parent_audit']).read_text(encoding='utf-8'))
    if (parent['status'] != 'complete' or parent['seed'] != plan['seed']
            or parent['spec']['objective'] != 'mae' or parent['spec']['source_weight'] != 1.0
            or parent['epoch_completed'] != 60
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
    if {key: parent['spec'][key] for key in ['d_model', 'n_heads', 'feedforward_dim']} != {'d_model': 192, 'n_heads': 6, 'feedforward_dim': 768}:
        raise ValueError('Expected the original192/6/FF768 MAE parent')
    spec = copy.deepcopy(parent['spec'])
    spec.update(plan['changes'], name=plan['candidate'])
    difference = {key for key in spec if spec[key] != parent['spec'][key]}
    if difference != {'name', 'd_model', 'n_heads', 'feedforward_dim'}:
        raise ValueError('More than the declared capacity group changed')
    hashes.update(parent['code_hashes'])
    verify_bindings(root, hashes)
    return plan, parent, spec, hashes

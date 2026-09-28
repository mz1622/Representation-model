"""Recipe-binding guards and complete numerical-core equivalence; no training."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('r9_confirmation', ROOT / 'scripts/train_foodnutrigpt_v9_r9_confirmation.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def records(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return sha(path)
    freeze = tmp_path / 'freeze.json'
    freeze_hash = write('freeze.json', {'status': 'frozen'})
    frozen = {'data_hash': 'data', 'name_cache_hash': 'names', 'panel_hash': 'panel'}
    checkpoint_hash = write('output/parent/best_model.pt', b'fixture-only-not-a-model')
    source_hash = write('src/parent.py', b'# fixture source\n')
    parent = {'status': 'complete', 'version': 'V9-R9', 'kind': 'transformer_direct',
        'candidate': 'fixture', 'seed': 20260922, 'spec': {'seed': 20260922, 'epochs': 60,
        'learning_rate': .0003, 'nested': {'fixed': [1, 2]}}, 'epoch_completed': 60,
        'complete_test_opened': False, 'data_modified': False, 'baseline_refit': False,
        'freeze_sha256': freeze_hash, 'checkpoint_sha256': checkpoint_hash,
        'code_hashes': {'src/parent.py': source_hash}, **frozen}
    parent_hash = write('output/parent/run_manifest.json', parent)
    audit = {'status': 'complete', 'complete_test_opened': False,
        'manifest_sha256': parent_hash, 'checkpoint_sha256': checkpoint_hash,
        'freeze_sha256': freeze_hash, 'both323809_predictions_public_loader_replayed_exact': True,
        'all_candidate_vectors_and19089_ranks_replayed_exact': True,
        'all_epoch_orders_exposures_and_schedule_verified': True}
    audit_hash = write('reports/audit.json', audit)
    supporting = [{'path': f'reports/evidence{i}.json', 'sha256': write(f'reports/evidence{i}.json', {'fixture': i})} for i in [1, 2]]
    decision = {'status': 'fixed_recipe_for_seed_confirmation', 'parent_run': 'output/parent',
        'parent_manifest_sha256': parent_hash, 'freeze_sha256': freeze_hash,
        'rationale': '合成测试记录，不能作为实际模型选择。', 'supporting_evidence': supporting}
    decision_hash = write('reports/decision.json', decision)
    plan = {'schema_version': 1, 'purpose': 'fixed_r9_transformer_seed_confirmation',
        'seeds': [20260922, 20260923, 20260924], 'freeze_sha256': freeze_hash,
        'parent_run': 'output/parent', 'parent_manifest_sha256': parent_hash,
        'parent_audit': 'reports/audit.json', 'parent_audit_sha256': audit_hash,
        'decision_record': 'reports/decision.json', 'decision_record_sha256': decision_hash}
    write('plan.json', plan)
    return tmp_path, write, frozen, freeze, parent, audit, decision, plan


def call(records, seed=20260923):
    root, _, frozen, freeze, *_ = records
    return MODULE.load_confirmation(root, root / 'plan.json', seed, frozen, freeze)


@pytest.mark.parametrize('seed', [20260923, 20260924])
def test_only_seed_changes_and_unicode_decision_is_read(records, seed):
    _, parent, spec, bindings = call(records, seed)
    expected = copy.deepcopy(parent['spec'])
    expected['seed'] = seed
    assert spec == expected and parent['spec']['seed'] == 20260922
    spec['nested']['fixed'][0] = 99
    assert parent['spec']['nested']['fixed'][0] == 1
    assert len(bindings) == 8


@pytest.mark.parametrize('seed', [20260922, 42])
def test_unregistered_or_duplicate_seed_rejected(records, seed):
    with pytest.raises(ValueError, match='only seeds23/24'):
        call(records, seed)


@pytest.mark.parametrize('field,value', [('status', 'running'), ('kind', 'mlp'),
    ('epoch_completed', 59), ('data_modified', True), ('data_hash', 'different')])
def test_incomplete_or_incompatible_parent_rejected(records, field, value):
    _, write, _, _, parent, _, _, plan = records
    parent[field] = value
    plan['parent_manifest_sha256'] = write('output/parent/run_manifest.json', parent)
    write('plan.json', plan)
    with pytest.raises(ValueError):
        call(records)


def test_hyperparameter_override_rejected(records):
    _, write, *_, plan = records
    plan['learning_rate'] = .001
    write('plan.json', plan)
    with pytest.raises(ValueError, match='overrides are prohibited'):
        call(records)


def test_missing_full_replay_rejected(records):
    _, write, _, _, _, audit, _, plan = records
    audit['all_candidate_vectors_and19089_ranks_replayed_exact'] = False
    plan['parent_audit_sha256'] = write('reports/audit.json', audit)
    write('plan.json', plan)
    with pytest.raises(ValueError, match='independent parent replay'):
        call(records)


def test_unselected_recipe_rejected(records):
    _, write, _, _, _, _, decision, plan = records
    decision['status'] = 'draft'
    plan['decision_record_sha256'] = write('reports/decision.json', decision)
    write('plan.json', plan)
    with pytest.raises(ValueError, match='explicit recipe decision'):
        call(records)


@pytest.mark.parametrize('path', ['output/parent/best_model.pt', 'src/parent.py', 'reports/evidence1.json'])
def test_changed_checkpoint_source_or_supporting_evidence_rejected(records, path):
    root, write, *_ = records
    write(path, b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        call(records)
    assert not (root / 'output/confirmation').exists()


@pytest.mark.parametrize('changed_path', ['reports/decision.json', 'plan.json'])
def test_bindings_are_rechecked_after_loading(records, changed_path):
    root, write, *_ = records
    *_, bindings = call(records)
    write(changed_path, b'changed later')
    with pytest.raises(ValueError, match='evidence changed'):
        MODULE.verify_bindings(root, bindings)


def test_duplicate_evidence_is_not_two_distinct_records(records):
    _, write, _, _, _, _, decision, plan = records
    decision['supporting_evidence'][1] = decision['supporting_evidence'][0]
    plan['decision_record_sha256'] = write('reports/decision.json', decision)
    write('plan.json', plan)
    with pytest.raises(ValueError, match='Distinct supporting'):
        call(records)


def test_all_training_selection_and_evaluation_statements_match_original():
    old = (ROOT / 'scripts/train_foodnutrigpt_v9_r9_transformer.py').read_text(encoding='utf-8')
    new = (ROOT / 'scripts/train_foodnutrigpt_v9_r9_confirmation.py').read_text(encoding='utf-8')
    def numerical_tail(text):
        tail = text[text.index('    def checkpoint(epoch):'):]
        return '\n'.join(line for line in tail.splitlines()
                         if line.strip() != 'verify_bindings(ROOT, confirmation_hashes)')
    # Includes checkpoint content, warmup/RNG restore, optimizer/schedule, every
    # training/selection step, both prediction tables and retrieval replay.
    assert numerical_tail(old) == numerical_tail(new)
    def initialization(text):
        start = text.index('    torch.set_num_threads(4)')
        end = text.index('    args.output_dir.mkdir(parents=True)', start)
        return text[start:end]
    assert initialization(old) == initialization(new)
    ast.parse(old)
    ast.parse(new)

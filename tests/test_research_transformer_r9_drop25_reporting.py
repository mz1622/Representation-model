"""Report contracts use synthetic values and never evaluate a running candidate."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = script('report_foodnutrigpt_r9_drop25')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_report_rejects_nonfinite_metrics(value):
    with pytest.raises(ValueError, match='Nonfinite'):
        report.number(value)


def test_rebase_preserves_historical_evidence_and_new_counterpart_links(tmp_path):
    prior = tmp_path/'prior'
    prior.mkdir()
    (prior/'curve.svg').write_text('<svg/>')
    (prior/'REPORT_EN.md').write_text('old')
    new = tmp_path/'new'
    content = '[curve](curve.svg#axis) [English](REPORT_EN.md) [official](https://example.org/x)'
    result = report.rebase(content, prior/'REPORT_ZH.md', new)
    assert '[curve](../prior/curve.svg#axis)' in result
    assert '[English](REPORT_EN.md)' in result
    assert '[official](https://example.org/x)' in result
    with pytest.raises(FileNotFoundError):
        report.rebase('[missing](absent.json)', prior/'REPORT_ZH.md', new)


@pytest.mark.parametrize('name', ['review_foodnutrigpt_r9_drop25_axes', 'review_foodnutrigpt_r9_drop25_cases'])
def test_review_requires_completed_analysis_before_any_current_run_read(tmp_path, monkeypatch, name):
    module = script(name)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    if name.endswith('_axes'):
        monkeypatch.setitem(module.paths, 'analysis', tmp_path/'absent_analysis.json')
    # No model construction or current run read occurs on missing-analysis readiness.
    assert module.ready()


@pytest.mark.parametrize('passes', [False, True])
def test_bilingual_supplement_stays_a_draft_even_when_screen_passes(tmp_path, monkeypatch, passes):
    plan_root = tmp_path/'r9/drop25_v1'
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'PLAN', plan_root)
    monkeypatch.setattr(report, 'ANALYSIS', tmp_path/'analysis')
    monkeypatch.setattr(report, 'FIT', tmp_path/'fit')
    spec = {'name': 'parent', 'learning_rate': .0003, 'dropout': .15}
    parent = dict(spec=spec, candidate='parent', initial_state_sha256='same',
        code_commit='code', checkpoint_sha256='checkpoint', data_hash='data',
        panel_hash='panel', name_cache_hash='cache', parameter_count=100,
        trainable_parameter_count=90, seed=20260922, best_epoch=50,
        epoch_completed=60, elapsed_seconds=100.)
    candidate = dict(parent, candidate='tf192_mae_drop25_lr3e4_60',
        spec=dict(spec,name='tf192_mae_drop25_lr3e4_60',dropout=.25))
    analysis = {'records': {'mae': {'manifest': parent}, 'drop25': {'manifest': candidate}},
                'screening_gates': {'primary': passes, 'interval': passes, 'legacy': passes}}
    axes = {'axis_count':142, 'axes_point_better_than_mae':70,
        'axes_unadjusted_interval_supports_improvement':10,
        'axes_unadjusted_interval_supports_regression':11,'sparse_validation_axes_below30':9,
        'drop25_axes_better_than_rf':60,'amino_acid_axes_drop25_worse_than_mae':3,'input_hashes':{}}
    data = {
        plan_root/'config.json': {'estimated_training_seconds':11236.425175,'followup_delay_minutes':198},
        plan_root/'launch.json': {'plan_sha256':'hash'},
        tmp_path/'reports/v9_r9_drop25_functional_v1/verification.json':
            dict(small_batch_overfit={'steps':50,'before':.3,'after':.1}, **diagnostic_fixture()[0]),
        tmp_path/'reports/v9_r9_drop25_preparation_v1/verification.json': diagnostic_fixture()[1],
        tmp_path/'reports/v9_r9_drop25_axis_changes_v1/summary.json':axes,
        tmp_path/'reports/v9_r9_drop25_cases_v1/summary.json':
            {'candidate':candidate['candidate'],'input_hashes':{}}}
    monkeypatch.setattr(report,'read',lambda path:data[path])
    monkeypatch.setattr(report,'digest',lambda path:'hash')
    names=['completion','name_only','subsets','retrieval','intervals',
           'retrieval_intervals','decomposition','fit','clipping','gates']
    shared={key:f'|{key}|Value|\n|---|---|\n|synthetic|0.123456|' for key in names}
    zh=report.supplement('ZH',analysis,{'elapsed_seconds':3.},shared,tmp_path/'new')
    en=report.supplement('EN',analysis,{'elapsed_seconds':3.},shared,tmp_path/'new')
    assert '机器证据草稿' in zh and 'machine-evidence draft' in en
    assert '实际' in zh and 'still require review' in en
    for table in shared.values():
        assert zh.count(table)==en.count(table)==1
    assert ('满足三个' if passes else '未满足三个') in zh
    assert ('Meets all three' if passes else 'Does not meet all three') in en
    assert '### 16.8.' in zh and '### 16.8.' in en
    assert '14处' in zh and 'all 14 dropout settings' in en
    assert '训练损失及梯度按预期不同' in zh and 'gradients differ as intended' in en
    assert 'Failed preflight' not in en and 'two thirds' not in en
    # A second method factor cannot pass the declared single-factor check.
    candidate['spec']['objective']='mse'
    with pytest.raises(ValueError,match='operational evidence'):
        report.supplement('EN',analysis,{'elapsed_seconds':3.},shared,tmp_path/'new')


def diagnostic_fixture():
    functional = {
        'plan_sha256':'hash','same_seed_parent_initialization_exact':True,'constructor_rng_exact':True,
        'legacy_control_initialization_forward_and_initial_mae_exact':True,
        'matched_initial_training_rng_consumption':True,
        'training_loss_and_gradients_differ_under_matched_initial_rng':True,
        'all_configured_dropout_sites_verified':True,
        'dropout_sites_control':{str(i):.15 for i in range(14)},
        'dropout_sites_candidate':{str(i):.25 for i in range(14)}}
    prepared = {'plan_sha256':'hash','tests_passed':33,
        'production_default_backend_in_fresh_process':{
            'deterministic_algorithms':False,'CUBLAS_WORKSPACE_CONFIG':None,
            'flash_sdp':True,'memory_efficient_sdp':True,'math_sdp':True}}
    return functional, prepared


@pytest.mark.parametrize('change', ['formal_backend','dropout_rate','missing_site','training_function','stale_hash'])
def test_report_rejects_incorrect_dropout_or_backend_claims(change):
    functional, prepared = diagnostic_fixture()
    report.verify_dropout_scope(functional,prepared,'hash')
    if change == 'formal_backend':
        prepared['production_default_backend_in_fresh_process']['deterministic_algorithms']=True
    elif change == 'dropout_rate':
        functional['dropout_sites_candidate']['0']=.15
    elif change == 'missing_site':
        del functional['dropout_sites_candidate']['0']
    elif change == 'training_function':
        functional['training_loss_and_gradients_differ_under_matched_initial_rng']=False
    else:
        prepared['plan_sha256']='stale'
    with pytest.raises(ValueError,match='scope changed'):
        report.verify_dropout_scope(functional,prepared,'hash')

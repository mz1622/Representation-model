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


report = script('report_foodnutrigpt_r9_lr6e4')


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


@pytest.mark.parametrize('name', ['review_foodnutrigpt_r9_lr6e4_axes', 'review_foodnutrigpt_r9_lr6e4_cases'])
def test_review_requires_completed_analysis_before_any_current_run_read(tmp_path, monkeypatch, name):
    module = script(name)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    if name.endswith('_axes'):
        monkeypatch.setitem(module.paths, 'analysis', tmp_path/'absent_analysis.json')
    # No model construction or current run read occurs on missing-analysis readiness.
    assert module.ready()


@pytest.mark.parametrize('passes', [False, True])
def test_bilingual_supplement_stays_a_draft_even_when_screen_passes(tmp_path, monkeypatch, passes):
    plan_root = tmp_path/'r9/lr6e4_v1'
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report, 'PLAN', plan_root)
    monkeypatch.setattr(report, 'ANALYSIS', tmp_path/'analysis')
    monkeypatch.setattr(report, 'FIT', tmp_path/'fit')
    spec = {'name': 'parent', 'learning_rate': .0003}
    parent = dict(spec=spec, candidate='parent', initial_state_sha256='same',
        code_commit='code', checkpoint_sha256='checkpoint', data_hash='data',
        panel_hash='panel', name_cache_hash='cache', parameter_count=100,
        trainable_parameter_count=90, seed=20260922, best_epoch=50,
        epoch_completed=60, elapsed_seconds=100.)
    candidate = dict(parent, candidate='tf192_mae_lr6e4_60',
        spec=dict(spec,name='tf192_mae_lr6e4_60',learning_rate=.0006))
    analysis = {'records': {'mae': {'manifest': parent}, 'lr6e4': {'manifest': candidate}},
                'screening_gates': {'primary': passes, 'interval': passes, 'legacy': passes}}
    axes = {'axis_count':142, 'axes_point_better_than_mae':70,
        'axes_unadjusted_interval_supports_improvement':10,
        'axes_unadjusted_interval_supports_regression':11,'sparse_validation_axes_below30':9,
        'lr6e4_axes_better_than_rf':60,'amino_acid_axes_lr6e4_worse_than_mae':3,'input_hashes':{}}
    data = {
        plan_root/'config.json': {'estimated_training_seconds':13479.7949,'followup_delay_minutes':235},
        plan_root/'launch.json': {'plan_sha256':'hash'},
        tmp_path/'reports/v9_r9_lr6e4_functional_v1/verification.json':
            {'small_batch_overfit':{'steps':50,'before':.3,'after':.1}},
        tmp_path/'reports/v9_r9_lr6e4_launch_failure_v1/record.json':
            {'optimizer_steps':0,'method_registration_changed':False},
        tmp_path/'reports/v9_r9_lr6e4_axis_changes_v1/summary.json':axes,
        tmp_path/'reports/v9_r9_lr6e4_cases_v1/summary.json':
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
    assert '### 14.8.' in zh and '### 14.8.' in en
    # Neither a second method factor nor a failed launch with optimization can pass provenance checks.
    candidate['spec']['objective']='mse'
    with pytest.raises(ValueError,match='operational evidence'):
        report.supplement('EN',analysis,{'elapsed_seconds':3.},shared,tmp_path/'new')

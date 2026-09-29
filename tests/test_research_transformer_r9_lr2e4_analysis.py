"""Decision-boundary tests use synthetic intervals, not experimental outcomes."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_foodnutrigpt_r9_lr2e4.py'
spec = importlib.util.spec_from_file_location('r9_lr2e4_analysis', path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize('gain,lower,legacy,expected', [
    (.001, .0001, -.02, True),  # No invented5percent requirement; inclusive2percent guard.
    (0., .0001, 0., False),
    (.01, 0., 0., False),
    (.01, -.001, .02, False),
    (.05, .01, -.02001, False),
])
def test_registered_three_conditions_are_jointly_required(gain, lower, legacy, expected):
    comparison = {'paired_intervals': {
        'scaled_log_mae': {'relative_improvement': gain, 'relative_improvement_95_interval': [lower, .1]},
        'log_mae': {'relative_improvement': legacy}}}
    assert all(analysis.screening_gates(comparison).values()) == expected


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_screening_evidence_cannot_pass(invalid):
    comparison = {'paired_intervals': {
        'scaled_log_mae': {'relative_improvement': .01, 'relative_improvement_95_interval': [.001, .1]},
        'log_mae': {'relative_improvement': invalid}}}
    with pytest.raises(ValueError, match='Nonfinite'):
        analysis.screening_gates(comparison)


def control_fixture():
    parent = dict(seed=22, training_tasks=10, observed_target_cells=20, data_hash='d',
        panel_hash='p', name_cache_hash='n', initial_state_sha256='base',
        parameter_count=100, trainable_parameter_count=90,
        spec=dict(name='parent', objective='mae', source_weight=1., learning_rate=.0003))
    spec = dict(parent['spec'], name='lr2e4', learning_rate=.0002)
    candidate = dict(parent, spec=spec, kind='transformer_direct',
        same_seed_parent_initialization_exact=True,
        numerical_recipe_changes=['learning_rate'])
    return candidate, parent, spec


def test_exact_control_and_learning_rate_only_change_pass():
    analysis.verify_method_control(*control_fixture())


@pytest.mark.parametrize('field,value', [
    ('seed', 23), ('panel_hash', 'changed'), ('initial_state_sha256', 'changed'),
    ('parameter_count', 101), ('trainable_parameter_count', 89),
    ('kind', 'transformer_axisvalue_direct_v1'),
    ('numerical_recipe_changes', ['learning_rate', 'objective']),
    ('same_seed_parent_initialization_exact', False)])
def test_changed_controls_fail(field, value):
    candidate, parent, spec = control_fixture()
    candidate[field] = value
    with pytest.raises(ValueError):
        analysis.verify_method_control(candidate, parent, spec)


def test_matching_candidate_spec_cannot_hide_second_intervention():
    candidate, parent, spec = control_fixture()
    spec['source_weight'] = 0.
    with pytest.raises(ValueError, match='only changed numerical'):
        analysis.verify_method_control(candidate, parent, spec)


def partitions():
    import pandas as pd
    common = dict(stratification=['label_stratum']*2, stratum=['positive', 'explicit_zero'],
        source_cells=[4,3], candidate_groups=[2,2], supported_axes=[2,2],
        macro_weight_mass=[.6,.4], tree_under_contribution=[.07,0.],
        tree_over_contribution=[.06,.11], tree_mae_contribution=[.13,.11])
    parent = pd.DataFrame(dict(common, neural_under_contribution=[.1,0.],
        neural_over_contribution=[.05,.1], neural_mae_contribution=[.15,.1]))
    candidate = pd.DataFrame(dict(common, neural_under_contribution=[.08,0.],
        neural_over_contribution=[.09,.06], neural_mae_contribution=[.17,.06]))
    return parent, candidate


def test_common_denominator_decomposition_reconstructs_primary_change():
    import numpy as np
    frame, result = analysis.partition_change(*partitions(), -.02)
    np.testing.assert_allclose(
        [result['positive_under'], result['positive_over'], result['explicit_zero']],
        [-.02,.04,-.04], atol=1e-12, rtol=0)
    assert frame.lr2e4_minus_mae_mae.sum() == pytest.approx(-.02)


@pytest.mark.parametrize('column,value', [('tree_mae_contribution', .2),
    ('macro_weight_mass', .5), ('neural_mae_contribution', float('nan'))])
def test_incompatible_or_nonfinite_partition_cannot_be_reported(column, value):
    parent, candidate = partitions()
    candidate.loc[0,column] = value
    with pytest.raises((ValueError, AssertionError)):
        analysis.partition_change(parent, candidate, -.02)

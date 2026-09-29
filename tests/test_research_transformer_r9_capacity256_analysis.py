"""Decision-boundary tests use synthetic intervals, not experimental outcomes."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_foodnutrigpt_r9_capacity256.py'
spec = importlib.util.spec_from_file_location('r9_capacity256_analysis', path)
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
        spec=dict(name='parent', objective='mae', source_weight=1., learning_rate=.0003,
                  d_model=192, n_heads=6, feedforward_dim=768))
    spec = dict(parent['spec'], name='capacity256', d_model=256, n_heads=8,
                feedforward_dim=1024)
    candidate = dict(parent, spec=spec, kind='transformer_direct',
        same_seed_parent_initialization_exact=False, initial_state_sha256='larger',
        parent_initial_state_sha256='base', parameter_count=180,
        trainable_parameter_count=170,
        numerical_recipe_changes=['d_model', 'n_heads', 'feedforward_dim'])
    functional = dict(status='complete', same_seed_parent_initialization_exact=False,
        new_initialization_matches_explicit_reference_exact=True,
        only_capacity_configuration_triplet_changed=True,
        initial_state_sha256='larger', parameter_count=180, trainable_parameter_count=170)
    return candidate, parent, spec, functional


def test_capacity_group_uses_its_own_verified_initialization():
    analysis.verify_method_control(*control_fixture())


@pytest.mark.parametrize('field,value', [
    ('seed', 23), ('panel_hash', 'changed'), ('initial_state_sha256', 'base'),
    ('parameter_count', 181), ('trainable_parameter_count', 171),
    ('kind', 'transformer_axisvalue_direct_v1'),
    ('numerical_recipe_changes', ['d_model', 'n_heads', 'feedforward_dim', 'objective']),
    ('same_seed_parent_initialization_exact', True),
    ('parent_initial_state_sha256', 'different')])
def test_changed_controls_or_false_same_initialization_claim_fail(field, value):
    candidate, parent, spec, functional = control_fixture()
    candidate[field] = value
    with pytest.raises(ValueError):
        analysis.verify_method_control(candidate, parent, spec, functional)


@pytest.mark.parametrize('field,value', [('learning_rate', .001), ('source_weight', 0.),
                                        ('objective', 'mse'), ('n_heads', 4)])
def test_matching_candidate_spec_cannot_hide_second_intervention(field, value):
    candidate, parent, spec, functional = control_fixture()
    spec[field] = value
    with pytest.raises(ValueError, match='only changed numerical'):
        analysis.verify_method_control(candidate, parent, spec, functional)


@pytest.mark.parametrize('field,value', [('status', 'running'),
    ('same_seed_parent_initialization_exact', True),
    ('new_initialization_matches_explicit_reference_exact', False),
    ('only_capacity_configuration_triplet_changed', False),
    ('initial_state_sha256', 'wrong'), ('parameter_count', 179)])
def test_missing_capacity_preflight_evidence_cannot_pass(field, value):
    candidate, parent, spec, functional = control_fixture()
    functional[field] = value
    with pytest.raises(ValueError):
        analysis.verify_method_control(candidate, parent, spec, functional)


def test_common_denominator_changes_do_not_require_matching_model_sizes():
    candidate, parent, spec, functional = control_fixture()
    assert candidate['parameter_count'] > parent['parameter_count']
    assert candidate['initial_state_sha256'] != parent['initial_state_sha256']
    analysis.verify_method_control(candidate, parent, spec, functional)


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
    assert frame.capacity256_minus_mae_mae.sum() == pytest.approx(-.02)


@pytest.mark.parametrize('column,value', [('tree_mae_contribution', .2),
    ('macro_weight_mass', .5), ('neural_mae_contribution', float('nan'))])
def test_incompatible_or_nonfinite_partition_cannot_be_reported(column, value):
    parent, candidate = partitions()
    candidate.loc[0,column] = value
    with pytest.raises((ValueError, AssertionError)):
        analysis.partition_change(parent, candidate, -.02)

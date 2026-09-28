"""Decision-boundary tests use synthetic intervals, not experimental outcomes."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_foodnutrigpt_r9_axisvalue.py'
spec = importlib.util.spec_from_file_location('r9_axisvalue_analysis', path)
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
        parameter_count=100, trainable_parameter_count=90)
    spec = dict(objective='mae', axis_value_residual='zero_init_linear_v1')
    candidate = dict(parent, spec=spec, kind='transformer_axisvalue_direct_v1',
        parent_initial_state_sha256='base', added_parameters=48384,
        residual_initialization='zeros_no_rng_draw', numerical_recipe_changes=['axis_value_residual'])
    candidate.update(parameter_count=48484, trainable_parameter_count=48474)
    return candidate, parent, spec


def test_exact_shared_control_and_declared_parameter_increment_pass():
    analysis.verify_method_control(*control_fixture())


@pytest.mark.parametrize('field,value', [
    ('seed', 23), ('panel_hash', 'changed'), ('added_parameters', 48383),
    ('parameter_count', 48485), ('trainable_parameter_count', 48473),
    ('parent_initial_state_sha256', 'changed'), ('kind', 'transformer_direct'),
    ('numerical_recipe_changes', ['axis_value_residual', 'objective']),
    ('residual_initialization', 'random')])
def test_changed_controls_fail(field, value):
    candidate, parent, spec = control_fixture()
    candidate[field] = value
    with pytest.raises(ValueError):
        analysis.verify_method_control(candidate, parent, spec)

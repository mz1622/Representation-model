"""Decision-boundary tests use synthetic intervals, not experimental outcomes."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analyze_foodnutrigpt_r9_mse.py'
spec = importlib.util.spec_from_file_location('r9_mse_analysis', path)
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

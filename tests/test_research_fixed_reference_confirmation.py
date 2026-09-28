import numpy as np
import pandas as pd
import pytest

from foodcomp.research_confirmation import SEEDS
from foodcomp.research_fixed_reference_confirmation import confirm_fixed_references, acceptance_gates


def fixture():
    base = pd.DataFrame({'exact_name_group_id': np.repeat([f'g{i}' for i in range(20)], 2),
        'axis_index': np.tile([0, 1], 20), 'scaled_log_mae': np.arange(1, 41, dtype=float),
        'log_mae': np.arange(1, 41, dtype=float)/10, 'raw_mae': np.arange(1, 41, dtype=float)*2})
    refs = {'rf': base.copy(), 'xgb': base.copy()}
    cols = ['scaled_log_mae', 'log_mae', 'raw_mae']
    refs['xgb'][cols] *= .9
    candidates = {}
    for seed, factor in zip(SEEDS, [.96, .97, .98]):
        candidates[seed] = base.copy()
        candidates[seed][cols] *= factor
    return candidates, refs


def test_positive_gain_below_five_percent_can_pass_without_beating_xgb():
    candidates, refs = fixture()
    result, axes = confirm_fixed_references(candidates, refs, [0, 1], repeats=100)
    assert result['conditional_fixed_rf_milestone_passed']
    assert result['comparisons']['rf']['paired_intervals']['scaled_log_mae']['relative_improvement'] == pytest.approx(.03)
    assert result['comparisons']['xgb']['paired_intervals']['scaled_log_mae']['relative_improvement'] < 0
    assert result['comparisons']['rf']['reference_seed_count'] == 1
    assert result['comparisons']['rf']['paired_intervals']['scaled_log_mae']['group_count'] == 20
    assert result['seed_summary']['scaled_log_mae']['sample_std'] > 0
    assert axes['rf'].candidate_support.tolist() == [20, 20]


def test_averages_errors_from_all_seeds_and_is_order_invariant():
    candidates, refs = fixture()
    result, _ = confirm_fixed_references(candidates, refs, [0, 1], repeats=100)
    shuffled = {seed: frame.sample(frac=1, random_state=seed) for seed, frame in candidates.items()}
    other, _ = confirm_fixed_references(shuffled, refs, [0, 1], repeats=100)
    assert result == other
    assert result['seed_summary']['scaled_log_mae']['mean'] == pytest.approx(.97*20.5)
    assert result['seed_summary']['scaled_log_mae']['mean'] > min(row['scaled_log_mae'] for row in result['per_seed'])


def test_legacy_guardrail_can_fail_despite_primary_win():
    candidates, refs = fixture()
    for frame in candidates.values():
        frame['log_mae'] *= 1.06
    result, _ = confirm_fixed_references(candidates, refs, [0, 1], repeats=100)
    assert result['gates']['primary_mean_improvement_over_fixed_rf']
    assert not result['gates']['legacy_mean_regression_at_most_2_percent']
    assert not result['conditional_fixed_rf_milestone_passed']


@pytest.mark.parametrize('gain,lower,legacy,expected', [(.01,0.,0.,False),(0.,.01,0.,False),(.01,.001,-.02,True),(.01,.001,-.020001,False)])
def test_strict_primary_and_interval_but_inclusive_legacy_boundary(gain,lower,legacy,expected):
    gates=acceptance_gates({'relative_improvement':gain,'relative_improvement_95_interval':[lower,.1]}, {'relative_improvement':legacy})
    assert all(gates.values()) is expected


@pytest.mark.parametrize('kind', ['missing_seed','missing_cell','duplicate_cell','nonfinite','negative','reference_panel','zero_reference','duplicate_axis'])
def test_invalid_evidence_cannot_be_accepted(kind):
    candidates, refs = fixture(); axes=[0,1]
    if kind=='missing_seed': del candidates[SEEDS[-1]]
    elif kind=='missing_cell': candidates[SEEDS[-1]]=candidates[SEEDS[-1]].iloc[:-1]
    elif kind=='duplicate_cell': candidates[SEEDS[0]]=pd.concat([candidates[SEEDS[0]],candidates[SEEDS[0]].iloc[:1]])
    elif kind=='nonfinite': candidates[SEEDS[0]].loc[0,'log_mae']=np.nan
    elif kind=='negative': refs['rf'].loc[0,'scaled_log_mae']=-1
    elif kind=='reference_panel': refs['xgb']=refs['xgb'].iloc[:-1]
    elif kind=='zero_reference': refs['rf']['scaled_log_mae']=0.
    elif kind=='duplicate_axis': axes=[0,1,1]
    with pytest.raises(ValueError): confirm_fixed_references(candidates,refs,axes,repeats=100)


def test_no_nonfinite_acceptance_metadata():
    with pytest.raises(ValueError,match='finite'):
        acceptance_gates({'relative_improvement':.1,'relative_improvement_95_interval':[np.nan,.2]}, {'relative_improvement':0.})

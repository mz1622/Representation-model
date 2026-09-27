import copy
import numpy as np
import pandas as pd
import pytest
from foodcomp.research_confirmation import SEEDS,seed_average_errors,confirm_completion


def runs_fixture():
    frame=pd.DataFrame({"exact_name_group_id":np.repeat([f"g{i}" for i in range(20)],2),
        "axis_index":np.tile([0,1],20),"scaled_log_mae":np.arange(1,41,dtype=float),"log_mae":np.arange(1,41,dtype=float)/10})
    runs={}
    for role,factor in [("candidate",.8),("rf",1.0),("xgb",.9)]:
        runs[role]={}
        for seed,variation in zip(SEEDS,[.99,1.,1.01]):
            f=frame.copy();f[["scaled_log_mae","log_mae"]]*=factor*variation;runs[role][seed]=f
    return runs


def test_all_registered_seeds_and_stronger_tree_are_used():
    result,axes=confirm_completion(runs_fixture(),[0,1],repeats=100)
    assert result["stronger_tree"]=="xgb"
    assert result["conditional_completion_milestone_passed"]
    assert result["food_group_intervals"]["scaled_log_mae"]["relative_improvement"]==pytest.approx(1-.8/.9)
    assert result["food_group_intervals"]["scaled_log_mae"]["group_count"]==20  # not 60 seed-foods or 120 cells
    assert result["seed_summaries"]["candidate"]["metrics"]["scaled_log_mae"]["sample_std"]>0
    assert axes.candidate_support.tolist()==[20,20]


def test_legacy_guardrail_cannot_be_hidden_by_primary_improvement():
    runs=runs_fixture()
    for frame in runs["candidate"].values():frame.log_mae*=1.2
    result,_=confirm_completion(runs,[0,1],repeats=100)
    assert result["gates"]["primary_mean_gain_at_least_5_percent"]
    assert not result["gates"]["legacy_mean_regression_at_most_2_percent"]
    assert not result["conditional_completion_milestone_passed"]


def test_no_missing_seed_mismatched_cells_or_nonfinite_errors():
    runs=runs_fixture();del runs["candidate"][SEEDS[-1]]
    with pytest.raises(ValueError,match="three"):confirm_completion(runs,[0,1],repeats=100)
    runs=runs_fixture();runs["candidate"][SEEDS[-1]]=runs["candidate"][SEEDS[-1]].iloc[:-1]
    with pytest.raises(ValueError,match="coverage"):confirm_completion(runs,[0,1],repeats=100)
    runs=runs_fixture();runs["candidate"][SEEDS[0]].loc[0,"log_mae"]=np.nan
    with pytest.raises(ValueError,match="Invalid"):confirm_completion(runs,[0,1],repeats=100)


def test_error_averaging_is_order_independent_and_not_best_seed_selection():
    runs=runs_fixture()["candidate"];changed=copy.deepcopy(runs)
    for seed in SEEDS:changed[seed]=changed[seed].sample(frac=1,random_state=5)
    a,summary=seed_average_errors(runs,[0,1]);b,_=seed_average_errors(changed,[0,1])
    pd.testing.assert_frame_equal(a,b)
    means=[r["scaled_log_mae"] for r in summary["per_seed"]]
    assert summary["metrics"]["scaled_log_mae"]["mean"]==pytest.approx(np.mean(means))
    assert summary["metrics"]["scaled_log_mae"]["mean"]>min(means)

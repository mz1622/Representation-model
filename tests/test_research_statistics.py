import numpy as np
import pandas as pd
import pytest
from foodcomp.research_statistics import paired_interval,paired_axis_intervals
from foodcomp.research_statistics import paired_group_rates

def test_group_resampling_preserves_paired_proportional_effect():
    a=pd.DataFrame({"exact_name_group_id":np.repeat([f"g{i}" for i in range(20)],2),
                    "axis_index":np.tile([0,1],20),"scaled_log_mae":np.arange(1,41,dtype=float)})
    b=a.copy();b.scaled_log_mae*=.9
    result=paired_interval(a,b,[0,1],repeats=100)
    assert result["group_count"]==20
    assert result["relative_improvement"]==pytest.approx(.1)
    assert result["relative_improvement_95_interval"]==pytest.approx([.1,.1])
    with pytest.raises(ValueError):paired_interval(a,b.iloc[:-1],[0,1],repeats=100)

def test_axis_intervals_are_paired_and_report_support():
    a=pd.DataFrame({"exact_name_group_id":np.repeat([f"g{i}" for i in range(20)],2),
                    "axis_index":np.tile([0,1],20),"scaled_log_mae":np.arange(1,41,dtype=float)})
    b=a.copy();b.scaled_log_mae-=.5
    result=paired_axis_intervals(a,b,repeats=100)
    np.testing.assert_allclose(result.difference_95_low,-.5)
    np.testing.assert_allclose(result.difference_95_high,-.5)
    assert result.candidate_support.tolist()==[20,20]
    assert result.valid_resamples.tolist()==[100,100]


def test_retrieval_zero_baseline_and_paired_group_rates():
    a=pd.DataFrame({"exact_name_group_id":["a","b","c"],"recall_at_1":[0.,0.,0.],"mrr":[.1,.2,.3]})
    b=a.copy();b.recall_at_1+=.1;b.mrr+=.05
    scores=paired_group_rates(a,b,["recall_at_1","mrr"],repeats=100)
    assert scores["recall_at_1"]["difference_95_interval"]==pytest.approx([.1,.1])
    assert scores["mrr"]["difference_95_interval"]==pytest.approx([.05,.05])
    assert scores["mrr"]["group_count"]==3
    with pytest.raises(ValueError,match="groups"):paired_group_rates(a,b.iloc[:-1],["mrr"],repeats=100)
    b.loc[0,"mrr"]=np.nan
    with pytest.raises(ValueError,match="finite"):paired_group_rates(a,b,["mrr"],repeats=100)

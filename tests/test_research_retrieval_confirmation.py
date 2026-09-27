"""Check scientific aggregation, especially source weights and seed/rank distinction."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('confirmation',Path(__file__).resolve().parents[1]/'scripts/confirm_foodnutrigpt_alignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_sources_equal_and_metrics_averaged_after_ranks():
    a=pd.DataFrame({'visible_fraction':[1.]*4,'exact_name_group_id':['a','a','a','b'],
        'source_key':['s1','s1','s2','s1'],'rank':[1,1,4,2]})
    b=a.copy();b['rank']=[4,4,1,8]
    ga=module.group_rates(a,1.);gb=module.group_rates(b,1.)
    assert ga.loc['a','mrr']==pytest.approx(.625)  # equal sources, not .75 profile mean
    combined=module.mean_seed_rates([ga,gb])
    assert combined.loc['a','mrr']==pytest.approx(.625)  # not reciprocal mean rank (.4)
    assert combined.loc['b','mrr']==pytest.approx(.3125)
    assert combined.loc['a','recall_at_1']==pytest.approx(.5)
    assert combined.mrr.mean()==pytest.approx(.46875)
    with pytest.raises(AssertionError):module.mean_seed_rates([ga,gb.iloc[:1]])
    gb.loc['a','mrr']=np.nan
    with pytest.raises(ValueError,match='Invalid'):module.mean_seed_rates([ga,gb])

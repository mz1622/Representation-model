import numpy as np
import pandas as pd
import pytest
from foodcomp.research_statistics import paired_interval

def test_group_resampling_preserves_paired_proportional_effect():
    a=pd.DataFrame({"exact_name_group_id":np.repeat([f"g{i}" for i in range(20)],2),
                    "axis_index":np.tile([0,1],20),"scaled_log_mae":np.arange(1,41,dtype=float)})
    b=a.copy();b.scaled_log_mae*=.9
    result=paired_interval(a,b,[0,1],repeats=100)
    assert result["group_count"]==20
    assert result["relative_improvement"]==pytest.approx(.1)
    assert result["relative_improvement_95_interval"]==pytest.approx([.1,.1])
    with pytest.raises(ValueError):paired_interval(a,b.iloc[:-1],[0,1],repeats=100)

import numpy as np
import pytest
from foodcomp.research_name_intervention import intervene_name_features


def test_name_permutation_preserves_duplicate_names_and_is_row_order_independent():
    names=np.array(['pear','apple','pear','beef','rice','apple'])
    base=np.arange(4*128,dtype=np.float32).reshape(4,128)
    x=base[[2,0,2,1,3,0]]
    original=x.copy()
    altered,replacement=intervene_name_features(x,names,'permute_name')
    assert np.all(replacement!=names)
    np.testing.assert_array_equal(altered[0],altered[2])
    np.testing.assert_array_equal(altered[1],altered[5])
    reverse,names_reverse=intervene_name_features(x[::-1],names[::-1],'permute_name')
    np.testing.assert_array_equal(reverse[::-1],altered)
    np.testing.assert_array_equal(names_reverse[::-1],replacement)
    np.testing.assert_array_equal(x,original)


def test_extra_direction_removal_retains_prefix_and_original_input():
    x=np.arange(2*128,dtype=np.float32).reshape(2,128)
    changed,_=intervene_name_features(x,['a','b'],'drop_extra96')
    np.testing.assert_array_equal(changed[:,:32],x[:,:32])
    assert not changed[:,32:].any() and x[:,32:].any()
    zero,_=intervene_name_features(x,['a','b'],'zero_name')
    assert not zero.any()
    with pytest.raises(ValueError):intervene_name_features(x,['a','a'],'permute_name')
    bad=x.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):intervene_name_features(bad,['a','b'],'zero_name')

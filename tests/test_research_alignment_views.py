import numpy as np
import pytest
import torch
from foodcomp.research_alignment_views import partial_view_masks,partial_view_features
from foodcomp.research_alignment import numeric_features


def test_zero_probability_exact_default_and_preserves_inputs():
    observed=np.ones((8,142),dtype=bool);observed[:,20:]=False
    x=numeric_features(np.ones((8,142)),observed)
    out,mask,assigned=partial_view_features(x,0.,22,1)
    np.testing.assert_array_equal(out,x);np.testing.assert_array_equal(mask,observed);assert not assigned.any()


def test_subset_minimum_unassigned_and_three_observations():
    observed=np.zeros((100,142),dtype=bool);observed[:50,:3]=True;observed[50:,:80]=True
    result,assigned=partial_view_masks(observed,.5,22,1)
    assert assigned.any() and (~assigned).any()
    assert not (result&~observed).any() and (result.sum(1)>=3).all()
    np.testing.assert_array_equal(result[:50],observed[:50])
    np.testing.assert_array_equal(result[~assigned],observed[~assigned])
    assert (result[50:][assigned[50:]].sum(1)<80).all()


def test_reproducible_independent_rng_and_epoch_varies():
    observed=np.ones((40,142),dtype=bool)
    np.random.seed(7);before=np.random.get_state();torch.manual_seed(7);torch_before=torch.get_rng_state()
    a,aa=partial_view_masks(observed,.5,22,1);b,bb=partial_view_masks(observed,.5,22,1)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(aa,bb)
    after=np.random.get_state();assert before[0]==after[0] and before[2:]==after[2:];np.testing.assert_array_equal(before[1],after[1])
    torch.testing.assert_close(torch_before,torch.get_rng_state(),rtol=0,atol=0)
    other,_=partial_view_masks(observed,.5,22,2);assert not np.array_equal(other,a)


def test_views_value_blind_and_keep_explicit_zero_indicator():
    observed=np.ones((20,142),dtype=bool)
    a=numeric_features(np.zeros((20,142)),observed);b=numeric_features(np.full((20,142),9.),observed)
    x,ma,sa=partial_view_features(a,.5,22,1);_,mb,sb=partial_view_features(b,.5,22,1)
    np.testing.assert_array_equal(ma,mb);np.testing.assert_array_equal(sa,sb)
    assert np.count_nonzero(x[:,:142])==0
    np.testing.assert_array_equal(x[:,142:],ma.astype(np.float32))
    assert (x[:,142:]==1).any() and (x[:,142:]==0).any()


@pytest.mark.parametrize('probability',[-.1,1.1,float('nan')])
def test_bad_probability_rejected(probability):
    with pytest.raises(ValueError):partial_view_masks(np.ones((2,142),bool),probability,22,1)


def test_insufficient_original_context_rejected():
    with pytest.raises(ValueError):partial_view_masks(np.zeros((2,142),bool),.5,22,1)

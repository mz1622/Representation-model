import numpy as np
import pytest
import torch
from foodcomp.research_name_neighbors import ObservedAxisNeighbors
from foodcomp.research_profile_retrieval import predicted_profile_ranks


def test_original_neighbor_formula_with_explicit_zero_and_source_weights():
    x=np.array([[0.],[2.]],dtype=np.float32);y=np.array([0.,np.log(3)],dtype=np.float32);w=np.array([1.,2.],dtype=np.float32)
    model=ObservedAxisNeighbors(x,y,w,.5,n_jobs=1)
    result,indices,distances=model.predict(np.array([[1.]],dtype=np.float32),return_neighbors=True)
    expected=.5*np.expm1(float(y[1])*2/3)
    np.testing.assert_allclose(result,[expected],rtol=1e-12,atol=0)
    assert set(indices[0])=={0,1};np.testing.assert_array_equal(distances,[[1.,1.]])


def test_missing_label_must_be_excluded_not_imputed():
    with pytest.raises(ValueError):ObservedAxisNeighbors(np.zeros((2,1)),np.array([0.,np.nan]),np.ones(2),1.)


def test_tree_tied_neighbors_independent_of_query_batch_and_order():
    x=np.zeros((20,32),dtype=np.float32);x[10:]=1.
    model=ObservedAxisNeighbors(x,np.arange(20,dtype=np.float32)/20,np.ones(20),1.,n_jobs=1,backend='kd_tree')
    queries=np.array([x[0],x[-1],np.full(32,.5)],dtype=np.float32)
    whole=model.predict(queries)
    np.testing.assert_array_equal(model.predict(queries[::-1])[::-1],whole)
    np.testing.assert_array_equal(np.concatenate([model.predict(q[None]) for q in queries]),whole)


def test_ranks_use_only_visible_values_and_keep_zero_observed():
    candidates=torch.tensor([[0.,8.],[1.,0.],[0.,9.]])
    query=torch.tensor([[0.,float('nan')],[99.,0.]])
    visible=torch.tensor([[True,False],[False,True]])
    correct=torch.tensor([2,1])
    ranks=predicted_profile_ranks(candidates,query,visible,correct)
    torch.testing.assert_close(ranks,torch.tensor([2,1]),rtol=0,atol=0)
    query[~visible]=float('inf')
    torch.testing.assert_close(predicted_profile_ranks(candidates,query,visible,correct),ranks,rtol=0,atol=0)


def test_nonfinite_candidate_fails_even_on_hidden_coordinate():
    with pytest.raises(FloatingPointError):predicted_profile_ranks(torch.tensor([[0.,float('nan')]]),torch.zeros((1,2)),torch.tensor([[True,False]]),torch.tensor([0]))

import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_alignment import numeric_features,source_equal_weights,mapping_loss,exact_ranks,selection_score,NumericNameMapper,state_fingerprint


def test_numeric_hidden_values_irrelevant_and_zero_observed():
    values=np.array([[0.,2.,100.]],np.float32);mask=np.array([[True,True,False]])
    expected=numeric_features(values,mask)
    values[0,2]=np.nan;np.testing.assert_array_equal(expected,numeric_features(values,mask))
    values[0,2]=np.inf;np.testing.assert_array_equal(expected,numeric_features(values,mask))
    assert expected[0,0]==0 and expected[0,3]==1 and expected[0,5]==0
    values[0,1]=np.nan
    with pytest.raises(ValueError):numeric_features(values,mask)
    with pytest.raises(ValueError):numeric_features(np.ones((1,3)),np.ones((1,3)))


def test_weights_equalize_sources_and_profiles_without_pooling_labels():
    p=pd.DataFrame({'exact_name_group_id':['a','a','a','b'],'source_key':['x','x','y','x'],'profile_id':[1,2,3,4]})
    weights=source_equal_weights(p)
    np.testing.assert_array_equal(weights,np.array([.5,.5,1.,2.]))
    assert weights[:3].sum()==weights[3] and weights[:2].sum()==weights[2]


def test_mse_uniform_minibatch_estimate_matches_global_weighted_objective():
    mapped=torch.tensor([[1.,3.],[2.,5.],[7.,9.]],dtype=torch.float64,requires_grad=True)
    target=torch.zeros_like(mapped);ids=torch.tensor([0,1,2]);w=torch.tensor([.25,.75,2.],dtype=torch.float64)
    def loss(s):return mapping_loss(mapped[s],target[s],ids[s],w[s],objective='mse',population_size=3,weight_sum=3.)
    full=loss(slice(None));split=(loss(slice(0,1))+2*loss(slice(1,3)))/3
    expected=(mapped.square().mean(1)*w).sum()/3
    torch.testing.assert_close(full,expected);torch.testing.assert_close(full,split)
    torch.testing.assert_close(torch.autograd.grad(full,mapped)[0],2*mapped*w[:,None]/(3*2))


def test_contrastive_exact_names_are_one_column_with_shared_positive():
    mapped=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=torch.float64,requires_grad=True)
    target=mapped.detach().clone();ids=torch.tensor([5,5,9]);w=torch.ones(3,dtype=torch.float64)
    loss=mapping_loss(mapped,target,ids,w,objective='contrastive',population_size=3,weight_sum=3.,temperature=1.)
    expected=torch.log1p(torch.exp(torch.tensor(-1.,dtype=torch.float64)))
    torch.testing.assert_close(loss,expected)
    loss.backward();assert torch.isfinite(mapped.grad).all()
    wrong=target.clone();wrong[1]=torch.tensor([0.,1.])
    with pytest.raises(ValueError):mapping_loss(mapped,wrong,ids,w,objective='contrastive',population_size=3,weight_sum=3.)


def test_contrastive_labels_are_supervision_not_model_inputs():
    torch.manual_seed(2);model=NumericNameMapper(input_dim=4,text_dim=2,width=8)
    x=torch.rand(3,4);before=model(x).detach().clone()
    targets=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    mapping_loss(model(x),targets,torch.tensor([4,5,6]),torch.ones(3),objective='contrastive',population_size=3,weight_sum=3.)
    torch.testing.assert_close(model(x),before,rtol=0,atol=0)


def test_exact_ranks_perfect_worst_and_lexical_ties():
    distance=torch.tensor([[0.,1.,2.],[0.,1.,2.],[0.,0.,0.]])
    assert exact_ranks(distance,torch.tensor([0,2,2])).tolist()==[1,3,3]
    with pytest.raises(FloatingPointError):exact_ranks(torch.full((1,2),float('nan')),torch.tensor([0]))
    with pytest.raises(ValueError):exact_ranks(distance,torch.tensor([0,1,3]))


def test_mapping_save_reload_and_bad_input(tmp_path):
    torch.manual_seed(3);m=NumericNameMapper(input_dim=4,text_dim=2,width=8);x=torch.ones(3,4)
    path=tmp_path/'state.pt';torch.save(m.state_dict(),path)
    restored=NumericNameMapper(input_dim=4,text_dim=2,width=8);restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0);assert state_fingerprint(m)==state_fingerprint(restored)
    with pytest.raises(FloatingPointError):m(torch.full((3,4),float('inf')))


def test_selection_uses_both_scenarios_equally():
    assert selection_score([{'visible_fraction':1.,'mrr':.3},{'visible_fraction':.3,'mrr':.1}])==.2
    with pytest.raises(ValueError):selection_score([{'visible_fraction':1.,'mrr':.3}])
    with pytest.raises(FloatingPointError):selection_score([{'visible_fraction':1.,'mrr':float('nan')},{'visible_fraction':.3,'mrr':.1}])

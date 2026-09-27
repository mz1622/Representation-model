import copy
import numpy as np
import pytest
import torch
from foodcomp.research_neural import DenseModel, batch_from_arrays
from foodcomp.research_r1 import model_loss
from foodcomp.research_views import extra_view_masks, subset_view, representation_distance, weighted_consistency, two_view_loss


def fixture():
    b = batch_from_arrays(np.array([[1., 0., 3., 0.], [4., 5., 0., 2.]], np.float32),
        np.array([[True, False, True, False], [False, True, False, True]]),
        np.array([[.2, .4], [.8, .3]], np.float32), "cpu")
    b.update(target=torch.tensor([[False, True, False, False], [True, False, False, False]]),
        positive=b["value"] > 0, cell_weight=torch.tensor([[0., 1., 0., 0.], [2., 0., 0., 0.]]),
        axis_total=torch.tensor([2., 1., 0., 0.]), objective_multiplier=.5)
    return b


def test_extra_masks_reproducible_nested_and_global_rng_independent():
    cpu = torch.get_rng_state(); nrng = np.random.get_state()
    a = extra_view_masks(1000, 4, .3, 22, 3)
    np.testing.assert_array_equal(a, extra_view_masks(1000, 4, .3, 22, 3))
    assert a.dtype == bool and 1000 < a.sum() < 1400
    assert (a <= extra_view_masks(1000, 4, .6, 22, 3)).all()
    assert not np.array_equal(a, extra_view_masks(1000, 4, .3, 22, 4))
    assert not extra_view_masks(4, 4, 0., 22, 3).any()
    assert extra_view_masks(4, 4, 1., 22, 3).all()
    torch.testing.assert_close(cpu, torch.get_rng_state(), rtol=0, atol=0)
    for x,y in zip(nrng, np.random.get_state()): np.testing.assert_array_equal(x,y)


def test_subset_cannot_reveal_targets_or_missing_and_does_not_mutate_labels():
    b = fixture(); before = copy.deepcopy(b)
    mask = torch.tensor([[True, False, False, False], [False, False, False, True]])
    second = subset_view(b, mask)
    assert second["masked"][b["masked"]].all()
    assert second["masked"][b["target"]].all()
    assert second["masked"][0,3]  # missing remains missing
    for key in b:
        if key != "masked": assert second[key] is b[key]
        if torch.is_tensor(b[key]): torch.testing.assert_close(b[key], before[key], rtol=0, atol=0)
    zero = copy.deepcopy(b); zero["value"][0,2] = 0.
    assert not subset_view(zero,mask)["masked"][0,2]  # retained explicit zero observed


def test_zero_coefficient_matches_original_loss_gradients_and_updates():
    torch.manual_seed(31)
    original = DenseModel(2,4,"mlp",width=12); matched = copy.deepcopy(original)
    a = torch.optim.AdamW(original.parameters(), lr=.001, weight_decay=.0001)
    c = torch.optim.AdamW(matched.parameters(), lr=.001, weight_decay=.0001)
    b = fixture()
    class Config: amount_loss_weight=1.
    for epoch in range(3):
        drop = extra_view_masks(2,4,.3,22,epoch)
        rng = torch.get_rng_state()
        a.zero_grad(); c.zero_grad()
        la = model_loss(original,b,"mlp",Config(),"mae")
        lc, parts = two_view_loss(matched,b,drop,0.)
        torch.testing.assert_close(la,lc,rtol=0,atol=0)
        la.backward(); lc.backward()
        for x,y in zip(original.parameters(),matched.parameters()):
            torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        torch.nn.utils.clip_grad_norm_(original.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(matched.parameters(),1.)
        a.step(); c.step()
        for x,y in zip(original.parameters(),matched.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        assert torch.isfinite(parts["consistency"])


def test_consistency_weighting_uses_global_axis_totals_and_batch_estimator():
    b = fixture(); distance = torch.tensor([.2,.8],requires_grad=True)
    value = weighted_consistency(distance,b)
    torch.testing.assert_close(value,torch.tensor(.5))
    values=[]
    for i in range(2):
        sub={k:(v[i:i+1] if torch.is_tensor(v) and v.ndim==2 else v) for k,v in b.items()}
        sub["objective_multiplier"]=1.
        values.append(weighted_consistency(distance[i:i+1],sub))
    torch.testing.assert_close(torch.stack(values).mean(),value)
    value.backward()
    torch.testing.assert_close(distance.grad,torch.tensor([.5,.5]))
    # Duplicating a source/profile observation while splitting its original weight
    # does not change the exhaustive objective.
    repeated={**b,"target":b["target"][[0,0,1]],"cell_weight":b["cell_weight"][[0,0,1]].clone()}
    repeated["cell_weight"][:2]/=2
    torch.testing.assert_close(weighted_consistency(torch.tensor([.2,.2,.8]),repeated),value)


def test_explicit_zero_supervised_missing_not_supervised_and_b_has_no_new_labels():
    m=DenseModel(2,4,"mlp",width=12); b=fixture()
    loss,_=two_view_loss(m,b,torch.ones_like(b["masked"]),.1)
    loss.backward()
    assert m.head.bias.grad[1] != 0  # original explicit-zero target
    assert m.head.bias.grad[2] == 0 and m.head.bias.grad[3] == 0


def test_hidden_values_source_and_target_labels_do_not_change_encodings_or_distance():
    m=DenseModel(2,4,"mlp",width=12); b=fixture(); other=copy.deepcopy(b)
    other["value"][b["masked"]]+=1000
    other["source"]+=99; other["positive"]=~other["positive"]
    other["target"]=~other["target"]
    extra=extra_view_masks(2,4,.3,22,1)
    for mask in [None,extra]:
        a=b if mask is None else subset_view(b,mask)
        c=other if mask is None else subset_view(other,mask)
        torch.testing.assert_close(m.encode(a),m.encode(c),rtol=0,atol=0)
        torch.testing.assert_close(m(a)["amount_normalized"],m(c)["amount_normalized"],rtol=0,atol=0)


def test_distance_is_scale_invariant_and_backpropagates_both_views():
    a=torch.tensor([[1.,2.],[2.,1.]],requires_grad=True)
    b=torch.tensor([[3.,1.],[1.,4.]],requires_grad=True)
    d=representation_distance(a,b)
    torch.testing.assert_close(d,representation_distance(2*a,3*b))
    torch.testing.assert_close(d,1-torch.nn.functional.cosine_similarity(a,b))
    d.sum().backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert torch.isfinite(representation_distance(torch.zeros_like(a),torch.zeros_like(b))).all()


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-.1,1.1])
def test_bad_drop_probability_fails(bad):
    with pytest.raises(ValueError): extra_view_masks(2,4,bad,22,1)


def test_nonfinite_and_invalid_masks_or_model_fail():
    b=fixture(); m=DenseModel(2,4,"mlp",width=12)
    with pytest.raises(ValueError): subset_view(b,torch.ones((2,4)))
    with pytest.raises(ValueError): subset_view(b,torch.ones((1,4),dtype=torch.bool))
    with pytest.raises(FloatingPointError): representation_distance(torch.full((2,4),float("nan")),torch.ones((2,4)))
    with pytest.raises(FloatingPointError): weighted_consistency(torch.tensor([1.,float("inf")]),b)
    with pytest.raises(ValueError): two_view_loss(m,b,torch.zeros_like(b["masked"]),float("nan"))
    with pytest.raises(ValueError): two_view_loss(DenseModel(2,4,"mlp",width=12,query_residual=True),b,torch.zeros_like(b["masked"]),.1)

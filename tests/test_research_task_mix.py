import copy
import numpy as np
import pytest
import torch
from foodcomp.research_task_mix import name_only_tasks,remove_numeric_context
from foodcomp.research_neural import DenseModel,batch_from_arrays
from foodcomp.research_r1 import panel_loss


def batch_fixture():
    batch=batch_from_arrays(np.array([[1.,0.,3.],[4.,5.,0.]],np.float32),
        np.array([[True,False,True],[False,True,False]]),np.ones((2,2),np.float32),"cpu")
    batch.update(target=torch.tensor([[False,True,False],[True,False,False]]),
        cell_weight=torch.ones((2,3)),axis_total=torch.ones(3),objective_multiplier=.5)
    return batch


def test_mixture_changes_only_context_without_mutation_or_extra_supervision():
    batch=batch_fixture();before=copy.deepcopy(batch)
    mixed=remove_numeric_context(batch,np.array([True,False]))
    assert mixed["masked"][0].all()
    torch.testing.assert_close(mixed["masked"][1],batch["masked"][1],rtol=0,atol=0)
    for key in batch:
        if torch.is_tensor(batch[key]):torch.testing.assert_close(batch[key],before[key],rtol=0,atol=0)
        if key!="masked":assert mixed[key] is batch[key]
    prediction=torch.ones((2,3),requires_grad=True)
    panel_loss({"amount_normalized":prediction},mixed,objective="mae").backward()
    assert prediction.grad[0,1]!=0  # explicit zero remains supervised
    assert prediction.grad[0,0]==0 and prediction.grad[1,2]==0


def test_probability_zero_preserves_predictions_and_name_only_ignores_all_values():
    model=DenseModel(2,3,"mlp");model.eval();batch=batch_fixture()
    none=remove_numeric_context(batch,np.zeros(2,bool))
    all_hidden=remove_numeric_context(batch,np.ones(2,bool))
    changed=copy.deepcopy(all_hidden);changed["value"]+=999
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["amount_normalized"],model(none)["amount_normalized"],rtol=0,atol=0)
        torch.testing.assert_close(model(all_hidden)["amount_normalized"],model(changed)["amount_normalized"],rtol=0,atol=0)


def test_task_assignments_are_nested_reproducible_and_rng_independent():
    before_torch=torch.get_rng_state();before_numpy=np.random.get_state()
    a=name_only_tasks(10000,.1,20260922,3);b=name_only_tasks(10000,.2,20260922,3)
    assert (a<=b).all() and 900<a.sum()<1100 and 1900<b.sum()<2100
    np.testing.assert_array_equal(a,name_only_tasks(10000,.1,20260922,3))
    assert not np.array_equal(a,name_only_tasks(10000,.1,20260922,4))
    assert not name_only_tasks(100,.0,20260922,3).any()
    assert name_only_tasks(100,1.,20260922,3).all()
    torch.testing.assert_close(torch.get_rng_state(),before_torch,rtol=0,atol=0)
    for before,after in zip(before_numpy,np.random.get_state()):np.testing.assert_array_equal(before,after)
    with pytest.raises(ValueError):name_only_tasks(10,float("nan"),20260922,1)
    with pytest.raises(ValueError):remove_numeric_context(batch_fixture(),np.array([1,0]))

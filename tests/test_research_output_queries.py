from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_neural import predictions_from_outputs,schema_query_axes,evaluate


def test_unrequested_finite_log_value_need_not_fit_raw_dtype_but_requested_must():
    out={"amount_normalized":torch.tensor([[1.,500.,0.]])}
    values,_=predictions_from_outputs(out,np.ones(3),target_axes=[2,0])
    assert values.shape==(1,2) and values[0,0]==0
    assert values[0,1]==torch.expm1(torch.tensor(1.))
    with pytest.raises(FloatingPointError):predictions_from_outputs(out,np.ones(3),target_axes=[1])
    with pytest.raises(FloatingPointError):predictions_from_outputs(out,np.ones(3))


@pytest.mark.parametrize("hurdle",[False,True])
def test_finite_subsets_and_order_match_all_axis_decoding_exactly(hurdle):
    torch.manual_seed(17)
    out={"amount_normalized":torch.randn(17,8)*3}
    if hurdle:out["positive_logit"]=torch.randn(17,8)
    scales=np.logspace(-9,1,8)
    all_values,all_prob=predictions_from_outputs(out,scales)
    chosen,prob=predictions_from_outputs(out,scales,target_axes=[7,1,3])
    torch.testing.assert_close(chosen,all_values[:,[7,1,3]],rtol=0,atol=0)
    if hurdle:torch.testing.assert_close(prob,all_prob[:,[7,1,3]],rtol=0,atol=0)
    reverse,_=predictions_from_outputs(out,scales,target_axes=np.arange(8)[::-1])
    torch.testing.assert_close(reverse,all_values.flip(1),rtol=0,atol=0)


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-float("inf")])
def test_nonfinite_forward_anywhere_still_fails_even_outside_request(bad):
    with pytest.raises(FloatingPointError):
        predictions_from_outputs({"amount_normalized":torch.tensor([[1.,bad]])},[1.,1.],target_axes=[0])
    with pytest.raises(FloatingPointError):
        predictions_from_outputs({"amount_normalized":torch.ones(1,2),"positive_logit":torch.tensor([[0.,bad]])},[1.,1.],target_axes=[0])


@pytest.mark.parametrize("axes",[[],[True],[0.],[0,0],[-1],[3],[[0]]])
def test_invalid_query_sets_fail(axes):
    with pytest.raises(ValueError):predictions_from_outputs({"amount_normalized":torch.ones(2,3)},np.ones(3),target_axes=axes)


def fixture():
    return SimpleNamespace(values=np.zeros((1,3),np.float32),observed=np.array([[True,False,True]]),
        validation=np.array([0]),axes=list(range(3)),targets=np.array([0,1,2]),
        scale=np.ones(3),families=np.array(["a","a","b"]),
        jobs=pd.DataFrame({"profile_index":[0],"axis_index":[0],"mask_family":["a"]}))


class Fixed(torch.nn.Module):
    def __init__(self,bad_axis):super().__init__();self.bad_axis=bad_axis
    def forward(self,batch):
        value=torch.ones_like(batch["value"]);value[:,self.bad_axis]=500.
        return {"amount_normalized":value}


def test_unlabelled_requested_family_axis_is_still_checked():
    data=fixture()
    assert schema_query_axes(data,family="a").tolist()==[0,1]
    # Axis1 has no label and no scoring job, but it remains in caller's request.
    with pytest.raises(FloatingPointError):evaluate(Fixed(1),data,np.ones((1,2),np.float32),"cpu")
    assert len(evaluate(Fixed(2),data,np.ones((1,2),np.float32),"cpu"))==1
    changed=copy.deepcopy(data);changed.observed[:]=False;changed.values[:]=999
    changed.jobs=changed.jobs.iloc[:0]
    np.testing.assert_array_equal(schema_query_axes(changed,family="a"),[0,1])
    np.testing.assert_array_equal(schema_query_axes(changed,mode="name_only"),[0,1,2])


def test_name_only_requests_all_supervised_axes_and_unknown_tasks_fail():
    data=fixture()
    with pytest.raises(FloatingPointError):evaluate(Fixed(2),data,np.ones((1,2),np.float32),"cpu",mode="name_only")
    for kwargs in [{"family":"unknown"},{"mode":"invalid"},{"mode":"name_only","family":"a"}]:
        with pytest.raises(ValueError):schema_query_axes(data,**kwargs)

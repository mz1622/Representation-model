import numpy as np
import pytest
import torch
from foodcomp.research_neural import predictions_from_outputs


@pytest.mark.parametrize("hurdle",[False,True])
def test_all_finite_historical_outputs_and_probability_are_bitwise_unchanged(hurdle):
    generator=torch.Generator().manual_seed(23)
    values=torch.randn(256,32,generator=generator)*7
    scales=torch.logspace(-10,1,32)
    out={"amount_normalized":values}
    old=torch.expm1(values.clamp_min(0))*scales
    old_prob=None
    if hurdle:
        out["positive_logit"]=torch.randn(256,32,generator=generator)*5
        old_prob=torch.sigmoid(out["positive_logit"]);old=old*old_prob
    raw,prob=predictions_from_outputs(out,scales)
    assert raw.dtype==old.dtype and torch.equal(raw.view(torch.int32),old.view(torch.int32))
    if hurdle:assert torch.equal(prob.view(torch.int32),old_prob.view(torch.int32))
    else:assert prob is None


@pytest.mark.parametrize("hurdle",[False,True])
def test_recover_intermediate_overflow_without_changing_finite_neighbours(hurdle):
    values=torch.tensor([[0.,-3.,1.,93.]],dtype=torch.float32)
    scales=torch.tensor([1.,.1,.03,.000039],dtype=torch.float32)
    out={"amount_normalized":values}
    old=torch.expm1(values.clamp_min(0))*scales
    expected=torch.expm1(values.double().clamp_min(0))*scales.double()
    if hurdle:
        out["positive_logit"]=torch.tensor([[1.,2.,3.,-.7]])
        p=torch.sigmoid(out["positive_logit"]);old=old*p;expected=expected*p.double()
    assert torch.isinf(old[0,3]) and expected[0,3]<torch.finfo(torch.float32).max
    raw,_=predictions_from_outputs(out,scales)
    assert torch.isfinite(raw).all() and raw[0,0]==0 and raw[0,1]==0
    assert torch.equal(raw[0,:3].view(torch.int32),old[0,:3].view(torch.int32))
    assert raw[0,3]==expected.float()[0,3]
    assert raw[0,3]>1e30  # No nutritional cap or plausibility substitution.


@pytest.mark.parametrize("value",[100.,1000.,float("nan"),float("inf"),-float("inf")])
def test_true_overflow_and_nonfinite_inputs_still_fail(value):
    with pytest.raises(FloatingPointError):
        predictions_from_outputs({"amount_normalized":torch.tensor([[value]])},np.array([1.]))


def test_nonfinite_probability_and_invalid_scale_fail():
    with pytest.raises(FloatingPointError):
        predictions_from_outputs({"amount_normalized":torch.ones(1,1),"positive_logit":torch.tensor([[float("nan")]])},[1.])
    for scale in [0.,-1.,float("nan"),float("inf")]:
        with pytest.raises(FloatingPointError):
            predictions_from_outputs({"amount_normalized":torch.ones(1,1)},[scale])

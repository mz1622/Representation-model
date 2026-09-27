import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_neural import make_model,batch_from_arrays,macro_loss,predictions_from_outputs
from foodcomp.research_inference import NutritionModel


def tiny_data():
    return SimpleNamespace(axes=list(range(4)),profiles=pd.DataFrame({"source_index":[0,1]}),train=np.array([0,1]))


@pytest.mark.parametrize("kind",["mlp","name_mlp","numeric_mlp","v9","v9_direct","v8_optimized"])
def test_hidden_values_and_target_availability_do_not_change_prediction(kind):
    torch.set_num_threads(1)
    model,_=make_model(tiny_data(),3,kind);model.eval()
    values=np.array([[1.,2.,0.,3.]])
    visible=np.array([[True,False,False,True]])
    a=batch_from_arrays(values,visible,np.ones((1,3)),"cpu")
    b=copy.deepcopy(a);b["value"][0,1:3]=999
    # A formerly unobserved requested axis has the same position as any other query.
    b["target"]=torch.ones((1,4),dtype=torch.bool)
    with torch.no_grad():
        x=model(a);y=model(b)
    torch.testing.assert_close(x["amount_normalized"],y["amount_normalized"],rtol=0,atol=0)
    assert x["amount_normalized"].shape==(1,4)


@pytest.mark.parametrize("kind",["v9","v9_direct"])
def test_v9_source_does_not_enter_encoder(kind):
    model,_=make_model(tiny_data(),3,kind);model.eval()
    a=batch_from_arrays(np.ones((1,4)),np.ones((1,4),bool),np.ones((1,3)),"cpu")
    b=copy.deepcopy(a);b["source"][:]=1
    with torch.no_grad():
        torch.testing.assert_close(model(a)["amount_normalized"],model(b)["amount_normalized"],rtol=0,atol=0)


def test_only_observed_targets_receive_gradients_including_explicit_zero():
    prediction=torch.tensor([[1.,1.,1.,1.]],requires_grad=True)
    batch={"target":torch.tensor([[True,False,False,True]]),"cell_weight":torch.ones((1,4)),
        "value":torch.tensor([[0.,900.,400.,2.]])}
    macro_loss({"amount_normalized":prediction},batch,1).backward()
    assert prediction.grad[0,0]!=0  # explicit zero remains supervised
    assert prediction.grad[0,1:3].abs().sum()==0
    assert prediction.grad[0,3]!=0


def test_nonfinite_loss_or_prediction_fails():
    with pytest.raises(FloatingPointError):predictions_from_outputs({"amount_normalized":torch.tensor([[float("nan")]])},[1])
    b={"target":torch.ones((1,1),dtype=torch.bool),"cell_weight":torch.ones((1,1)),"value":torch.zeros((1,1))}
    with pytest.raises(FloatingPointError):macro_loss({"amount_normalized":torch.tensor([[float("inf")]])},b,1)


@pytest.mark.parametrize("width",[256,512])
def test_checkpoint_reload_predictions_match(tmp_path,width):
    model,_=make_model(tiny_data(),3,"mlp",mlp_width=width);model.eval()
    batch=batch_from_arrays(np.ones((2,4)),np.zeros((2,4),bool),np.ones((2,3)),"cpu")
    torch.save(model.state_dict(),tmp_path/"model.pt")
    loaded,_=make_model(tiny_data(),3,"mlp",mlp_width=width)
    loaded.load_state_dict(torch.load(tmp_path/"model.pt",weights_only=True));loaded.eval()
    with torch.no_grad():torch.testing.assert_close(model(batch)["amount_normalized"],loaded(batch)["amount_normalized"],rtol=0,atol=0)


def test_retrieval_interface_consumes_candidate_predictions_only():
    model=object.__new__(NutritionModel)
    model.data=SimpleNamespace(axes=pd.DataFrame({"loss_group":["nutrition"]*2,"loss_eligible":[True]*2}),scale=np.ones(2))
    model.profile_arrays=lambda profile:(np.log1p([[profile["a"],0]]),np.array([[True,False]]))
    received=[]
    def profiles(names):
        received.extend(names)
        return np.array([[2.,999.],[8.,0.]])
    model.candidate_profiles=profiles
    results=model.retrieve_names({"a":2.},["pear","apple"],top_k=1)
    assert received==["apple","pear"]
    assert results[0]["name"]=="apple"
    assert results[0]["score"]==pytest.approx(0)


def test_direct_v9_matched_initialization_rng_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17);hurdle,_=make_model(tiny_data(),3,"v9")
    torch.manual_seed(17);direct,_=make_model(tiny_data(),3,"v9_direct")
    for key,value in hurdle.state_dict().items():
        torch.testing.assert_close(value,direct.state_dict()[key],rtol=0,atol=0)
    batch=batch_from_arrays(np.ones((2,4)),np.zeros((2,4),bool),np.ones((2,3)),"cpu")
    hurdle.train();direct.train()
    torch.manual_seed(41);a=hurdle(batch);rng_a=torch.get_rng_state()
    torch.manual_seed(41);b=direct(batch);rng_b=torch.get_rng_state()
    torch.testing.assert_close(a["amount_normalized"],b["amount_normalized"],rtol=0,atol=0)
    torch.testing.assert_close(rng_a,rng_b,rtol=0,atol=0)
    assert "positive_logit" not in b
    assert all(not p.requires_grad for p in direct.presence_head.parameters())
    assert not direct.source_presence_residual.weight.requires_grad
    torch.save(direct.state_dict(),tmp_path/"direct.pt")
    loaded,_=make_model(tiny_data(),3,"v9_direct")
    loaded.load_state_dict(torch.load(tmp_path/"direct.pt",weights_only=True));loaded.eval();direct.eval()
    with torch.no_grad():
        torch.testing.assert_close(direct(batch)["amount_normalized"],loaded(batch)["amount_normalized"],rtol=0,atol=0)


def test_direct_v9_supervises_explicit_zero_and_omits_missing():
    from foodcomp.research_r1 import model_loss
    model,config=make_model(tiny_data(),3,"v9_direct")
    batch=batch_from_arrays(np.array([[0.,0.,1.,0.]]),np.zeros((1,4),bool),np.ones((1,3)),"cpu")
    batch.update(target=torch.tensor([[True,False,True,False]]),positive=torch.tensor([[False,False,True,False]]),
                 cell_weight=torch.ones((1,4)),axis_total=torch.ones(4),objective_multiplier=.5)
    model.eval();outputs=model(batch);outputs["amount_normalized"].retain_grad()
    from foodcomp.research_r1 import panel_loss
    panel_loss(outputs,batch,objective="mae").backward()
    assert outputs["amount_normalized"].grad[0,0]!=0
    assert outputs["amount_normalized"].grad[0,1]==0
    assert outputs["amount_normalized"].grad[0,3]==0
    model.zero_grad();model_loss(model,batch,"v9_direct",config,objective="smooth_l1").backward()
    assert model.source_amount_residual.weight.grad is not None
    assert model.source_presence_residual.weight.grad is None
    assert all(p.grad is None for p in model.presence_head.parameters())

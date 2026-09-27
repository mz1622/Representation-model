import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from foodcomp.research_neural import make_model,batch_from_arrays


def fixture(axes=4):
    return SimpleNamespace(axes=list(range(axes)),profiles=pd.DataFrame({"source_index":[0,1]}),train=np.array([0,1]))


def test_cloned_heads_preserve_common_initialization_outputs_and_rng():
    torch.set_num_threads(1)
    torch.manual_seed(23);shared,_=make_model(fixture(252),32,"mlp",mlp_width=512);rng=torch.get_rng_state()
    torch.manual_seed(23);separate,_=make_model(fixture(252),32,"mlp",mlp_width=512,mlp_task_heads="separate")
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    for key,value in shared.state_dict().items():torch.testing.assert_close(value,separate.state_dict()[key],rtol=0,atol=0)
    for key,value in shared.head.state_dict().items():torch.testing.assert_close(value,separate.name_head.state_dict()[key],rtol=0,atol=0)
    assert sum(p.numel() for p in shared.parameters())==667900
    assert sum(p.numel() for p in separate.parameters())==797176
    visible=np.zeros((2,252),bool);visible[0,0]=True
    batch=batch_from_arrays(np.zeros((2,252)),visible,np.ones((2,32)),"cpu")
    with torch.no_grad():torch.testing.assert_close(shared(batch)["amount_normalized"],separate(batch)["amount_normalized"],rtol=0,atol=0)


def test_route_depends_on_visible_input_including_zero_not_hidden_labels():
    model,_=make_model(fixture(),3,"mlp",mlp_task_heads="separate");model.eval()
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.fill_(2)
        model.name_head.weight.zero_();model.name_head.bias.fill_(5)
    batch=batch_from_arrays(np.zeros((2,4)),np.array([[True,False,False,False],[False]*4]),np.ones((2,3)),"cpu")
    changed=copy.deepcopy(batch);changed["value"][batch["masked"]]=999;changed["target"]=~batch["masked"];changed["source"][:]=1
    with torch.no_grad():
        a=model(batch)["amount_normalized"];b=model(changed)["amount_normalized"]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a,torch.tensor([[2.]*4,[5.]*4]))


def test_only_routed_head_receives_gradient_and_reload_matches(tmp_path):
    model,_=make_model(fixture(),3,"mlp",mlp_task_heads="separate")
    for visible,selected,unused in [(np.ones((2,4),bool),model.head,model.name_head),(np.zeros((2,4),bool),model.name_head,model.head)]:
        model.zero_grad(set_to_none=True)
        batch=batch_from_arrays(np.ones((2,4)),visible,np.ones((2,3)),"cpu")
        model(batch)["amount_normalized"].sum().backward()
        assert selected.bias.grad.abs().sum()>0
        assert all(p.grad is None or not p.grad.any() for p in unused.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    torch.save(model.state_dict(),tmp_path/"model.pt")
    loaded,_=make_model(fixture(),3,"mlp",mlp_task_heads="separate")
    loaded.load_state_dict(torch.load(tmp_path/"model.pt",weights_only=True))
    with torch.no_grad():torch.testing.assert_close(model(batch)["amount_normalized"],loaded(batch)["amount_normalized"],rtol=0,atol=0)

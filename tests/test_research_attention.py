import pytest
import torch
from foodcomp.research_attention import attention_category_mass,replay_v9_attention


def test_uniform_attention_matches_key_fraction_and_weighted_queries():
    attention=torch.full((2,2,5,5),.2)
    categories={"visible":torch.tensor([[True,False,False,False,False],[True,True,True,False,False]]),
                "masked":torch.tensor([[False,True,True,True,True],[False,False,False,True,True]])}
    query=torch.tensor([[0.,1.,0.,0.,0.],[0.,3.,0.,0.,0.]])
    result=attention_category_mass(attention,query,categories)
    torch.testing.assert_close(result["visible"],torch.tensor([.5,.5],dtype=torch.float64))
    torch.testing.assert_close(result["masked"],torch.tensor([.5,.5],dtype=torch.float64))


def test_invalid_partition_and_nonfinite_attention_fail():
    attention=torch.full((1,2,3,3),1/3);query=torch.ones((1,3))
    with pytest.raises(ValueError,match="partition"):
        attention_category_mass(attention,query,{"a":torch.ones((1,3),dtype=torch.bool),"b":torch.ones((1,3),dtype=torch.bool)})
    attention[0,0,0,0]=float("nan")
    with pytest.raises(FloatingPointError):attention_category_mass(attention,query,{"a":torch.ones((1,3),dtype=torch.bool)})
    attention.fill_(1/3);attention[0,0,0]=torch.tensor([-.1,.5,.6])
    with pytest.raises(FloatingPointError):attention_category_mass(attention,query,{"a":torch.ones((1,3),dtype=torch.bool)})


def test_replay_preserves_predictions_rng_and_rejects_hooked_or_training_paths():
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from foodcomp.research_neural import make_model,batch_from_arrays
    torch.set_num_threads(1)
    data=SimpleNamespace(axes=list(range(4)),profiles=pd.DataFrame({"source_index":[0,1]}),train=np.array([0,1]))
    model,_=make_model(data,3,"v9")
    batch=batch_from_arrays(np.ones((2,4)),np.array([[True,False,True,False],[False]*4]),np.ones((2,3)),"cpu")
    with pytest.raises(ValueError,match="eval mode"):replay_v9_attention(model,batch,lambda *args:None)
    model.eval();seen=[];rng=torch.get_rng_state()
    with torch.no_grad():original=model(batch)
    replay=replay_v9_attention(model,batch,lambda layer,weights:seen.append((layer,weights.shape)))
    assert [layer for layer,_ in seen]==list(range(len(model.encoder.layers)))
    for _,shape in seen:assert shape==(2,model.encoder.layers[0].self_attn.num_heads,6,6)
    for key in original:torch.testing.assert_close(original[key],replay[key],rtol=0,atol=0)
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    handle=model.encoder.layers[0].self_attn.register_forward_pre_hook(lambda *args:None)
    try:
        with pytest.raises(ValueError,match="no forward hooks"):replay_v9_attention(model,batch,lambda *args:None)
    finally:handle.remove()

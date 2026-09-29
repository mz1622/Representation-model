"""Capacity-group configuration and hidden-input invariants, without outcome selection."""
from copy import deepcopy
from dataclasses import asdict
import json
import pytest
import torch

from foodcomp.research_transformer_r9 import make_transformer,transformer_loss
from foodcomp.research_transformer_r9_capacity256 import load_method
from test_research_transformer_r9 import example


def test_only_declared_capacity_configuration_changes():
    data,spec,batch=example()
    torch.manual_seed(22);parent,p_config=make_transformer(data,128,spec)
    torch.manual_seed(22);model,config=make_transformer(data,128,dict(spec,d_model=256,n_heads=8,feedforward_dim=1024))
    assert {key for key in asdict(config) if asdict(config)[key]!=asdict(p_config)[key]}=={'d_model','n_heads','feedforward_dim'}
    assert sum(p.numel() for p in model.parameters())>sum(p.numel() for p in parent.parameters())
    assert config.source_calibrated_loss_weight==1. and config.dropout==.15


def test_capacity_forward_excludes_hidden_labels_and_source():
    data,spec,batch=example()
    model,config=make_transformer(data,128,dict(spec,d_model=256,n_heads=8,feedforward_dim=1024))
    model.eval();changed=deepcopy(batch)
    changed['value'][batch['masked']]=987654
    changed['source'][:]=0;changed['target']=~changed['target'];changed['positive']=~changed['positive']
    torch.testing.assert_close(model(batch)['amount_normalized'],model(changed)['amount_normalized'],rtol=0,atol=0)
    loss=transformer_loss(model,batch,config,'mae');loss.backward()
    assert torch.isfinite(loss) and model.source_amount_residual.weight.grad.count_nonzero()>0


@pytest.mark.parametrize('change',[{'d_model':256},{'d_model':256,'n_heads':8,'feedforward_dim':1024,'source_weight':0.},
    {'d_model':256,'n_heads':8,'feedforward_dim':1024,'objective':'mse'},
    {'d_model':256,'n_heads':8,'feedforward_dim':1024,'dropout':.05}])
def test_registration_rejects_undeclared_or_partial_capacity_group(tmp_path,change):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'status':'registered','purpose':'r9_capacity_group_256_8_1024_contrast',
        'changes':change,'candidate':'tf256_mae_lr3e4_60','candidate_index':6,'max_candidates':12,'seed':20260922}),encoding='utf-8')
    with pytest.raises(ValueError,match='PDF capacity-group'):
        load_method(tmp_path,path,{},tmp_path/'unused_freeze')

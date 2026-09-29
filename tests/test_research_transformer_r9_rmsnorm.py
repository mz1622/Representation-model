from copy import deepcopy
import pytest
import torch
from torch import nn
from test_research_transformer_r9 import example
from foodcomp.research_transformer_r9 import make_transformer as control, transformer_loss
from foodcomp.research_transformer_r9_rmsnorm import (
    RMSNorm, ARCHITECTURE, make_transformer, replace_encoder_norms, norm_sites)
from foodcomp.research_transformer_r9_drop25 import dropout_sites


def pair():
    data, spec, batch = example()
    spec['architecture'] = ARCHITECTURE
    torch.manual_seed(20260922)
    old, conf = control(data, 128, spec)
    rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    new, _ = make_transformer(data, 128, spec)
    assert torch.equal(rng, torch.get_rng_state())
    return old, new, conf, batch


def test_rms_formula_and_gradient():
    norm = RMSNorm(nn.LayerNorm(4, eps=1e-5)).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    expected = x / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    torch.testing.assert_close(norm(x), expected, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(norm, (x,))
    assert not hasattr(norm, 'bias')


def test_only_seven_norm_biases_removed_and_no_other_initialization_or_dropout_change():
    old, new, _, _ = pair()
    removed = set(old.state_dict()) - set(new.state_dict())
    expected = {f'encoder.layers.{i}.norm{j}.bias' for i in range(3) for j in [1, 2]}
    expected.add('encoder.norm.bias')
    assert removed == expected
    for name, value in new.state_dict().items():
        assert torch.equal(value, old.state_dict()[name]), name
    assert sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in new.parameters()) == 1344
    assert dropout_sites(old) == dropout_sites(new)
    sites = norm_sites(new)
    assert sum(v['type'] == 'RMSNorm' for v in sites.values()) == 7
    assert sites['text_projection.0']['type'] == 'LayerNorm'


@pytest.mark.parametrize('training', [False, True])
def test_explicit_layernorm_bridge_preserves_function_and_gradients(training):
    old, _, conf, batch = pair()
    bridge = replace_encoder_norms(deepcopy(old), rms=False)
    old.train(training); bridge.train(training)
    torch.manual_seed(13)
    a = transformer_loss(old, batch, conf, 'mae'); a.backward()
    torch.manual_seed(13)
    b = transformer_loss(bridge, batch, conf, 'mae'); b.backward()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for (_, x), (_, y) in zip(old.named_parameters(), bridge.named_parameters()):
        if x.grad is None:
            assert y.grad is None
        else:
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-7)


def test_hidden_targets_and_source_never_change_base_prediction():
    _, new, _, batch = pair()
    new.eval()
    changed = deepcopy(batch)
    changed['value'][batch['masked']] = 1e5
    changed['source'][:] = 0
    changed['target'] = ~changed['target']
    with torch.no_grad():
        assert torch.equal(new(batch)['amount_normalized'], new(changed)['amount_normalized'])


def test_forward_changed_by_rmsnorm_and_state_reload_exact():
    old, new, conf, batch = pair()
    old.eval(); new.eval()
    with torch.no_grad():
        result = new(batch)['amount_normalized']
        assert not torch.equal(result, old(batch)['amount_normalized'])
        restored = deepcopy(new)
        restored.load_state_dict(new.state_dict())
        assert torch.equal(result, restored(batch)['amount_normalized'])
    new.train()
    loss = transformer_loss(new, batch, conf, 'mae')
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)


def test_wrong_architecture_is_rejected():
    data, spec, _ = example()
    with pytest.raises(ValueError):
        make_transformer(data, 128, spec)

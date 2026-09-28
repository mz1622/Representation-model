"""Architecture intervention: identity at zero, visible-only effect, and gradients."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import io
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_neural import batch_from_arrays
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_transformer_r9 import make_transformer, transformer_loss
from foodcomp.research_transformer_r9_axisvalue import make_axisvalue_transformer, base_state_fingerprint


def example():
    data = SimpleNamespace(axes=range(4), train=np.array([0, 1]),
        profiles=pd.DataFrame({'source_index': [1, 2, 0]}))
    spec = dict(d_model=24, n_layers=1, n_heads=3, feedforward_dim=48, dropout=.15,
        axis_residual_rank=4, source_weight=1., source_residual_l2=1e-4,
        objective='mae', axis_value_residual='zero_init_linear_v1')
    batch = batch_from_arrays(np.array([[0., 2., 3., 0.], [1., 0., 0., 4.]]),
        np.array([[False, True, True, False], [True, False, False, True]]),
        np.zeros((2, 8)), 'cpu', source=np.array([1, 2]),
        targets=np.array([[True, False, False, False], [False, True, False, False]]),
        weights=np.ones((2, 4)), positive=np.array([[False, True, True, False], [True, False, False, True]]))
    batch['axis_total'] = torch.ones(4)
    batch['objective_multiplier'] = .5
    return data, spec, batch


@pytest.mark.parametrize('training', [False, True])
def test_zero_residual_preserves_parent_weights_rng_forward_loss_and_base_gradients(training):
    data, spec, batch = example()
    torch.manual_seed(20260922)
    old, config = make_transformer(data, 8, spec)
    old_rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    new, new_config = make_axisvalue_transformer(data, 8, spec)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert asdict(config) == asdict(new_config)
    assert base_state_fingerprint(new) == state_fingerprint(old)
    assert not new.axis_value_residual.detach().count_nonzero()
    old.train(training); new.train(training)
    torch.manual_seed(111)
    old_loss = transformer_loss(old, batch, config, 'mae')
    old_loss.backward()
    after = torch.get_rng_state()
    torch.manual_seed(111)
    new_loss = transformer_loss(new, batch, config, 'mae')
    new_loss.backward()
    assert torch.equal(after, torch.get_rng_state())
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    for name, param in old.named_parameters():
        actual = dict(new.named_parameters())[name]
        if param.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(param.grad, actual.grad, rtol=0, atol=0)
    assert new.axis_value_residual.grad.abs().sum() > 0


def test_nonzero_residual_changes_visible_prediction_but_not_hidden_or_source_inputs():
    data, spec, batch = example()
    model, _ = make_axisvalue_transformer(data, 8, spec)
    model.eval()
    empty = deepcopy(batch)
    empty['masked'][:] = True
    with torch.no_grad():
        original = model(batch)['amount_normalized']
        original_empty = model(empty)['amount_normalized']
        model.axis_value_residual.copy_(torch.randn_like(model.axis_value_residual))
        changed = model(batch)['amount_normalized']
        assert not torch.equal(changed, original)
        torch.testing.assert_close(model(empty)['amount_normalized'], original_empty, rtol=0, atol=0)
        hidden = deepcopy(batch)
        hidden['value'][hidden['masked']] = 1e6
        hidden['target'] = ~hidden['target']
        hidden['positive'] = ~hidden['positive']
        hidden['source'][:] = 0
        torch.testing.assert_close(model(hidden)['amount_normalized'], changed, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    transformer_loss(model, empty, model_config(spec), 'mae').backward()
    assert not model.axis_value_residual.grad.count_nonzero()


def model_config(spec):
    return SimpleNamespace(source_calibrated_loss_weight=spec['source_weight'],
                           source_residual_l2=spec['source_residual_l2'])


def test_encode_uses_forward_tokens_and_state_reload_preserves_nonzero_residual():
    data, spec, batch = example()
    model, _ = make_axisvalue_transformer(data, 8, spec)
    with torch.no_grad():
        model.axis_value_residual.normal_()
    model.eval()
    seen = []
    hook = model.encoder.register_forward_hook(lambda module, inputs, output: seen.append(output.detach().clone()))
    with torch.no_grad():
        expected = model(batch)['amount_normalized']
        encoded = model.encode(batch)
    hook.remove()
    torch.testing.assert_close(encoded, seen[0][:, 2:].mean(1), rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    loaded, _ = make_axisvalue_transformer(data, 8, spec)
    loaded.load_state_dict(torch.load(stream, weights_only=True))
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(batch)['amount_normalized'], expected, rtol=0, atol=0)
        torch.testing.assert_close(loaded.encode(batch), encoded, rtol=0, atol=0)


def test_unsupported_method_is_rejected():
    data, spec, _ = example()
    spec['axis_value_residual'] = 'unregistered'
    with pytest.raises(ValueError):
        make_axisvalue_transformer(data, 8, spec)

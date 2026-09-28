from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_neural import make_model, batch_from_arrays, predictions_from_outputs
from foodcomp.research_r1 import model_loss
from foodcomp.research_transformer_r9 import make_transformer, transformer_loss


def example():
    data = SimpleNamespace(axes=range(4), train=np.array([0, 1]),
        profiles=pd.DataFrame({'source_index': [1, 2, 0]}))
    spec = {'d_model': 192, 'n_layers': 3, 'n_heads': 6, 'feedforward_dim': 768,
        'dropout': .15, 'axis_residual_rank': 16, 'source_weight': 1., 'source_residual_l2': 1e-4,
        'objective': 'mae'}
    batch = batch_from_arrays(np.array([[0., 2., 3., 0.], [1., 0., 0., 4.]]),
        np.array([[False, True, True, False], [True, False, False, True]]),
        np.zeros((2, 128)), 'cpu', source=np.array([1, 2]),
        targets=np.array([[True, False, False, False], [False, True, False, False]]),
        weights=np.ones((2, 4)), positive=np.array([[False, True, True, False], [True, False, False, True]]))
    batch['axis_total'] = torch.ones(4)
    batch['objective_multiplier'] = .5
    return data, spec, batch


def test_r9_control_matches_existing_transformer_initialization_forward_and_loss():
    data, spec, batch = example()
    torch.manual_seed(22)
    old, old_config = make_model(data, 128, 'v9_direct')
    torch.manual_seed(22)
    new, new_config = make_transformer(data, 128, spec)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    old.eval(); new.eval()
    torch.testing.assert_close(old(batch)['amount_normalized'], new(batch)['amount_normalized'], rtol=0, atol=0)
    torch.testing.assert_close(model_loss(old, batch, 'v9_direct', old_config, 'mae'),
        transformer_loss(new, batch, new_config, 'mae'), rtol=0, atol=0)


def test_hidden_labels_and_source_are_not_encoder_inputs():
    data, spec, batch = example()
    model, _ = make_transformer(data, 128, spec)
    model.eval()
    changed = deepcopy(batch)
    changed['value'][batch['masked']] = 90000
    changed['target'] = ~batch['target']
    changed['positive'] = ~batch['positive']
    changed['source'][:] = 0
    torch.testing.assert_close(model(batch)['amount_normalized'], model(changed)['amount_normalized'], rtol=0, atol=0)


@pytest.mark.parametrize('objective', ['mae', 'mse', 'smooth_l1'])
def test_zero_has_gradient_and_missing_label_has_none(objective):
    _, _, batch = example()
    output = torch.nn.Parameter(torch.ones((2, 4)))
    class Fixed:
        def __call__(self, batch):
            return {'amount_normalized': output}
        def calibrated_outputs(self, base, batch):
            return base
        def source_residual_penalty(self):
            return torch.zeros(())
    config = SimpleNamespace(source_calibrated_loss_weight=1., source_residual_l2=1e-4)
    loss = transformer_loss(Fixed(), batch, config, objective)
    loss.backward()
    assert (output.grad[batch['target']] > 0).all()
    assert (output.grad[~batch['target']] == 0).all()


def test_nonfinite_predictions_fail_before_metric_reduction():
    with pytest.raises(FloatingPointError):
        predictions_from_outputs({'amount_normalized': torch.tensor([[float('nan')]])}, np.ones(1), target_axes=np.array([0]))

"""Independent loss arithmetic/gradient checks for the registered MSE intervention."""
from types import SimpleNamespace

import pytest
import torch

from foodcomp.research_transformer_r9 import transformer_loss


@pytest.mark.parametrize('source_weight', [0., 1., 3.])
def test_weighted_masked_mse_and_calibration_gradient(source_weight):
    prediction = torch.nn.Parameter(torch.tensor([[1., 4.], [3., 2.]], dtype=torch.float64))
    residual = torch.nn.Parameter(torch.tensor([[.5, -.5]], dtype=torch.float64))
    batch = {'value': torch.tensor([[0., 999.], [1., 0.]], dtype=torch.float64),
             'target': torch.tensor([[True, False], [True, True]]),
             'cell_weight': torch.tensor([[.25, 10.], [.75, .5]], dtype=torch.float64),
             'axis_total': torch.tensor([2., 4.], dtype=torch.float64), 'objective_multiplier': 3.}
    class Model:
        def __call__(self, batch):
            return {'amount_normalized': prediction}
        def calibrated_outputs(self, base, batch):
            return {'amount_normalized': prediction + residual}
        def source_residual_penalty(self):
            return residual.square().mean()
    cfg = SimpleNamespace(source_calibrated_loss_weight=source_weight, source_residual_l2=.01)
    actual = transformer_loss(Model(), batch, cfg, 'mse')
    expected = prediction.new_zeros(())
    gradient = torch.zeros_like(prediction)
    for i, a in [(0, 0), (1, 0), (1, 1)]:
        factor = batch['cell_weight'][i, a] / batch['axis_total'][a] * 3.
        error = prediction[i, a] - batch['value'][i, a]
        calibrated_error = error + residual[0, a]
        expected = expected + factor * (error.square() + source_weight * calibrated_error.square()) / (1 + source_weight)
        gradient[i, a] = factor * 2 * (error + source_weight * calibrated_error) / (1 + source_weight)
    expected = expected + .01 * residual.square().mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    actual.backward()
    torch.testing.assert_close(prediction.grad, gradient, rtol=0, atol=1e-12)
    assert prediction.grad[0, 1] == 0
    assert prediction.grad[0, 0] != 0 and prediction.grad[1, 1] != 0

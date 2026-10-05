"""Tests for single-student masked-axis distillation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v12 import (  # noqa: E402
    macro_axis_mae, mean_teacher_prediction,
)


class ConstantTeacher(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, axis, value, padding, target_axis, target_padding):
        return torch.full(target_axis.shape, self.value).masked_fill(
            target_padding, 0.0
        )


def test_teacher_predictions_are_averaged_in_training_space():
    packed = (
        torch.tensor([[0]]), torch.tensor([[0.2]]), torch.tensor([[False]]),
        torch.tensor([[1, 2]]), torch.tensor([[False, True]]),
    )
    result = mean_teacher_prediction(
        [ConstantTeacher(1.0), ConstantTeacher(3.0)], packed
    )
    torch.testing.assert_close(result, torch.tensor([[2.0, 0.0]]))


def test_macro_axis_mae_gives_axes_equal_epoch_weight():
    prediction = torch.tensor([[2.0, 2.0], [2.0, 0.0]])
    target = torch.zeros_like(prediction)
    axis = torch.tensor([[0, 1], [0, 0]])
    padding = torch.tensor([[False, False], [False, True]])
    # Axis 0 occurs twice, axis 1 once. Both have MAE 2 and equal macro weight.
    result = macro_axis_mae(
        prediction, target, axis, padding, torch.tensor([2.0, 1.0]),
        task_count=2, active_axis_count=2,
    )
    torch.testing.assert_close(result, torch.tensor(2.0))

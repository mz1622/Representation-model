"""Tests for learned target-token layer fusion."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v22 import (  # noqa: E402
    TargetLayerFusionTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_initial_layer_fusion_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(43)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(43)
    candidate = TargetLayerFusionTransformer(4, config).eval()
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_layer_fusion_can_read_an_earlier_block():
    config = Config(
        d_model=8, n_heads=2, n_layers=2,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(47)
    model = TargetLayerFusionTransformer(4, config).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.target_layer_fusion.weight[:, :8] += 0.5 * torch.eye(8)
        changed = model(*_inputs())
    assert not torch.allclose(baseline, changed)

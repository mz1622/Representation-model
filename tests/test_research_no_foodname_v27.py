"""Tests for depth-shared axis-pair attention bias."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v27 import (  # noqa: E402
    SharedAxisPairBiasTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_shared_bias_starts_as_exact_parent_function():
    config = Config(
        d_model=8, n_heads=2, n_layers=3,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(79)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(79)
    candidate = SharedAxisPairBiasTransformer(4, config).eval()
    assert candidate.blocks[0].attention.pair_bias is candidate.blocks[1].attention.pair_bias
    assert candidate.blocks[1].attention.pair_bias is candidate.blocks[2].attention.pair_bias
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_shared_bias_update_is_visible_in_every_block():
    model = SharedAxisPairBiasTransformer(
        4,
        Config(d_model=8, n_heads=2, n_layers=3,
               feedforward_dim=16, dropout=0),
    )
    with torch.no_grad():
        model.blocks[2].attention.pair_bias[0, 0] = 3.0
    assert model.blocks[0].attention.pair_bias[0, 0].item() == 3.0

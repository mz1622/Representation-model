"""Tests for low-rank axis-pair attention bias."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config  # noqa: E402
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v21 import (  # noqa: E402
    LowRankAxisPairBiasTransformer,
)


def _inputs():
    return (
        torch.tensor([[0, 1]]),
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[False, False]]),
        torch.tensor([[2, 3]]),
        torch.tensor([[False, False]]),
    )


def test_zero_factorized_bias_exactly_matches_parent():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(37)
    parent = AxisPairBiasMaskedAxisTransformer(4, config).eval()
    torch.manual_seed(37)
    candidate = LowRankAxisPairBiasTransformer(4, config, rank=2).eval()
    assert torch.count_nonzero(
        candidate.blocks[0].attention.materialized_bias()
    ) == 0
    with torch.no_grad():
        torch.testing.assert_close(parent(*_inputs()), candidate(*_inputs()))


def test_factorized_bias_changes_attention_after_key_update():
    config = Config(
        d_model=8, n_heads=2, n_layers=1,
        feedforward_dim=16, dropout=0,
    )
    torch.manual_seed(41)
    model = LowRankAxisPairBiasTransformer(4, config, rank=2).eval()
    with torch.no_grad():
        baseline = model(*_inputs())
        model.blocks[0].attention.key_factor[0].fill_(5.0)
        changed = model(*_inputs())
    assert not torch.allclose(baseline, changed)

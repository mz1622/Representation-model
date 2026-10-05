"""Critical invariants for the no-name architecture ablations."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcomp.research_no_foodname_v1 import Config
from foodcomp.research_no_foodname_v2 import VariantSetTransformer


def test_variants_ignore_padding_and_handle_empty_context():
    torch.manual_seed(7)
    axis = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    value = torch.tensor([[0.0, 0.3, 0.0], [0.0, 0.0, 0.0]])
    padding = torch.tensor([[False, False, True], [True, True, True]])
    altered_axis = axis.clone()
    altered_value = value.clone()
    altered_axis[padding] = 4
    altered_value[padding] = 1e3
    knots = np.tile(np.linspace(0, 2, 9, dtype=np.float32), (5, 1))
    for variant in ("target_query", "piecewise_numeric", "mean_pool", "hurdle"):
        model = VariantSetTransformer(5, Config(), variant,
                                      knots if variant == "piecewise_numeric" else None).eval()
        with torch.no_grad():
            first = model(axis, value, padding)
            second = model(altered_axis, altered_value, padding)
            permuted = model(axis[:, [1, 0, 2]], value[:, [1, 0, 2]],
                             padding[:, [1, 0, 2]])
        assert first.shape == (2, 5)
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(first, permuted, rtol=1e-5, atol=1e-5)


def test_hurdle_output_is_nonnegative_and_probability_sensitive():
    model = VariantSetTransformer(5, Config(), "hurdle").eval()
    axis = torch.zeros((1, 1), dtype=torch.long)
    value = torch.zeros((1, 1))
    padding = torch.ones((1, 1), dtype=torch.bool)
    with torch.no_grad():
        predicted, logits, amount = model.forward_details(axis, value, padding)
    assert (predicted >= 0).all()
    assert (amount > 0).all()
    torch.testing.assert_close(predicted, torch.sigmoid(logits) * amount)

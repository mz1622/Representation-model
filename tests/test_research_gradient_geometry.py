import pytest
import torch
from foodcomp.research_gradient_geometry import gradient_geometry


def test_opposed_gradients_and_explicit_zero_denominators():
    r=gradient_geometry([torch.tensor([1.,0.])],[torch.tensor([-2.,0.])])
    assert r["cosine"]==pytest.approx(-1.)
    assert r["name_to_completion_norm"]==pytest.approx(2.)
    r=gradient_geometry([torch.zeros(2)],[torch.ones(2)])
    assert not r["both_nonzero"] and r["cosine"] is None and r["name_to_completion_norm"] is None
    with pytest.raises(FloatingPointError):gradient_geometry([torch.ones(2)],[torch.tensor([float("nan"),1.])])
    with pytest.raises(FloatingPointError):gradient_geometry([torch.tensor([1e200],dtype=torch.float64)],[torch.ones(1,dtype=torch.float64)])
    with pytest.raises(ValueError):gradient_geometry([torch.ones(2)],[torch.ones(3)])

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from foodcomp.research_auxiliary import metabolome_axis_scale
from foodcomp.research_r1 import panel_loss


def data():
    return SimpleNamespace(axes=pd.DataFrame({'axis_index':range(252),'loss_group':['nutrition']*142+['food_metabolome']*45+['context']*65}),targets=np.arange(187))


def test_fixed_coefficients_keep_nutrition_and_context_grid():
    d=data();assert metabolome_axis_scale(d,1,'cpu') is None
    scale=metabolome_axis_scale(d,.5,'cpu')
    assert torch.equal(scale[:142],torch.ones(142))
    assert torch.equal(scale[142:187],torch.full((45,),.5))
    assert torch.equal(scale[187:],torch.ones(65))
    d.axes=d.axes.iloc[::-1]
    with pytest.raises(ValueError):metabolome_axis_scale(d,.5,'cpu')


@pytest.mark.parametrize('weight',[0,-1,1.1,float('nan'),float('inf')])
def test_invalid_auxiliary_coefficient(weight):
    with pytest.raises(ValueError):metabolome_axis_scale(data(),weight,'cpu')


def test_weighted_loss_and_gradients_match_explicit_partition_without_renormalizing():
    y=torch.tensor([[0.,2.,8.],[4.,0.,1.]],dtype=torch.float64)
    p=torch.tensor([[1.,5.,100.],[2.,3.,100.]],dtype=torch.float64,requires_grad=True)
    b={'value':y,'target':torch.tensor([[True,True,False],[True,True,False]]),
       'cell_weight':torch.tensor([[.5,.25,0.],[.5,.75,0.]],dtype=torch.float64),
       'axis_total':torch.ones(3,dtype=torch.float64),'objective_multiplier':1/187}
    actual=panel_loss({'amount_normalized':p},b,objective='mae',axis_loss_scale=torch.tensor([1.,.5,1.],dtype=torch.float64))
    expected=(.5*(p[0,0]-0).abs()+.5*(p[1,0]-4).abs()+.5*(.25*(p[0,1]-2).abs()+.75*(p[1,1]-0).abs()))/187
    # The independent scalar formula divides last; the implementation multiplies
    # by the precomputed reciprocal. Mathematical equivalence, not bit identity.
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-16)
    ga=torch.autograd.grad(actual,p,retain_graph=True)[0];ge=torch.autograd.grad(expected,p)[0]
    torch.testing.assert_close(ga,ge,rtol=1e-14,atol=1e-16)
    assert ga[0,0]!=0 and ga[1,1]!=0 and torch.equal(ga[:,2],torch.zeros(2,dtype=torch.float64))


def test_loss_rejects_bad_scale_and_nonfinite_prediction():
    b={'value':torch.zeros(2,2),'target':torch.ones(2,2,dtype=torch.bool),'cell_weight':torch.ones(2,2),'axis_total':torch.ones(2),'objective_multiplier':1.}
    for s in [torch.ones(1),torch.tensor([1.,0.]),torch.tensor([1.,float('nan')])]:
        with pytest.raises(ValueError):panel_loss({'amount_normalized':torch.ones(2,2)},b,objective='mae',axis_loss_scale=s)
    with pytest.raises(FloatingPointError):panel_loss({'amount_normalized':torch.full((2,2),float('nan'))},b,objective='mae',axis_loss_scale=torch.ones(2))

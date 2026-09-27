from types import SimpleNamespace
import numpy as np
import torch
from foodcomp.research_r1 import panel_arrays,panel_loss

def test_every_observed_training_target_once_and_whole_family_hidden():
    data=SimpleNamespace(families=np.array(["a","a","b","b"]),targets=np.array([0,1,2]),axes=range(4),train=np.array([0,1]),
        observed=np.array([[True,False,True,True],[True,True,False,False],[True,True,True,True]]))
    rows,families,_,masks=panel_arrays(data)
    covered=np.zeros_like(data.observed,int)
    eligible=np.array([True,True,True,False])
    for row,family in zip(rows,families):
        target=data.observed[row]&masks[family]&eligible
        covered[row]+=target
        assert not ((data.observed[row]&~masks[family])&target).any()
    expected=np.zeros_like(covered);expected[data.train]=data.observed[data.train]&eligible
    np.testing.assert_array_equal(covered,expected)

def test_panel_loss_minibatches_equal_full_axis_weighted_objective():
    values=torch.tensor([[0.,2.],[4.,0.],[0.,1.]])
    target=torch.tensor([[True,True],[True,False],[False,True]])
    weight=torch.tensor([[.5,1.],[.5,0.],[0.,.5]])
    predictions=torch.tensor([[1.,4.],[2.,500.],[500.,2.]],requires_grad=True)
    def compute(rows):
        batch={"value":values[rows],"target":target[rows],"cell_weight":weight[rows],
            "axis_total":torch.tensor([1.,1.5]),"objective_multiplier":3/(len(rows)*2)}
        return panel_loss({"amount_normalized":predictions[rows]},batch,objective="mae")
    whole=compute([0,1,2]);split=(compute([0])+2*compute([1,2]))/3
    torch.testing.assert_close(whole,split)
    torch.testing.assert_close(whole,torch.tensor((1.5+2.5/1.5)/2))
    whole.backward()
    assert predictions.grad[0,0]!=0  # explicit zero supervised
    assert predictions.grad[1,1]==0 and predictions.grad[2,0]==0

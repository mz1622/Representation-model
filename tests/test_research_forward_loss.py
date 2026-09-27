import pytest
import torch
from foodcomp.research_forward_loss import observed_axis_mae


def fixture():
    prediction=torch.tensor([[0.,1.,5.],[0.,8.,4.]],requires_grad=True)
    batch={'target':torch.tensor([[True,True,False],[True,False,False]]),
        'value':torch.tensor([[2.,0.,float('nan')],[1.,float('nan'),float('inf')]]),
        'cell_weight':torch.tensor([[1.,2.,0.],[3.,0.,0.]])}
    return prediction,batch


def test_axis_weighting_hand_value_and_gradient_includes_zero():
    p,b=fixture();loss=observed_axis_mae(p,b)
    assert loss.item()==1.125;loss.backward()
    torch.testing.assert_close(p.grad,torch.tensor([[-.125,.5,0.],[-.375,0.,0.]]),rtol=0,atol=0)


def test_unobserved_labels_do_not_change_loss_or_gradient():
    p,b=fixture();before=observed_axis_mae(p,b);g=torch.autograd.grad(before,p)[0]
    b['value'][~b['target']]=-123456.
    after=observed_axis_mae(p,b);torch.testing.assert_close(after,before,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(after,p)[0],g,rtol=0,atol=0)


def test_empty_targets_rejected():
    p,b=fixture();b['target'].fill_(False)
    with pytest.raises(ValueError):observed_axis_mae(p,b)


@pytest.mark.parametrize('which',['hidden_prediction','observed_target','weight'])
def test_nonfinite_required_values_fail(which):
    p,b=fixture()
    if which=='hidden_prediction':p=p.detach();p[0,2]=float('nan')
    elif which=='observed_target':b['value'][0,0]=float('nan')
    else:b['cell_weight'][0,0]=float('nan')
    with pytest.raises((ValueError,FloatingPointError)):observed_axis_mae(p,b)

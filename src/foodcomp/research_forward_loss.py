"""Pointwise-loss ablation preserving the original forward baseline's batch axis weights."""
import torch
from .research_neural import loss as legacy_loss


def forward_loss(model,batch,config,objective='smooth_l1'):
    if objective=='smooth_l1':return legacy_loss(model,batch,'name_mlp',config)
    if objective!='mae':raise ValueError('Unknown forward objective.')
    return observed_axis_mae(model(batch)['amount_normalized'],batch)


def observed_axis_mae(prediction,batch):
    active=batch['target'];target=batch['value'];weights=batch['cell_weight']
    if prediction.ndim!=2 or active.dtype!=torch.bool or any(x.shape!=prediction.shape for x in [active,target,weights]):
        raise ValueError('Expected matching profile-axis matrices and boolean target mask.')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[active]).all():raise FloatingPointError('Nonfinite prediction or observed target.')
    if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('Finite nonnegative cell weights required.')
    weights=weights*active;total=weights.sum(0)
    if not (total>0).any():raise ValueError('No supervised targets.')
    safe_target=torch.where(active,target,0.)
    error=(prediction-safe_target).abs()
    result=((error*weights).sum(0)/total.clamp_min(1e-12))[total>0].mean()
    if not torch.isfinite(result):raise FloatingPointError('Nonfinite loss.')
    return result

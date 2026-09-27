"""Rank name-predicted nutrition vectors using only visible query coordinates."""
import torch
from .research_alignment import exact_ranks


def predicted_profile_ranks(candidate,query,visible,correct):
    if candidate.ndim!=2 or query.ndim!=2 or visible.shape!=query.shape or query.shape[1]!=candidate.shape[1]:raise ValueError('Mismatched retrieval matrices.')
    if visible.dtype!=torch.bool or not visible.any(1).all():raise ValueError('Each query needs observed nutrition, including explicit zero.')
    if not torch.isfinite(candidate).all() or not torch.isfinite(query[visible]).all():raise FloatingPointError('Nonfinite candidate or observed query.')
    if (candidate<0).any() or (query[visible]<0).any():raise ValueError('Nonnegative scaled-log nutrition required.')
    mask=visible.to(query.dtype);y=torch.where(visible,query,0.)
    distance=(mask@candidate.square().T-2*y@candidate.T+y.square().sum(1,keepdim=True))/mask.sum(1,keepdim=True)
    return exact_ranks(distance,correct)

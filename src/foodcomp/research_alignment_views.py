"""Independent, value-blind training views for the R5 numeric retrieval branch."""
import numpy as np
from .research_alignment import numeric_features


def partial_view_masks(visible,probability,seed,epoch,*,keep_probability=.3,minimum_visible=3):
    visible=np.asarray(visible)
    if visible.ndim!=2 or visible.dtype!=np.bool_:raise ValueError('Boolean profile-by-axis observedness required.')
    if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Assignment probability must be in [0,1].')
    if not np.isfinite(keep_probability) or not 0<keep_probability<=1:raise ValueError('Keep probability must be in (0,1].')
    if not isinstance(minimum_visible,int) or minimum_visible<1 or (visible.sum(1)<minimum_visible).any():raise ValueError('Every original profile must meet the minimum visibility.')
    if seed<0 or epoch<0:raise ValueError('Nonnegative seed and epoch required.')
    if probability==0:return visible.copy(),np.zeros(len(visible),dtype=bool)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,5105]))
    assigned=rng.random(len(visible))<probability
    kept=visible & (rng.random(visible.shape)<keep_probability)
    count=kept.sum(1)
    rows=np.flatnonzero(assigned & (count<minimum_visible))
    if len(rows):
        remaining=visible[rows] & ~kept[rows]
        priority=rng.random(remaining.shape);priority[~remaining]=np.inf
        order=np.argsort(priority,axis=1,kind='stable')
        needed=minimum_visible-count[rows]
        for j in range(minimum_visible):
            use=needed>j
            kept[rows[use],order[use,j]]=True
    result=np.where(assigned[:,None],kept,visible)
    if (result & ~visible).any() or (result.sum(1)<minimum_visible).any():raise AssertionError('Invalid generated subset view.')
    return result,assigned


def partial_view_features(base_features,probability,seed,epoch):
    base_features=np.asarray(base_features)
    if base_features.ndim!=2 or base_features.shape[1]!=284:raise ValueError('Expected142 values and142 masks.')
    bits=base_features[:,142:]
    if not np.isin(bits,[0.,1.]).all():raise ValueError('Nonbinary observedness.')
    visible=bits.astype(bool)
    result,assigned=partial_view_masks(visible,probability,seed,epoch)
    return numeric_features(base_features[:,:142],result),result,assigned

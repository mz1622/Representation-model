"""Paired uncertainty uses entire food candidate groups as resampling units."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def paired_interval(baseline, candidate, axes, metric="scaled_log_mae", repeats=1000, seed=20260922):
    keys=["exact_name_group_id","axis_index"]
    a=baseline[baseline.axis_index.isin(axes)][keys+[metric]]
    b=candidate[candidate.axis_index.isin(axes)][keys+[metric]]
    joined=a.merge(b,on=keys,suffixes=("_base","_candidate"),how="outer",validate="one_to_one",indicator=True)
    if not joined._merge.eq("both").all():raise ValueError("Paired bootstrap requires identical candidate/axis cells.")
    values=joined[[metric+"_base",metric+"_candidate"]].to_numpy()
    if not np.isfinite(values).all():raise ValueError("Nonfinite bootstrap observations.")
    groups,group_ids=np.unique(joined.exact_name_group_id,return_inverse=True)
    _,axis_ids=np.unique(joined.axis_index,return_inverse=True)
    if len(np.unique(axis_ids))!=len(axes):raise ValueError("Missing axis support.")
    shape=(len(groups),len(axes))
    support=coo_matrix((np.ones(len(joined)),(group_ids,axis_ids)),shape=shape).tocsr()
    matrices=[coo_matrix((values[:,i],(group_ids,axis_ids)),shape=shape).tocsr() for i in range(2)]
    rng=np.random.default_rng(seed);scores=[];missing=0
    for start in range(0,repeats,32):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(32,repeats-start))
        den=weights@support
        good=(den>0).all(1)
        missing+=int((~good).sum())
        if not good.any():continue
        scores.extend(np.stack([(weights[good]@matrix/den[good]).mean(1) for matrix in matrices],axis=1))
    scores=np.array(scores)
    point=joined.groupby("axis_index")[[metric+"_base",metric+"_candidate"]].mean().mean().to_numpy()
    if len(scores)<repeats*.95:raise ValueError("Too many resamples lack rare-axis support; report axes separately.")
    relative=1-scores[:,1]/scores[:,0]
    return {"metric":metric,"group_count":len(groups),"repeats":repeats,"seed":seed,
        "valid_resamples":len(scores),"resamples_without_all_axis_support":missing,
        "baseline":float(point[0]),"candidate":float(point[1]),"relative_improvement":float(1-point[1]/point[0]),
        "relative_improvement_95_interval":np.quantile(relative,[.025,.975]).tolist(),
        "candidate_minus_baseline_95_interval":np.quantile(scores[:,1]-scores[:,0],[.025,.975]).tolist(),
        "scope":"validation group sampling uncertainty conditional on fitted models; excludes training seed variation and label validity"}

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


def paired_axis_intervals(baseline, candidate, repeats=1000, seed=20260922):
    keys=["exact_name_group_id","axis_index"]
    joined=baseline[keys+["scaled_log_mae"]].merge(candidate[keys+["scaled_log_mae"]],on=keys,
        suffixes=("_base","_candidate"),how="outer",validate="one_to_one",indicator=True)
    if not joined._merge.eq("both").all():raise ValueError("Different per-axis bootstrap panels.")
    values=joined[["scaled_log_mae_base","scaled_log_mae_candidate"]].to_numpy()
    if not np.isfinite(values).all():raise ValueError("Nonfinite observed errors.")
    groups,gi=np.unique(joined.exact_name_group_id,return_inverse=True)
    axes,ai=np.unique(joined.axis_index,return_inverse=True)
    shape=(len(groups),len(axes))
    support=coo_matrix((np.ones(len(joined)),(gi,ai)),shape=shape).tocsr()
    difference=coo_matrix((values[:,1]-values[:,0],(gi,ai)),shape=shape).tocsr()
    rng=np.random.default_rng(seed);samples=[];valid=[]
    for start in range(0,repeats,32):
        w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(32,repeats-start))
        den=w@support;num=w@difference
        samples.append(np.divide(num,den,out=np.zeros_like(num),where=den>0));valid.append(den>0)
    samples=np.concatenate(samples);valid=np.concatenate(valid)
    counts=np.asarray(support.sum(0)).ravel()
    point=np.asarray(difference.sum(0)).ravel()/counts
    rows=[]
    for i,axis in enumerate(axes):
        kept=samples[valid[:,i],i]
        lo,hi=np.quantile(kept,[.025,.975])
        rows.append({"axis_index":int(axis),"candidate_support":int(counts[i]),"candidate_minus_baseline":float(point[i]),
            "difference_95_low":float(lo),"difference_95_high":float(hi),"valid_resamples":len(kept),
            "resamples_without_axis_support":int(repeats-len(kept)),"sparse_support_below_30":bool(counts[i]<30)})
    return pd.DataFrame(rows)


def paired_group_rates(baseline, candidate, metrics, repeats=1000, seed=20260922):
    """Absolute paired differences for source-equal food-group retrieval rates.

    Relative changes are deliberately omitted because a sparse Recall@1 baseline
    or bootstrap draw can be zero. Higher values are better for these metrics.
    """
    key="exact_name_group_id"
    joined=baseline[[key]+list(metrics)].merge(candidate[[key]+list(metrics)],on=key,
        suffixes=("_base","_candidate"),how="outer",validate="one_to_one",indicator=True)
    if not len(joined) or not joined._merge.eq("both").all():raise ValueError("Retrieval groups differ or are empty.")
    values=[joined[[m+suffix for m in metrics]].to_numpy(float) for suffix in ["_base","_candidate"]]
    if not all(np.isfinite(v).all() and (v>=0).all() and (v<=1).all() for v in values):
        raise ValueError("Retrieval rates must be finite in [0,1].")
    differences=values[1]-values[0];rng=np.random.default_rng(seed);samples=[];n=len(joined)
    for start in range(0,repeats,32):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(32,repeats-start))
        samples.append(weights@differences/n)
    interval=np.quantile(np.concatenate(samples),[.025,.975],axis=0)
    return {metric:{"baseline":float(values[0][:,i].mean()),"candidate":float(values[1][:,i].mean()),
        "candidate_minus_baseline":float(differences[:,i].mean()),
        "difference_95_interval":interval[:,i].tolist(),"group_count":n,"repeats":repeats,"seed":seed,
        "higher_is_better":True,"scope":"paired food-group uncertainty conditional on fixed trained models, candidate names and exact-name relevance; excludes seeds, aliases and selection uncertainty"}
        for i,metric in enumerate(metrics)}

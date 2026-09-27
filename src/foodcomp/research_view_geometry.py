"""Descriptive, finite-only statistics for a fixed training representation probe."""
import numpy as np


def geometry_summary(first, second, task_weights, task_count, axis_count):
    a=np.asarray(first,dtype=np.float64);b=np.asarray(second,dtype=np.float64)
    q=np.asarray(task_weights,dtype=np.float64)
    if a.ndim!=2 or a.shape!=b.shape or min(a.shape)<1 or q.shape!=(len(a),):
        raise ValueError("Matching nonempty representations and one task weight per row required.")
    if not all(np.isfinite(x).all() for x in [a,b,q]):
        raise FloatingPointError("Nonfinite probe representation or weight.")
    if (q<0).any() or q.sum()<=0 or task_count<len(a) or axis_count<1:
        raise ValueError("Invalid task weights or full-panel dimensions.")
    norms=[np.linalg.norm(x,axis=1) for x in [a,b]]
    unit=[x/np.maximum(n[:,None],1e-12) for x,n in zip([a,b],norms)]
    distance=.5*np.sum((unit[0]-unit[1])**2,axis=1)
    result={"tasks":len(a),"representation_dim":a.shape[1],
        "uniform_task_view_distance":float(distance.mean()),
        "axis_weighted_mean_view_distance":float(np.sum(q*distance)/q.sum()),
        "full_panel_consistency_estimate":float(np.sum(q*distance)*task_count/(len(a)*axis_count)),
        "full_panel_weight_mass_estimate":float(q.sum()*task_count/(len(a)*axis_count))}
    for label,h,n in zip(["A","B"],unit,norms):
        centre=h.mean(0)
        result[label]={"mean_representation_norm":float(n.mean()),
            "zero_norm_fraction":float((n<=1e-12).mean()),
            "mean_unit_feature_population_std":float(h.std(0,ddof=0).mean()),
            "unit_mean_squared_distance_from_mean":float(np.sum((h-centre)**2,axis=1).mean()),
            "unit_mean_vector_norm":float(np.linalg.norm(centre))}
    return result

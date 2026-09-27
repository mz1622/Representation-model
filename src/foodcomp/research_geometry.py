"""Label-free name-neighborhood diagnostics, with explicit identity exclusion."""
import numpy as np


def nearest_other_rows(features,queries,k=50,batch_size=64):
    features=np.asarray(features,dtype=np.float64)
    queries=np.asarray(queries,dtype=np.int64)
    if features.ndim!=2 or not np.isfinite(features).all():raise ValueError("Finite 2D name features required.")
    if queries.ndim!=1 or (queries<0).any() or (queries>=len(features)).any():raise ValueError("Invalid query indices.")
    if not 0<k<len(features) or batch_size<1:raise ValueError("Invalid neighbor/batch sizes.")
    norms=np.einsum("ij,ij->i",features,features)
    neighbors=[];distances=[]
    for start in range(0,len(queries),batch_size):
        q=queries[start:start+batch_size]
        square=norms[q,None]+norms[None,:]-2*features[q]@features.T
        if not np.isfinite(square).all() or square.min() < -1e-8:raise FloatingPointError("Invalid squared name distances.")
        square=np.maximum(square,0.)
        square[np.arange(len(q)),q]=np.inf
        # Stable candidate-index tie rule also applies at the top-k boundary.
        order=np.argsort(square,axis=1,kind="stable")[:,:k]
        neighbors.append(order);distances.append(np.take_along_axis(square,order,axis=1))
    return np.concatenate(neighbors),np.concatenate(distances)


def neighbor_overlap(reference,candidate,k):
    reference=np.asarray(reference);candidate=np.asarray(candidate)
    if reference.shape!=candidate.shape or reference.ndim!=2 or not 0<k<=reference.shape[1]:
        raise ValueError("Neighbor arrays or top-k differ.")
    return np.array([len(set(a[:k]).intersection(b[:k]))/k for a,b in zip(reference,candidate)])

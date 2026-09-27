"""Original per-axis name-neighbor prediction with only train labels and query text."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ObservedAxisNeighbors:
    def __init__(self,train_text,observed_values,cell_weights,scale,*,n_jobs=8,backend='auto'):
        if backend not in {'auto','kd_tree'}:raise ValueError('Unregistered neighbor backend.')
        x=np.asarray(train_text);y=np.asarray(observed_values);w=np.asarray(cell_weights)
        if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or not len(x):raise ValueError('Aligned observed training rows required.')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any():raise ValueError('Finite nonnegative observed labels and finite text required.')
        if not np.isfinite(w).all() or not (w>0).all() or not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite weights/scale required.')
        self.values=y;self.weights=w;self.scale=scale;self.width=x.shape[1]
        self.neighbors=NearestNeighbors(n_neighbors=min(10,len(x)),n_jobs=n_jobs,algorithm=backend).fit(x)

    def predict(self,query_text,*,return_neighbors=False):
        x=np.asarray(query_text)
        if x.ndim!=2 or x.shape[1]!=self.width or not len(x) or not np.isfinite(x).all():raise ValueError('Finite query name features required.')
        distances,indexes=self.neighbors.kneighbors(x)
        weights=1/(distances+1e-3)*self.weights[indexes]
        predicted=np.sum(self.values[indexes]*weights,axis=1)/weights.sum(axis=1)
        result=self.scale*np.expm1(predicted)
        if not np.isfinite(result).all() or (result<0).any():raise FloatingPointError('Invalid neighbor prediction.')
        return (result,indexes,distances) if return_neighbors else result

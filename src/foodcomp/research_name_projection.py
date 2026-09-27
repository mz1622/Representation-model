"""Frozen train-only exact PCA with explicit active dimensions and padded slots."""
import json
from pathlib import Path
import numpy as np
from .research_r0 import digest


def fit_projection(raw,train_rows,slots=128):
    x=np.asarray(raw)[np.asarray(train_rows)]
    if x.ndim!=2 or len(x)<2 or not 0<slots<=min(x.shape) or not np.isfinite(x).all():raise ValueError('Invalid training names or PCA dimensions.')
    x=x.astype(np.float64);mean=x.mean(0);centred=x-mean
    covariance=centred.T@centred/(len(x)-1)
    values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    if not np.isfinite(values).all() or values.min()<-1e-10:raise FloatingPointError('Invalid covariance eigenvalues.')
    components=vectors[:,:slots].T.copy()
    pivots=np.argmax(np.abs(components),axis=1);sign=np.sign(components[np.arange(slots),pivots]);components*=sign[:,None]
    np.testing.assert_allclose(components@components.T,np.eye(slots),rtol=0,atol=1e-12)
    return {'mean':mean,'components':components,'eigenvalues':values}


def project_names(raw,projection):
    x=np.asarray(raw,dtype=np.float64);mean=np.asarray(projection['mean']);components=np.asarray(projection['components'])
    active=int(projection['active_components'])
    if x.ndim!=2 or mean.shape!=(x.shape[1],) or components.ndim!=2 or components.shape[1]!=x.shape[1] or not 0<active<=len(components):raise ValueError('Invalid name projection shape.')
    if not all(np.isfinite(v).all() for v in [x,mean,components]):raise FloatingPointError('Nonfinite name projection.')
    # Per-row/per-direction reduction order is independent of row count and selected prefix.
    scores=np.einsum('ij,kj->ik',x-mean,components,optimize=False).astype(np.float32)
    scores[:,active:]=0
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite projected names.')
    return scores


def load_variant(cache,*,data_hash=None,active_components=None):
    cache=Path(cache);m=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if m.get('projection_kind')!='exact_train_pca_padded_v1' or m.get('text_field')!='original_name' or m.get('fit_partition')!='train':raise ValueError('Unregistered name cache.')
    if data_hash is not None and m['data_sha256']!=data_hash:raise ValueError('Stale name data fingerprint.')
    if active_components is not None and m['active_components']!=active_components:raise ValueError('Wrong active dimensions.')
    for name,expected in m['hashes'].items():
        if digest(cache/name)!=expected:raise ValueError(f'Changed name cache: {name}')
    features=np.load(cache/'features.npy',allow_pickle=False)
    with np.load(cache/'pca.npz',allow_pickle=False) as f:projection=dict(f)
    if features.shape!=(m['rows'],m['pca_components']) or features.dtype!=np.float32 or not np.isfinite(features).all():raise ValueError('Invalid name features.')
    if np.any(features[:,m['active_components']:]) or int(projection['active_components'])!=m['active_components']:raise ValueError('Inactive dimensions are nonzero or stale.')
    return features,projection,m

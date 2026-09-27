"""Deterministic post-fit text interventions; no labels, sources or fitting inputs."""
import numpy as np


def intervene_name_features(features, names, mode, seed=20260922):
    x = np.asarray(features)
    names = np.asarray(names, dtype=str)
    if x.ndim != 2 or x.shape != (len(names), 128) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Expected finite128-slot float32 name features.')
    unique, first, inverse = np.unique(names, return_index=True, return_inverse=True)
    if not np.array_equal(x, x[first][inverse]):
        raise ValueError('Identical raw names must have identical input vectors.')
    changed = x.copy(); replacements = None
    if mode == 'zero_name':
        changed.fill(0)
    elif mode == 'drop_extra96':
        changed[:, 32:] = 0
    elif mode == 'permute_name':
        if len(unique) < 2:
            raise ValueError('A name permutation requires at least two unique names.')
        order = np.random.default_rng(seed).permutation(len(unique))
        permutation = np.empty(len(unique), dtype=np.int64)
        permutation[order] = np.roll(order, 1)
        assert not np.any(permutation == np.arange(len(unique)))
        changed = x[first][permutation[inverse]].copy()
        replacements = unique[permutation[inverse]]
    else:
        raise ValueError('Unknown name intervention.')
    return changed, replacements

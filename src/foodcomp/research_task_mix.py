"""Deterministic task-level numeric-context removal; labels and weights stay fixed."""
import numpy as np
import torch


def name_only_tasks(count, probability, seed, epoch):
    if count < 1 or epoch < 1 or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid task mixture parameters.")
    if probability == 0:return np.zeros(count, dtype=bool)
    # Independent stream; same uniforms across probabilities, keyed by task ID.
    rng=np.random.default_rng(np.random.SeedSequence([seed, epoch, 3103]))
    return rng.random(count) < probability


def remove_numeric_context(batch, selected):
    selected=torch.as_tensor(selected, device=batch["masked"].device)
    if selected.dtype != torch.bool or selected.shape != (len(batch["masked"]),):
        raise ValueError("Expected one Boolean name-only flag per training task.")
    changed=dict(batch)
    changed["masked"]=batch["masked"] | selected[:, None]
    return changed

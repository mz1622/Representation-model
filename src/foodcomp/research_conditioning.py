"""Training-only statistics for an optional internal name-input conditioning probe."""
import hashlib
import numpy as np
import torch
from torch import nn


class FrozenNameStandardizer(nn.Module):
    """Model-internal affine transform, fitted once and serialized with the model."""
    def __init__(self, dimensions):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("std", torch.ones(dimensions))
        self.register_buffer("fitted", torch.tensor(False))

    def validate(self):
        if not bool(self.fitted):
            raise ValueError("Name conditioning statistics have not been fitted or loaded.")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 1e-8).any():
            raise ValueError("Invalid saved name conditioning statistics.")

    @torch.no_grad()
    def set_statistics(self, mean, std):
        if bool(self.fitted):
            raise ValueError("Name conditioning statistics are already frozen.")
        mean = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device)
        std = torch.as_tensor(std, dtype=self.std.dtype, device=self.std.device)
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError("Conditioning statistic shape mismatch.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 1e-8).any():
            raise ValueError("Invalid name conditioning statistics.")
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.fitted.fill_(True)

    def forward(self, text):
        if not bool(self.fitted):
            raise ValueError("Name conditioning statistics have not been fitted or loaded.")
        return (text-self.mean)/self.std


def fit_training_name_statistics(data, text):
    """Weight each distinct original training string once; never inspect held-out vectors."""
    rows = np.asarray(data.train)
    if rows.ndim != 1 or len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("Nonempty unique training rows required.")
    if not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= len(data.profiles)).any():
        raise ValueError("Invalid training row indices.")
    if text.ndim != 2 or len(text) != len(data.profiles):
        raise ValueError("Name feature/profile shape mismatch.")
    profiles = data.profiles.iloc[rows]
    if not profiles.partition.eq("train").all():
        raise ValueError("Held-out profile in conditioning fit.")
    names = profiles.original_name.to_numpy()
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Original training names must be nonempty strings.")
    _, first, inverse = np.unique(names, return_index=True, return_inverse=True)
    values = np.asarray(text[rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite training name features.")
    representatives = values[first]
    if not np.array_equal(values, representatives[inverse]):
        raise ValueError("Same original name has inconsistent cached vectors.")
    mean = representatives.mean(axis=0)
    std = representatives.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 1e-8).any():
        raise ValueError("Degenerate training name coordinate; no silent scale clipping.")
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    return mean, std, {
        "fit_partition": "train", "weighting": "one per exact original_name string",
        "train_profiles": len(rows), "train_unique_names": len(first), "dimensions": text.shape[1],
        "representative_rows_sha256": hashlib.sha256(rows[first].astype("<i8").tobytes()).hexdigest(),
        "statistics_sha256": hashlib.sha256(mean.astype("<f4").tobytes() + std.astype("<f4").tobytes()).hexdigest(),
        "population_std_ddof": 0, "nutrition_labels_used": False, "held_out_vectors_used": False,
    }

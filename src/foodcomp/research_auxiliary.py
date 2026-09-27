"""Training-only auxiliary-axis coefficient; never changes inference or scoring."""
import numpy as np
import torch


def metabolome_axis_scale(data, weight, device):
    if not np.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("Metabolome weight must be finite and in (0,1].")
    if weight == 1:
        return None  # Keep the original reduction/gradient path exactly.
    if not np.array_equal(data.axes.axis_index.to_numpy(),np.arange(len(data.axes))):
        raise ValueError("Noncanonical axis order.")
    eligible=np.zeros(len(data.axes),dtype=bool);eligible[data.targets]=True
    nutrition=data.axes.loss_group.eq("nutrition").to_numpy() & eligible
    metabolome=data.axes.loss_group.eq("food_metabolome").to_numpy() & eligible
    if nutrition.sum()!=142 or metabolome.sum()!=45 or eligible.sum()!=187:
        raise ValueError("Auxiliary-weight registration requires142 nutrition/45 metabolome axes.")
    scale=torch.ones(len(data.axes),dtype=torch.float32,device=device)
    scale[torch.as_tensor(metabolome,device=device)]=weight
    return scale

"""Source-free hurdle selection on the frozen complete validation family panel.

This retains the historical BCE + positive-only SmoothL1 objective, but uses
the registered R0 labels/scales and a full-panel axis macro, not the historical
mean of minibatch macros. Both checkpoint selectors see the same trajectory.
"""
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .research_neural import batch_from_arrays, predictions_from_outputs
from .research_r0 import cell_weights


@torch.no_grad()
def evaluate_dual_selection(model, data, text, device, *, amount_weight=1., batch_size=128):
    if batch_size < 1 or not np.isfinite(amount_weight) or amount_weight < 0:
        raise ValueError("Invalid selection evaluator parameters.")
    jobs = data.jobs.reset_index(drop=True).copy()
    keys = ["profile_index", "axis_index"]
    if jobs.duplicated(keys).any() or not len(jobs):
        raise ValueError("Validation jobs must be nonempty and unique.")
    if not jobs.profile_index.isin(data.validation).all():
        raise ValueError("Selection must use validation profiles only.")
    if not jobs.axis_index.isin(data.targets).all():
        raise ValueError("Selection includes an ineligible target axis.")
    rows, axes = jobs.profile_index.to_numpy(), jobs.axis_index.to_numpy()
    if not data.observed[rows, axes].all():
        raise ValueError("Missing cells cannot enter validation loss.")
    if not np.array_equal(jobs.mask_family.to_numpy(), data.families[axes]):
        raise ValueError("Frozen family/query mismatch.")
    metadata = data.profiles.iloc[rows][["profile_id", "source_key", "exact_name_group_id"]].reset_index(drop=True)
    metadata["axis_index"] = axes
    jobs["selection_weight"] = cell_weights(metadata)
    totals = np.bincount(axes, weights=jobs.selection_weight, minlength=len(data.axes))
    if (totals[data.targets] <= 0).any():
        raise ValueError("Every registered loss axis needs validation support.")
    amount_sum = np.zeros(len(data.axes), dtype=np.float64)
    presence_sum = np.zeros_like(amount_sum)
    pieces = []
    model.eval()
    for family, family_jobs in jobs.groupby("mask_family"):
        family_rows = np.sort(family_jobs.profile_index.unique())
        for start in range(0, len(family_rows), batch_size):
            current = family_rows[start:start + batch_size]
            current_jobs = family_jobs[family_jobs.profile_index.isin(current)]
            visible = data.observed[current].copy()
            visible[:, data.families == family] = False
            batch = batch_from_arrays(data.values[current], visible, text[current], device)
            outputs = model(batch)
            if "positive_logit" not in outputs:
                raise ValueError("Hurdle selection requires a presence head.")
            raw, probability = predictions_from_outputs(outputs, data.scale)
            p = np.searchsorted(current, current_jobs.profile_index)
            a = current_jobs.axis_index.to_numpy()
            pt = torch.as_tensor(p, device=device)
            at = torch.as_tensor(a, device=device)
            y = torch.as_tensor(data.values[current_jobs.profile_index, a], device=device, dtype=torch.float64)
            positive = (y > 0).double()
            amount = F.smooth_l1_loss(outputs["amount_normalized"][pt, at].double(), y, reduction="none") * positive
            presence = F.binary_cross_entropy_with_logits(outputs["positive_logit"][pt, at].double(), positive, reduction="none")
            if not torch.isfinite(amount).all() or not torch.isfinite(presence).all():
                raise FloatingPointError("Nonfinite validation hurdle component.")
            w = current_jobs.selection_weight.to_numpy()
            amount_sum += np.bincount(a, weights=amount.cpu().numpy()*w, minlength=len(data.axes))
            presence_sum += np.bincount(a, weights=presence.cpu().numpy()*w, minlength=len(data.axes))
            frame = current_jobs[keys].copy()
            frame["prediction"] = raw[pt, at].cpu().numpy()
            frame["positive_probability"] = probability[pt, at].cpu().numpy()
            pieces.append(frame)
    targets = data.targets
    presence = float((presence_sum[targets]/totals[targets]).mean())
    amount = float((amount_sum[targets]/totals[targets]).mean())
    result = {"source_free_hurdle": presence + amount_weight*amount,
              "presence_bce": presence, "positive_amount_smooth_l1": amount,
              "amount_weight": amount_weight, "axis_count": len(targets), "jobs": len(jobs)}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite selection aggregate.")
    return pd.concat(pieces, ignore_index=True), result

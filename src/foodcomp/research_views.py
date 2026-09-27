"""R4 matched two-view representation loss on the unchanged supervised panel."""
import numpy as np
import torch
from torch.nn import functional as F
from .research_r1 import panel_loss


def extra_view_masks(task_count, axis_count, probability, seed, epoch):
    if any(not isinstance(v, int) or v < 1 for v in (task_count, axis_count)):
        raise ValueError("Positive integer task and axis counts required.")
    if any(not isinstance(v, int) or v < 0 for v in (seed, epoch)):
        raise ValueError("Nonnegative integer seed and epoch required.")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Finite drop probability in [0,1] required.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 4104]))
    return rng.random((task_count, axis_count), dtype=np.float32) < probability


def subset_view(batch, extra_hidden):
    extra_hidden = torch.as_tensor(extra_hidden, device=batch["masked"].device)
    if extra_hidden.dtype != torch.bool or extra_hidden.shape != batch["masked"].shape:
        raise ValueError("Additional mask must be boolean with the original context shape.")
    return {**batch, "masked": batch["masked"] | extra_hidden}


def representation_distance(first, second, centering="none"):
    if first.ndim != 2 or first.shape != second.shape or min(first.shape) == 0:
        raise ValueError("Matching nonempty batch-by-feature representations required.")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Nonfinite representation.")
    if centering not in {"none", "joint_batch"}:
        raise ValueError("Unknown representation centering.")
    if centering == "joint_batch":
        centre = (first.mean(0) + second.mean(0)) / 2
        first = first - centre
        second = second - centre
    a = F.normalize(first, dim=1, eps=1e-12)
    b = F.normalize(second, dim=1, eps=1e-12)
    return .5 * (a-b).square().sum(1)


def weighted_consistency(distance, batch):
    if distance.ndim != 1 or len(distance) != len(batch["target"]):
        raise ValueError("One distance per supervised family task required.")
    weights = batch["cell_weight"] * batch["target"] / batch["axis_total"].clamp_min(1e-12)
    if not torch.isfinite(distance).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite consistency distance or weights.")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Nonnegative weights with observed supervision required.")
    value = (weights.sum(1) * distance).sum() * batch["objective_multiplier"]
    if not torch.isfinite(value):
        raise FloatingPointError("Nonfinite weighted consistency.")
    return value


def two_view_loss(model, batch, extra_hidden, coefficient, centering="none"):
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Finite nonnegative consistency coefficient required.")
    if model.kind != "mlp" or any(hasattr(model, name) for name in ("name_head", "query_residual", "name_standardizer")):
        raise ValueError("Registered two-view experiment requires the original shared-head fused MLP.")
    second = subset_view(batch, extra_hidden)
    first_hidden = model.encode(batch)
    second_hidden = model.encode(second)
    supervised = panel_loss({"amount_normalized": model.head(first_hidden)}, batch, objective="mae")
    consistency = weighted_consistency(representation_distance(first_hidden, second_hidden, centering), batch)
    # Keep both graphs even in the zero-weight control; do not skip B or detach
    # its representation. The supervised coefficient stays exactly one.
    value = supervised + coefficient * consistency
    if not torch.isfinite(value):
        raise FloatingPointError("Nonfinite two-view loss.")
    unit = F.normalize(first_hidden.detach(), dim=1, eps=1e-12)
    centred = first_hidden.detach() - (first_hidden.detach().mean(0) + second_hidden.detach().mean(0)) / 2
    return value, {
        "supervised": supervised.detach(),
        "consistency": consistency.detach(),
        "mean_representation_norm": first_hidden.detach().norm(dim=1).mean(),
        "unit_batch_std": unit.std(dim=0, unbiased=False).mean(),
        "centred_representation_norm": centred.norm(dim=1).mean(),
        "centred_unit_batch_std": F.normalize(centred, dim=1, eps=1e-12).std(dim=0, unbiased=False).mean(),
        "visible_cells": (~batch["masked"]).sum(),
        "removed_visible_cells": ((~batch["masked"]) & second["masked"]).sum(),
    }

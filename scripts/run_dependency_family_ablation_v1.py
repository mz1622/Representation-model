#!/usr/bin/env python3
"""Expose and perturb normally hidden dependency relatives at validation time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    checkpoint_metadata, dependency_visible, load_dependency_model, predict_normalized,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.infoods_component_groups_v1 import component_group  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261028
JOBS_PER_TARGET = 250
FACTORS = (0.9, 1.1, 10.0)


def weak_component(policy, start):
    found, pending = {int(start)}, [int(start)]
    while pending:
        axis = pending.pop()
        for other in policy.parents[axis] | policy.children[axis]:
            if other not in found:
                found.add(other)
                pending.append(other)
    return found


def component_label(policy, axes):
    roots = sorted(axis for axis in axes if not (policy.parents[axis] & axes))
    return " + ".join(policy.name_by_axis[axis] for axis in roots)


def macro_summary(jobs, prediction, base_prediction, condition):
    frame = jobs[["profile_index", "target_axis", "component"]].copy()
    truth = jobs.truth_normalized.to_numpy(dtype=float)
    frame["absolute_error"] = np.abs(prediction - truth)
    frame["prediction_shift_vs_masked"] = np.abs(prediction - base_prediction)
    per_axis = frame.groupby(["component", "target_axis"], as_index=False).agg(
        absolute_error=("absolute_error", "mean"),
        prediction_shift_vs_masked=("prediction_shift_vs_masked", "mean"),
        jobs=("profile_index", "size"), profiles=("profile_index", "nunique"),
    )
    by_component = per_axis.groupby("component", as_index=False).agg(
        absolute_error=("absolute_error", "mean"),
        prediction_shift_vs_masked=("prediction_shift_vs_masked", "mean"),
        target_axes=("target_axis", "nunique"), jobs=("jobs", "sum"),
    )
    overall = pd.DataFrame([{
        "component": "ALL", "absolute_error": per_axis.absolute_error.mean(),
        "prediction_shift_vs_masked": per_axis.prediction_shift_vs_masked.mean(),
        "target_axes": per_axis.target_axis.nunique(), "jobs": per_axis.jobs.sum(),
    }])
    result = pd.concat([overall, by_component], ignore_index=True)
    result.insert(0, "condition", condition)
    return result


def perturb_summary(jobs, family_clean, family_pred, control_clean, control_pred, factor):
    frame = jobs[["profile_index", "target_axis", "component"]].copy()
    truth = jobs.truth_normalized.to_numpy(dtype=float)
    frame["family_shift"] = np.abs(family_pred - family_clean)
    frame["control_shift"] = np.abs(control_pred - control_clean)
    frame["family_error_delta"] = np.abs(family_pred - truth) - np.abs(family_clean - truth)
    frame["control_error_delta"] = np.abs(control_pred - truth) - np.abs(control_clean - truth)
    per_axis = frame.groupby(["component", "target_axis"], as_index=False).agg(
        family_shift=("family_shift", "mean"), control_shift=("control_shift", "mean"),
        family_error_delta=("family_error_delta", "mean"),
        control_error_delta=("control_error_delta", "mean"),
        jobs=("profile_index", "size"),
    )
    by_component = per_axis.groupby("component", as_index=False).agg(
        family_shift=("family_shift", "mean"), control_shift=("control_shift", "mean"),
        family_error_delta=("family_error_delta", "mean"),
        control_error_delta=("control_error_delta", "mean"),
        target_axes=("target_axis", "nunique"), jobs=("jobs", "sum"),
    )
    overall = pd.DataFrame([{
        "component": "ALL", "family_shift": per_axis.family_shift.mean(),
        "control_shift": per_axis.control_shift.mean(),
        "family_error_delta": per_axis.family_error_delta.mean(),
        "control_error_delta": per_axis.control_error_delta.mean(),
        "target_axes": per_axis.target_axis.nunique(), "jobs": per_axis.jobs.sum(),
    }])
    result = pd.concat([overall, by_component], ignore_index=True)
    result.insert(0, "factor", factor)
    return result, frame


def paired_bootstrap(frame, factor, draws=2000):
    profile = frame.assign(difference=frame.family_shift - frame.control_shift).groupby(
        "profile_index", as_index=False
    ).difference.mean()
    values = profile.difference.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED + int(factor * 100))
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        estimates[draw] = rng.choice(values, len(values), replace=True).mean()
    return {
        "factor": factor, "profiles": len(values), "draws": draws,
        "family_minus_control_shift": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_family_greater": float((estimates > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    started = time.monotonic()
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    policy = DependencyMaskPolicy(data.axes)
    model, checkpoint, _ = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, device
    )

    rng = np.random.default_rng(SEED)
    job_parts = []
    components = {}
    for target in data.targets:
        target = int(target)
        hidden_relatives = set(policy.mask_indices(target)) - {target}
        if not hidden_relatives:
            continue
        component = weak_component(policy, target)
        label = component_label(policy, component)
        components[target] = (component, label)
        candidates = data.validation[data.observed[data.validation, target]]
        eligible = []
        relative_array = np.asarray(sorted(hidden_relatives), dtype=np.int64)
        for row in candidates:
            if np.any(data.observed[row, relative_array] & (data.raw[row, relative_array] > 0)):
                eligible.append(int(row))
        eligible = np.asarray(eligible, dtype=np.int64)
        if len(eligible) > JOBS_PER_TARGET:
            eligible = np.sort(rng.choice(eligible, JOBS_PER_TARGET, replace=False))
        if len(eligible):
            job_parts.append(pd.DataFrame({
                "profile_index": eligible, "target_axis": target,
                "target_name": policy.name_by_axis[target], "component": label,
                "truth_normalized": data.values[eligible, target],
            }))
    jobs = pd.concat(job_parts, ignore_index=True)
    rows = jobs.profile_index.to_numpy(dtype=np.int64)
    targets = jobs.target_axis.to_numpy(dtype=np.int64)
    base_visible = dependency_visible(data, policy, rows, targets)
    family_axis, control_axis, observed_mates, half_mates = [], [], [], []
    for position, (row, target) in enumerate(zip(rows, targets)):
        component, _ = components[int(target)]
        hidden = np.asarray(sorted(set(policy.mask_indices(int(target))) - {int(target)}), dtype=np.int64)
        positive = hidden[data.observed[row, hidden] & (data.raw[row, hidden] > 0)]
        offset = (SEED + 1009 * int(row) + 9176 * int(target)) % len(positive)
        selected = int(positive[offset])
        family_axis.append(selected)
        all_observed = hidden[data.observed[row, hidden]]
        observed_mates.append(all_observed)
        perm = np.random.default_rng(SEED + int(row) * 17 + int(target)).permutation(all_observed)
        half_mates.append(np.sort(perm[:max(1, int(np.ceil(len(perm) / 2)))]))

        available = np.flatnonzero(base_visible[position] & (data.raw[row] > 0))
        available = np.asarray([axis for axis in available if axis not in component], dtype=np.int64)
        selected_group = component_group(policy.name_by_axis[selected])
        same = np.asarray([
            axis for axis in available
            if component_group(policy.name_by_axis[int(axis)]) == selected_group
        ], dtype=np.int64)
        pool = same if len(same) else available
        if not len(pool):
            raise ValueError("dependency ablation job lacks a positive unrelated control")
        distance = np.abs(data.values[row, pool] - data.values[row, selected])
        control_axis.append(int(pool[np.argmin(distance)]))
    jobs["family_input_axis"] = family_axis
    jobs["family_input_name"] = [policy.name_by_axis[axis] for axis in family_axis]
    jobs["control_input_axis"] = control_axis
    jobs["control_input_name"] = [policy.name_by_axis[axis] for axis in control_axis]
    jobs.to_parquet(args.output_dir / "selected_jobs.parquet", index=False)
    family_axis = np.asarray(family_axis, dtype=np.int64)
    control_axis = np.asarray(control_axis, dtype=np.int64)

    one_visible = base_visible.copy()
    one_visible[np.arange(len(jobs)), family_axis] = True
    half_visible = base_visible.copy()
    all_visible = base_visible.copy()
    control_hidden_visible = base_visible.copy()
    control_hidden_visible[np.arange(len(jobs)), control_axis] = False
    for i, (half, all_axes) in enumerate(zip(half_mates, observed_mates)):
        half_visible[i, half] = True
        all_visible[i, all_axes] = True

    base = predict_normalized(model, data, rows, targets, base_visible, device, 512, two_pass=True)
    one = predict_normalized(model, data, rows, targets, one_visible, device, 512, two_pass=True)
    half = predict_normalized(model, data, rows, targets, half_visible, device, 512, two_pass=True)
    all_pred = predict_normalized(model, data, rows, targets, all_visible, device, 512, two_pass=True)
    control_hidden = predict_normalized(
        model, data, rows, targets, control_hidden_visible, device, 512, two_pass=True
    )
    exposure = pd.concat([
        macro_summary(jobs, base, base, "standard_mask"),
        macro_summary(jobs, one, base, "expose_one"),
        macro_summary(jobs, half, base, "expose_half"),
        macro_summary(jobs, all_pred, base, "expose_all"),
    ], ignore_index=True)
    exposure.to_csv(args.output_dir / "exposure_summary.csv", index=False)

    reveal_frame = jobs[["profile_index", "target_axis", "component"]].copy()
    reveal_frame["family_shift"] = np.abs(one - base)
    reveal_frame["control_shift"] = np.abs(base - control_hidden)
    reveal_axis = reveal_frame.groupby(["component", "target_axis"], as_index=False).agg(
        family_reveal_shift=("family_shift", "mean"),
        control_reveal_shift=("control_shift", "mean"), jobs=("profile_index", "size"),
    )
    reveal_summary = reveal_axis.groupby("component", as_index=False).agg(
        family_reveal_shift=("family_reveal_shift", "mean"),
        control_reveal_shift=("control_reveal_shift", "mean"),
        target_axes=("target_axis", "nunique"), jobs=("jobs", "sum"),
    )
    reveal_summary = pd.concat([pd.DataFrame([{
        "component": "ALL", "family_reveal_shift": reveal_axis.family_reveal_shift.mean(),
        "control_reveal_shift": reveal_axis.control_reveal_shift.mean(),
        "target_axes": reveal_axis.target_axis.nunique(), "jobs": reveal_axis.jobs.sum(),
    }]), reveal_summary], ignore_index=True)
    reveal_summary.to_csv(args.output_dir / "reveal_comparison.csv", index=False)
    reveal_bootstrap = paired_bootstrap(reveal_frame, 1.0)
    reveal_bootstrap["comparison"] = "reveal dependency relative versus reveal matched unrelated control"
    pd.DataFrame([reveal_bootstrap]).to_csv(
        args.output_dir / "reveal_paired_bootstrap.csv", index=False
    )

    perturb_parts, bootstrap = [], []
    for factor in FACTORS:
        family_values = data.values[rows].copy()
        family_raw = data.raw[rows, family_axis] * factor
        family_values[np.arange(len(jobs)), family_axis] = np.log1p(
            family_raw / data.scale[family_axis]
        )
        family_pred = predict_normalized(
            model, data, rows, targets, one_visible, device, 512, two_pass=True,
            normalized_values=family_values,
        )
        control_values = data.values[rows].copy()
        control_raw = data.raw[rows, control_axis] * factor
        control_values[np.arange(len(jobs)), control_axis] = np.log1p(
            control_raw / data.scale[control_axis]
        )
        control_pred = predict_normalized(
            model, data, rows, targets, base_visible, device, 512, two_pass=True,
            normalized_values=control_values,
        )
        summary, frame = perturb_summary(
            jobs, one, family_pred, base, control_pred, factor
        )
        perturb_parts.append(summary)
        bootstrap.append(paired_bootstrap(frame, factor))
        print(f"dependency-family factor {factor:g} complete", flush=True)
    perturbation = pd.concat(perturb_parts, ignore_index=True)
    perturbation.to_csv(args.output_dir / "perturbation_summary.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(args.output_dir / "paired_bootstrap.csv", index=False)

    overall_exposure = exposure[exposure.component.eq("ALL")].set_index("condition")
    order = ["standard_mask", "expose_one", "expose_half", "expose_all"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].plot(
        range(4), overall_exposure.loc[order, "absolute_error"], marker="o", color="#17365D"
    )
    axes[0].set_xticks(range(4), ["Masked", "Expose 1", "Expose half", "Expose all"])
    axes[0].set_ylabel("Macro scaled-log MAE")
    axes[0].set_title("Normally hidden dependency values exposed")
    component_base = exposure[exposure.condition.eq("standard_mask")].set_index("component")
    component_all = exposure[exposure.condition.eq("expose_all")].set_index("component")
    names = [name for name in component_base.index if name != "ALL"]
    change = component_all.loc[names].absolute_error - component_base.loc[names].absolute_error
    axes[1].barh(range(len(names)), change, color=np.where(change < 0, "#59A14F", "#E15759"))
    axes[1].set_yticks(range(len(names)), names, fontsize=7)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Expose-all error minus masked error")
    axes[1].set_title("Change by dependency component")
    fig.tight_layout()
    fig.savefig(args.output_dir / "dependency_exposure_ablation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    overall_perturb = perturbation[perturbation.component.eq("ALL")].sort_values("factor")
    fig, axis = plt.subplots(figsize=(7.8, 4.5))
    axis.plot(overall_perturb.factor, overall_perturb.family_shift, marker="o", label="Dependency relative")
    axis.plot(overall_perturb.factor, overall_perturb.control_shift, marker="s", label="Matched unrelated control")
    axis.set_xscale("log")
    axis.set_xticks(list(FACTORS), ["×0.9", "×1.1", "×10"])
    axis.set_ylabel("Macro absolute prediction shift")
    axis.set_title("Perturbation after exposing one normally hidden value")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dependency_perturbation_ablation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    component_table = []
    for target, (component, label) in components.items():
        component_table.append({
            "target_axis": target, "target_name": policy.name_by_axis[target],
            "dependency_component": label,
            "weak_component_axes": ";".join(map(str, sorted(component))),
            "target_specific_hidden_axes": ";".join(map(str, policy.mask_indices(target))),
        })
    pd.DataFrame(component_table).to_csv(
        args.output_dir / "dependency_components.csv", index=False
    )
    manifest = {
        "status": "complete", "version": "dependency_family_ablation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "jobs": len(jobs), "profiles": int(jobs.profile_index.nunique()),
        "targets": int(jobs.target_axis.nunique()),
        "dependency_components": int(jobs.component.nunique()),
        "jobs_per_target_cap": JOBS_PER_TARGET,
        "standard_policy": "target plus directed ancestors and descendants hidden; siblings visible",
        "ablation": "expose one, half, or all observed target-specific hidden relatives",
        "perturbation": "multiply one exposed positive dependency relative or one unrelated visible control",
        "control_matching": "same selected INFOODS group when available, then nearest normalized value",
        "bootstrap_draws": 2000, "bootstrap_unit": "validation profile",
        "evaluation_only": True, "training_changed": False,
        "causal_claim": False,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint": checkpoint_metadata(args.checkpoint, checkpoint),
        "exposure_summary": exposure.to_dict(orient="records"),
        "perturbation_summary": perturbation.to_dict(orient="records"),
        "paired_bootstrap": bootstrap,
        "reveal_summary": reveal_summary.to_dict(orient="records"),
        "reveal_paired_bootstrap": reveal_bootstrap,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "jobs": len(jobs), "targets": int(jobs.target_axis.nunique())}))


if __name__ == "__main__":
    main()

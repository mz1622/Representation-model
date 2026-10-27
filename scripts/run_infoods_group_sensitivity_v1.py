#!/usr/bin/env python3
"""Measure V5+CCE sensitivity between selected FAO/INFOODS component groups."""
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
from foodcomp.infoods_component_groups_v1 import (  # noqa: E402
    AUTHORITY, AUTHORITY_URL, TAGNAME_URL, VERSION as GROUP_VERSION, group_frame,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261028
JOBS_PER_TARGET = 120
CONDITIONS = (("minus_10pct", 0.9), ("plus_10pct", 1.1), ("deletion", None))
GROUP_ORDER = [
    "Amino acids", "Fatty acids", "Sugars", "Dietary fibre",
    "Minerals", "Vitamins", "Organic acids",
]


def aggregate(frame):
    per_axis = frame.groupby(
        ["condition", "input_group", "target_group", "target_axis"], as_index=False
    ).agg(
        prediction_shift=("prediction_shift", "mean"),
        error_delta=("error_delta", "mean"),
        perturbed_error=("perturbed_error", "mean"),
        jobs=("profile_index", "size"),
        profiles=("profile_index", "nunique"),
        input_axes=("input_axis", "nunique"),
    )
    return per_axis.groupby(
        ["condition", "input_group", "target_group"], as_index=False
    ).agg(
        prediction_shift=("prediction_shift", "mean"),
        error_delta=("error_delta", "mean"),
        perturbed_error=("perturbed_error", "mean"),
        target_axes=("target_axis", "nunique"),
        jobs=("jobs", "sum"), profiles=("profiles", "max"),
        input_axes=("input_axes", "max"),
    )


def heatmap(summary, condition, path, title):
    current = summary[summary.condition.eq(condition)].pivot(
        index="input_group", columns="target_group", values="prediction_shift"
    ).reindex(index=GROUP_ORDER, columns=GROUP_ORDER)
    fig, axis = plt.subplots(figsize=(9.6, 7.8))
    image = axis.imshow(current.to_numpy(dtype=float), cmap="magma", aspect="auto")
    axis.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=35, ha="right")
    axis.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    axis.set_xlabel("Target component group")
    axis.set_ylabel("Perturbed input component group")
    axis.set_title(title)
    for row in range(len(GROUP_ORDER)):
        for col in range(len(GROUP_ORDER)):
            value = current.iloc[row, col]
            if np.isfinite(value):
                maximum = np.nanmax(current.to_numpy())
                axis.text(
                    col, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if value > 0.65 * maximum else "white",
                )
    fig.colorbar(image, ax=axis, label="Target-macro absolute prediction shift")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


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
    mapping = group_frame(data.axes)
    mapping.to_csv(args.output_dir / "infoods_axis_groups.csv", index=False)
    group_by_axis = mapping.set_index("axis_index").infoods_group.to_dict()
    axes_by_group = {
        group: mapping.loc[mapping.infoods_group.eq(group), "axis_index"].to_numpy(dtype=np.int64)
        for group in GROUP_ORDER
    }

    target_axes = [
        int(axis) for axis in data.targets if pd.notna(group_by_axis.get(int(axis)))
    ]
    selected = []
    rng = np.random.default_rng(SEED)
    for axis in target_axes:
        candidates = data.validation[data.observed[data.validation, axis]]
        if len(candidates) > JOBS_PER_TARGET:
            candidates = np.sort(rng.choice(candidates, JOBS_PER_TARGET, replace=False))
        selected.append(pd.DataFrame({
            "profile_index": candidates, "target_axis": axis,
            "target_group": group_by_axis[axis],
        }))
    target_jobs = pd.concat(selected, ignore_index=True)
    rows = target_jobs.profile_index.to_numpy(dtype=np.int64)
    targets = target_jobs.target_axis.to_numpy(dtype=np.int64)
    base_visible = dependency_visible(data, policy, rows, targets)
    clean = predict_normalized(
        model, data, rows, targets, base_visible, device, 512, two_pass=True
    )
    truth = data.values[rows, targets]

    pair_records = []
    for job in range(len(target_jobs)):
        for group in GROUP_ORDER:
            available = axes_by_group[group]
            available = available[
                base_visible[job, available] & (data.raw[rows[job], available] > 0)
            ]
            if not len(available):
                continue
            offset = (SEED + 1009 * int(rows[job]) + 9176 * int(targets[job]) + 37 * GROUP_ORDER.index(group))
            chosen = int(available[offset % len(available)])
            pair_records.append({
                "job_index": job, "profile_index": int(rows[job]),
                "target_axis": int(targets[job]),
                "target_group": group_by_axis[int(targets[job])],
                "input_axis": chosen, "input_group": group,
            })
    pairs = pd.DataFrame(pair_records)
    if pairs.empty:
        raise ValueError("no eligible grouped input-target pairs")
    pairs.to_parquet(args.output_dir / "selected_pairs.parquet", index=False)
    pair_job = pairs.job_index.to_numpy(dtype=np.int64)
    pair_rows = rows[pair_job]
    pair_targets = targets[pair_job]
    pair_inputs = pairs.input_axis.to_numpy(dtype=np.int64)
    clean_pair = clean[pair_job]
    truth_pair = truth[pair_job]
    pair_visible = base_visible[pair_job].copy()
    result_parts = []
    for condition, factor in CONDITIONS:
        visible = pair_visible.copy()
        values = None
        if factor is None:
            visible[np.arange(len(pairs)), pair_inputs] = False
        else:
            values = data.values[pair_rows].copy()
            raw = data.raw[pair_rows, pair_inputs] * factor
            values[np.arange(len(pairs)), pair_inputs] = np.log1p(
                raw / data.scale[pair_inputs]
            )
        perturbed = predict_normalized(
            model, data, pair_rows, pair_targets, visible, device, 512,
            two_pass=True, normalized_values=values,
        )
        frame = pairs.copy()
        frame["condition"] = condition
        frame["prediction_shift"] = np.abs(perturbed - clean_pair)
        frame["clean_error"] = np.abs(clean_pair - truth_pair)
        frame["perturbed_error"] = np.abs(perturbed - truth_pair)
        frame["error_delta"] = frame.perturbed_error - frame.clean_error
        result_parts.append(frame)
        print(f"group sensitivity {condition}: {len(frame)} pairs", flush=True)
    results = pd.concat(result_parts, ignore_index=True)
    summary = aggregate(results)
    summary.to_csv(args.output_dir / "group_sensitivity.csv", index=False)
    diagonal = summary.assign(
        relation=np.where(summary.input_group.eq(summary.target_group), "within_group", "between_group")
    ).groupby(["condition", "relation"], as_index=False).agg(
        prediction_shift=("prediction_shift", "mean"),
        error_delta=("error_delta", "mean"), pairs=("jobs", "sum"),
    )
    diagonal.to_csv(args.output_dir / "within_between_summary.csv", index=False)
    heatmap(
        summary, "plus_10pct", args.output_dir / "group_sensitivity_plus10.png",
        "V5+CCE cross-group sensitivity: +10% one input",
    )
    heatmap(
        summary, "deletion", args.output_dir / "group_sensitivity_deletion.png",
        "V5+CCE cross-group sensitivity: delete one input",
    )
    manifest = {
        "status": "complete", "version": "infoods_group_sensitivity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "component_group_version": GROUP_VERSION,
        "authority": AUTHORITY, "authority_url": AUTHORITY_URL,
        "tagname_url": TAGNAME_URL,
        "classification_policy": "explicit conservative registry; unlisted axes remain unclassified",
        "groups": GROUP_ORDER,
        "classified_axes": int(mapping.classified.sum()),
        "classified_target_axes": len(target_axes),
        "target_jobs": len(target_jobs), "selected_input_target_pairs": len(pairs),
        "jobs_per_target_cap": JOBS_PER_TARGET,
        "mask_policy": "dependency mask applied before selecting visible grouped inputs",
        "estimand": "target-axis macro mean absolute normalized-log prediction shift",
        "causal_claim": False,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint": checkpoint_metadata(args.checkpoint, checkpoint),
        "summary": summary.to_dict(orient="records"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-validation sampled single-input sensitivity under dependency-mask v1.

Every observed validation target is included.  For each target job, one
strictly-positive visible input is selected deterministically after applying
the dependency mask.  The same selected input and the same task cells are used
for every model and perturbation, enabling paired comparisons without the
combinatorial cost of perturbing every visible input separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    checkpoint_metadata, dependency_visible, load_dependency_model, pack_visible,
    predict_normalized,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    dense_features, dependency_tasks, make_baseline,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261027
FACTORS = (0.001, 0.01, 0.1, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 10.0, 100.0, 1000.0)


def deterministic_selected(mask, profile, target):
    """Choose one eligible input per task with a deterministic rotated scan."""
    selected = np.full(len(mask), -1, dtype=np.int16)
    start = (profile.astype(np.int64) * 17 + target.astype(np.int64) * 31) % mask.shape[1]
    for offset in range(mask.shape[1]):
        axis = (start + offset) % mask.shape[1]
        take = (selected < 0) & mask[np.arange(len(mask)), axis]
        selected[take] = axis[take]
    return selected


@torch.inference_mode()
def predict_condition(
    model, data, rows, targets, base_visible, selected, device, batch_size,
    *, two_pass, kind, factor=None,
):
    pieces = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        batch_targets = targets[start:stop]
        batch_selected = selected[start:stop].astype(np.int64)
        visible = base_visible[start:stop].copy()
        values = data.values[batch_rows].copy()
        local = np.arange(len(batch_rows))
        if kind == "deletion":
            visible[local, batch_selected] = False
        elif kind == "factor":
            raw = data.raw[batch_rows, batch_selected] * float(factor)
            values[local, batch_selected] = np.log1p(
                raw / data.scale[batch_selected]
            ).astype(np.float32)
        elif kind == "zero_probe":
            raw = 0.1 * data.scale[batch_selected]
            values[local, batch_selected] = np.log1p(
                raw / data.scale[batch_selected]
            ).astype(np.float32)
        else:
            raise ValueError(kind)
        packed = pack_visible(
            data, batch_rows, batch_targets, visible, device, values
        )
        prediction = (
            model.forward_prompt_details(*packed)[1]
            if two_pass else model.forward_details(*packed)[0]
        )
        pieces.append(prediction[:, 0].detach().cpu().numpy())
    result = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite perturbed neural prediction")
    return result


def score_condition(method, condition, rows, targets, truth, clean, perturbed):
    frame = pd.DataFrame({
        "profile_index": rows, "axis_index": targets,
        "prediction_shift": np.abs(perturbed - clean),
        "clean_error": np.abs(clean - truth),
        "perturbed_error": np.abs(perturbed - truth),
    })
    frame["error_delta"] = frame.perturbed_error - frame.clean_error
    per_axis = frame.groupby("axis_index", as_index=False).agg(
        jobs=("profile_index", "size"),
        prediction_shift=("prediction_shift", "mean"),
        clean_error=("clean_error", "mean"),
        error_delta=("error_delta", "mean"),
        perturbed_error=("perturbed_error", "mean"),
    )
    return {
        "method": method, "condition": condition,
        "jobs": len(frame), "profiles": int(frame.profile_index.nunique()),
        "axes": int(frame.axis_index.nunique()),
        "prediction_shift": float(per_axis.prediction_shift.mean()),
        "clean_error": float(per_axis.clean_error.mean()),
        "error_delta": float(per_axis.error_delta.mean()),
        "perturbed_error": float(per_axis.perturbed_error.mean()),
        "nonfinite": int((~np.isfinite(perturbed)).sum()),
    }, frame


def bootstrap_difference(left, right, rows, targets, draws=500):
    profiles = np.unique(rows)
    axes = np.unique(targets)
    profile_map = {value: i for i, value in enumerate(profiles)}
    axis_map = {value: i for i, value in enumerate(axes)}
    rr = np.asarray([profile_map[value] for value in rows], dtype=np.int64)
    aa = np.asarray([axis_map[value] for value in targets], dtype=np.int64)
    rng = np.random.default_rng(SEED + 999)
    weights = np.stack([
        np.bincount(rng.integers(len(profiles), size=len(profiles)), minlength=len(profiles))
        for _ in range(draws)
    ]).astype(np.float64)
    results = {}
    for metric in ("prediction_shift", "error_delta", "perturbed_error"):
        diff = left[metric].to_numpy(float) - right[metric].to_numpy(float)
        point = float(pd.DataFrame({"axis": targets, "difference": diff})
                      .groupby("axis").difference.mean().mean())
        total = np.zeros((len(profiles), len(axes)), dtype=np.float64)
        count = np.zeros_like(total)
        np.add.at(total, (rr, aa), diff)
        np.add.at(count, (rr, aa), 1.0)
        numerator = weights @ total
        denominator = weights @ count
        sample = np.nanmean(np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        ), axis=1)
        results[metric] = {
            "difference": point,
            "ci_low": float(np.quantile(sample, 0.025)),
            "ci_high": float(np.quantile(sample, 0.975)),
            "probability_left_lower": float((sample < 0).mean()),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=768)
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
    rows, targets = dependency_tasks(data, data.validation)
    order = np.lexsort((targets, rows))
    rows, targets = rows[order], targets[order]
    if len(rows) != 323_809:
        raise ValueError("unexpected validation target-job count")
    visible = dependency_visible(data, policy, rows, targets)
    positive = visible & (data.raw[rows] > 0)
    zero = visible & (data.raw[rows] == 0)
    selected_positive = deterministic_selected(positive, rows, targets)
    selected_zero = deterministic_selected(zero, rows, targets)
    positive_keep = selected_positive >= 0
    zero_keep = selected_zero >= 0
    selection = pd.DataFrame({
        "profile_index": rows, "axis_index": targets,
        "selected_positive_axis": selected_positive,
        "selected_zero_axis": selected_zero,
        "visible_count": visible.sum(1),
        "visible_positive_count": positive.sum(1),
        "visible_zero_count": zero.sum(1),
    })
    selection.to_parquet(args.output_dir / "task_input_selection.parquet", index=False)

    checkpoint_specs = {
        "V4": (args.checkpoint_dir / "v4_best_validation.pt", False),
        "V5": (args.checkpoint_dir / "v5_best_validation.pt", True),
        "V5_CCE": (args.checkpoint_dir / "cce_best_validation.pt", True),
    }
    conditions = [(f"factor_{factor:g}", "factor", factor) for factor in FACTORS]
    conditions += [("deletion", "deletion", None), ("zero_plus_0.1scale", "zero_probe", None)]
    outputs, cell_frames, summary_rows, checkpoint_info = {}, {}, [], {}
    for method, (path, two_pass) in checkpoint_specs.items():
        model, checkpoint, config = load_dependency_model(
            path, data, args.policy_manifest, device
        )
        checkpoint_info[method] = checkpoint_metadata(path, checkpoint)
        print(f"clean sensitivity prediction {method}", flush=True)
        clean = predict_normalized(
            model, data, rows, targets, visible, device, args.batch_size,
            two_pass=two_pass,
        )
        outputs[method] = {"clean": clean}
        truth_all = data.values[rows, targets]
        clean_axis = pd.DataFrame({
            "axis": targets, "absolute_error": np.abs(clean - truth_all),
        }).groupby("axis").absolute_error.mean()
        summary_rows.append({
            "method": method, "condition": "clean", "jobs": len(rows),
            "profiles": len(np.unique(rows)), "axes": len(clean_axis),
            "prediction_shift": 0.0, "clean_error": float(clean_axis.mean()),
            "error_delta": 0.0, "perturbed_error": float(clean_axis.mean()),
            "nonfinite": 0,
        })
        for condition, kind, factor in conditions:
            keep = zero_keep if kind == "zero_probe" else positive_keep
            chosen = selected_zero[keep] if kind == "zero_probe" else selected_positive[keep]
            print(f"{method} {condition}: {keep.sum()} jobs", flush=True)
            perturbed = predict_condition(
                model, data, rows[keep], targets[keep], visible[keep], chosen,
                device, args.batch_size, two_pass=two_pass, kind=kind, factor=factor,
            )
            outputs[method][condition] = perturbed
            record, cells = score_condition(
                method, condition, rows[keep], targets[keep], truth_all[keep],
                clean[keep], perturbed,
            )
            summary_rows.append(record)
            if condition in ("factor_0.9", "factor_1.1", "deletion", "factor_1000"):
                cell_frames[(method, condition)] = cells
        del model
        torch.cuda.empty_cache()

    # The formal RF baseline is refit on the original training split and then
    # evaluated on exactly the same task/input pairs as the neural models.
    method = "RF"
    rf_clean = np.zeros(len(rows), dtype=np.float32)
    rf_outputs = {condition: np.full(positive_keep.sum() if kind != "zero_probe" else zero_keep.sum(), np.nan, dtype=np.float32)
                  for condition, kind, factor in conditions}
    positive_position = np.full(len(rows), -1, dtype=np.int64)
    positive_position[positive_keep] = np.arange(positive_keep.sum())
    zero_position = np.full(len(rows), -1, dtype=np.int64)
    zero_position[zero_keep] = np.arange(zero_keep.sum())
    for number, axis in enumerate(np.unique(targets), start=1):
        train = data.train[data.observed[data.train, axis]]
        train_axes = np.full(len(train), axis, dtype=np.int64)
        x_train = dense_features(data, train, train_axes, policy)
        estimator = make_baseline("rf", int(axis), config)
        weight = data.weights[train, axis]
        estimator.fit(
            x_train, data.values[train, axis], sample_weight=weight / weight.mean()
        )
        estimator.n_jobs = 1
        positions = np.flatnonzero(targets == axis)
        x = np.concatenate((
            np.where(visible[positions], data.values[rows[positions]], 0),
            visible[positions].astype(np.float32),
        ), axis=1).astype(np.float32)
        rf_clean[positions] = estimator.predict(x)
        for condition, kind, factor in conditions:
            keep_local = zero_keep[positions] if kind == "zero_probe" else positive_keep[positions]
            local_positions = positions[keep_local]
            if not len(local_positions):
                continue
            selected = selected_zero[local_positions] if kind == "zero_probe" else selected_positive[local_positions]
            xx = x[keep_local].copy()
            local = np.arange(len(xx))
            if kind == "deletion":
                xx[local, selected] = 0
                xx[local, len(data.axes) + selected] = 0
            elif kind == "factor":
                raw = data.raw[rows[local_positions], selected] * float(factor)
                xx[local, selected] = np.log1p(raw / data.scale[selected])
            else:
                xx[local, selected] = np.log1p(0.1)
            destination = zero_position[local_positions] if kind == "zero_probe" else positive_position[local_positions]
            rf_outputs[condition][destination] = estimator.predict(xx)
        if number % 10 == 0:
            print(f"RF sensitivity axis {number}/{len(np.unique(targets))}", flush=True)
    if not np.isfinite(rf_clean).all() or any(not np.isfinite(value).all() for value in rf_outputs.values()):
        raise FloatingPointError("nonfinite RF sensitivity prediction")
    outputs[method] = {"clean": rf_clean, **rf_outputs}
    truth_all = data.values[rows, targets]
    clean_axis = pd.DataFrame({
        "axis": targets, "absolute_error": np.abs(rf_clean - truth_all),
    }).groupby("axis").absolute_error.mean()
    summary_rows.append({
        "method": method, "condition": "clean", "jobs": len(rows),
        "profiles": len(np.unique(rows)), "axes": len(clean_axis),
        "prediction_shift": 0.0, "clean_error": float(clean_axis.mean()),
        "error_delta": 0.0, "perturbed_error": float(clean_axis.mean()),
        "nonfinite": 0,
    })
    for condition, kind, factor in conditions:
        keep = zero_keep if kind == "zero_probe" else positive_keep
        perturbed = rf_outputs[condition]
        record, cells = score_condition(
            method, condition, rows[keep], targets[keep], truth_all[keep],
            rf_clean[keep], perturbed,
        )
        summary_rows.append(record)
        if condition in ("factor_0.9", "factor_1.1", "deletion", "factor_1000"):
            cell_frames[(method, condition)] = cells

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    bootstrap = []
    for condition in ("factor_0.9", "factor_1.1", "deletion", "factor_1000"):
        left = cell_frames[("V5_CCE", condition)]
        right = cell_frames[("RF", condition)]
        if not left[["profile_index", "axis_index"]].equals(right[["profile_index", "axis_index"]]):
            raise ValueError("paired sensitivity cells differ")
        result = bootstrap_difference(
            left, right,
            left.profile_index.to_numpy(dtype=np.int64),
            left.axis_index.to_numpy(dtype=np.int64),
        )
        for metric, values in result.items():
            bootstrap.append({
                "condition": condition, "left": "V5_CCE", "right": "RF",
                "metric": metric, "draws": 500, **values,
            })
    pd.DataFrame(bootstrap).to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    input_support = selection.groupby("selected_positive_axis", as_index=False).size()
    input_support = input_support[input_support.selected_positive_axis.ge(0)].merge(
        data.axes[["axis_index", "canonical_name"]],
        left_on="selected_positive_axis", right_on="axis_index", how="left",
    )
    input_support.to_csv(args.output_dir / "selected_input_support.csv", index=False)
    manifest = {
        "status": "complete", "version": "dependency_mask_sensitivity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "validation_profiles": len(data.validation), "jobs": len(rows),
        "targets": int(len(np.unique(targets))), "input_axes": len(data.axes),
        "positive_input_jobs": int(positive_keep.sum()),
        "zero_input_jobs": int(zero_keep.sum()),
        "selection": "one deterministic eligible input per target job after dependency masking",
        "estimand": "target-macro prediction shift, error delta, and perturbed error",
        "conditions": [value[0] for value in conditions],
        "bootstrap_draws": 500, "bootstrap_unit": "validation profile",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoints": checkpoint_info,
        "legacy_family_column_used": False,
        "summary": summary.to_dict(orient="records"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

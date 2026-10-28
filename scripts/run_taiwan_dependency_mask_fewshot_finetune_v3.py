#!/usr/bin/env python3
"""Fine-tune dependency-mask V5+CCE on the exact Taiwan few-shot baseline subsets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    attach_taiwan_panel,
    load_dependency_model,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    axis_balanced_loss,
    masked_axis_batch,
    predict_transformer,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SHOTS = (8, 32, 128)
SEEDS = (20261027, 20261028, 20261029)
SPLIT_SEED = 20261026
TEST_PROFILES = 400


def coverage_nested_order(pool, axes, observed, seed):
    """Match v2 baselines: cover all axes using missingness only, then randomize."""
    rng = np.random.default_rng(seed)
    remaining = np.asarray(pool, dtype=np.int64).copy()
    uncovered = np.ones(len(axes), dtype=bool)
    selected = []
    while uncovered.any():
        score = observed[np.ix_(remaining, axes)][:, uncovered].sum(1)
        best = score.max(initial=0)
        if best <= 0:
            raise ValueError("adaptation pool cannot cover all Taiwan axes")
        tied = np.flatnonzero(score == best)
        position = int(rng.choice(tied))
        row = int(remaining[position])
        selected.append(row)
        uncovered &= ~observed[row, axes]
        remaining = np.delete(remaining, position)
    if len(selected) > min(SHOTS):
        raise ValueError(
            f"axis coverage requires {len(selected)} profiles, more than 8 shots"
        )
    tail = rng.permutation(remaining)
    return np.concatenate((np.asarray(selected, dtype=np.int64), tail)), len(selected)


def row_hash(rows):
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_baseline_artifacts(args, computed_split):
    manifest_path = args.baseline_dir / "manifest.json"
    summary_path = args.baseline_dir / "fewshot_summary.csv"
    split_path = args.baseline_dir / "profile_split.csv"
    for path in (manifest_path, summary_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("baseline few-shot run is not complete")
    expected = {
        "shots": list(SHOTS),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES,
        "test_axes": 80,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"baseline {key} mismatch: {manifest.get(key)} != {value}")
    if manifest.get("checkpoint_sha256") != digest(args.checkpoint):
        raise ValueError("baseline checkpoint hash mismatch")
    if manifest.get("policy_manifest_sha256") != digest(args.policy_manifest):
        raise ValueError("baseline policy hash mismatch")
    if manifest.get("data_sha256") != digest(args.data_dir / "manifest.json"):
        raise ValueError("baseline data hash mismatch")

    baseline_split = pd.read_csv(split_path)[
        ["profile_index", "fewshot_partition"]
    ].sort_values("profile_index").reset_index(drop=True)
    computed_split = computed_split[
        ["profile_index", "fewshot_partition"]
    ].sort_values("profile_index").reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_split, computed_split, check_dtype=False)
    return manifest, pd.read_csv(summary_path), digest(manifest_path), digest(split_path)


def fine_tune_once(
    data,
    jobs,
    policy,
    checkpoint,
    policy_manifest,
    train_rows,
    test_jobs,
    seed,
    device,
    epochs,
    learning_rate,
    batch_size,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, _, config = load_dependency_model(
        checkpoint, data, policy_manifest, device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )

    train_jobs = jobs[jobs.profile_index.isin(train_rows)].copy().reset_index(drop=True)
    counts = np.bincount(
        train_jobs.axis_index.to_numpy(dtype=np.int64), minlength=len(data.axes)
    ).astype(np.float32)
    active = np.flatnonzero(counts > 0)
    if len(active) != 80:
        raise ValueError(f"fine-tuning subset covers {len(active)} axes instead of 80")
    totals = torch.from_numpy(np.maximum(counts, 1)).to(device)
    rng = np.random.default_rng(seed + 100_000)
    task_rows = train_jobs.profile_index.to_numpy(dtype=np.int64)
    task_axes = train_jobs.axis_index.to_numpy(dtype=np.int64)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_jobs))
        first_total = 0.0
        second_total = 0.0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            rows = task_rows[chosen]
            axes = task_axes[chosen]
            packed = masked_axis_batch(data, rows, axes, policy, device)
            first, second, _, _, _, _ = model.forward_prompt_details(*packed)
            label = torch.from_numpy(
                data.values[rows, axes].astype(np.float32)
            ).to(device)
            axis_tensor = torch.from_numpy(axes).to(device)
            first_loss = axis_balanced_loss(
                first[:, 0],
                label,
                axis_tensor,
                totals,
                len(active),
                len(train_jobs),
                len(chosen),
            )
            second_loss = axis_balanced_loss(
                second[:, 0],
                label,
                axis_tensor,
                totals,
                len(active),
                len(train_jobs),
                len(chosen),
            )
            loss = 0.5 * (first_loss + second_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Taiwan fine-tuning loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            first_total += float(first_loss.detach()) * len(chosen)
            second_total += float(second_loss.detach()) * len(chosen)
        history.append(
            {
                "epoch": epoch,
                "first_loss": first_total / len(train_jobs),
                "second_loss": second_total / len(train_jobs),
            }
        )

    prediction = predict_transformer(
        model, data, policy, test_jobs, device, batch_size, two_pass=True
    )
    metrics, per_axis, _ = score_profile_axis_macro(
        data, test_jobs, prediction
    )
    metadata = {
        "train_profiles": len(train_rows),
        "train_jobs": len(train_jobs),
        "active_training_axes": len(active),
        "min_axis_labels": int(counts[active].min()),
        "max_axis_labels": int(counts[active].max()),
    }
    return model, metrics, per_axis, pd.DataFrame(history), metadata


def build_comparison(finetune_summary, baseline_summary):
    metrics = [
        "macro_scaled_log_mae",
        "macro_scaled_log_rmse",
        "macro_raw_mae",
    ]
    tree = baseline_summary[
        baseline_summary.method.isin(["RF", "XGBoost", "V5+CCE frozen"])
    ].copy()
    tree = tree[[
        "method",
        "shots",
        *[f"{metric}_mean" for metric in metrics],
        *[f"{metric}_sd" for metric in metrics],
    ]]
    tuned = finetune_summary[[
        "shots",
        *[f"{metric}_mean" for metric in metrics],
        *[f"{metric}_sd" for metric in metrics],
    ]].copy()
    tuned.insert(0, "method", "V5+CCE finetuned")
    comparison = pd.concat([tree, tuned], ignore_index=True)
    comparison = comparison.sort_values(["shots", "method"]).reset_index(drop=True)
    frozen = comparison[comparison.method == "V5+CCE frozen"].set_index("shots")
    comparison["scaled_log_mae_change_vs_frozen"] = comparison.apply(
        lambda row: row.macro_scaled_log_mae_mean
        - frozen.loc[row.shots, "macro_scaled_log_mae_mean"],
        axis=1,
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if args.epochs < 1 or args.learning_rate <= 0 or args.batch_size < 1:
        raise ValueError("invalid fine-tuning hyperparameters")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    started = time.monotonic()

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, novel_rows, panel, axes = attach_taiwan_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    order = np.random.default_rng(SPLIT_SEED).permutation(novel_rows)
    test_rows = np.sort(order[:TEST_PROFILES])
    adaptation_pool = np.sort(order[TEST_PROFILES:])
    test_jobs = jobs[jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    if test_jobs.axis_index.nunique() != 80:
        raise ValueError("fixed Taiwan test does not cover all 80 axes")
    split = panel[["profile_index"]].copy()
    split["fewshot_partition"] = np.where(
        split.profile_index.isin(test_rows),
        "fixed_test",
        np.where(
            split.profile_index.isin(adaptation_pool),
            "adaptation_pool",
            "excluded",
        ),
    )
    baseline_manifest, baseline_summary, baseline_manifest_hash, split_hash = (
        validate_baseline_artifacts(args, split)
    )

    manifest = {
        "status": "running",
        "version": "taiwan_dependency_mask_fewshot_finetune_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES,
        "test_jobs": len(test_jobs),
        "test_axes": 80,
        "adaptation_pool_profiles": len(adaptation_pool),
        "initialization": "validation-selected dependency-mask V5+CCE checkpoint",
        "parameters_trainable": "all",
        "fine_tuning_objective": "0.5 first-pass completion + 0.5 second-pass completion",
        "cce_during_fine_tuning": False,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "checkpoint_selection": "fixed final epoch; no Taiwan test-based selection",
        "subset_sampling": (
            "identical deterministic coverage-first nested subsets as RF/XGBoost v2"
        ),
        "baseline_manifest_sha256": baseline_manifest_hash,
        "baseline_split_sha256": split_hash,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "model_checkpoint_saved": False,
        "row_level_predictions_saved": False,
        "runs": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    records = []
    history_parts = []
    axis_parts = []
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            run_started = time.monotonic()
            train_rows = np.sort(nested[:shots])
            print(f"fine-tuning V5+CCE: seed={seed} shots={shots}", flush=True)
            model, metrics, per_axis, history, metadata = fine_tune_once(
                data,
                jobs,
                policy,
                args.checkpoint,
                args.policy_manifest,
                train_rows,
                test_jobs,
                seed,
                device,
                args.epochs,
                args.learning_rate,
                args.batch_size,
            )
            record = {
                "method": "V5+CCE finetuned",
                "shots": shots,
                "seed": seed,
                **metrics,
                **metadata,
                "coverage_profiles": coverage_profiles,
                "selected_profile_rows_sha256": row_hash(train_rows),
                "elapsed_seconds": time.monotonic() - run_started,
            }
            records.append(record)
            manifest["runs"].append(record)
            history.insert(0, "seed", seed)
            history.insert(0, "shots", shots)
            history_parts.append(history)
            per_axis.insert(0, "seed", seed)
            per_axis.insert(0, "shots", shots)
            per_axis.insert(0, "method", "V5+CCE finetuned")
            axis_parts.append(per_axis)
            write_json(args.output_dir / "manifest.json", manifest)
            print(
                f"  SLMAE={metrics['macro_scaled_log_mae']:.6f} "
                f"SLRMSE={metrics['macro_scaled_log_rmse']:.6f} "
                f"raw={metrics['macro_raw_mae']:.6f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    runs = pd.DataFrame(records)
    runs.to_csv(args.output_dir / "finetune_runs.csv", index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(
        args.output_dir / "training_history.csv", index=False
    )
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    metrics = [
        "macro_scaled_log_mae",
        "macro_scaled_log_rmse",
        "macro_raw_mae",
    ]
    summary = runs.groupby("shots", as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sd": (metric, "std") for metric in metrics},
        min_axis_labels=("min_axis_labels", "min"),
        max_axis_labels=("max_axis_labels", "max"),
    )
    summary.to_csv(args.output_dir / "finetune_summary.csv", index=False)
    comparison = build_comparison(summary, baseline_summary)
    comparison.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    manifest.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        completed_runs=len(runs),
        baseline_protocol_version=baseline_manifest["version"],
        finetune_summary=summary.to_dict(orient="records"),
        comparison_summary=comparison.to_dict(orient="records"),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["comparison_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

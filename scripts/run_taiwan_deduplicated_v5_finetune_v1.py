#!/usr/bin/env python3
"""Fine-tune V5+CCE on deduplicated, identity-group-isolated Taiwan few-shot data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from foodcomp.dependency_mask_analysis_v1 import load_dependency_model  # noqa: E402
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    axis_balanced_loss, masked_axis_batch, predict_transformer,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.taiwan_deduplicated_v1 import (  # noqa: E402
    SEEDS, SHOTS, SPLIT_SEED, TEST_PROFILES, attach_clean_panel,
    coverage_nested_order, group_aware_split, row_hash,
)


def fine_tune_once(
    data, jobs, policy, checkpoint, policy_manifest, train_rows, test_jobs,
    seed, device, epochs, learning_rate, batch_size,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _, config = load_dependency_model(checkpoint, data, policy_manifest, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    train_jobs = jobs[jobs.profile_index.isin(train_rows)].reset_index(drop=True)
    counts = np.bincount(
        train_jobs.axis_index.to_numpy(np.int64), minlength=len(data.axes)
    ).astype(np.float32)
    active = np.flatnonzero(counts > 0)
    if len(active) != 80:
        raise ValueError(f"fine-tuning subset covers {len(active)} axes, expected 80")
    totals = torch.from_numpy(np.maximum(counts, 1)).to(device)
    task_rows = train_jobs.profile_index.to_numpy(np.int64)
    task_axes = train_jobs.axis_index.to_numpy(np.int64)
    rng = np.random.default_rng(seed + 100_000)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        first_total = second_total = 0.0
        order = rng.permutation(len(train_jobs))
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            rows, axes = task_rows[chosen], task_axes[chosen]
            packed = masked_axis_batch(data, rows, axes, policy, device)
            first, second, _, _, _, _ = model.forward_prompt_details(*packed)
            label = torch.from_numpy(data.values[rows, axes].astype(np.float32)).to(device)
            axis_tensor = torch.from_numpy(axes).to(device)
            first_loss = axis_balanced_loss(
                first[:, 0], label, axis_tensor, totals, len(active),
                len(train_jobs), len(chosen),
            )
            second_loss = axis_balanced_loss(
                second[:, 0], label, axis_tensor, totals, len(active),
                len(train_jobs), len(chosen),
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
        history.append({
            "epoch": epoch,
            "first_loss": first_total / len(train_jobs),
            "second_loss": second_total / len(train_jobs),
        })
    prediction = predict_transformer(
        model, data, policy, test_jobs, device, batch_size, two_pass=True
    )
    metrics, per_axis, _ = score_profile_axis_macro(data, test_jobs, prediction)
    metadata = {
        "train_profiles": len(train_rows), "train_jobs": len(train_jobs),
        "active_training_axes": len(active),
        "min_axis_labels": int(counts[active].min()),
        "max_axis_labels": int(counts[active].max()),
    }
    return model, metrics, per_axis, pd.DataFrame(history), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--overlap-ledger", type=Path, required=True)
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
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    started = time.monotonic()

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, clean_jobs, clean, _, axes = attach_clean_panel(
        data, args.audit_dir, args.overlap_ledger
    )
    test_rows, adaptation_pool = group_aware_split(clean)
    test_jobs = clean_jobs[clean_jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    policy = DependencyMaskPolicy(data.axes)
    baseline_manifest = json.loads(
        (args.baseline_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if baseline_manifest.get("status") != "complete":
        raise ValueError("deduplicated Taiwan baseline run is incomplete")
    for key, value in {
        "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES, "test_jobs": len(test_jobs),
        "overlap_ledger_sha256": digest(args.overlap_ledger),
    }.items():
        if baseline_manifest.get(key) != value:
            raise ValueError(f"baseline {key} mismatch")

    manifest = {
        "status": "running", "version": "taiwan_deduplicated_v5_finetune_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "test_profiles": len(test_rows), "test_jobs": len(test_jobs),
        "test_axes": len(axes), "adaptation_pool_profiles": len(adaptation_pool),
        "identity_group_aware_split": True,
        "initialization": "validation-selected dependency-mask V5+CCE checkpoint",
        "parameters_trainable": "all", "cce_during_fine_tuning": False,
        "objective": "0.5 first-pass completion + 0.5 second-pass completion",
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "optimizer": "AdamW",
        "checkpoint_selection": "fixed epoch 20; no Taiwan test selection",
        "model_checkpoint_saved": False, "row_predictions_saved": False,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "overlap_ledger_sha256": digest(args.overlap_ledger),
        "baseline_manifest_sha256": digest(args.baseline_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint), "runs": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    records, histories, axis_parts = [], [], []
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            run_started = time.monotonic()
            train_rows = np.sort(nested[:shots])
            print(f"V5+CCE finetune seed={seed} shots={shots}", flush=True)
            model, metrics, per_axis, history, metadata = fine_tune_once(
                data, clean_jobs, policy, args.checkpoint, args.policy_manifest,
                train_rows, test_jobs, seed, device, args.epochs,
                args.learning_rate, args.batch_size,
            )
            record = {
                "method": "V5+CCE finetuned", "shots": shots, "seed": seed,
                **metrics, **metadata, "coverage_profiles": coverage_profiles,
                "selected_profile_rows_sha256": row_hash(train_rows),
                "elapsed_seconds": time.monotonic() - run_started,
            }
            records.append(record)
            manifest["runs"].append(record)
            history.insert(0, "seed", seed)
            history.insert(0, "shots", shots)
            histories.append(history)
            per_axis.insert(0, "seed", seed)
            per_axis.insert(0, "shots", shots)
            per_axis.insert(0, "method", "V5+CCE finetuned")
            axis_parts.append(per_axis)
            write_json(args.output_dir / "manifest.json", manifest)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    runs = pd.DataFrame(records)
    metrics = ["macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"]
    summary = runs.groupby("shots", as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sd": (metric, "std") for metric in metrics},
        min_axis_labels=("min_axis_labels", "min"),
        max_axis_labels=("max_axis_labels", "max"),
    )
    baseline = pd.read_csv(args.baseline_dir / "fewshot_summary.csv")
    tuned = summary.copy()
    tuned.insert(0, "method", "V5+CCE finetuned")
    comparison = pd.concat((baseline, tuned), ignore_index=True).sort_values(
        ["shots", "method"]
    )
    runs.to_csv(args.output_dir / "finetune_runs.csv", index=False)
    summary.to_csv(args.output_dir / "finetune_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        args.output_dir / "training_history.csv", index=False
    )
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    manifest.update(
        status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started, completed_runs=len(runs),
        summary=summary.to_dict(orient="records"),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()

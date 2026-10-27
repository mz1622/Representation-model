#!/usr/bin/env python3
"""Train V4 -> V5 -> V5+CCE from scratch under dependency-mask protocol v1."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy, VERSION as MASK_VERSION  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    FoodPromptCceTransformer, axis_balanced_loss, dependency_tasks,
    masked_axis_batch, multi_positive_cce_loss, predict_transformer,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


VERSION = "dependency_mask_v1_v4_v5_cce"


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, model, config, data_sha, stage, epoch, metrics, policy_sha):
    torch.save({
        "version": VERSION,
        "mask_policy_version": MASK_VERSION,
        "mask_policy_sha256": policy_sha,
        "stage": stage,
        "completed_epoch": epoch,
        "data_sha256": data_sha,
        "food_name_model_input": False,
        "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "official_inference_pass": "first" if stage == "v4" else "second",
        "config": asdict(config),
        "validation_metrics": metrics,
        "state_dict": cpu_state(model),
    }, path)


def train_stage(
    model, data, policy, task_rows, task_axes, validation_jobs, device,
    config, stage, epochs, learning_rate, eval_batch_size, output_dir,
    data_sha, policy_sha, cce_weight, cce_temperature, tasks_per_epoch,
):
    if stage not in {"v4", "v5", "cce"}:
        raise ValueError(stage)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    full_totals = np.bincount(task_axes, minlength=len(data.axes)).astype(np.float32)
    active = np.flatnonzero(full_totals > 0)
    if set(map(int, data.targets)) - set(map(int, active)):
        raise ValueError("a supervised axis has no training task")
    pools = {axis: np.flatnonzero(task_axes == axis) for axis in active}
    rng = np.random.default_rng(config.seed + {"v4": 0, "v5": 1, "cce": 2}[stage])
    history, best_metric, best_state, best_epoch = [], math.inf, None, None
    stage_start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_count = min(tasks_per_epoch, len(task_rows)) if tasks_per_epoch else len(task_rows)
        if epoch_count < len(task_rows):
            quotient, remainder = divmod(epoch_count, len(active))
            selected = []
            for position, axis in enumerate(active):
                count = quotient + (position < remainder)
                pool = pools[int(axis)]
                selected.append(rng.choice(pool, count, replace=count > len(pool)))
            epoch_indices = np.concatenate(selected)
        else:
            epoch_indices = np.arange(len(task_rows), dtype=np.int64)
        order = rng.permutation(epoch_indices)
        epoch_axes = task_axes[epoch_indices]
        totals_np = np.bincount(epoch_axes, minlength=len(data.axes)).astype(np.float32)
        totals = torch.as_tensor(totals_np, device=device).clamp_min(1.0)
        accumulated = {"first": 0.0, "second": 0.0, "cce": 0.0, "total": 0.0}
        epoch_start = time.monotonic()
        for start in range(0, len(order), config.batch_size):
            chosen = order[start:start + config.batch_size]
            rows, axes = task_rows[chosen], task_axes[chosen]
            packed = masked_axis_batch(
                data, rows, axes, policy, device, with_targets=True
            )
            axis_tensor = packed[3][:, 0]
            label = packed[5][:, 0]
            batch_count = len(chosen)
            optimizer.zero_grad(set_to_none=True)
            if stage == "v4":
                first_prediction, _, _ = model.forward_details(*packed[:5])
                first_loss = axis_balanced_loss(
                    first_prediction[:, 0], label, axis_tensor, totals,
                    len(active), len(order), batch_count,
                )
                second_loss = first_loss.detach() * 0
                cce = first_loss.detach() * 0
                loss = first_loss
            else:
                first_prediction, prompt_food, _ = model.forward_details(*packed[:5])
                second_prediction, _, _ = model.forward_details_with_food_prompt(
                    *packed[:5], prompt_food.detach()
                )
                first_loss = axis_balanced_loss(
                    first_prediction[:, 0], label, axis_tensor, totals,
                    len(active), len(order), batch_count,
                )
                second_loss = axis_balanced_loss(
                    second_prediction[:, 0], label, axis_tensor, totals,
                    len(active), len(order), batch_count,
                )
                completion = 0.5 * (first_loss + second_loss)
                if stage == "cce" and batch_count >= 2:
                    _, contrastive_food, _ = model.forward_details(*packed[:5])
                    profile_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
                    raw_cce = multi_positive_cce_loss(
                        prompt_food, contrastive_food, profile_tensor,
                        temperature=cce_temperature,
                    )
                    unique = int(profile_tensor.unique().numel())
                    cce = raw_cce / math.log(unique) if unique > 1 else raw_cce * 0
                else:
                    cce = completion.detach() * 0
                loss = completion + (cce_weight * cce if stage == "cce" else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            accumulated["first"] += float(first_loss.detach()) * batch_count
            accumulated["second"] += float(second_loss.detach()) * batch_count
            accumulated["cce"] += float(cce.detach()) * batch_count
            accumulated["total"] += float(loss.detach()) * batch_count

        two_pass = stage != "v4"
        prediction = predict_transformer(
            model, data, policy, validation_jobs, device, eval_batch_size, two_pass
        )
        metrics, per_axis, _ = score_profile_axis_macro(data, validation_jobs, prediction)
        elapsed = time.monotonic() - epoch_start
        row = {
            "stage": stage,
            "epoch": epoch,
            "training_tasks": len(order),
            "train_first_loss": accumulated["first"] / len(order),
            "train_second_loss": accumulated["second"] / len(order),
            "train_cce_normalized": accumulated["cce"] / len(order),
            "train_total_loss": accumulated["total"] / len(order),
            **{f"validation_{key}": value for key, value in metrics.items()},
            "epoch_seconds": elapsed,
        }
        history.append(row)
        current = metrics["macro_scaled_log_mae"]
        if current < best_metric:
            best_metric, best_state, best_epoch = current, cpu_state(model), epoch
        prediction.to_parquet(output_dir / f"{stage}_validation_predictions_latest.parquet", index=False)
        per_axis.to_csv(output_dir / f"{stage}_validation_axis_metrics_latest.csv", index=False)
        pd.DataFrame(history).to_csv(output_dir / f"{stage}_history.csv", index=False)
        print(
            f"{stage} epoch {epoch}/{epochs}: validation SLMAE={current:.6f}, "
            f"seconds={elapsed:.1f}", flush=True,
        )
    final_metrics = metrics
    save_checkpoint(
        output_dir / f"{stage}_final.pt", model, config, data_sha, stage,
        epochs, final_metrics, policy_sha,
    )
    final_state = cpu_state(model)
    model.load_state_dict(best_state)
    save_checkpoint(
        output_dir / f"{stage}_best_validation.pt", model, config, data_sha,
        stage, best_epoch, {"macro_scaled_log_mae": best_metric}, policy_sha,
    )
    model.load_state_dict(final_state)
    return {
        "stage": stage,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "tasks_per_epoch": int(min(tasks_per_epoch, len(task_rows)) if tasks_per_epoch else len(task_rows)),
        "best_epoch": best_epoch,
        "best_validation_macro_scaled_log_mae": best_metric,
        "final_validation_metrics": final_metrics,
        "elapsed_seconds": time.monotonic() - stage_start,
        "final_checkpoint_sha256": digest(output_dir / f"{stage}_final.pt"),
        "best_checkpoint_sha256": digest(output_dir / f"{stage}_best_validation.pt"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs-v4", type=int, default=20)
    parser.add_argument("--epochs-v5", type=int, default=20)
    parser.add_argument("--epochs-cce", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--cce-weight", type=float, default=0.25)
    parser.add_argument("--cce-temperature", type=float, default=0.5)
    parser.add_argument(
        "--tasks-per-epoch", type=int, default=337048,
        help="axis-balanced task budget; 337048 matches the legacy family-task forward budget",
    )
    parser.add_argument("--max-train-tasks", type=int, default=0)
    parser.add_argument("--max-validation-jobs", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs_v4, args.epochs_v5, args.epochs_cce, args.batch_size) < 1:
        raise ValueError("positive epochs and batch size required")
    if args.tasks_per_epoch < 0:
        raise ValueError("tasks-per-epoch must be zero (all) or positive")
    args.output_dir.mkdir(parents=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.n_jobs)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    config = replace(
        Config(), d_model=192, n_heads=6, n_layers=3,
        feedforward_dim=256, dropout=0.15, batch_size=args.batch_size,
        epochs=20, learning_rate=3e-4, seed=args.seed, n_jobs=args.n_jobs,
    )
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    policy = DependencyMaskPolicy(data.axes)
    task_rows, task_axes = dependency_tasks(data, data.train)
    validation_jobs = data.jobs.sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    pilot = bool(args.max_train_tasks or args.max_validation_jobs)
    if args.max_train_tasks:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(task_rows), min(args.max_train_tasks, len(task_rows)), replace=False)
        task_rows, task_axes = task_rows[selected], task_axes[selected]
    if args.max_validation_jobs:
        validation_jobs = validation_jobs.iloc[:args.max_validation_jobs].copy()
    if not pilot and (len(validation_jobs) != 323_809 or validation_jobs.axis_index.nunique() != 187):
        raise ValueError("full validation panel contract failed")
    if pilot:
        # Pilot must still contain all active task axes in its validation view.
        active = np.unique(task_axes)
        validation_jobs = validation_jobs[validation_jobs.axis_index.isin(active)]
        data.targets = active
    policy_sha = digest(args.policy_manifest)
    data_sha = digest(args.data_dir / "manifest.json")
    manifest = {
        "status": "running", "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": data_sha, "mask_policy_version": MASK_VERSION,
        "mask_policy_manifest_sha256": policy_sha,
        "code_sha256": {
            "scripts/run_dependency_mask_neural_v1.py": digest(Path(__file__)),
            "src/foodcomp/dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/dependency_mask_v1.py"
            ),
            "src/foodcomp/research_dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/research_dependency_mask_v1.py"
            ),
        },
        "food_name_model_input": False, "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "available_training_labels": int(len(task_rows)),
        "tasks_per_epoch": int(
            min(args.tasks_per_epoch, len(task_rows))
            if args.tasks_per_epoch else len(task_rows)
        ),
        "validation_jobs": int(len(validation_jobs)),
        "supervised_axes": int(len(np.unique(task_axes))),
        "pilot": pilot, "device": str(device), "config": asdict(config),
        "protocol": {
            "v4": "from-scratch one-pass V4, 3 layers / 192 hidden / 6 heads",
            "v5": "continue final V4 with detached two-pass FOOD prompt",
            "cce": "continue final V5 with two-view multi-positive CCE",
            "parent_checkpoint_policy": "final epoch of preceding stage",
            "model_selection": "validation only; final epoch remains the next-stage parent",
            "epoch_task_sampling": (
                "axis-balanced sampling from observed training labels; the default "
                "337048-task budget matches the legacy family-task forward-pass budget"
            ),
        },
        "stages": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        model = FoodPromptCceTransformer(len(data.axes), config).to(device)
        stages = (
            ("v4", args.epochs_v4, 3e-4),
            ("v5", args.epochs_v5, 1e-4),
            ("cce", args.epochs_cce, 1e-4),
        )
        for stage, epochs, learning_rate in stages:
            record = train_stage(
                model, data, policy, task_rows, task_axes, validation_jobs,
                device, config, stage, epochs, learning_rate,
                args.eval_batch_size, args.output_dir, data_sha, policy_sha,
                args.cce_weight, args.cce_temperature, args.tasks_per_epoch,
            )
            manifest["stages"].append(record)
            write_json(args.output_dir / "manifest.json", manifest)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

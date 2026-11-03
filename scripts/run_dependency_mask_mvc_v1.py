#!/usr/bin/env python3
"""Continue dependency-mask V5 with an scGPT-style MVC auxiliary objective."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
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

from foodcomp.dependency_mask_v1 import (  # noqa: E402
    DependencyMaskPolicy, VERSION as MASK_VERSION,
)
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    FoodPromptMvcTransformer, axis_balanced_loss, dependency_tasks,
    masked_axis_batch, predict_transformer, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


VERSION = "dependency_mask_v1_v5_mvc_v1"
MVC_KEYS = {
    "mvc_decoder.axis_to_query.weight",
    "mvc_decoder.axis_to_query.bias",
    "mvc_decoder.query_projection.weight",
}


def cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_checkpoint(
    path, model, config, parent, epoch, metrics, mvc_weight, parent_sha
):
    torch.save({
        "version": VERSION,
        "mask_policy_version": parent["mask_policy_version"],
        "mask_policy_sha256": parent["mask_policy_sha256"],
        "stage": "mvc",
        "parent_stage": "v5",
        "parent_checkpoint_sha256": parent_sha,
        "completed_epoch": epoch,
        "data_sha256": parent["data_sha256"],
        "food_name_model_input": False,
        "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "official_inference_pass": "second",
        "mvc_inference_used": False,
        "mvc_weight": mvc_weight,
        "config": asdict(config),
        "validation_metrics": metrics,
        "state_dict": cpu_state(model),
    }, path)


def load_parent(path, data_dir, policy_manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "stage": "v5",
        "mask_policy_version": MASK_VERSION,
        "mask_policy_sha256": digest(policy_manifest),
        "data_sha256": digest(data_dir / "manifest.json"),
        "food_name_model_input": False,
        "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
    }
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"parent checkpoint mismatch for {key}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mvc-weight", type=float, default=0.25)
    parser.add_argument("--tasks-per-epoch", type=int, default=337048)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs, args.learning_rate, args.mvc_weight, args.tasks_per_epoch) <= 0:
        raise ValueError("epochs, learning rate, MVC weight and task budget must be positive")
    args.output_dir.mkdir(parents=True)

    parent = load_parent(args.parent_checkpoint, args.data_dir, args.policy_manifest)
    config = Config(**parent["config"])
    seed = config.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.n_jobs)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    policy = DependencyMaskPolicy(data.axes)
    task_rows, task_axes = dependency_tasks(data, data.train)
    validation_jobs = data.jobs.sort_values(
        ["profile_index", "axis_index"]
    ).reset_index(drop=True)
    if len(validation_jobs) != 323_809 or validation_jobs.axis_index.nunique() != 187:
        raise ValueError("full validation panel contract failed")

    model = FoodPromptMvcTransformer(len(data.axes), config)
    incompatible = model.load_state_dict(parent["state_dict"], strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != MVC_KEYS:
        raise ValueError(f"unexpected parent/model keys: {incompatible}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay
    )

    full_totals = np.bincount(task_axes, minlength=len(data.axes)).astype(np.float32)
    active = np.flatnonzero(full_totals > 0)
    if set(map(int, data.targets)) - set(map(int, active)):
        raise ValueError("a supervised axis has no training task")
    pools = {axis: np.flatnonzero(task_axes == axis) for axis in active}
    rng = np.random.default_rng(seed + 2)
    parent_sha = digest(args.parent_checkpoint)
    manifest = {
        "status": "running",
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": args.parent_checkpoint.name,
        "parent_checkpoint_sha256": parent_sha,
        "parent_stage": "v5",
        "data_sha256": parent["data_sha256"],
        "mask_policy_version": MASK_VERSION,
        "mask_policy_manifest_sha256": parent["mask_policy_sha256"],
        "food_name_model_input": False,
        "source_model_input": False,
        "family_label_model_input": False,
        "legacy_mask_family_used_for_masking": False,
        "device": str(device),
        "config": asdict(config),
        "protocol": {
            "architecture": "V5 plus scGPT-style bilinear MVC decoder",
            "mvc_input": "first-forward FOOD state plus raw target-axis embedding",
            "mvc_target": "same dependency-masked normalized-log target",
            "mvc_loss": "axis-balanced MAE",
            "mvc_weight": args.mvc_weight,
            "completion_loss": "0.5 first-pass MAE + 0.5 second-pass MAE",
            "cce_used": False,
            "official_inference": "unchanged V5 second-pass target-token decoder",
            "tasks_per_epoch": args.tasks_per_epoch,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "model_selection": "validation macro scaled-log MAE",
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    history = []
    best_metric, best_state, best_epoch, best_metrics = math.inf, None, None, None
    started = time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_count = min(args.tasks_per_epoch, len(task_rows))
            quotient, remainder = divmod(epoch_count, len(active))
            selected = []
            for position, axis in enumerate(active):
                count = quotient + (position < remainder)
                pool = pools[int(axis)]
                selected.append(rng.choice(pool, count, replace=count > len(pool)))
            order = rng.permutation(np.concatenate(selected))
            epoch_axes = task_axes[order]
            totals = torch.as_tensor(
                np.bincount(epoch_axes, minlength=len(data.axes)).astype(np.float32),
                device=device,
            ).clamp_min(1.0)
            accumulated = {"first": 0.0, "second": 0.0, "mvc": 0.0, "total": 0.0}
            epoch_start = time.monotonic()
            for start in range(0, len(order), config.batch_size):
                chosen = order[start:start + config.batch_size]
                rows, axes = task_rows[chosen], task_axes[chosen]
                packed = masked_axis_batch(
                    data, rows, axes, policy, device, with_targets=True
                )
                axis_tensor = packed[3][:, 0]
                label = packed[5][:, 0]
                count = len(chosen)
                optimizer.zero_grad(set_to_none=True)
                first, second, mvc, *_ = model.forward_prompt_mvc_details(*packed[:5])
                losses = {
                    name: axis_balanced_loss(
                        prediction[:, 0], label, axis_tensor, totals,
                        len(active), len(order), count,
                    )
                    for name, prediction in (
                        ("first", first), ("second", second), ("mvc", mvc)
                    )
                }
                completion = 0.5 * (losses["first"] + losses["second"])
                loss = completion + args.mvc_weight * losses["mvc"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite MVC training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key in ("first", "second", "mvc"):
                    accumulated[key] += float(losses[key].detach()) * count
                accumulated["total"] += float(loss.detach()) * count

            predictions = predict_transformer(
                model, data, policy, validation_jobs, device,
                args.eval_batch_size, two_pass=True,
            )
            metrics, per_axis, _ = score_profile_axis_macro(
                data, validation_jobs, predictions
            )
            elapsed = time.monotonic() - epoch_start
            row = {
                "epoch": epoch,
                "training_tasks": len(order),
                **{f"train_{key}_loss": value / len(order)
                   for key, value in accumulated.items()},
                **{f"validation_{key}": value for key, value in metrics.items()},
                "epoch_seconds": elapsed,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
            predictions.to_parquet(
                args.output_dir / "validation_predictions_latest.parquet", index=False
            )
            per_axis.to_csv(args.output_dir / "validation_axis_metrics_latest.csv", index=False)
            current = metrics["macro_scaled_log_mae"]
            if current < best_metric:
                best_metric = current
                best_state = cpu_state(model)
                best_epoch = epoch
                best_metrics = metrics
            print(
                f"mvc epoch {epoch}/{args.epochs}: validation SLMAE={current:.6f}, "
                f"mvc_loss={row['train_mvc_loss']:.6f}, seconds={elapsed:.1f}",
                flush=True,
            )

        final_metrics = metrics
        save_checkpoint(
            args.output_dir / "mvc_final.pt", model, config, parent,
            args.epochs, final_metrics, args.mvc_weight, parent_sha,
        )
        model.load_state_dict(best_state, strict=True)
        save_checkpoint(
            args.output_dir / "mvc_best_validation.pt", model, config, parent,
            best_epoch, best_metrics, args.mvc_weight, parent_sha,
        )
        manifest.update({
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "mvc_parameter_count": sum(
                parameter.numel() for parameter in model.mvc_decoder.parameters()
            ),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "final_validation_metrics": final_metrics,
            "best_checkpoint_sha256": digest(args.output_dir / "mvc_best_validation.pt"),
            "final_checkpoint_sha256": digest(args.output_dir / "mvc_final.pt"),
        })
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

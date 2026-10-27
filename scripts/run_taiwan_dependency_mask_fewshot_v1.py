#!/usr/bin/env python3
"""Taiwan few-shot adaptation under dependency-mask v1."""
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

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    attach_taiwan_panel, load_dependency_model,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    FoodPromptCceTransformer, axis_balanced_loss, masked_axis_batch,
    predict_transformer, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SHOTS = (8, 32, 128)
SEEDS = (20261027, 20261028, 20261029)
SPLIT_SEED = 20261026
TEST_PROFILES = 400


def train_once(
    data, jobs, policy, checkpoint_path, policy_manifest, train_rows, test_jobs,
    initialization, seed, device, epochs, learning_rate, batch_size,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config(**saved["config"])
    if initialization == "pretrained":
        model, _, _ = load_dependency_model(
            checkpoint_path, data, policy_manifest, device
        )
    elif initialization == "scratch":
        model = FoodPromptCceTransformer(len(data.axes), config).to(device)
    else:
        raise ValueError(initialization)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    train_jobs = jobs[jobs.profile_index.isin(train_rows)].copy().reset_index(drop=True)
    counts = np.bincount(
        train_jobs.axis_index.to_numpy(dtype=np.int64), minlength=len(data.axes)
    ).astype(np.float32)
    active = np.flatnonzero(counts > 0)
    totals = torch.from_numpy(np.maximum(counts, 1)).to(device)
    rng = np.random.default_rng(seed + 100_000)
    history = []
    task_rows = train_jobs.profile_index.to_numpy(dtype=np.int64)
    task_axes = train_jobs.axis_index.to_numpy(dtype=np.int64)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_jobs))
        first_total = second_total = 0.0
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            rows = task_rows[chosen]
            axes = task_axes[chosen]
            packed = masked_axis_batch(data, rows, axes, policy, device)
            first, second, _, _, _, _ = model.forward_prompt_details(*packed)
            label = torch.from_numpy(data.values[rows, axes].astype(np.float32)).to(device)
            axis_tensor = torch.from_numpy(axes).to(device)
            first_loss = axis_balanced_loss(
                first[:, 0], label, axis_tensor, totals,
                len(active), len(train_jobs), len(chosen),
            )
            second_loss = axis_balanced_loss(
                second[:, 0], label, axis_tensor, totals,
                len(active), len(train_jobs), len(chosen),
            )
            loss = 0.5 * (first_loss + second_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Taiwan few-shot loss")
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
        "initialization": initialization, "seed": seed,
        "train_profiles": len(train_rows), "train_jobs": len(train_jobs),
        "active_training_axes": len(active),
        "active_axis_indices": active.tolist(),
        "epochs": epochs, "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    return model, prediction, metrics, per_axis, pd.DataFrame(history), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zero-shot-predictions", type=Path, required=True)
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
    jobs, novel_rows, panel, axes = attach_taiwan_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    order = np.random.default_rng(SPLIT_SEED).permutation(novel_rows)
    test_rows = np.sort(order[:TEST_PROFILES])
    adaptation_pool = np.sort(order[TEST_PROFILES:])
    test_jobs = jobs[jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    if test_jobs.axis_index.nunique() != 80:
        raise ValueError("fixed TFDA test set does not cover all axes")
    split = panel[["food_id", "profile_index", "english_name_seen_in_pretraining"]].copy()
    split["fewshot_partition"] = np.where(
        split.profile_index.isin(test_rows), "fixed_test",
        np.where(split.profile_index.isin(adaptation_pool), "adaptation_pool", "excluded"),
    )
    split.to_csv(args.output_dir / "profile_split.csv", index=False)
    zero_prediction = pd.read_parquet(args.zero_shot_predictions)[
        ["profile_index", "axis_index", "prediction"]
    ]
    zero_prediction = zero_prediction[zero_prediction.profile_index.isin(test_rows)]
    zero_metrics, zero_axis, _ = score_profile_axis_macro(
        data, test_jobs, zero_prediction
    )
    zero_axis.to_csv(args.output_dir / "zero_shot_axis_metrics.csv", index=False)
    write_json(args.output_dir / "zero_shot_metrics.json", zero_metrics)

    records, predictions, axes_parts = [], [], []
    manifest = {
        "status": "running", "version": "taiwan_dependency_mask_fewshot_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS), "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED, "test_profiles": TEST_PROFILES,
        "test_jobs": len(test_jobs), "test_axes": int(test_jobs.axis_index.nunique()),
        "adaptation_pool_profiles": len(adaptation_pool),
        "zero_shot_metrics": zero_metrics,
        "checkpoint_sha256": digest(args.checkpoint),
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_policy": "fixed epoch 20; no test-based selection",
        "mask_policy": "target plus dependency ancestors and descendants in training and evaluation",
        "runs": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    for seed in SEEDS:
        nested = np.random.default_rng(seed).permutation(adaptation_pool)
        for shots in SHOTS:
            train_rows = np.sort(nested[:shots])
            for initialization in ("pretrained", "scratch"):
                run_started = time.monotonic()
                name = f"{initialization}_shots{shots}_seed{seed}"
                print(f"training {name}", flush=True)
                model, prediction, metrics, per_axis, history, metadata = train_once(
                    data, jobs, policy, args.checkpoint, args.policy_manifest,
                    train_rows, test_jobs, initialization, seed, device,
                    args.epochs, args.learning_rate, args.batch_size,
                )
                history.to_csv(args.output_dir / f"history_{name}.csv", index=False)
                tagged = prediction.copy()
                tagged.insert(0, "seed", seed)
                tagged.insert(0, "shots", shots)
                tagged.insert(0, "initialization", initialization)
                predictions.append(tagged)
                per_axis.insert(0, "seed", seed)
                per_axis.insert(0, "shots", shots)
                per_axis.insert(0, "initialization", initialization)
                axes_parts.append(per_axis)
                record = {
                    "run": name, "initialization": initialization,
                    "shots": shots, "seed": seed, **metrics, **metadata,
                    "elapsed_seconds": time.monotonic() - run_started,
                }
                records.append(record)
                manifest["runs"].append(record)
                write_json(args.output_dir / "manifest.json", manifest)
                print(
                    f"{name}: {metrics['macro_scaled_log_mae']:.6f} / "
                    f"{metrics['macro_scaled_log_rmse']:.6f} / "
                    f"{metrics['macro_raw_mae']:.6f}", flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    runs = pd.DataFrame(records)
    runs.to_csv(args.output_dir / "fewshot_runs.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "test_predictions.parquet", index=False
    )
    pd.concat(axes_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    metric_columns = [
        "macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae",
    ]
    summary = runs.groupby(["initialization", "shots"], as_index=False).agg(
        **{f"{column}_mean": (column, "mean") for column in metric_columns},
        **{f"{column}_sd": (column, "std") for column in metric_columns},
        active_training_axes_min=("active_training_axes", "min"),
        active_training_axes_max=("active_training_axes", "max"),
    )
    summary.to_csv(args.output_dir / "fewshot_summary.csv", index=False)
    paired = runs.pivot(
        index=["shots", "seed"], columns="initialization", values=metric_columns
    )
    paired.columns = [f"{metric}_{initialization}" for metric, initialization in paired.columns]
    paired = paired.reset_index()
    for metric in metric_columns:
        paired[f"{metric}_pretrained_minus_scratch"] = (
            paired[f"{metric}_pretrained"] - paired[f"{metric}_scratch"]
        )
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    manifest.update(
        status="complete", elapsed_seconds=time.monotonic() - started,
        completed_runs=len(runs), summary=summary.to_dict(orient="records"),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

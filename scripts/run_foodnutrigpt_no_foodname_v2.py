#!/usr/bin/env python3
"""Run one no-food-name architecture ablation on the frozen validation protocol.

Each variant starts from the same v1 training configuration and numeric data.
Baselines are fixed by the separate complete v1 control run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config, family_tasks, numeric_only_training_view, score_subset, set_batch,
)
from foodcomp.research_no_foodname_v2 import (  # noqa: E402
    VARIANTS, VariantSetTransformer, training_knots,
)
from run_foodnutrigpt_no_foodname_v1 import predict_transformer, select_rows  # noqa: E402


def train_variant(data, config, variant, axes, train_rows, validation_rows, device, knots):
    tasks, family = family_tasks(data, train_rows)
    if not len(tasks):
        raise ValueError("empty training task panel")
    totals = data.weights[train_rows].sum(axis=0, dtype=np.float64)
    positive_totals = (data.weights[train_rows] *
                       (data.values[train_rows] > 0)).sum(axis=0, dtype=np.float64)
    active = data.targets[totals[data.targets] > 0]
    if not set(axes).issubset(set(active)):
        raise ValueError("selected axes lack training support")
    total_tensor = torch.as_tensor(totals, device=device, dtype=torch.float32).clamp_min(1e-12)
    positive_tensor = torch.as_tensor(positive_totals, device=device,
                                      dtype=torch.float32).clamp_min(1e-12)
    model = VariantSetTransformer(len(data.axes), config, variant, knots).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history = []
    best_score, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(tasks))
        loss_sum = 0.0
        epoch_start = time.monotonic()
        for start in range(0, len(tasks), config.batch_size):
            chosen = order[start:start + config.batch_size]
            packed = set_batch(data, tasks[chosen], family[chosen], device, with_targets=True)
            prediction, logits, amount = model.forward_details(*packed[:3])
            label, target, cell_weight = packed[3:]
            weighted = target * cell_weight / total_tensor
            multiplier = len(tasks) / (len(chosen) * len(active))
            if variant == "hurdle":
                binary = (label > 0).float()
                classification = (weighted * F.binary_cross_entropy_with_logits(
                    logits, binary, reduction="none")).sum() * multiplier
                positive_weight = target * binary * cell_weight / positive_tensor
                positive_error = F.smooth_l1_loss(amount, label, reduction="none")
                regression = (positive_weight * positive_error).sum() * multiplier
                loss = 0.5 * (classification + regression)
            else:
                loss = (weighted * (prediction - label).abs()).sum() * multiplier
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)
        pred = predict_transformer(model, data, axes, device, config.batch_size,
                                   validation_rows)
        metric, _ = score_subset(data, pred, axes, validation_rows=validation_rows)
        primary = metric["nutrition"]["scaled_log_mae"]
        if primary is None:
            primary = metric["all"]["scaled_log_mae"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(tasks),
                        "validation_primary": primary,
                        "epoch_seconds": time.monotonic() - epoch_start})
        print(f"{variant} epoch {epoch}/{config.epochs}: "
              f"train={history[-1]['train_loss']:.6f} validation={primary:.6f} "
              f"seconds={history[-1]['epoch_seconds']:.1f}", flush=True)
        if primary < best_score:
            best_score, best_epoch = primary, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict_transformer(model, data, axes, device, config.batch_size,
                               validation_rows)
    return model, pred, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes", help="comma-separated target axis IDs; default all 187")
    parser.add_argument("--max-train-profiles", type=int, default=0)
    parser.add_argument("--max-validation-profiles", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs, args.batch_size, args.n_jobs) < 1:
        raise ValueError("epochs, batch size and jobs must be positive")
    config = replace(Config(), epochs=args.epochs, batch_size=args.batch_size,
                     n_jobs=args.n_jobs, seed=args.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.n_jobs)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    data = ResearchData(args.data_dir, "quarantined")
    train_rows = select_rows(data.train, args.max_train_profiles, config.seed)
    validation_rows = select_rows(data.validation, args.max_validation_profiles,
                                  config.seed + 1)
    data = numeric_only_training_view(data, train_rows)
    axes = list(map(int, args.axes.split(","))) if args.axes else list(map(int, data.targets))
    if not axes or len(set(axes)) != len(axes) or not set(axes).issubset(set(data.targets)):
        raise ValueError("axes must be distinct supervised IDs")
    selected_jobs = data.jobs[data.jobs.axis_index.isin(axes) &
                              data.jobs.profile_index.isin(validation_rows)]
    if set(selected_jobs.axis_index) != set(axes):
        raise ValueError("validation subset does not cover all selected axes")
    knots = training_knots(data, train_rows) if args.variant == "piecewise_numeric" else None
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "status": "running", "version": "no_foodname_v2",
        "causal_change_from_v1": args.variant,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                        (ROOT / "src/foodcomp/research_no_foodname_v1.py",
                         ROOT / "src/foodcomp/research_no_foodname_v2.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_v1.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_v2.py")},
        "config": asdict(config), "device": str(device), "axes": axes,
        "train_profiles": len(train_rows), "validation_profiles": len(validation_rows),
        "validation_jobs": len(selected_jobs), "complete_test_opened": False,
        "food_name_model_input": False, "source_model_input": False,
        "training_scale": "per-axis positive median from selected train numeric values only",
        "training_weight": "one per observed source-record/axis",
        "pilot_subset": bool(args.max_train_profiles or args.max_validation_profiles or args.axes),
        "piecewise_bins": 8 if knots is not None else None,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        model, prediction, history, best_epoch = train_variant(
            data, config, args.variant, axes, train_rows, validation_rows, device, knots)
        metrics, per_axis = score_subset(data, prediction, axes,
                                         validation_rows=validation_rows)
        torch.save({"version": "no_foodname_v2", "variant": args.variant,
                    "state_dict": model.state_dict(), "config": asdict(config),
                    "data_sha256": manifest["data_sha256"], "best_epoch": best_epoch,
                    "food_name_model_input": False}, args.output_dir / "transformer.pt")
        prediction.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
        per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        write_json(args.output_dir / "metrics.json", metrics)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        best_epoch=best_epoch, metrics=metrics,
                        parameter_count=sum(p.numel() for p in model.parameters()))
        write_json(args.output_dir / "manifest.json", manifest)
        print(f"{args.variant} complete: {metrics['all']['scaled_log_mae']:.6f}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validation-only composition pretraining and RF/XGBoost/KNN comparison.

Example pilot (no frozen test access):
  python scripts/run_foodnutrigpt_no_foodname_v1.py --axes 74,90,143 \
    --max-train-profiles 5000 --max-validation-profiles 300 --epochs 3 \
    --output-dir output/no_foodname_v1_pilot
Omit --axes and both caps for the complete validation experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config, CompositionSetTransformer, baseline_predictions, family_tasks,
    inverse_values, numeric_only_training_view, score_subset, set_batch,
)


def select_rows(rows, cap, seed):
    if cap < 0:
        raise ValueError("row cap must be nonnegative")
    if not cap or cap >= len(rows):
        return np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=cap, replace=False))


@torch.no_grad()
def predict_transformer(model, data, axes, device, batch_size, validation_rows):
    jobs = data.jobs[data.jobs.axis_index.isin(axes)]
    if validation_rows is not None:
        jobs = jobs[jobs.profile_index.isin(validation_rows)]
    output = []
    model.eval()
    for family in sorted(set(data.families[np.asarray(axes, dtype=int)])):
        part = jobs[jobs.mask_family.eq(family)]
        if part.empty:
            continue
        rows, inverse = np.unique(part.profile_index.to_numpy(dtype=np.int64), return_inverse=True)
        blocks = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            packed = set_batch(data, batch_rows, np.repeat(family, len(batch_rows)), device)
            prediction = model(*packed).cpu().numpy()
            blocks.append(prediction)
        transformed = np.concatenate(blocks, axis=0)
        target_axes = part.axis_index.to_numpy(dtype=np.int64)
        raw = inverse_values(transformed[inverse, target_axes], data.scale[target_axes])
        output.append(pd.DataFrame({"profile_index": part.profile_index.to_numpy(dtype=np.int64),
                                    "axis_index": target_axes, "prediction": raw}))
    if not output:
        raise ValueError("selected validation panel is empty")
    return pd.concat(output, ignore_index=True)


def train_transformer(data, config, axes, train_rows, validation_rows, device):
    tasks, family = family_tasks(data, train_rows)
    if not len(tasks):
        raise ValueError("training task panel is empty")
    totals = data.weights[train_rows].sum(axis=0, dtype=np.float64)
    active_targets = data.targets[totals[data.targets] > 0]
    if not set(axes).issubset(set(active_targets)):
        raise ValueError("training subset lacks one or more selected supervised axes")
    axis_total = torch.as_tensor(totals, device=device, dtype=torch.float32).clamp_min(1e-12)
    model = CompositionSetTransformer(len(data.axes), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history = []
    best = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(tasks))
        loss_sum = 0.0
        for start in range(0, len(tasks), config.batch_size):
            chosen = order[start:start + config.batch_size]
            packed = set_batch(data, tasks[chosen], family[chosen], device, with_targets=True)
            prediction = model(*packed[:3])
            label, target, cell_weight = packed[3:]
            error = (prediction - label).abs()
            weighted = target * cell_weight / axis_total
            loss = (weighted * error).sum() * len(tasks) / (len(chosen) * len(active_targets))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite masked-value loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)
        pred = predict_transformer(model, data, axes, device, config.batch_size, validation_rows)
        metric, _ = score_subset(data, pred, axes, validation_rows=validation_rows)
        primary = metric["nutrition"]["scaled_log_mae"]
        if primary is None:
            primary = metric["all"]["scaled_log_mae"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(tasks),
                        "validation_primary": primary})
        print(f"Transformer epoch {epoch}/{config.epochs}: train={history[-1]['train_loss']:.6f} "
              f"validation={primary:.6f}", flush=True)
        if primary < best:
            best = primary
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict_transformer(model, data, axes, device, config.batch_size, validation_rows)
    return model, pred, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes", help="Comma-separated target axis IDs; default is all 187")
    parser.add_argument("--max-train-profiles", type=int, default=0,
                        help="Pilot cap; 0 uses all frozen training profiles")
    parser.add_argument("--max-validation-profiles", type=int, default=0,
                        help="Pilot cap; 0 uses all frozen validation profiles")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--rf-trees", type=int, default=Config.rf_trees)
    parser.add_argument("--xgb-trees", type=int, default=Config.xgb_trees)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs, args.batch_size, args.rf_trees, args.xgb_trees, args.n_jobs) < 1:
        raise ValueError("epochs, batch size, tree counts and jobs must be positive")
    config = replace(Config(), epochs=args.epochs, batch_size=args.batch_size,
                     rf_trees=args.rf_trees, xgb_trees=args.xgb_trees, n_jobs=args.n_jobs)
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
    validation_rows = select_rows(data.validation, args.max_validation_profiles, config.seed + 1)
    data = numeric_only_training_view(data, train_rows)
    axes = list(map(int, args.axes.split(","))) if args.axes else list(map(int, data.targets))
    if not axes or len(set(axes)) != len(axes) or not set(axes).issubset(set(data.targets)):
        raise ValueError("axes must be distinct supervised IDs")
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("empty selected partition")
    selected_jobs = data.jobs[data.jobs.axis_index.isin(axes) &
                              data.jobs.profile_index.isin(validation_rows)]
    if set(selected_jobs.axis_index) != set(axes):
        raise ValueError("validation subset does not cover all selected axes")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"status": "running", "version": "no_foodname_v1",
                "scientific_status": "exploratory validation only",
                "data_sha256": digest(args.data_dir / "manifest.json"),
                "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                (ROOT / "src/foodcomp/research_no_foodname_v1.py",
                                 ROOT / "scripts/run_foodnutrigpt_no_foodname_v1.py")},
                "config": asdict(config), "device": str(device), "axes": axes,
                "train_profiles": len(train_rows), "validation_profiles": len(validation_rows),
                "validation_jobs": len(selected_jobs), "complete_test_opened": False,
                "food_name_model_input": False, "source_model_input": False,
                "training_scale": "per-axis positive median from selected train numeric values only",
                "training_weight": "one per observed source-record/axis; no name-derived weights",
                "pilot_subset": bool(args.max_train_profiles or args.max_validation_profiles or args.axes)}
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        model, neural, history, best_epoch = train_transformer(
            data, config, axes, train_rows, validation_rows, device)
        torch.save({"version": "no_foodname_v1", "state_dict": model.state_dict(),
                    "config": asdict(config), "data_sha256": manifest["data_sha256"],
                    "best_epoch": best_epoch, "food_name_model_input": False},
                   args.output_dir / "transformer.pt")
        pd.DataFrame(history).to_csv(args.output_dir / "transformer_history.csv", index=False)
        results = {}
        for method in ("transformer", "rf", "xgb", "knn"):
            if method == "transformer":
                predictions = neural
                fit = None
            else:
                print(f"Fitting {method} on {len(axes)} axes", flush=True)
                predictions, fit = baseline_predictions(
                    data, method, axes, config, train_rows=train_rows,
                    validation_rows=validation_rows)
            metrics, per_axis = score_subset(data, predictions, axes,
                                             validation_rows=validation_rows)
            target = args.output_dir / method
            target.mkdir()
            predictions.to_parquet(target / "validation_predictions.parquet", index=False)
            per_axis.to_csv(target / "axis_metrics.csv", index=False)
            if fit is not None:
                fit.to_csv(target / "fit_support.csv", index=False)
            write_json(target / "metrics.json", metrics)
            results[method] = metrics
            print(f"{method}: {metrics['all']['scaled_log_mae']:.6f}", flush=True)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        best_epoch=best_epoch, metrics=results)
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

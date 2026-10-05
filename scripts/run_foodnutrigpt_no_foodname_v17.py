#!/usr/bin/env python3
"""Train one ReGLU masked-axis Transformer with Talking-Heads Attention."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config, family_tasks, numeric_only_training_view, score_subset,
)
from foodcomp.research_no_foodname_v3 import masked_axis_batch  # noqa: E402
from foodcomp.research_no_foodname_v17 import (  # noqa: E402
    TalkingHeadsMaskedAxisTransformer,
)
from run_foodnutrigpt_no_foodname_v1 import select_rows  # noqa: E402
from run_foodnutrigpt_no_foodname_v3 import predict_masked_axis  # noqa: E402


def train_model(data, config, axes, train_rows, validation_rows, device):
    tasks, families = family_tasks(data, train_rows)
    totals = data.weights[train_rows].sum(axis=0, dtype=np.float64)
    active = data.targets[totals[data.targets] > 0]
    if not len(tasks) or not set(axes).issubset(set(active)):
        raise ValueError("training tasks or supported axes are incomplete")
    total_tensor = torch.as_tensor(
        totals, device=device, dtype=torch.float32
    ).clamp_min(1e-12)
    model = TalkingHeadsMaskedAxisTransformer(len(data.axes), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    history = []
    best_score, best_epoch, best_state = float("inf"), None, None
    identity = torch.eye(config.n_heads, device=device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(tasks))
        loss_sum = 0.0
        epoch_start = time.monotonic()
        for start in range(0, len(tasks), config.batch_size):
            chosen = order[start:start + config.batch_size]
            packed = masked_axis_batch(
                data, tasks[chosen], families[chosen], device, with_targets=True
            )
            prediction = model(*packed[:5])
            target_axis, target_padding = packed[3:5]
            label, observed, cell_weight = packed[5:]
            weighted = observed * cell_weight / total_tensor[target_axis]
            weighted = weighted.masked_fill(target_padding, 0.0)
            multiplier = len(tasks) / (len(chosen) * len(active))
            loss = (weighted * (prediction - label).abs()).sum() * multiplier
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite target-query training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)

        prediction = predict_masked_axis(
            model, data, axes, device, config.batch_size, validation_rows
        )
        metrics, _ = score_subset(
            data, prediction, axes, validation_rows=validation_rows
        )
        primary = metrics["nutrition"]["scaled_log_mae"]
        if primary is None:
            primary = metrics["all"]["scaled_log_mae"]
        mixing_rms = float(np.mean([
            torch.cat((
                (block.attention.pre_softmax_mix - identity).flatten(),
                (block.attention.post_softmax_mix - identity).flatten(),
            )).square().mean().sqrt().item()
            for block in model.blocks
        ]))
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(tasks),
            "validation_primary": primary,
            "validation_all": metrics["all"]["scaled_log_mae"],
            "head_mixing_delta_rms": mixing_rms,
            "epoch_seconds": time.monotonic() - epoch_start,
        })
        print(
            f"talking_heads epoch {epoch}/{config.epochs}: "
            f"train={history[-1]['train_loss']:.6f} "
            f"nutrition={primary:.6f} "
            f"all={history[-1]['validation_all']:.6f} "
            f"mix_rms={mixing_rms:.5f} "
            f"seconds={history[-1]['epoch_seconds']:.1f}",
            flush=True,
        )
        if primary < best_score:
            best_score, best_epoch = primary, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    prediction = predict_masked_axis(
        model, data, axes, device, config.batch_size, validation_rows
    )
    return model, prediction, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes")
    parser.add_argument("--max-train-profiles", type=int, default=0)
    parser.add_argument("--max-validation-profiles", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs, args.batch_size, args.n_jobs) < 1:
        raise ValueError("positive training arguments required")

    config = replace(
        Config(), epochs=args.epochs, seed=args.seed,
        batch_size=args.batch_size, n_jobs=args.n_jobs,
    )
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.n_jobs)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    data = ResearchData(args.data_dir, "quarantined")
    train_rows = select_rows(data.train, args.max_train_profiles, config.seed)
    validation_rows = select_rows(
        data.validation, args.max_validation_profiles, config.seed + 1
    )
    data = numeric_only_training_view(data, train_rows)
    axes = (
        list(map(int, args.axes.split(","))) if args.axes
        else list(map(int, data.targets))
    )
    if (not axes or len(set(axes)) != len(axes)
            or not set(axes).issubset(set(data.targets))):
        raise ValueError("axes must be distinct supervised IDs")
    selected_jobs = data.jobs[
        data.jobs.axis_index.isin(axes)
        & data.jobs.profile_index.isin(validation_rows)
    ]
    if set(selected_jobs.axis_index) != set(axes):
        raise ValueError("validation subset does not cover all selected axes")

    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    source_files = (
        ROOT / "src/foodcomp/research_no_foodname_v1.py",
        ROOT / "src/foodcomp/research_no_foodname_v3.py",
        ROOT / "src/foodcomp/research_no_foodname_v7.py",
        ROOT / "src/foodcomp/research_no_foodname_v17.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v3.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v17.py",
    )
    manifest = {
        "status": "running",
        "version": "no_foodname_v17_talking_heads_reglu",
        "causal_change_from_v7": (
            "identity-initialized learned linear mixing across attention heads "
            "immediately before and after softmax; all else fixed"
        ),
        "literature_basis": "Shazeer et al., 2020, Talking-Heads Attention",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "code_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in source_files
        },
        "config": asdict(config),
        "device": str(device),
        "axes": axes,
        "train_profiles": len(train_rows),
        "validation_profiles": len(validation_rows),
        "validation_jobs": len(selected_jobs),
        "complete_test_opened": False,
        "food_name_model_input": False,
        "source_model_input": False,
        "prediction_source": "one scalar decoded from each masked target axis token",
        "context_token_role": "sample representation only",
        "ffn_activation": "ReGLU",
        "attention": "pre/post-softmax cross-head linear mixing",
        "head_mixing_initialization": "identity",
        "teacher_models": 0,
        "distillation": False,
        "deployed_model_count": 1,
        "selection_metric": "nutrition scaled_log_mae",
        "pilot_subset": bool(
            args.max_train_profiles or args.max_validation_profiles or args.axes
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        model, prediction, history, best_epoch = train_model(
            data, config, axes, train_rows, validation_rows, device
        )
        metrics, per_axis = score_subset(
            data, prediction, axes, validation_rows=validation_rows
        )
        torch.save({
            "version": manifest["version"],
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "data_sha256": manifest["data_sha256"],
            "best_epoch": best_epoch,
            "food_name_model_input": False,
            "source_model_input": False,
            "architecture": "TalkingHeadsMaskedAxisTransformer",
            "distillation": False,
        }, args.output_dir / "transformer.pt")
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        prediction.to_parquet(
            args.output_dir / "validation_predictions.parquet", index=False
        )
        per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
        write_json(args.output_dir / "metrics.json", metrics)
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            best_epoch=best_epoch,
            metrics=metrics,
            parameter_count=sum(
                parameter.numel() for parameter in model.parameters()
            ),
        )
        write_json(args.output_dir / "manifest.json", manifest)
        print(
            f"talking_heads complete: {metrics['all']['scaled_log_mae']:.6f}",
            flush=True,
        )
    except Exception as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

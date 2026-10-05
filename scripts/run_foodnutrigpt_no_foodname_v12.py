#!/usr/bin/env python3
"""Distill complementary masked-axis teachers into one ReGLU student."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    Config, family_tasks, numeric_only_training_view, score_subset,
)
from foodcomp.research_no_foodname_v3 import (  # noqa: E402
    MaskedAxisTokenTransformer, masked_axis_batch,
)
from foodcomp.research_no_foodname_v7 import GatedMaskedAxisTransformer  # noqa: E402
from foodcomp.research_no_foodname_v12 import (  # noqa: E402
    distillation_axis_totals, macro_axis_mae, write_teacher_cache,
)
from run_foodnutrigpt_no_foodname_v1 import select_rows  # noqa: E402
from run_foodnutrigpt_no_foodname_v3 import predict_masked_axis  # noqa: E402


DEFAULT_TEACHERS = (
    ROOT / "output/no_foodname_masked_axis_20ep_20261005/transformer.pt",
    ROOT / "output/no_foodname_masked_axis_20ep_seed20261006/transformer.pt",
    ROOT / "output/no_foodname_masked_axis_20ep_seed20261007/transformer.pt",
    ROOT / "output/no_foodname_v7_reglu_20ep_20261005/transformer.pt",
)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_teachers(checkpoints, data_sha256, data, device):
    """Load only audited full-validation, no-name teacher checkpoints."""
    teachers, metadata = [], []
    for checkpoint_path in checkpoints:
        checkpoint_path = Path(checkpoint_path).resolve()
        manifest_path = checkpoint_path.parent / "manifest.json"
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"teacher is incomplete: {checkpoint_path}")
        if manifest.get("pilot_subset") is not False:
            raise ValueError(f"teacher is not a full run: {checkpoint_path}")
        if manifest.get("complete_test_opened") is not False:
            raise ValueError(f"teacher opened complete test: {checkpoint_path}")
        if manifest.get("food_name_model_input") is not False:
            raise ValueError(f"teacher used food name: {checkpoint_path}")
        if manifest.get("source_model_input") is not False:
            raise ValueError(f"teacher used source identity: {checkpoint_path}")
        if manifest.get("data_sha256") != data_sha256:
            raise ValueError(f"teacher data hash differs: {checkpoint_path}")
        if manifest.get("axes") != list(map(int, data.targets)):
            raise ValueError(f"teacher target panel differs: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("data_sha256") != data_sha256:
            raise ValueError(f"checkpoint data hash differs: {checkpoint_path}")
        if checkpoint.get("food_name_model_input") is not False:
            raise ValueError(f"checkpoint used food name: {checkpoint_path}")
        config = Config(**checkpoint["config"])
        version = checkpoint.get("version")
        if version == "no_foodname_v3_masked_axis":
            model = MaskedAxisTokenTransformer(len(data.axes), config)
        elif version == "no_foodname_v7_reglu_masked_axis":
            model = GatedMaskedAxisTransformer(len(data.axes), config)
        else:
            raise ValueError(f"unsupported teacher version: {version}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval().requires_grad_(False)
        teachers.append(model)
        metadata.append({
            "version": version,
            "seed": config.seed,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": digest(checkpoint_path),
            "validation_all_scaled_log_mae":
                manifest["metrics"]["all"]["scaled_log_mae"],
        })
    if len({item["checkpoint_sha256"] for item in metadata}) != len(metadata):
        raise ValueError("duplicate teacher checkpoints")
    return teachers, metadata


def supervised_loss(data, packed, prediction, total_tensor, task_count, active_count):
    target_axis, target_padding = packed[3:5]
    label, observed, cell_weight = packed[5:]
    weighted = observed * cell_weight / total_tensor[target_axis]
    weighted = weighted.masked_fill(target_padding, 0.0)
    multiplier = task_count / (len(prediction) * active_count)
    return (weighted * (prediction - label).abs()).sum() * multiplier


def train_student(
    data, config, axes, train_rows, validation_rows, device, teacher_targets,
):
    tasks, families = family_tasks(data, train_rows)
    if teacher_targets.shape[0] != len(tasks):
        raise ValueError("teacher cache does not match training tasks")
    totals = data.weights[train_rows].sum(axis=0, dtype=np.float64)
    active = data.targets[totals[data.targets] > 0]
    if not len(tasks) or not set(axes).issubset(set(active)):
        raise ValueError("training tasks or supported axes are incomplete")
    total_tensor = torch.as_tensor(
        totals, device=device, dtype=torch.float32
    ).clamp_min(1e-12)
    distill_totals = distillation_axis_totals(data, families)
    if (distill_totals[active] <= 0).any():
        raise ValueError("an active supervised axis lacks distillation tasks")
    distill_total_tensor = torch.as_tensor(
        distill_totals,
        device=device, dtype=torch.float32,
    )
    model = GatedMaskedAxisTransformer(len(data.axes), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    history = []
    best_score, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(tasks))
        loss_sum = supervised_sum = distill_sum = 0.0
        epoch_start = time.monotonic()
        for start in range(0, len(tasks), config.batch_size):
            chosen = order[start:start + config.batch_size]
            packed = masked_axis_batch(
                data, tasks[chosen], families[chosen], device, with_targets=True
            )
            prediction = model(*packed[:5])
            width = prediction.shape[1]
            teacher = torch.from_numpy(
                np.asarray(teacher_targets[chosen, :width]).copy()
            ).to(device)
            teacher = teacher.masked_fill(packed[4], 0.0)
            label_loss = supervised_loss(
                data, packed, prediction, total_tensor, len(tasks), len(active)
            )
            teacher_loss = macro_axis_mae(
                prediction, teacher, packed[3], packed[4], distill_total_tensor,
                task_count=len(tasks), active_axis_count=len(active),
            )
            loss = 0.5 * (label_loss + teacher_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite student training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_count = len(chosen)
            loss_sum += float(loss.detach()) * batch_count
            supervised_sum += float(label_loss.detach()) * batch_count
            distill_sum += float(teacher_loss.detach()) * batch_count

        prediction_frame = predict_masked_axis(
            model, data, axes, device, config.batch_size, validation_rows
        )
        metrics, _ = score_subset(
            data, prediction_frame, axes, validation_rows=validation_rows
        )
        primary = metrics["nutrition"]["scaled_log_mae"]
        if primary is None:
            primary = metrics["all"]["scaled_log_mae"]
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(tasks),
            "supervised_loss": supervised_sum / len(tasks),
            "distillation_loss": distill_sum / len(tasks),
            "validation_primary": primary,
            "validation_all": metrics["all"]["scaled_log_mae"],
            "epoch_seconds": time.monotonic() - epoch_start,
        })
        print(
            f"distilled_student epoch {epoch}/{config.epochs}: "
            f"train={history[-1]['train_loss']:.6f} "
            f"supervised={history[-1]['supervised_loss']:.6f} "
            f"teacher={history[-1]['distillation_loss']:.6f} "
            f"nutrition={primary:.6f} "
            f"all={history[-1]['validation_all']:.6f} "
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
    prediction_frame = predict_masked_axis(
        model, data, axes, device, config.batch_size, validation_rows
    )
    return model, prediction_frame, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", action="append", type=Path)
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

    data_sha256 = digest(args.data_dir / "manifest.json")
    checkpoints = args.teacher_checkpoint or DEFAULT_TEACHERS
    teachers, teacher_metadata = load_teachers(
        checkpoints, data_sha256, data, device
    )
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    source_files = (
        ROOT / "src/foodcomp/research_no_foodname_v1.py",
        ROOT / "src/foodcomp/research_no_foodname_v3.py",
        ROOT / "src/foodcomp/research_no_foodname_v7.py",
        ROOT / "src/foodcomp/research_no_foodname_v12.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v3.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v12.py",
    )
    manifest = {
        "status": "running",
        "version": "no_foodname_v12_single_student_distillation",
        "causal_change_from_v7": (
            "equal-weight dense scaled-log teacher targets are combined with "
            "observed-label loss while training one fresh ReGLU student"
        ),
        "data_sha256": data_sha256,
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
        "student_architecture": "one GatedMaskedAxisTransformer",
        "deployed_model_count": 1,
        "teacher_dependency_at_inference": False,
        "prediction_source": "one scalar from each masked target axis token",
        "context_token_role": "sample representation only",
        "teacher_targets": "equal mean in train-only scaled-log space",
        "distillation_scope": "training row-family tasks only",
        "loss_mix": {"observed_label_macro_mae": 0.5,
                     "dense_teacher_macro_mae": 0.5},
        "teachers": teacher_metadata,
        "selection_metric": "nutrition scaled_log_mae",
        "pilot_subset": bool(
            args.max_train_profiles or args.max_validation_profiles or args.axes
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        tasks, families = family_tasks(data, train_rows)
        cache_path = args.output_dir / "teacher_targets.npy"
        cache_started = time.monotonic()
        cache_shape = write_teacher_cache(
            cache_path, teachers, data, tasks, families, device, config.batch_size
        )
        manifest.update(
            teacher_cache_shape=list(cache_shape),
            teacher_cache_sha256=digest(cache_path),
            teacher_cache_seconds=time.monotonic() - cache_started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        del teachers
        if device.type == "cuda":
            torch.cuda.empty_cache()
        teacher_targets = np.load(cache_path, mmap_mode="r")
        model, prediction, history, best_epoch = train_student(
            data, config, axes, train_rows, validation_rows, device,
            teacher_targets,
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
            "architecture": "GatedMaskedAxisTransformer",
            "teacher_dependency_at_inference": False,
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
            f"single student complete: {metrics['all']['scaled_log_mae']:.6f}",
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

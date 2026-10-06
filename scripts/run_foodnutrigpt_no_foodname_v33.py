#!/usr/bin/env python3
"""Fit a train-only axis shrinkage head on the fixed 20-epoch v18 model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
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
from foodcomp.research_no_foodname_v16 import (  # noqa: E402
    AxisPairBiasMaskedAxisTransformer,
)
from foodcomp.research_no_foodname_v33 import (  # noqa: E402
    TrainCalibratedShrinkageTransformer, fit_axis_retention,
)
from run_foodnutrigpt_no_foodname_v1 import select_rows  # noqa: E402
from run_foodnutrigpt_no_foodname_v3 import predict_masked_axis  # noqa: E402


def training_completion_observations(
    model, data, train_rows, device, batch_size
):
    """Collect only observed training targets from the fixed family tasks."""
    tasks, families = family_tasks(data, train_rows)
    if not len(tasks):
        raise ValueError("empty calibration task panel")
    axis_parts, prediction_parts, target_parts, weight_parts = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            packed = masked_axis_batch(
                data, tasks[start:start + batch_size],
                families[start:start + batch_size], device,
                with_targets=True,
            )
            prediction = model(*packed[:5])
            target_axis, target_padding = packed[3:5]
            target, observed, weight = packed[5:]
            keep = observed & ~target_padding
            if keep.any():
                axis_parts.append(target_axis[keep].cpu().numpy())
                prediction_parts.append(prediction[keep].cpu().numpy())
                target_parts.append(target[keep].cpu().numpy())
                weight_parts.append(weight[keep].cpu().numpy())
    if not axis_parts:
        raise ValueError("calibration task panel has no observed targets")
    return tuple(map(np.concatenate, (
        prediction_parts, target_parts, weight_parts, axis_parts,
    )))


def train_only_axis_prior(data, train_rows):
    prior = np.zeros(len(data.axes), dtype=np.float32)
    for axis_index in data.targets:
        observed = data.observed[train_rows, axis_index]
        values = data.values[train_rows, axis_index][observed]
        if len(values):
            prior[axis_index] = np.median(values)
    return prior


def load_parent(parent_dir, axis_count, device):
    manifest_path = parent_dir / "manifest.json"
    checkpoint_path = parent_dir / "transformer.pt"
    if not manifest_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("parent manifest or checkpoint is missing")
    import json
    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "complete":
        raise ValueError("parent experiment is not complete")
    if parent_manifest.get("complete_test_opened") is not False:
        raise ValueError("parent complete-test state is not closed")
    if parent_manifest.get("food_name_model_input") is not False:
        raise ValueError("parent model used food-name input")
    if parent_manifest.get("selection_metric") != "all187 macro-axis scaled_log_mae":
        raise ValueError("parent checkpoint selection metric differs")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = Config(**checkpoint["config"])
    expected = {
        "d_model": 192, "n_heads": 6, "n_layers": 3,
        "feedforward_dim": 256, "dropout": 0.15,
        "batch_size": 128, "epochs": 20, "learning_rate": 3e-4,
        "weight_decay": 1e-4, "seed": 20261005,
    }
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        raise ValueError(f"parent training configuration changed: {actual}")
    if checkpoint.get("architecture") != "AxisPairBiasMaskedAxisTransformer":
        raise ValueError("parent checkpoint architecture differs")
    model = AxisPairBiasMaskedAxisTransformer(axis_count, config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint, parent_manifest, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1",
    )
    parser.add_argument(
        "--parent-output-dir", type=Path,
        default=ROOT / "output/no_foodname_v18_unified187_20ep_20261006",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes")
    parser.add_argument("--max-train-profiles", type=int, default=0)
    parser.add_argument("--max-validation-profiles", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    data = ResearchData(args.data_dir, "quarantined")
    model, parent_checkpoint, parent_manifest, config = load_parent(
        args.parent_output_dir, len(data.axes), device
    )
    torch.set_num_threads(config.n_jobs)
    train_rows = select_rows(data.train, args.max_train_profiles, config.seed)
    validation_rows = select_rows(
        data.validation, args.max_validation_profiles, config.seed + 1
    )
    # The parent checkpoint was trained with the complete training partition's
    # scale. A calibration-row subset must not silently change its input space.
    data = numeric_only_training_view(data, data.train)
    if digest(args.data_dir / "manifest.json") != parent_checkpoint["data_sha256"]:
        raise ValueError("parent checkpoint data hash differs")
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
        ROOT / "src/foodcomp/research_no_foodname_v16.py",
        ROOT / "src/foodcomp/research_no_foodname_v33.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v3.py",
        ROOT / "scripts/run_foodnutrigpt_no_foodname_v33.py",
    )
    manifest = {
        "status": "running",
        "version": "no_foodname_v33_train_only_axis_shrinkage",
        "parent_version": parent_manifest["version"],
        "parent_output_dir": str(args.parent_output_dir),
        "parent_checkpoint_sha256": digest(
            args.parent_output_dir / "transformer.pt"
        ),
        "causal_change_from_v18": (
            "append one bounded per-axis shrinkage readout fitted analytically "
            "from v18 predictions and labels on training rows only; the 20-epoch "
            "Transformer checkpoint and all training parameters are unchanged"
        ),
        "registered_capacity_changes": {
            "d_model": [192, 192], "n_heads": [6, 6],
            "n_layers": [3, 3], "dropout": [0.15, 0.15],
        },
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
        "prediction_source": (
            "single Transformer target-token prediction shrunk toward a "
            "train-only axis median"
        ),
        "context_token_role": "sample representation only",
        "ffn_activation": "ReGLU",
        "attention": "per-layer ordered axis-pair logit bias",
        "calibration_fit_rows": "training only",
        "calibration_validation_labels_used": False,
        "calibration_objective": (
            "exact weighted-L1 per-axis retention constrained to [0, 1]"
        ),
        "family_labels_used_by_model": False,
        "teacher_models": 0,
        "distillation": False,
        "deployed_model_count": 1,
        "selection_metric": "all187 macro-axis scaled_log_mae",
        "parent_best_epoch": parent_checkpoint["best_epoch"],
        "pilot_subset": bool(
            args.max_train_profiles or args.max_validation_profiles or args.axes
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        prior = train_only_axis_prior(data, train_rows)
        base_prediction, target, weight, target_axis = (
            training_completion_observations(
                model, data, train_rows, device, config.batch_size
            )
        )
        retention, support = fit_axis_retention(
            base_prediction, target, weight, target_axis, prior, len(data.axes)
        )
        calibrated = TrainCalibratedShrinkageTransformer(
            len(data.axes), config, prior, retention
        ).to(device)
        incompatible = calibrated.load_state_dict(
            parent_checkpoint["state_dict"], strict=False
        )
        if (set(incompatible.missing_keys) != {"axis_prior", "axis_retention"}
                or incompatible.unexpected_keys):
            raise ValueError(f"parent checkpoint keys differ: {incompatible}")
        prediction = predict_masked_axis(
            calibrated, data, axes, device, config.batch_size, validation_rows
        )
        metrics, per_axis = score_subset(
            data, prediction, axes, validation_rows=validation_rows
        )
        torch.save({
            "version": manifest["version"],
            "state_dict": calibrated.state_dict(),
            "config": asdict(config),
            "data_sha256": manifest["data_sha256"],
            "best_epoch": parent_checkpoint["best_epoch"],
            "food_name_model_input": False,
            "source_model_input": False,
            "architecture": "TrainCalibratedShrinkageTransformer",
            "distillation": False,
        }, args.output_dir / "transformer.pt")
        pd.read_csv(args.parent_output_dir / "history.csv").to_csv(
            args.output_dir / "history.csv", index=False
        )
        prediction.to_parquet(
            args.output_dir / "validation_predictions.parquet", index=False
        )
        per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
        pd.DataFrame({
            "axis_index": np.arange(len(data.axes)),
            "axis_prior": prior,
            "axis_retention": retention,
            "calibration_support": support,
        }).to_csv(args.output_dir / "axis_calibration.csv", index=False)
        write_json(args.output_dir / "metrics.json", metrics)
        target_retention = retention[data.targets]
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            best_epoch=parent_checkpoint["best_epoch"],
            metrics=metrics,
            calibration_observations=int(len(target_axis)),
            calibrated_axes=int(np.count_nonzero(support[data.targets])),
            retention_min=float(target_retention.min()),
            retention_mean=float(target_retention.mean()),
            retention_median=float(np.median(target_retention)),
            retention_max=float(target_retention.max()),
            parameter_count=sum(
                parameter.numel() for parameter in calibrated.parameters()
            ),
        )
        write_json(args.output_dir / "manifest.json", manifest)
        print(
            f"v33 train-only shrinkage complete: "
            f"{metrics['all']['scaled_log_mae']:.6f}", flush=True,
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

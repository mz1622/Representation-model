#!/usr/bin/env python3
"""Neural-only longer-budget replication of a no-name validation experiment.

The RF/XGBoost/KNN comparison is computed once in the complete v1 control run.
"""
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
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view, score_subset  # noqa: E402
from foodcomp.research_no_foodname_v2 import VARIANTS, training_knots  # noqa: E402
from run_foodnutrigpt_no_foodname_v1 import select_rows, train_transformer  # noqa: E402
from run_foodnutrigpt_no_foodname_v2 import train_variant  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control",) + VARIANTS, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(args.epochs, args.batch_size, args.n_jobs) < 1:
        raise ValueError("epochs, batch size and jobs must be positive")
    config = replace(Config(), epochs=args.epochs, seed=args.seed,
                     batch_size=args.batch_size, n_jobs=args.n_jobs)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.n_jobs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ResearchData(args.data_dir, "quarantined")
    train_rows = select_rows(data.train, 0, config.seed)
    validation_rows = select_rows(data.validation, 0, config.seed + 1)
    data = numeric_only_training_view(data, train_rows)
    axes = list(map(int, data.targets))
    knots = training_knots(data, train_rows) if args.variant == "piecewise_numeric" else None
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "status": "running", "version": "no_foodname_extended",
        "variant": args.variant, "comparison_stage": "longer training budget",
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                        (ROOT / "src/foodcomp/research_no_foodname_v1.py",
                         ROOT / "src/foodcomp/research_no_foodname_v2.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_v1.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_v2.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_extended.py")},
        "config": asdict(config), "device": str(device), "axes": axes,
        "train_profiles": len(train_rows), "validation_profiles": len(validation_rows),
        "validation_jobs": len(data.jobs), "complete_test_opened": False,
        "food_name_model_input": False, "source_model_input": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        if args.variant == "control":
            model, prediction, history, best_epoch = train_transformer(
                data, config, axes, train_rows, validation_rows, device)
        else:
            model, prediction, history, best_epoch = train_variant(
                data, config, args.variant, axes, train_rows, validation_rows,
                device, knots)
        metrics, per_axis = score_subset(data, prediction, axes,
                                         validation_rows=validation_rows)
        torch.save({"variant": args.variant, "state_dict": model.state_dict(),
                    "config": asdict(config), "data_sha256": manifest["data_sha256"],
                    "best_epoch": best_epoch, "food_name_model_input": False},
                   args.output_dir / "transformer.pt")
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        prediction.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
        per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
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

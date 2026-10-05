#!/usr/bin/env python3
"""Width-matched v1 control for the piecewise numeric embedding experiment."""
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
from run_foodnutrigpt_no_foodname_v1 import train_transformer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    config = replace(Config(), d_model=176, feedforward_dim=352, epochs=20,
                     seed=args.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.n_jobs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data/processed/foodnutrigpt_v9_r0_v1"
    data = numeric_only_training_view(ResearchData(data_dir, "quarantined"))
    axes = list(map(int, data.targets))
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "status": "running", "version": "no_foodname_capacity_control",
        "reason": "match piecewise embedding 620412 parameters with 620476 control parameters",
        "data_sha256": digest(data_dir / "manifest.json"),
        "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                        (ROOT / "src/foodcomp/research_no_foodname_v1.py",
                         ROOT / "scripts/run_foodnutrigpt_no_foodname_v1.py",
                         ROOT / "scripts/run_nofoodname_capacity_control.py")},
        "config": asdict(config), "device": str(device), "axes": axes,
        "train_profiles": len(data.train), "validation_profiles": len(data.validation),
        "validation_jobs": len(data.jobs), "complete_test_opened": False,
        "food_name_model_input": False, "source_model_input": False,
    }
    write_json(output / "manifest.json", manifest)
    try:
        model, prediction, history, best_epoch = train_transformer(
            data, config, axes, data.train, data.validation, device)
        metrics, per_axis = score_subset(data, prediction, axes,
                                         validation_rows=data.validation)
        torch.save({"variant": "capacity_control", "state_dict": model.state_dict(),
                    "config": asdict(config), "data_sha256": manifest["data_sha256"],
                    "best_epoch": best_epoch, "food_name_model_input": False},
                   output / "transformer.pt")
        prediction.to_parquet(output / "validation_predictions.parquet", index=False)
        per_axis.to_csv(output / "axis_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "metrics.json", metrics)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                        best_epoch=best_epoch, metrics=metrics,
                        parameter_count=sum(p.numel() for p in model.parameters()))
        write_json(output / "manifest.json", manifest)
        print(f"capacity control complete: {metrics['all']['scaled_log_mae']:.6f}",
              flush=True)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Append one validation-first FoodNutriGPT V8/V2 experiment to the ledger.

Colab:
    python scripts/record_foodnutrigpt_v8_v2_experiment.py \
      --experiment-id all_axis_20_epoch_schedule \
      --output-dir output/... \
      --parent-id baseline_unified_all_axis \
      --changed-factor "optimization horizon: 8 to 20 epochs" \
      --hypothesis "The baseline remained underfit at epoch eight."
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments/foodnutrigpt_v8_v2/experiment_ledger.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--changed-factor", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--conclusion", default="Pending causal interpretation.")
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing experiment summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("complete_test_opened"):
        raise ValueError("Candidate experiments must keep the complete test panel unopened.")
    validation = summary.get("validation_metrics")
    if not validation:
        raise ValueError("Validation metrics are required before recording an experiment.")

    with LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise ValueError(f"Experiment ledger has no header: {LEDGER}")
    if any(row["experiment_id"] == args.experiment_id for row in rows):
        raise ValueError(f"Experiment ID already exists in ledger: {args.experiment_id}")
    record = {
        "experiment_id": args.experiment_id,
        "status": "validation_complete",
        "parent_id": args.parent_id,
        "changed_factor": args.changed_factor,
        "hypothesis": args.hypothesis,
        "selection_protocol": "fixed source-unknown validation; complete test unopened",
        "complete_test_opened": "false",
        "best_validation_joint_hurdle_loss": summary["best_validation_joint_hurdle_loss"],
        "validation_macro_axis_log_mae": validation["macro_axis_log_mae"],
        "validation_macro_axis_log_rmse": validation["macro_axis_log_rmse"],
        "validation_macro_axis_raw_mae_g_per_100g": validation["macro_axis_raw_mae_g_per_100g"],
        "test_macro_axis_log_mae": "",
        "test_macro_axis_log_rmse": "",
        "test_macro_axis_raw_mae_g_per_100g": "",
        "conclusion": args.conclusion,
        "output_dir": str(args.output_dir),
    }
    with LEDGER.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

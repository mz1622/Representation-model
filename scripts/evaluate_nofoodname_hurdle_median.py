#!/usr/bin/env python3
"""Re-score a frozen hurdle checkpoint with its MAE-aligned median readout."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view, score_subset  # noqa: E402
from foodcomp.research_no_foodname_v2 import VariantSetTransformer  # noqa: E402
from run_foodnutrigpt_no_foodname_v1 import predict_transformer  # noqa: E402


class MedianReadout(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, axis, value, padding):
        _, logits, amount = self.model.forward_details(axis, value, padding)
        return torch.where(logits >= 0, amount, torch.zeros_like(amount))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed/foodnutrigpt_v9_r0_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    origin = args.checkpoint_dir / "manifest.json"
    import json
    parent = json.loads(origin.read_text(encoding="utf-8"))
    if parent["status"] != "complete" or parent["causal_change_from_v1"] != "hurdle":
        raise ValueError("checkpoint directory must contain completed hurdle run")
    data = ResearchData(args.data_dir, "quarantined")
    data = numeric_only_training_view(data, data.train)
    if digest(args.data_dir / "manifest.json") != parent["data_sha256"]:
        raise ValueError("data hash differs from training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(**parent["config"])
    checkpoint = torch.load(args.checkpoint_dir / "transformer.pt", map_location=device,
                            weights_only=False)
    if checkpoint["variant"] != "hurdle" or checkpoint["data_sha256"] != parent["data_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    model = VariantSetTransformer(len(data.axes), config, "hurdle").to(device)
    model.load_state_dict(checkpoint["state_dict"])
    axes = list(map(int, data.targets))
    prediction = predict_transformer(MedianReadout(model).eval(), data, axes,
                                     device, config.batch_size, data.validation)
    metrics, per_axis = score_subset(data, prediction, axes,
                                     validation_rows=data.validation)
    args.output_dir.mkdir(parents=True)
    prediction.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    per_axis.to_csv(args.output_dir / "axis_metrics.csv", index=False)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "manifest.json", {
        "status": "complete", "version": "no_foodname_hurdle_median_readout",
        "parent_manifest_sha256": digest(origin),
        "checkpoint_sha256": digest(args.checkpoint_dir / "transformer.pt"),
        "code_sha256": digest(Path(__file__)),
        "data_sha256": parent["data_sha256"], "metrics": metrics,
        "decision_rule": "p(positive) >= 0.5 => predicted positive amount, else 0",
        "threshold_selected_on_validation": False,
        "food_name_model_input": False, "complete_test_opened": False,
    })
    print(f"hurdle median readout: {metrics['all']['scaled_log_mae']:.6f}")


if __name__ == "__main__":
    main()

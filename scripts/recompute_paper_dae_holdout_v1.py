#!/usr/bin/env python3
"""Recompute the paper-native masked-cell RMSE for saved DAE checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

from run_paper_dae_external_baseline_v1 import (
    encode_food_text,
    load_paper_data,
    paper_holdout_score,
    paper_split_and_mask,
)

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from foodcomp.paper_dae_baseline_v1 import PaperDenoisingAutoencoder  # noqa: E402


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PaperDenoisingAutoencoder(int(checkpoint["dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--text-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--text-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()
    device = torch.device("cpu")

    value, columns, names, _, _ = load_paper_data(args.paper_repo)
    train, test, test_mask, means = paper_split_and_mask(value, args.seed)
    paper_text, _, _ = encode_food_text(
        names, ["placeholder"], args.text_model, args.text_device, args.seed
    )

    configurations = (
        (
            "Paper DAE (no name)",
            None,
            args.output_dir / "paper_dae_no_name.pt",
            args.output_dir / "paper_holdout_axis_rmse_no_name.csv",
        ),
        (
            "Paper-data DAE + FoodText",
            paper_text,
            args.output_dir / "paper_dae_food_text.pt",
            args.output_dir / "paper_holdout_axis_rmse_food_text.csv",
        ),
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, text, checkpoint_path, axis_path in configurations:
        raw = value if text is None else np.concatenate((value, text), axis=1)
        scaler = MinMaxScaler().fit(raw[train])
        model = load_checkpoint(checkpoint_path, device)
        summary, axis_rmse = paper_holdout_score(
            model, scaler, value, text, test, test_mask, means, device
        )
        manifest["models"][name]["paper_holdout"] = summary
        pd.DataFrame(
            {"paper_nutrient_id": columns, "rmse_scaled_0_1": axis_rmse}
        ).to_csv(axis_path, index=False)
        print(name, summary, flush=True)

    manifest["paper_holdout_metric"] = (
        "sum and macro mean of per-nutrient RMSE on masked cells in the "
        "paper training split's MinMax-scaled [0,1] space"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

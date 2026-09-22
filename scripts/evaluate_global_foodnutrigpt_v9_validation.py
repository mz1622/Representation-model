#!/usr/bin/env python3
"""Generate the source-free complete validation panel for a trained V9 checkpoint.

This is deliberately separate from training so an interrupted long validation
pass can be rerun without changing the selected model or reopening test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_global_foodnutrigpt_v8_single_stage as v8  # noqa: E402
import train_global_foodnutrigpt_v9_source_calibrated as v9  # noqa: E402


DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_dir, split_dir, output_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.output_dir.resolve()
    checkpoint_path = output_dir / "validation_selected_source_free_base_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V9 validation-selected checkpoint is absent: {checkpoint_path}")
    cache_dir = output_dir / "text_cache"
    ids_path, matrix_path = cache_dir / "train_validation_profile_ids.csv", cache_dir / "train_validation_embeddings.npy"
    if not ids_path.exists() or not matrix_path.exists():
        raise FileNotFoundError("V9 train/validation text cache is absent.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = v9.Config(**checkpoint["config"])
    corpus = v9.SourceEqualizedCorpus(data_dir, split_dir)
    profile_ids = pd.read_csv(ids_path)["profile_id"].tolist()
    embeddings = np.load(matrix_path).astype(np.float32)
    if len(profile_ids) != len(embeddings):
        raise ValueError("V9 text cache has mismatched profile IDs and embeddings.")
    expected = {corpus.profiles.iloc[position].profile_id for position in corpus.positions("train") + corpus.positions("validation")}
    if set(profile_ids) != expected:
        raise ValueError("V9 text cache is not exactly the train plus validation partition.")
    text_embeddings = np.zeros((len(corpus.profiles), embeddings.shape[1]), dtype=np.float32)
    for profile_id, embedding in zip(profile_ids, embeddings):
        text_embeddings[corpus.profile_index[profile_id]] = embedding

    device = v8.device_for_training()
    model = v9.SourceCalibratedFoodNutriGPT(
        text_embeddings.shape[1], len(corpus.axes), int(corpus.profiles["source_index"].max()) + 1,
        corpus.train_source_indices, config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = v9.complete_panel_dataset(corpus, corpus.positions("validation"), config)
    print(f"Evaluating {len(dataset):,} validation food-family masks across {int(corpus.axes['loss_eligible'].sum())} axes.", flush=True)
    predictions = v9.evaluate_profile_axis(
        model, v9.make_loader(dataset, text_embeddings, config.batch_size, False), corpus, device,
    )
    cells, axis_metrics, metrics = v9.summarize_source_free_cells(predictions, corpus)
    predictions.to_csv(output_dir / "validation_complete_axis_panel_source_free_profile_predictions.csv", index=False)
    cells.to_csv(output_dir / "validation_complete_axis_panel_source_free_candidate_cells.csv", index=False)
    axis_metrics.to_csv(output_dir / "validation_complete_axis_panel_source_free_axis_metrics.csv", index=False)
    config_payload = {**v9.asdict(config), "data_dir": str(data_dir), "split_dir": str(split_dir)}
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    summary = {
        "training_mode": "single_stage_source_calibrated_encoder_source_free",
        "checkpoint": str(checkpoint_path),
        "best_validation_source_free_hurdle_loss": checkpoint["validation_source_free_hurdle_loss"],
        "encoder_source_policy": "No source token or source embedding is present in the Transformer input.",
        "evaluation_source_policy": "The complete validation panel uses only the source-free base head; source is used only for equal-source offline target aggregation.",
        "equalization": corpus.equalization_summary,
        "validation_complete_panel_protocol": "Leave-one-mask-family-out across every validation loss axis; candidate-food axis scoring gives each observed source equal weight.",
        "validation_metrics": metrics,
        "complete_test_opened": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the validation-selected V9 source-free base head on the complete test panel.

This script must be run only after the V9 checkpoint and its settings have been
selected on validation.  It encodes complete-test food text after selection and
never uses any test value for fitting, normalization, or model choice.
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = v9.Config(**checkpoint["config"])
    corpus = v9.SourceEqualizedCorpus(data_dir, split_dir)
    cache_dir = output_dir / "text_cache"
    train_validation_ids = pd.read_csv(cache_dir / "train_validation_profile_ids.csv")["profile_id"].tolist()
    train_validation_embeddings = np.load(cache_dir / "train_validation_embeddings.npy").astype(np.float32)
    if len(train_validation_ids) != len(train_validation_embeddings):
        raise ValueError("V9 train/validation text cache has mismatched IDs and embeddings.")
    expected_train_validation = {
        corpus.profiles.iloc[position].profile_id
        for position in corpus.positions("train") + corpus.positions("validation")
    }
    if set(train_validation_ids) != expected_train_validation:
        raise ValueError("V9 train/validation text cache does not match the fixed split.")
    text_embeddings = np.zeros((len(corpus.profiles), train_validation_embeddings.shape[1]), dtype=np.float32)
    for profile_id, embedding in zip(train_validation_ids, train_validation_embeddings):
        text_embeddings[corpus.profile_index[profile_id]] = embedding

    test_positions = corpus.positions("test_complete_axis_panel")
    test_profiles = corpus.profiles.iloc[test_positions].copy()
    test_ids, test_embeddings = v8._encode_texts(
        test_profiles, config, cache_dir, "complete_test_after_selection",
    )
    test_embedding_by_id = dict(zip(test_ids, test_embeddings))
    for profile_index in test_positions:
        profile_id = corpus.profiles.iloc[profile_index].profile_id
        text_embeddings[profile_index] = test_embedding_by_id[profile_id]

    device = v8.device_for_training()
    model = v9.SourceCalibratedFoodNutriGPT(
        text_embeddings.shape[1], len(corpus.axes), int(corpus.profiles["source_index"].max()) + 1,
        corpus.train_source_indices, config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = v9.complete_panel_dataset(corpus, test_positions, config)
    print(f"Evaluating {len(dataset):,} complete-test food-family masks across {int(corpus.axes['loss_eligible'].sum())} axes.", flush=True)
    predictions = v9.evaluate_profile_axis(
        model, v9.make_loader(dataset, text_embeddings, config.batch_size, False), corpus, device,
    )
    cells, axis_metrics, metrics = v9.summarize_source_free_cells(predictions, corpus)
    predictions.to_csv(output_dir / "complete_test_source_free_profile_predictions.csv", index=False)
    cells.to_csv(output_dir / "complete_test_source_free_candidate_cells.csv", index=False)
    axis_metrics.to_csv(output_dir / "complete_test_source_free_axis_metrics.csv", index=False)
    summary = {
        "training_mode": "single_stage_source_calibrated_encoder_source_free",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "validation-only source-free hurdle loss",
        "encoder_source_policy": "No source token or source embedding is present in the Transformer input.",
        "test_source_policy": "The complete test panel uses only the source-free base head; source is used only for equal-source offline target aggregation.",
        "equalization": corpus.equalization_summary,
        "complete_test_protocol": "Leave-one-mask-family-out across all 187 loss axes; candidate-food axis scoring gives each observed source equal weight.",
        "test_metrics": metrics,
        "complete_test_opened": True,
    }
    (output_dir / "complete_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one validation-selected V8/V2 checkpoint on the frozen complete test.

The script loads an existing checkpoint rather than retraining.  It is intended
for the single prespecified candidate selected on validation and writes a new,
immutable complete-test artifact directory.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_global_foodnutrigpt_v8_single_stage as train  # noqa: E402


def load_train_validation_text(
    corpus: train.SourceNativeCorpus, model_output: Path,
) -> np.ndarray:
    cache_dir = model_output / "text_cache"
    ids = pd.read_csv(cache_dir / "train_validation_profile_ids.csv")["profile_id"].tolist()
    embeddings = np.load(cache_dir / "train_validation_embeddings.npy").astype(np.float32)
    if len(ids) != len(embeddings):
        raise ValueError("Cached train/validation text IDs and embeddings differ in length.")
    by_id = dict(zip(ids, embeddings))
    text = np.zeros((len(corpus.profiles), embeddings.shape[1]), dtype=np.float32)
    for profile_index in corpus.positions("train") + corpus.positions("validation"):
        profile_id = corpus.profiles.iloc[profile_index].profile_id
        if profile_id not in by_id:
            raise ValueError(f"Missing cached text embedding: {profile_id}")
        text[profile_index] = by_id[profile_id]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=train.DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=train.SPLIT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    config_raw = json.loads((args.model_output / "config.json").read_text(encoding="utf-8"))
    config_keys = {field.name for field in fields(train.Config)}
    config = train.Config(**{key: value for key, value in config_raw.items() if key in config_keys})
    corpus = train.SourceNativeCorpus(args.data_dir.resolve(), args.split_dir.resolve())
    test_positions = corpus.positions("test_complete_axis_panel")
    text = load_train_validation_text(corpus, args.model_output)

    device = train.device_for_training()
    source_count = int(corpus.profiles["source_index"].max()) + 1
    model = train.SourceAwareFoodNutriGPT(text.shape[1], len(corpus.axes), source_count, config).to(device)
    checkpoint = torch.load(args.model_output / "validation_selected_joint_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    test_profiles = corpus.profiles.iloc[test_positions].copy()
    ids, test_embeddings = train._encode_texts(
        test_profiles, config, args.output_dir / "text_cache", "complete_test_after_selection",
    )
    by_id = dict(zip(ids, test_embeddings))
    for profile_index in test_positions:
        text[profile_index] = by_id[corpus.profiles.iloc[profile_index].profile_id]
    evaluation_axes = corpus.axes.loc[
        corpus.axes["target_axis_id"].isin(corpus.splits["test_axis_ids"]), "axis_index"
    ].to_numpy(dtype=np.int64)
    dataset = train.MaskedProfileDataset(
        corpus, test_positions, config, training=False, evaluation_axes=evaluation_axes,
        force_unknown_source=True, cover_all_evaluation_families=True,
    )
    dataset.set_epoch(0)
    predictions, metrics = train.evaluate(
        model, train.make_loader(dataset, text, config.batch_size, False), corpus, device,
    )
    predictions.to_csv(args.output_dir / "complete_axis_panel_source_unknown_predictions.csv", index=False)
    summary = {
        "model_output": str(args.model_output.resolve()),
        "test_protocol": "frozen v2 complete 187-axis leave-one-mask-family-out panel; SOURCE_UNKNOWN",
        "selection_statement": "Checkpoint was selected on validation before this complete test evaluation.",
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

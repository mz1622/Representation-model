#!/usr/bin/env python3
"""Evaluate an existing V8/V2 checkpoint on the fixed source-unknown validation set.

This utility never loads or encodes complete-test foods. It provides comparable
validation metrics for historical checkpoints created before validation
predictions were written by the training script.
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
    valid_positions = corpus.positions("validation")
    cache_dir = args.model_output / "text_cache"
    ids = pd.read_csv(cache_dir / "train_validation_profile_ids.csv")["profile_id"].tolist()
    embeddings = np.load(cache_dir / "train_validation_embeddings.npy").astype(np.float32)
    if len(ids) != len(embeddings):
        raise ValueError("Cached train/validation text IDs and embeddings differ in length.")
    embeddings_by_id = dict(zip(ids, embeddings))
    text = np.zeros((len(corpus.profiles), embeddings.shape[1]), dtype=np.float32)
    for profile_index in corpus.positions("train") + valid_positions:
        profile_id = corpus.profiles.iloc[profile_index].profile_id
        if profile_id not in embeddings_by_id:
            raise ValueError(f"Missing cached text embedding: {profile_id}")
        text[profile_index] = embeddings_by_id[profile_id]

    device = train.device_for_training()
    source_count = int(corpus.profiles["source_index"].max()) + 1
    model = train.SourceAwareFoodNutriGPT(text.shape[1], len(corpus.axes), source_count, config).to(device)
    checkpoint = torch.load(args.model_output / "validation_selected_joint_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = train.MaskedProfileDataset(corpus, valid_positions, config, training=False, force_unknown_source=True)
    dataset.set_epoch(0)
    predictions, metrics = train.evaluate(
        model, train.make_loader(dataset, text, config.batch_size, False), corpus, device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_dir / "validation_source_unknown_predictions.csv", index=False)
    summary = {
        "model_output": str(args.model_output.resolve()),
        "validation_protocol": "fixed sampled mask families; SOURCE_UNKNOWN; no complete-test data loaded",
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

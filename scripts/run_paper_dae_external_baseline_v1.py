#!/usr/bin/env python3
"""Reproduce the 2022 food DAE and zero-shot it on the current completion task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.paper_dae_baseline_v1 import (  # noqa: E402
    VERSION, PaperDenoisingAutoencoder, build_paper_axis_crosswalk,
    paper_external_features, target_link_by_axis,
)
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    attach_complete_test, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_paper_data(paper_repo: Path):
    value_path = paper_repo / "datasets/average datasets/byColumns/average_dataset_198.csv"
    nutrient_path = paper_repo / "datasets/original data/nutrient.csv"
    food_path = paper_repo / "datasets/original data/food.csv"
    frame = pd.read_csv(value_path, index_col=0, low_memory=False)
    columns = [str(item) for item in frame.columns]
    if frame.shape != (13_105, 198):
        raise ValueError(f"unexpected paper matrix shape {frame.shape}")
    value = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(value).all():
        raise FloatingPointError("paper average-value matrix contains nonfinite values")
    food = pd.read_csv(food_path, usecols=["fdc_id", "description"], low_memory=False)
    if food.fdc_id.duplicated().any():
        raise ValueError("paper food table contains duplicate fdc_id")
    names = frame.index.to_series().map(food.set_index("fdc_id").description)
    if names.isna().any():
        raise ValueError("paper nutrient row lacks a food description")
    return value, columns, names.astype(str).tolist(), pd.read_csv(nutrient_path), value_path


def encode_food_text(
    paper_names: list[str], external_names: list[str], model_name: str,
    device: str, seed: int,
):
    encoder = SentenceTransformer(model_name, device=device)
    paper_embedding = encoder.encode(
        paper_names, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    pca = PCA(n_components=50, svd_solver="full")
    paper_text = pca.fit_transform(paper_embedding).astype(np.float32)
    external_embedding = encoder.encode(
        external_names, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    external_text = pca.transform(external_embedding).astype(np.float32)
    return paper_text, external_text, pca


def paper_split_and_mask(value: np.ndarray, seed: int):
    index = np.arange(len(value), dtype=np.int64)
    train, test = train_test_split(index, test_size=0.2, random_state=seed)
    rng = np.random.RandomState(seed + 1)
    test_mask = rng.uniform(size=(len(test), value.shape[1])) <= 0.2
    train_mask = rng.uniform(size=(len(train), value.shape[1])) <= 0.2
    corrupted_train = value[train].astype(np.float64).copy()
    corrupted_train[train_mask] = np.nan
    imputation_mean = np.nanmean(corrupted_train, axis=0)
    if not np.isfinite(imputation_mean).all():
        raise FloatingPointError("paper imputation mean is nonfinite")
    return train, test, test_mask, imputation_mean.astype(np.float32)


def train_dae(
    input_scaled: np.ndarray, *, seed: int, epochs: int, batch_size: int,
    learning_rate: float, device: torch.device, checkpoint: Path,
):
    set_seed(seed)
    model = PaperDenoisingAutoencoder(input_scaled.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(input_scaled.astype(np.float32))),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, drop_last=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total, examples = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            examples += len(batch)
        epoch_loss = total / examples
        history.append({"epoch": epoch, "reconstruction_mse": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"DAE dim={input_scaled.shape[1]} epoch {epoch}/{epochs} "
                f"mse={epoch_loss:.8f} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    torch.save({
        "model_state": model.state_dict(), "dim": input_scaled.shape[1],
        "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "history": history,
    }, checkpoint)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_scaled(model, value: np.ndarray, device: torch.device, batch_size=8192):
    model.eval()
    output = []
    for start in range(0, len(value), batch_size):
        batch = torch.from_numpy(value[start:start + batch_size].astype(np.float32)).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def paper_holdout_score(
    model, scaler, paper_value, paper_text, test, test_mask,
    imputation_mean, device,
):
    nutrient_count = paper_value.shape[1]
    corrupted = paper_value[test].copy()
    corrupted[test_mask] = np.broadcast_to(
        imputation_mean, corrupted.shape
    )[test_mask]
    model_input = corrupted if paper_text is None else np.concatenate(
        (corrupted, paper_text[test]), axis=1
    )
    prediction = predict_scaled(
        model, scaler.transform(model_input).astype(np.float32), device
    )[:, :nutrient_count]
    clean_input = paper_value[test] if paper_text is None else np.concatenate(
        (paper_value[test], paper_text[test]), axis=1
    )
    truth = scaler.transform(clean_input)[:, :nutrient_count]
    rmse = []
    for column in range(nutrient_count):
        delta = prediction[:, column][test_mask[:, column]] - truth[:, column][test_mask[:, column]]
        rmse.append(float(np.sqrt(np.mean(np.square(delta)))))
    return {
        "masked_cells": int(test_mask.sum()),
        "nutrients": nutrient_count,
        "rmse_sum": float(np.sum(rmse)),
        "macro_rmse": float(np.mean(rmse)),
    }, np.asarray(rmse)


@torch.no_grad()
def predict_external_jobs(
    model, scaler, data, policy, jobs, links, paper_means, device,
    external_text_by_row=None, batch_size=8192,
):
    target_link = target_link_by_axis(links)
    rows = jobs.profile_index.to_numpy(dtype=np.int64)
    axes = jobs.axis_index.to_numpy(dtype=np.int64)
    pieces = []
    model.eval()
    for start in range(0, len(jobs), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_axes = axes[start:start + batch_size]
        raw = paper_external_features(
            data, batch_rows, batch_axes, policy, links, paper_means
        )
        if external_text_by_row is not None:
            raw = np.concatenate((raw, external_text_by_row[batch_rows]), axis=1)
        scaled = scaler.transform(raw).astype(np.float32)
        reconstructed = scaler.inverse_transform(
            model(torch.from_numpy(scaled).to(device)).cpu().numpy()
        )
        selected = np.asarray([
            reconstructed[i, target_link[int(axis)].paper_column]
            / target_link[int(axis)].grams_to_paper
            for i, axis in enumerate(batch_axes)
        ])
        pieces.append(pd.DataFrame({
            "profile_index": batch_rows,
            "axis_index": batch_axes,
            "prediction": np.maximum(selected, 0.0),
        }))
    return pd.concat(pieces, ignore_index=True)


def existing_prediction_sources(baseline_dir: Path, cce_dir: Path):
    return {
        "KNN": baseline_dir / "knn",
        "RF": baseline_dir / "rf",
        "XGBoost": baseline_dir / "xgb",
        "V5+CCE": cce_dir,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--frozen-data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--cce-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--train-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--text-device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--text-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    train_device = args.train_device
    if train_device == "auto":
        train_device = "cuda" if torch.cuda.is_available() else "cpu"
    text_device = args.text_device
    if text_device == "auto":
        text_device = "cuda" if torch.cuda.is_available() else "cpu"
    if (train_device == "cuda" or text_device == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(train_device)
    paper_value, paper_columns, paper_names, nutrient, paper_value_path = load_paper_data(args.paper_repo)
    train, test, test_mask, paper_means = paper_split_and_mask(paper_value, args.seed)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    validation_jobs = data.jobs[["profile_index", "axis_index", "target"]].copy()
    _, test_jobs, complete_audit = attach_complete_test(data, args.frozen_data_dir, args.split_path)
    policy = DependencyMaskPolicy(data.axes)
    links, crosswalk = build_paper_axis_crosswalk(paper_columns, nutrient, data.axes)
    target_axes = sorted(link.axis_index for link in links if link.loss_eligible)
    validation_jobs = validation_jobs[validation_jobs.axis_index.isin(target_axes)].reset_index(drop=True)
    test_jobs = test_jobs[test_jobs.axis_index.isin(target_axes)].reset_index(drop=True)
    if len(target_axes) != 127:
        raise ValueError(f"expected 127 strict common targets, found {len(target_axes)}")
    crosswalk.to_csv(args.output_dir / "paper_to_current_axis_crosswalk.csv", index=False)

    eval_rows = np.unique(np.concatenate((
        validation_jobs.profile_index.to_numpy(np.int64),
        test_jobs.profile_index.to_numpy(np.int64),
    )))
    external_names = data.profiles.loc[eval_rows, "original_name"].fillna("").astype(str).tolist()
    paper_text, external_text, pca = encode_food_text(
        paper_names, external_names, args.text_model,
        text_device, args.seed,
    )
    external_text_by_row = np.zeros((len(data.profiles), 50), dtype=np.float32)
    external_text_by_row[eval_rows] = external_text

    manifest = {
        "status": "running", "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_device": str(device), "text_encoding_device": text_device,
        "seed": args.seed,
        "paper_data_sha256": digest(paper_value_path),
        "current_data_sha256": digest(args.data_dir / "manifest.json"),
        "paper_training_rows": int(len(train)), "paper_holdout_rows": int(len(test)),
        "paper_nutrients": int(paper_value.shape[1]),
        "strict_common_supervised_axes": len(target_axes),
        "validation_jobs": len(validation_jobs), "complete_test_jobs": len(test_jobs),
        "complete_test_original_panel": complete_audit,
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "optimizer": "SGD", "learning_rate": args.learning_rate,
            "loss": "full-vector MSE", "input_dropout": 0.2,
            "checkpoint": "fixed final epoch; current validation never used for selection",
        },
        "models": {
            "Paper DAE (no name)": {
                "input": "198 nutrient values", "food_name_model_input": False,
            },
            "Paper-data DAE + FoodText": {
                "input": "198 nutrient values + 50 PCA components of frozen MiniLM food-name embedding",
                "food_name_model_input": True,
                "text_encoder": args.text_model,
                "pca_fit": "paper food names only",
                "note": "adapted transfer variant; replaces the public notebook PCA contaminated by fdc_id",
            },
        },
        "results": {},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        trained = {}
        for key, text in (("Paper DAE (no name)", None), ("Paper-data DAE + FoodText", paper_text)):
            raw = paper_value if text is None else np.concatenate((paper_value, text), axis=1)
            scaler = MinMaxScaler().fit(raw[train])
            scaled_train = scaler.transform(raw[train]).astype(np.float32)
            checkpoint = args.output_dir / (
                "paper_dae_no_name.pt" if text is None else "paper_dae_food_text.pt"
            )
            model, history = train_dae(
                scaled_train, seed=args.seed, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                device=device, checkpoint=checkpoint,
            )
            history.to_csv(
                args.output_dir / ("history_no_name.csv" if text is None else "history_food_text.csv"),
                index=False,
            )
            holdout, per_nutrient_rmse = paper_holdout_score(
                model, scaler, paper_value, text, test, test_mask,
                paper_means, device,
            )
            pd.DataFrame({
                "paper_nutrient_id": paper_columns,
                "rmse_scaled_0_1": per_nutrient_rmse,
            }).to_csv(
                args.output_dir / ("paper_holdout_axis_rmse_no_name.csv" if text is None else "paper_holdout_axis_rmse_food_text.csv"),
                index=False,
            )
            manifest["models"][key]["paper_holdout"] = holdout
            trained[key] = (model, scaler, text is not None)
            write_json(args.output_dir / "manifest.json", manifest)

        comparison_rows = []
        job_sets = {"validation": validation_jobs, "complete_test": test_jobs}
        for method, (model, scaler, uses_text) in trained.items():
            method_dir = args.output_dir / method.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            method_dir.mkdir()
            for partition, jobs in job_sets.items():
                prediction = predict_external_jobs(
                    model, scaler, data, policy, jobs, links, paper_means,
                    device, external_text_by_row if uses_text else None,
                )
                metrics, per_axis, _ = score_profile_axis_macro(data, jobs, prediction)
                prediction.to_parquet(method_dir / f"{partition}_predictions.parquet", index=False)
                per_axis.to_csv(method_dir / f"{partition}_axis_metrics.csv", index=False)
                comparison_rows.append({"method": method, "partition": partition, **metrics})
                manifest["results"].setdefault(method, {})[partition] = metrics
                write_json(args.output_dir / "manifest.json", manifest)
                print(f"{method} {partition}: {metrics['macro_scaled_log_mae']:.6f}", flush=True)

        for method, source in existing_prediction_sources(args.baseline_dir, args.cce_eval_dir).items():
            for partition, jobs in job_sets.items():
                prediction = pd.read_parquet(source / f"{partition}_predictions.parquet")
                prediction = prediction[prediction.axis_index.isin(target_axes)].reset_index(drop=True)
                metrics, per_axis, _ = score_profile_axis_macro(data, jobs, prediction)
                per_axis.to_csv(args.output_dir / f"{method.lower().replace('+', '_')}_{partition}_axis_metrics.csv", index=False)
                comparison_rows.append({"method": method, "partition": partition, **metrics})
                manifest["results"].setdefault(method, {})[partition] = metrics

        comparison = pd.DataFrame(comparison_rows).sort_values(
            ["partition", "macro_scaled_log_mae", "method"]
        )
        comparison.to_csv(args.output_dir / "common_127_axis_metrics.csv", index=False)
        manifest.update(
            status="complete", elapsed_seconds=time.monotonic() - started,
            text_pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed", error_type=type(exc).__name__, error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

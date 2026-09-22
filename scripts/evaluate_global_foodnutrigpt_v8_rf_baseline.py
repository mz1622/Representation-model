#!/usr/bin/env python3
"""Fit a leakage-safe Random Forest baseline for the v2 complete-axis panel.

The baseline uses the same profile-level leave-one-mask-family-out test
examples as FoodNutriGPT.  It receives frozen food text plus visible
composition cells and their observedness indicators.  Source IDs are not input
features because the Transformer is evaluated with SOURCE_UNKNOWN.

Colab:
    python scripts/evaluate_global_foodnutrigpt_v8_rf_baseline.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_global_foodnutrigpt_v8_single_stage as model_code  # noqa: E402


DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
TRANSFORMER_OUTPUT = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_complete_test"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v8_single_stage_v2_rf_baseline"
SEED = 20260920


def stable_rng(axis_index: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|rf|{axis_index}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def load_cached_embeddings(corpus: model_code.SourceNativeCorpus, transformer_output: Path) -> np.ndarray:
    """Load frozen text vectors generated without test-time model fitting."""
    text = np.zeros((len(corpus.profiles), 384), dtype=np.float32)
    cache_dir = transformer_output / "text_cache"
    cache_specs = [
        ("train_validation", corpus.positions("train") + corpus.positions("validation")),
        ("complete_test_after_selection", corpus.positions("test_complete_axis_panel")),
    ]
    for cache_name, positions in cache_specs:
        ids_path = cache_dir / f"{cache_name}_profile_ids.csv"
        matrix_path = cache_dir / f"{cache_name}_embeddings.npy"
        if not ids_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"Missing frozen text cache for RF baseline: {matrix_path}")
        ids = pd.read_csv(ids_path)["profile_id"].tolist()
        matrix = np.load(matrix_path).astype(np.float32)
        if len(ids) != len(matrix):
            raise ValueError(f"Text cache has mismatched IDs and embeddings: {cache_name}")
        for profile_id, embedding in zip(ids, matrix):
            profile_index = corpus.profile_index.get(profile_id)
            if profile_index is None:
                raise ValueError(f"Cached text profile is absent from corpus: {profile_id}")
            text[profile_index] = embedding
        expected = {corpus.profiles.iloc[position].profile_id for position in positions}
        if set(ids) != expected:
            raise ValueError(f"Text cache does not match the expected split: {cache_name}")
    return text


def build_dense_profile_cells(corpus: model_code.SourceNativeCorpus, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build one tabular cell per profile-axis using median source-native z values.

    Source-native rows remain the Transformer training representation.  RF
    requires a rectangular feature matrix, so only this baseline uses a median
    within a profile-axis cell; an observedness indicator retains the difference
    between missing and a genuine zero.
    """
    tokens = pd.read_csv(
        data_dir / "source_native_axis_tokens.csv.gz",
        usecols=["profile_id", "axis_index", "normalized_log1p_value"], low_memory=False,
    )
    tokens["profile_index"] = tokens["profile_id"].map(corpus.profile_index)
    if tokens["profile_index"].isna().any():
        raise ValueError("RF cell builder found a token without a food profile.")
    cells = tokens.groupby(["profile_index", "axis_index"], as_index=False)["normalized_log1p_value"].median()
    values = np.zeros((len(corpus.profiles), len(corpus.axes)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.float32)
    rows = cells["profile_index"].to_numpy(dtype=np.int64)
    columns = cells["axis_index"].to_numpy(dtype=np.int64)
    values[rows, columns] = cells["normalized_log1p_value"].to_numpy(dtype=np.float32)
    observed[rows, columns] = 1.0
    return values, observed


def complete_test_jobs(
    corpus: model_code.SourceNativeCorpus, config: model_code.Config,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return profile rows and hidden columns for every v2 test target axis."""
    test_positions = corpus.positions("test_complete_axis_panel")
    evaluation_axes = corpus.axes.loc[
        corpus.axes["target_axis_id"].isin(corpus.splits["test_axis_ids"]), "axis_index"
    ].to_numpy(dtype=np.int64)
    dataset = model_code.MaskedProfileDataset(
        corpus, test_positions, config, training=False, evaluation_axes=evaluation_axes,
        force_unknown_source=True, cover_all_evaluation_families=True,
    )
    dataset.set_epoch(0)
    jobs: dict[int, set[int]] = {int(axis): set() for axis in evaluation_axes}
    family_columns: dict[int, np.ndarray] = {}
    for profile_index, family_id in dataset.sample_specs:
        assert family_id is not None
        indexes = dataset._token_indexes(profile_index)
        target = (
            corpus.loss_eligible[indexes]
            & np.isin(corpus.axis[indexes], evaluation_axes)
            & (corpus.family[indexes] == family_id)
        )
        axes = np.unique(corpus.axis[indexes][target])
        columns = np.unique(corpus.axis[corpus.family == family_id])
        family_columns[int(family_id)] = columns
        for axis in axes:
            jobs[int(axis)].add(int(profile_index))
    jobs_array = {axis: np.array(sorted(rows), dtype=np.int64) for axis, rows in jobs.items()}
    axis_family = {
        int(axis): int(corpus.family[np.flatnonzero(corpus.axis == axis)[0]])
        for axis in evaluation_axes
    }
    hidden = {axis: family_columns[axis_family[axis]] for axis in jobs_array}
    missing = [axis for axis, rows in jobs_array.items() if not len(rows)]
    if missing:
        raise ValueError(f"Complete test evaluation did not mask axes: {missing[:5]}")
    return jobs_array, hidden


def inverse_raw(z: np.ndarray, median: float, scale: float) -> np.ndarray:
    return np.expm1(np.maximum(z * scale + median, 0.0))


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = predictions.copy()
    predictions["absolute_error_g_per_100g"] = np.abs(
        predictions["prediction_g_per_100g"] - predictions["target_g_per_100g"]
    )
    predictions["absolute_log1p_error"] = np.abs(
        np.log1p(predictions["prediction_g_per_100g"])
        - np.log1p(predictions["target_g_per_100g"])
    )
    per_axis = predictions.groupby(
        ["target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        labels=("profile_id", "size"),
        log_mae=("absolute_log1p_error", "mean"),
        log_mse=("absolute_log1p_error", lambda x: float((x * x).mean())),
        raw_mae_g_per_100g=("absolute_error_g_per_100g", "mean"),
    )
    metrics = {
        "prediction_records": int(len(predictions)),
        "axes_scored": int(per_axis["target_axis_id"].nunique()),
        "macro_axis_log_mae": float(per_axis["log_mae"].mean()),
        "macro_axis_log_rmse": float(math.sqrt(per_axis["log_mse"].mean())),
        "macro_axis_raw_mae_g_per_100g": float(per_axis["raw_mae_g_per_100g"].mean()),
    }
    return per_axis, metrics


def transformer_profile_axis_predictions(
    path: Path,
    corpus: model_code.SourceNativeCorpus,
    values: np.ndarray,
) -> pd.DataFrame:
    """Aggregate Transformer predictions onto the RF profile-axis cell grid.

    The Transformer retains source-native target records, whereas the RF
    baseline needs one dense profile-axis target.  Replacing the grouped raw
    target here with the same median-log profile cell used by RF guarantees
    that both methods are scored against exactly the same target value.
    """
    predictions = pd.read_csv(path)
    required = {"profile_id", "target_axis_id", "canonical_name", "loss_group", "target_g_per_100g", "prediction_g_per_100g"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Transformer predictions lack profile-cell fields: {sorted(missing)}")
    grouped = predictions.groupby(
        ["profile_id", "target_axis_id", "canonical_name", "loss_group"], as_index=False,
    ).agg(
        prediction_g_per_100g=("prediction_g_per_100g", "median"),
    )
    axis_index = corpus.axes.set_index("target_axis_id")["axis_index"]
    axis_rows = corpus.axes.set_index("axis_index")
    profile_rows = grouped["profile_id"].map(corpus.profile_index)
    columns = grouped["target_axis_id"].map(axis_index)
    if profile_rows.isna().any() or columns.isna().any():
        raise ValueError("Transformer prediction contains an unknown profile or target axis.")
    columns = columns.to_numpy(dtype=np.int64)
    medians = axis_rows.loc[columns, "median_log1p"].to_numpy(dtype=np.float64)
    scales = axis_rows.loc[columns, "scale_log1p"].to_numpy(dtype=np.float64)
    grouped["target_g_per_100g"] = inverse_raw(
        values[profile_rows.to_numpy(dtype=np.int64), columns], medians, scales,
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--transformer-output", type=Path, default=TRANSFORMER_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--max-train-rows-per-axis", type=int, default=20_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir, split_dir = args.data_dir.resolve(), args.split_dir.resolve()
    transformer_output, output_dir = args.transformer_output.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = model_code.SourceNativeCorpus(data_dir, split_dir)
    train_rows = np.array(corpus.positions("train"), dtype=np.int64)
    config = model_code.Config()
    text = load_cached_embeddings(corpus, transformer_output)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED)
    text_pca = np.zeros((len(corpus.profiles), 32), dtype=np.float32)
    text_pca[train_rows] = pca.fit_transform(text[train_rows]).astype(np.float32)
    other_rows = np.setdiff1d(np.arange(len(corpus.profiles), dtype=np.int64), train_rows, assume_unique=False)
    text_pca[other_rows] = pca.transform(text[other_rows]).astype(np.float32)
    values, observed = build_dense_profile_cells(corpus, data_dir)
    base = np.concatenate([text_pca, values, observed], axis=1).astype(np.float32)
    jobs, hidden_by_axis = complete_test_jobs(corpus, config)
    loss_axes = corpus.axes[corpus.axes["training_role"].eq("masked_loss_target")].copy()
    if len(loss_axes) != 187 or set(loss_axes["axis_index"]) != set(jobs):
        raise ValueError("RF target axes do not match the frozen 187-axis complete test panel.")

    rows: list[pd.DataFrame] = []
    feature_count = base.shape[1]
    for completed, axis_row in enumerate(loss_axes.itertuples(index=False), start=1):
        axis = int(axis_row.axis_index)
        labels = train_rows[observed[train_rows, axis].astype(bool)]
        if len(labels) < 20:
            raise ValueError(f"RF target has fewer than 20 train labels: {axis_row.canonical_name}")
        if len(labels) > args.max_train_rows_per_axis:
            labels = np.sort(stable_rng(axis).choice(labels, size=args.max_train_rows_per_axis, replace=False))
        hidden = hidden_by_axis[axis]
        x_train = base[labels].copy()
        x_train[:, 32 + hidden] = 0.0
        x_train[:, 32 + len(corpus.axes) + hidden] = 0.0
        estimator = RandomForestRegressor(
            n_estimators=args.trees, max_depth=12, max_features=0.35,
            min_samples_leaf=5, n_jobs=args.n_jobs, random_state=SEED + axis,
        )
        estimator.fit(x_train, values[labels, axis])
        test_rows = jobs[axis]
        if not observed[test_rows, axis].all():
            raise AssertionError(f"An RF test job has no observed target cell: {axis_row.canonical_name}")
        x_test = base[test_rows].copy()
        x_test[:, 32 + hidden] = 0.0
        x_test[:, 32 + len(corpus.axes) + hidden] = 0.0
        prediction_z = estimator.predict(x_test).astype(np.float32)
        rows.append(pd.DataFrame({
            "profile_id": corpus.profiles.iloc[test_rows]["profile_id"].tolist(),
            "target_axis_id": axis_row.target_axis_id,
            "canonical_name": axis_row.canonical_name,
            "loss_group": axis_row.loss_group,
            "target_g_per_100g": inverse_raw(values[test_rows, axis], axis_row.median_log1p, axis_row.scale_log1p),
            "prediction_g_per_100g": inverse_raw(prediction_z, axis_row.median_log1p, axis_row.scale_log1p),
        }))
        print(f"RF axis {completed:03d}/{len(loss_axes)}: {axis_row.canonical_name} ({len(labels):,} train, {len(test_rows):,} test)")

    rf_predictions = pd.concat(rows, ignore_index=True)
    rf_axis, rf_metrics = summarize(rf_predictions)
    transformer_predictions = transformer_profile_axis_predictions(
        transformer_output / "complete_axis_panel_source_unknown_predictions.csv", corpus, values,
    )
    transformer_axis, transformer_metrics = summarize(transformer_predictions)
    rf_keys = set(zip(rf_predictions.profile_id, rf_predictions.target_axis_id))
    transformer_keys = set(zip(transformer_predictions.profile_id, transformer_predictions.target_axis_id))
    if rf_keys != transformer_keys:
        raise AssertionError(
            f"RF and Transformer do not score identical profile-axis cells: {len(rf_keys)} vs {len(transformer_keys)}"
        )
    paired_targets = rf_predictions.merge(
        transformer_predictions[["profile_id", "target_axis_id", "target_g_per_100g"]],
        on=["profile_id", "target_axis_id"], suffixes=("_rf", "_transformer"), validate="one_to_one",
    )
    # The dense RF matrix stores normalized values as float32 while the
    # Transformer CSV has already round-tripped through text.  A 1e-4 g/100 g
    # absolute tolerance is far below reporting precision and guards against
    # a material target mismatch without rejecting this representation detail.
    if not np.allclose(
        paired_targets["target_g_per_100g_rf"], paired_targets["target_g_per_100g_transformer"],
        rtol=1e-7, atol=1e-4,
    ):
        raise AssertionError("RF and Transformer target values are not identical after profile-axis aggregation.")
    comparison = pd.DataFrame([
        {"method": "FoodNutriGPT_group_balanced", **transformer_metrics},
        {"method": "RandomForest_tabular", **rf_metrics},
    ])
    rf_predictions.to_csv(output_dir / "rf_complete_axis_panel_predictions.csv", index=False)
    rf_axis.to_csv(output_dir / "rf_profile_axis_metrics.csv", index=False)
    transformer_axis.to_csv(output_dir / "foodnutrigpt_group_balanced_profile_axis_metrics.csv", index=False)
    comparison.to_csv(output_dir / "comparison_metrics.csv", index=False)
    manifest = {
        "dataset_version": json.loads((data_dir / "build_manifest.json").read_text())["dataset_version"],
        "test_protocol": "same complete 187-axis leave-one-mask-family-out profile-axis cells as FoodNutriGPT",
        "source_policy": "no source feature; FoodNutriGPT comparison uses SOURCE_UNKNOWN",
        "rf_cell_policy": "median normalized log1p value within profile-axis, with a separate observedness indicator",
        "text_policy": "frozen MiniLM embeddings reduced to 32 PCA components fitted on train only",
        "rf_hyperparameters": {
            "n_estimators": args.trees, "max_depth": 12, "max_features": 0.35,
            "min_samples_leaf": 5, "max_train_rows_per_axis": args.max_train_rows_per_axis,
        },
        "axes": int(rf_predictions["target_axis_id"].nunique()),
        "prediction_records": int(len(rf_predictions)),
        "metrics": comparison.to_dict(orient="records"),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

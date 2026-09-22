#!/usr/bin/env python3
"""Fit the V9 source-free Random Forest validation baseline.

RF receives frozen MiniLM text features, visible composition values, and
observedness indicators.  It never receives a source feature.  For each target
axis, the entire corresponding chemical family is hidden in both training and
validation rows, matching FoodNutriGPT's leave-one-mask-family-out protocol.

The RF and Transformer are scored with the same source-free candidate-cell
aggregation: sources have equal offline label influence but are not inputs and
are absent from the released prediction table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_global_foodnutrigpt_v8_single_stage as v8  # noqa: E402
import train_global_foodnutrigpt_v9_source_calibrated as v9  # noqa: E402


DATA_DIR = ROOT / "data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test"
SPLIT_DIR = ROOT / "data/splits/global_foodnutrigpt_v8_single_stage_v2_complete_test"
TRANSFORMER_OUTPUT = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_validation"
OUTPUT_DIR = ROOT / "output/global_foodnutrigpt_v9_source_calibrated_rf_validation"
SEED = 20260922


def stable_rng(axis_index: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|v9-rf|{axis_index}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def load_cached_embeddings(corpus: v9.SourceEqualizedCorpus, transformer_output: Path) -> np.ndarray:
    cache_dir = transformer_output / "text_cache"
    ids_path = cache_dir / "train_validation_profile_ids.csv"
    matrix_path = cache_dir / "train_validation_embeddings.npy"
    if not ids_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(f"Missing V9 frozen text cache: {matrix_path}")
    ids = pd.read_csv(ids_path)["profile_id"].tolist()
    matrix = np.load(matrix_path).astype(np.float32)
    if len(ids) != len(matrix):
        raise ValueError("V9 text cache has mismatched profile IDs and embeddings.")
    expected_positions = corpus.positions("train") + corpus.positions("validation")
    expected_ids = {corpus.profiles.iloc[position].profile_id for position in expected_positions}
    if set(ids) != expected_ids:
        raise ValueError("V9 text cache does not exactly cover train plus validation profiles.")
    text = np.zeros((len(corpus.profiles), matrix.shape[1]), dtype=np.float32)
    for profile_id, embedding in zip(ids, matrix):
        text[corpus.profile_index[profile_id]] = embedding
    return text


def build_dense_profile_cells(corpus: v9.SourceEqualizedCorpus, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Median the duplicate measurements only within each source-native profile cell."""
    tokens = pd.read_csv(
        data_dir / "source_native_axis_tokens.csv.gz",
        usecols=["profile_id", "axis_index", "normalized_log1p_value"], low_memory=False,
    )
    tokens["profile_index"] = tokens["profile_id"].map(corpus.profile_index)
    if tokens["profile_index"].isna().any():
        raise ValueError("RF cell builder found a token without a V9 profile.")
    cells = tokens.groupby(["profile_index", "axis_index"], as_index=False)["normalized_log1p_value"].median()
    values = np.zeros((len(corpus.profiles), len(corpus.axes)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.float32)
    rows = cells["profile_index"].to_numpy(dtype=np.int64)
    columns = cells["axis_index"].to_numpy(dtype=np.int64)
    values[rows, columns] = cells["normalized_log1p_value"].to_numpy(dtype=np.float32)
    observed[rows, columns] = 1.0
    return values, observed


def complete_validation_jobs(
    corpus: v9.SourceEqualizedCorpus, config: v9.Config,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return exactly the V9 complete-validation profile/axis mask jobs."""
    validation_positions = corpus.positions("validation")
    evaluation_axes = corpus.axes.loc[corpus.axes["loss_eligible"], "axis_index"].to_numpy(dtype=np.int64)
    dataset = v9.complete_panel_dataset(corpus, validation_positions, config)
    jobs: dict[int, set[int]] = {int(axis): set() for axis in evaluation_axes}
    # Chemical-family membership is corpus-global.  Precompute it once rather
    # than scanning every token for every validation food-family mask.
    family_columns = {
        int(family_id): np.unique(corpus.axis[corpus.family == family_id])
        for family_id in np.unique(corpus.family)
    }
    for profile_index, family_id in dataset.sample_specs:
        assert family_id is not None
        indexes = dataset._token_indexes(profile_index)
        target = (
            corpus.loss_eligible[indexes]
            & np.isin(corpus.axis[indexes], evaluation_axes)
            & (corpus.family[indexes] == family_id)
        )
        axes = np.unique(corpus.axis[indexes][target])
        for axis in axes:
            jobs[int(axis)].add(int(profile_index))
    jobs_array = {axis: np.asarray(sorted(rows), dtype=np.int64) for axis, rows in jobs.items()}
    axis_family = {
        int(axis): int(corpus.family[np.flatnonzero(corpus.axis == axis)[0]])
        for axis in evaluation_axes
    }
    hidden = {axis: family_columns[axis_family[axis]] for axis in jobs_array}
    missing = [axis for axis, rows in jobs_array.items() if not len(rows)]
    if missing:
        raise ValueError(f"The V9 complete validation panel omitted masked axes: {missing[:5]}")
    return jobs_array, hidden


def train_axis_sample_weights(corpus: v9.SourceEqualizedCorpus, rows: np.ndarray, axis: int) -> np.ndarray:
    """Equal total RF fitting weight per candidate food-axis source cell."""
    frame = corpus.profiles.iloc[rows][["exact_name_group_id", "source_key", "profile_id"]].copy()
    frame["axis_index"] = axis
    source_count = frame.groupby(["exact_name_group_id", "axis_index"])["source_key"].transform("nunique")
    profiles_per_source = frame.groupby(["exact_name_group_id", "axis_index", "source_key"])["profile_id"].transform("nunique")
    weights = 1.0 / (source_count.to_numpy(dtype=np.float32) * profiles_per_source.to_numpy(dtype=np.float32))
    return weights / weights.mean()


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
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    corpus = v9.SourceEqualizedCorpus(data_dir, split_dir)
    config_path = transformer_output / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"V9 config is absent: {config_path}")
    config_values = json.loads(config_path.read_text())
    config_values.pop("data_dir", None)
    config_values.pop("split_dir", None)
    config = v9.Config(**config_values)
    train_rows = np.asarray(corpus.positions("train"), dtype=np.int64)
    text = load_cached_embeddings(corpus, transformer_output)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED)
    text_pca = np.zeros((len(corpus.profiles), 32), dtype=np.float32)
    text_pca[train_rows] = pca.fit_transform(text[train_rows]).astype(np.float32)
    validation_rows = np.asarray(corpus.positions("validation"), dtype=np.int64)
    text_pca[validation_rows] = pca.transform(text[validation_rows]).astype(np.float32)
    values, observed = build_dense_profile_cells(corpus, data_dir)
    base = np.concatenate([text_pca, values, observed], axis=1).astype(np.float32)
    jobs, hidden_by_axis = complete_validation_jobs(corpus, config)
    loss_axes = corpus.axes[corpus.axes["loss_eligible"]].copy()
    if set(loss_axes["axis_index"]) != set(jobs):
        raise ValueError("RF loss axes do not match the V9 complete validation panel.")

    rows: list[pd.DataFrame] = []
    for completed, axis_row in enumerate(loss_axes.itertuples(index=False), start=1):
        axis = int(axis_row.axis_index)
        labels = train_rows[observed[train_rows, axis].astype(bool)]
        if len(labels) < 20:
            raise ValueError(f"RF axis has fewer than 20 observed train labels: {axis_row.canonical_name}")
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
        estimator.fit(x_train, values[labels, axis], sample_weight=train_axis_sample_weights(corpus, labels, axis))
        validation_rows = jobs[axis]
        if not observed[validation_rows, axis].all():
            raise AssertionError(f"An RF validation job lacks an observed target: {axis_row.canonical_name}")
        x_validation = base[validation_rows].copy()
        x_validation[:, 32 + hidden] = 0.0
        x_validation[:, 32 + len(corpus.axes) + hidden] = 0.0
        prediction_z = estimator.predict(x_validation).astype(np.float32)
        target_raw = np.expm1(np.maximum(values[validation_rows, axis] * axis_row.scale_log1p + axis_row.median_log1p, 0.0))
        prediction_raw = np.expm1(np.maximum(prediction_z * axis_row.scale_log1p + axis_row.median_log1p, 0.0))
        rows.append(pd.DataFrame({
            "profile_id": corpus.profiles.iloc[validation_rows]["profile_id"].tolist(),
            "target_axis_id": axis_row.target_axis_id,
            "canonical_name": axis_row.canonical_name,
            "loss_group": axis_row.loss_group,
            "target_g_per_100g": target_raw,
            "prediction_g_per_100g": prediction_raw,
            "positive_probability": np.nan,
            "target_positive": (target_raw > 0).astype(np.int64),
        }))
        print(f"RF validation axis {completed:03d}/{len(loss_axes)}: {axis_row.canonical_name} ({len(labels):,} train; {len(validation_rows):,} validation)")

    rf_profile = pd.concat(rows, ignore_index=True)
    rf_cells, rf_axis, rf_metrics = v9.summarize_source_free_cells(rf_profile, corpus)
    transformer_profile_path = transformer_output / "validation_complete_axis_panel_source_free_profile_predictions.csv"
    transformer_profile = pd.read_csv(transformer_profile_path)
    transformer_cells, transformer_axis, transformer_metrics = v9.summarize_source_free_cells(transformer_profile, corpus)
    rf_keys = set(zip(rf_profile["profile_id"], rf_profile["target_axis_id"]))
    transformer_keys = set(zip(transformer_profile["profile_id"], transformer_profile["target_axis_id"]))
    if rf_keys != transformer_keys:
        raise AssertionError(f"RF and Transformer do not score identical V9 validation profile-axis cells: {len(rf_keys)} vs {len(transformer_keys)}")
    comparison = pd.DataFrame([
        {"method": "FoodNutriGPT_v9_source_free_base", **transformer_metrics},
        {"method": "RandomForest_source_free", **rf_metrics},
    ])
    rf_profile.to_csv(output_dir / "rf_validation_complete_axis_panel_source_free_profile_predictions.csv", index=False)
    rf_cells.to_csv(output_dir / "rf_validation_complete_axis_panel_source_free_candidate_cells.csv", index=False)
    rf_axis.to_csv(output_dir / "rf_validation_complete_axis_panel_source_free_axis_metrics.csv", index=False)
    transformer_cells.to_csv(output_dir / "foodnutrigpt_validation_complete_axis_panel_source_free_candidate_cells.csv", index=False)
    transformer_axis.to_csv(output_dir / "foodnutrigpt_validation_complete_axis_panel_source_free_axis_metrics.csv", index=False)
    comparison.to_csv(output_dir / "validation_complete_axis_panel_comparison.csv", index=False)
    manifest = {
        "dataset_version": json.loads((data_dir / "build_manifest.json").read_text())["dataset_version"],
        "protocol": "same V9 complete validation profile-axis jobs; source-free encoder/features; equal-source candidate-cell scoring",
        "source_policy": "No source feature in RF or Transformer. Source is only an offline equal-weight label aggregation unit.",
        "rf_features": "32 train-fitted PCA components of frozen MiniLM text, visible normalized values, and observedness indicators",
        "mask_policy": "all axes in the target chemical family hidden before RF fitting and inference",
        "rf_hyperparameters": {
            "n_estimators": args.trees, "max_depth": 12, "max_features": 0.35,
            "min_samples_leaf": 5, "max_train_rows_per_axis": args.max_train_rows_per_axis,
        },
        "metrics": comparison.to_dict(orient="records"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

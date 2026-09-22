"""Locked, family-aware benchmark masks and baseline evaluation.

By default this module runs only on an internal training fold. Access to the
locked validation outcomes requires an explicit version confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DATASET_VERSION, RANDOM_SEED
from .util import read_component_csv, stable_id, write_csv, write_json


VISIBILITY_LEVELS = (0.10, 0.30, 0.50, 0.70)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().casefold() == "true"


def load_release(release_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    archive = np.load(release_dir / "canonical_profile_matrix.npz", allow_pickle=False)
    values = archive["values"].astype(np.float64)
    food_ids = archive["food_ids"].astype(str).tolist()
    component_ids = archive["component_ids"].astype(str).tolist()
    foods = pd.read_csv(release_dir / "ml_partition.csv", low_memory=False).set_index("food_concept_id").loc[food_ids].reset_index()
    components = read_component_csv(release_dir / "component_concept.csv.gz").set_index("component_concept_id").loc[component_ids].reset_index()
    return values, foods, components, {key: i for i, key in enumerate(food_ids)}, {key: j for j, key in enumerate(component_ids)}


def fit_train_normalizer(values: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    log_values = np.log1p(values[train_rows])
    center = np.nanmedian(log_values, axis=0)
    q1 = np.nanquantile(log_values, 0.25, axis=0)
    q3 = np.nanquantile(log_values, 0.75, axis=0)
    mad = np.nanmedian(np.abs(log_values - center), axis=0)
    scale = np.maximum((q3 - q1) / 1.349, mad * 1.4826)
    degenerate = ~np.isfinite(scale) | (scale <= 1e-8)
    scale[degenerate] = 1.0
    center[~np.isfinite(center)] = 0.0
    stats = pd.DataFrame({
        "train_log_median": center, "train_log_q1": q1, "train_log_q3": q3,
        "train_log_mad": mad, "train_robust_scale": scale,
        "degenerate_scale": degenerate,
    })
    return center, scale, stats


def transform(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.log1p(values) - center[None, :]) / scale[None, :]


def inverse_transform(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(values * scales + centers))


def select_hidden_families(families: dict[str, list[int]], ordered: list[str], visibility: float) -> list[str]:
    """Nearest feasible visibility, keeping at least one whole family visible."""
    total = sum(len(columns) for columns in families.values())
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for family in ordered:
        size = len(families[family])
        for count, chosen in list(reachable.items()):
            if count + size < total:
                reachable.setdefault(count + size, chosen + (family,))
    possible = [count for count in reachable if count > 0]
    if not possible:
        return []
    desired_hidden = total * (1 - visibility)
    best = min(possible, key=lambda count: (abs(count - desired_hidden), count))
    return list(reachable[best])


def make_fixed_masks(values: np.ndarray, foods: pd.DataFrame, components: pd.DataFrame, output_dir: Path,
                     *, dataset_version: str = DATASET_VERSION) -> pd.DataFrame:
    target_columns = np.flatnonzero(components["training_role"].eq("maskable_target").to_numpy())
    family_by_col = components["component_family"].fillna(components["component_concept_id"]).to_dict()
    rows: list[dict[str, Any]] = []
    for food_index, food in foods.iterrows():
        if not _truthy(food.get("benchmark_eligible", True)):
            continue
        observed = [j for j in target_columns if np.isfinite(values[food_index, j])]
        if _truthy(food.get("text_task_eligible", False)):
            for column in observed:
                rows.append({
                    "mask_id": stable_id("mask", "cold_start", food.food_concept_id, components.iloc[column]["component_concept_id"]),
                    "task": "text_cold_start", "visibility": 0.0,
                    "food_concept_id": food.food_concept_id,
                    "family_cluster_id": food.family_cluster_id,
                    "component_concept_id": components.iloc[column]["component_concept_id"],
                    "component_family": components.iloc[column]["component_family"], "mask_step": 1,
                    "partition": food.partition, "validation_panel": food.validation_panel,
                })
        if not _truthy(food.get("reconstruction_task_eligible", False)):
            continue
        families: dict[str, list[int]] = {}
        for j in observed:
            families.setdefault(str(family_by_col[j]), []).append(j)
        if len(families) < 2:
            continue
        for visibility in VISIBILITY_LEVELS:
            ordered = sorted(families, key=lambda family: hashlib.sha256(f"{RANDOM_SEED}:{food.food_concept_id}:{visibility}:{family}".encode()).hexdigest())
            hidden: list[tuple[str, int, int]] = []
            for step, family in enumerate(select_hidden_families(families, ordered, visibility), start=1):
                for column in families[family]:
                    hidden.append((family, column, step))
            visible_count = len(observed) - len(hidden)
            for family, column, step in hidden:
                rows.append({
                    "mask_id": stable_id("mask", "reconstruction", visibility, food.food_concept_id, components.iloc[column]["component_concept_id"]),
                    "task": "family_reconstruction", "visibility": visibility,
                    "food_concept_id": food.food_concept_id,
                    "family_cluster_id": food.family_cluster_id,
                    "component_concept_id": components.iloc[column]["component_concept_id"],
                    "component_family": family, "mask_step": step,
                    "observed_target_count": len(observed), "visible_target_count": visible_count,
                    "actual_target_visibility": visible_count / len(observed),
                    "partition": food.partition, "validation_panel": food.validation_panel,
                })
    masks = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(masks, output_dir / "fixed_benchmark_masks.csv.gz")
    write_json({
        "dataset_version": dataset_version, "seed": RANDOM_SEED,
        "selection_policy": "nearest_feasible_whole_family_visibility_with_nonempty_context",
        "visibility_definition": "Fraction of observed maskable target axes retained; component families are hidden together.",
        "visibility_levels": list(VISIBILITY_LEVELS), "rows": len(masks),
        "mask_sha256": hashlib.sha256(masks.to_csv(index=False).encode()).hexdigest(),
    }, output_dir / "mask_manifest.json")
    return masks


def _text_embeddings(texts: list[str], train_rows: np.ndarray, backend: str, cache_path: Path) -> tuple[np.ndarray, str]:
    corpus_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    train_hash = hashlib.sha256(np.asarray(train_rows, dtype=np.int64).tobytes()).hexdigest()
    manifest_path = cache_path.with_suffix(".json")
    if cache_path.exists():
        cached = np.load(cache_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if cached.shape[0] == len(texts) and manifest.get("corpus_sha256") == corpus_hash and manifest.get("train_rows_sha256") == train_hash:
                return cached, backend
    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for the preregistered text baseline; install requirements-colab.txt") from exc
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    elif backend == "tfidf-smoke":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=512)
        vectorizer.fit([texts[index] for index in train_rows])
        embedding = vectorizer.transform(texts).toarray().astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding /= np.maximum(norm, 1e-12)
    else:
        raise ValueError(f"Unknown text backend: {backend}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    write_json({
        "backend": backend,
        "model": "sentence-transformers/all-MiniLM-L6-v2" if backend == "sentence-transformers" else "TF-IDF(1,2)-gram smoke only",
        "corpus_sha256": corpus_hash,
        "train_rows_sha256": train_hash,
        "rows": len(texts),
        "fit_scope": "pretrained frozen encoder" if backend == "sentence-transformers" else "training rows only",
    }, manifest_path)
    return embedding, backend


def _food_text(foods: pd.DataFrame) -> list[str]:
    result = []
    for row in foods.itertuples(index=False):
        fields = [f"name: {row.canonical_name}"]
        for label, value in (("scientific name", row.scientific_name), ("group", row.food_group), ("subgroup", row.food_subgroup), ("type", row.food_type), ("part", row.part), ("processing", row.processing)):
            if pd.notna(value) and str(value).strip():
                fields.append(f"{label}: {value}")
        result.append(". ".join(fields))
    return result


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(y_true, y_pred).statistic)
    except ImportError:
        return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def _axis_metrics(group: pd.DataFrame) -> dict[str, Any]:
    y = group["true_raw"].to_numpy(float)
    p = group["pred_raw"].to_numpy(float)
    yz = group["true_z"].to_numpy(float)
    pz = group["pred_z"].to_numpy(float)
    raw_error = p - y
    z_error = pz - yz
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(group), "primary_robust_log_mae": float(np.mean(np.abs(z_error))),
        "robust_log_mse": float(np.mean(z_error**2)), "robust_log_rmse": float(np.sqrt(np.mean(z_error**2))),
        "raw_mae_g_per_100g": float(np.mean(np.abs(raw_error))),
        "raw_mse_g2_per_100g2": float(np.mean(raw_error**2)),
        "raw_rmse_g_per_100g": float(np.sqrt(np.mean(raw_error**2))),
        "raw_r2": 1.0 - float(np.sum(raw_error**2)) / sst if sst > 0 else np.nan,
        "spearman": _safe_spearman(y, p), "validation_min_g_per_100g": float(np.min(y)),
        "validation_max_g_per_100g": float(np.max(y)), "validation_sd_g_per_100g": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        "validation_q1_g_per_100g": float(np.quantile(y, 0.25)), "validation_q3_g_per_100g": float(np.quantile(y, 0.75)),
    }


def summarize_predictions(predictions: pd.DataFrame, components: pd.DataFrame, bootstrap_replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    axis_rows = []
    group_columns = ["evaluation_scope", "task", "visibility", "method", "component_concept_id"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row.update(_axis_metrics(group))
        axis_rows.append(row)
    axis = pd.DataFrame(axis_rows).merge(components[["component_concept_id", "canonical_name", "nutritional_role", "component_family", "train_raw_min", "train_raw_max", "train_raw_sd", "train_raw_q1", "train_raw_q3", "train_count"]], on="component_concept_id", how="left")
    overall_rows = []
    metric_columns = ["primary_robust_log_mae", "robust_log_mse", "robust_log_rmse", "raw_mae_g_per_100g", "raw_mse_g2_per_100g2", "raw_rmse_g_per_100g", "raw_r2", "spearman"]
    keys = ["evaluation_scope", "task", "visibility", "method"]
    for group_key, group in axis.groupby(keys, sort=True):
        row = dict(zip(keys, group_key, strict=True))
        row["axis_count"] = group["component_concept_id"].nunique()
        row["prediction_count"] = int(group["n"].sum())
        for metric in metric_columns:
            row[metric] = float(group[metric].mean())
        lo, hi = _cluster_bootstrap_primary(predictions, row, bootstrap_replicates)
        row["primary_mae_ci95_low"] = lo
        row["primary_mae_ci95_high"] = hi
        overall_rows.append(row)
    return axis, pd.DataFrame(overall_rows)


def _cluster_bootstrap_primary(predictions: pd.DataFrame, selection: dict[str, Any], replicates: int) -> tuple[float, float]:
    subset = predictions.copy()
    for column in ("evaluation_scope", "task", "visibility", "method"):
        subset = subset[subset[column].eq(selection[column])]
    families = subset["family_cluster_id"].dropna().unique()
    if len(families) < 2 or replicates <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    scores = []
    grouped = {key: value for key, value in subset.groupby("family_cluster_id")}
    for _ in range(replicates):
        sampled = rng.choice(families, size=len(families), replace=True)
        boot = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        per_axis = boot.assign(error=(boot["pred_z"] - boot["true_z"]).abs()).groupby("component_concept_id")["error"].mean()
        scores.append(float(per_axis.mean()))
    return float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def holm_primary_task_comparisons(predictions: pd.DataFrame, reference_method: str = "train_median") -> pd.DataFrame:
    """Paired one-sided tests over the two preregistered co-primary tasks."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return pd.DataFrame(columns=["method", "task", "visibility", "raw_p", "holm_p", "status"])
    primary = predictions[
        ((predictions["task"].eq("family_reconstruction")) & predictions["visibility"].eq(0.5))
        | predictions["task"].eq("text_cold_start")
    ].copy()
    primary["absolute_error"] = (primary["pred_z"] - primary["true_z"]).abs()
    clustered = primary.groupby(
        ["method", "task", "visibility", "family_cluster_id"], as_index=False
    )["absolute_error"].mean()
    reference = clustered[clustered["method"].eq(reference_method)][
        ["family_cluster_id", "task", "visibility", "absolute_error"]
    ].rename(columns={"absolute_error": "reference_error"})
    rows = []
    for method in sorted(set(clustered["method"]) - {reference_method}):
        method_rows = clustered[clustered["method"].eq(method)].merge(
            reference, on=["family_cluster_id", "task", "visibility"], how="inner"
        )
        method_tests = []
        for (task, visibility), group in method_rows.groupby(["task", "visibility"]):
            difference = group["absolute_error"] - group["reference_error"]
            if len(group) < 10 or np.allclose(difference, 0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(group["absolute_error"], group["reference_error"], alternative="less", zero_method="pratt").pvalue)
            method_tests.append({
                "method": method, "reference_method": reference_method, "task": task,
                "visibility": visibility, "family_cluster_n": len(group), "raw_p": p_value,
                "mean_error_difference": float(difference.mean()),
            })
        ordered = sorted(range(len(method_tests)), key=lambda index: method_tests[index]["raw_p"])
        running = 0.0
        total = len(method_tests)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * method_tests[index]["raw_p"])
            running = max(running, adjusted)
            method_tests[index]["holm_p"] = running
            method_tests[index]["status"] = "better_after_holm" if running < 0.05 and method_tests[index]["mean_error_difference"] < 0 else "not_significant"
        rows.extend(method_tests)
    return pd.DataFrame(rows)


def _knn_predictions(similarity: np.ndarray, train_y: np.ndarray, eval_rows: np.ndarray, k: int = 10) -> np.ndarray:
    available = np.flatnonzero(np.isfinite(train_y))
    if not len(available):
        return np.zeros(len(eval_rows))
    result = np.empty(len(eval_rows), dtype=float)
    for index, eval_row in enumerate(eval_rows):
        scores = similarity[eval_row, available]
        take = available[np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]]
        weights = np.maximum(similarity[eval_row, take], 0) + 1e-6
        result[index] = float(np.average(train_y[take], weights=weights))
    return result


def _composition_features(
    row_indices: np.ndarray,
    visibility: float,
    target_family: str,
    embeddings: np.ndarray,
    base_filled: np.ndarray,
    observed_indicator: np.ndarray,
    family_columns: dict[str, list[int]],
    hidden_lookup: dict[tuple[str, float], set[str]],
    food_ids: np.ndarray,
) -> np.ndarray:
    keep = np.ones(base_filled.shape[1], dtype=bool)
    keep[family_columns.get(target_family, [])] = False
    profile = base_filled[row_indices].copy()
    observed = observed_indicator[row_indices].copy()
    for position, row_index in enumerate(row_indices):
        food_id = str(food_ids[row_index])
        hidden_families = set(hidden_lookup.get((food_id, float(visibility)), set()))
        hidden_families.add(target_family)
        for family in hidden_families:
            columns = family_columns.get(str(family), [])
            profile[position, columns] = 0.0
            observed[position, columns] = 0.0
    return np.concatenate([embeddings[row_indices], profile[:, keep], observed[:, keep]], axis=1)


def run_baselines(
    release_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    *,
    unlock_validation: bool,
    confirm_version: str,
    smoke_target_limit: int,
    text_backend: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    dataset_version = json.loads((release_dir / "dataset_summary.json").read_text())["dataset_version"]
    values, foods, components, food_index, component_index = load_release(release_dir)
    masks_path = benchmark_dir / "fixed_benchmark_masks.csv.gz"
    all_masks = pd.read_csv(masks_path) if masks_path.exists() else make_fixed_masks(values, foods, components, benchmark_dir, dataset_version=dataset_version)
    masks = all_masks.copy()
    if unlock_validation:
        if confirm_version != dataset_version:
            raise ValueError(f"Locked validation requires --confirm-version {dataset_version}")
        train_rows = np.flatnonzero(foods["partition"].eq("train").to_numpy())
        eval_food_mask = foods["partition"].eq("validation")
        evaluation_scope = "locked_validation"
        masks = masks[masks["partition"].eq("validation")].copy()
    else:
        train_rows = np.flatnonzero(foods["partition"].eq("train").to_numpy() & foods["cv_fold"].ne(0).to_numpy())
        eval_food_mask = foods["partition"].eq("train") & foods["cv_fold"].eq(0)
        evaluation_scope = "internal_fold_0_smoke"
        smoke_ids = set(foods.loc[eval_food_mask, "food_concept_id"])
        masks = masks[masks["food_concept_id"].isin(smoke_ids)].copy()

    target_ids = components.loc[components["training_role"].eq("maskable_target"), "component_concept_id"].tolist()
    if not unlock_validation and smoke_target_limit:
        target_ids = sorted(target_ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:smoke_target_limit]
        masks = masks[masks["component_concept_id"].isin(target_ids)].copy()
    if masks.empty:
        raise ValueError("No benchmark mask rows are available for the selected evaluation scope.")

    center, scale, normalization = fit_train_normalizer(values, train_rows)
    normalization.insert(0, "component_concept_id", components["component_concept_id"].values)
    write_csv(normalization, output_dir / "train_only_normalization.csv")
    z = transform(values, center, scale)
    texts = _food_text(foods)
    embeddings, actual_backend = _text_embeddings(texts, train_rows, text_backend, output_dir / f"food_text_embeddings_{text_backend}.npy")
    train_embeddings = embeddings[train_rows]
    similarity = embeddings @ train_embeddings.T
    group_values = foods["food_group"].fillna("unknown").astype(str).to_numpy()
    train_groups = group_values[train_rows]

    predictions = []
    methods = ["train_mean", "train_median", "food_group_mean", "text_knn"]
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    methods.extend(["random_forest", "xgboost"])

    family_columns: dict[str, list[int]] = {}
    for j, row in components.iterrows():
        family_columns.setdefault(str(row.component_family), []).append(j)
    base_profile = z.copy()
    observed_indicator = np.isfinite(base_profile).astype(np.float32)
    base_filled = np.nan_to_num(base_profile, nan=0.0).astype(np.float32)
    hidden_lookup = all_masks[all_masks["task"].eq("family_reconstruction")].groupby(
        ["food_concept_id", "visibility"]
    )["component_family"].apply(set).to_dict()
    all_food_ids = foods["food_concept_id"].astype(str).to_numpy()
    reconstruction_eligible = foods["reconstruction_task_eligible"].map(_truthy).to_numpy()

    for component_id, mask_group in masks.groupby("component_concept_id", sort=True):
        if component_id not in component_index:
            continue
        target_col = component_index[component_id]
        train_y = z[train_rows, target_col]
        train_available = np.isfinite(train_y)
        if train_available.sum() < 20:
            continue
        axis_center = center[target_col]
        axis_scale = scale[target_col]
        eval_rows = mask_group["food_concept_id"].map(food_index).to_numpy(int)
        true_z = z[eval_rows, target_col]
        valid = np.isfinite(true_z)
        mask_group = mask_group.iloc[np.flatnonzero(valid)].copy().reset_index(drop=True)
        eval_rows = eval_rows[valid]
        true_z = true_z[valid]
        if not len(eval_rows):
            continue
        train_axis = train_y[train_available]
        mean_pred = float(np.mean(train_axis))
        median_pred = float(np.median(train_axis))
        group_means = pd.DataFrame({"group": train_groups[train_available], "y": train_axis}).groupby("group")["y"].mean().to_dict()
        baseline_predictions = {
            "train_mean": np.full(len(eval_rows), mean_pred),
            "train_median": np.full(len(eval_rows), median_pred),
            "food_group_mean": np.asarray([group_means.get(group_values[row], mean_pred) for row in eval_rows]),
            "text_knn": _knn_predictions(similarity, train_y, eval_rows, k=10),
        }

        task_models: dict[tuple[str, float, str], Any] = {}
        for task, visibility in mask_group[["task", "visibility"]].drop_duplicates().itertuples(index=False, name=None):
            if task == "text_cold_start":
                model_available = train_available
                model_train_rows = train_rows[model_available]
                x_train = embeddings[model_train_rows]
            else:
                family = str(components.iloc[target_col].component_family)
                model_available = train_available & reconstruction_eligible[train_rows]
                model_train_rows = train_rows[model_available]
                x_train = _composition_features(
                    model_train_rows, float(visibility), family, embeddings, base_filled,
                    observed_indicator, family_columns, hidden_lookup, all_food_ids,
                )
            model_y = train_y[model_available]
            if len(model_y) < 20:
                continue
            if RandomForestRegressor is not None:
                rf = RandomForestRegressor(n_estimators=120 if unlock_validation else 30, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED)
                rf.fit(x_train, model_y)
                task_models[(task, float(visibility), "random_forest")] = rf
            if XGBRegressor is not None:
                xgb = XGBRegressor(n_estimators=300 if unlock_validation else 60, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_SEED)
                xgb.fit(x_train, model_y)
                task_models[(task, float(visibility), "xgboost")] = xgb

        for (task, visibility), task_group in mask_group.groupby(["task", "visibility"], sort=True):
            local_positions = task_group.index.to_numpy(int)
            task_eval_rows = eval_rows[local_positions]
            if task == "text_cold_start":
                x_eval = embeddings[task_eval_rows]
            else:
                family = str(components.iloc[target_col].component_family)
                x_eval = _composition_features(
                    task_eval_rows, float(visibility), family, embeddings, base_filled,
                    observed_indicator, family_columns, hidden_lookup, all_food_ids,
                )
            task_predictions = {method: values_[local_positions] for method, values_ in baseline_predictions.items()}
            for method in ("random_forest", "xgboost"):
                model = task_models.get((task, float(visibility), method))
                if model is not None:
                    task_predictions[method] = model.predict(x_eval)
            for method, pred_z in task_predictions.items():
                pred_raw = inverse_transform(np.asarray(pred_z), np.full(len(pred_z), axis_center), np.full(len(pred_z), axis_scale))
                true_raw = values[task_eval_rows, target_col]
                for position, (_, mask_row) in enumerate(task_group.iterrows()):
                    predictions.append({
                        "mask_id": mask_row.mask_id, "evaluation_scope": evaluation_scope,
                        "task": task, "visibility": mask_row.visibility, "method": method,
                        "food_concept_id": mask_row.food_concept_id, "family_cluster_id": mask_row.family_cluster_id,
                        "component_concept_id": component_id, "true_z": true_z[local_positions[position]],
                        "pred_z": float(pred_z[position]), "true_raw": float(true_raw[position]),
                        "pred_raw": float(pred_raw[position]),
                    })
    prediction_df = pd.DataFrame(predictions)
    if prediction_df.empty:
        raise RuntimeError("Baseline evaluation produced no predictions.")
    axis, overall = summarize_predictions(prediction_df, components, bootstrap_replicates)
    holm = holm_primary_task_comparisons(prediction_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(prediction_df, output_dir / "baseline_predictions.csv.gz")
    write_csv(axis, output_dir / "baseline_axis_metrics.csv")
    write_csv(overall, output_dir / "baseline_overall_metrics.csv")
    write_csv(holm, output_dir / "holm_primary_task_comparisons.csv")
    summary = {
        "dataset_version": dataset_version, "evaluation_scope": evaluation_scope,
        "locked_validation_opened": unlock_validation, "text_backend": actual_backend,
        "methods": sorted(prediction_df["method"].unique()), "mask_rows_evaluated": prediction_df["mask_id"].nunique(),
        "prediction_rows": len(prediction_df), "axis_count": prediction_df["component_concept_id"].nunique(),
        "bootstrap_replicates": bootstrap_replicates,
        "warning": "Internal-fold smoke results are implementation checks, not reportable locked-validation benchmark results." if not unlock_validation else "Validation was explicitly unlocked; any data repair now requires a new dataset version and complete rerun.",
    }
    write_json(summary, output_dir / "baseline_run_manifest.json")
    print(json.dumps(summary, indent=2))
    print(overall.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-dir", type=Path, default=Path("data/processed/scientific_food_composition_v1/release"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/splits/scientific_food_composition_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/scientific_food_composition_v1_baselines_smoke"))
    parser.add_argument("--masks-only", action="store_true")
    parser.add_argument("--unlock-validation", action="store_true")
    parser.add_argument("--confirm-version", default="")
    parser.add_argument("--smoke-target-limit", type=int, default=8)
    parser.add_argument("--text-backend", choices=["sentence-transformers", "tfidf-smoke"], default="tfidf-smoke")
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    release_dir = root / args.release_dir
    benchmark_dir = root / args.benchmark_dir
    values, foods, components, _, _ = load_release(release_dir)
    version = json.loads((release_dir / "dataset_summary.json").read_text())["dataset_version"]
    masks = make_fixed_masks(values, foods, components, benchmark_dir, dataset_version=version)
    print(f"Fixed mask rows: {len(masks):,}")
    if not args.masks_only:
        run_baselines(
            release_dir, benchmark_dir, root / args.output_dir,
            unlock_validation=args.unlock_validation, confirm_version=args.confirm_version,
            smoke_target_limit=args.smoke_target_limit, text_backend=args.text_backend,
            bootstrap_replicates=args.bootstrap_replicates,
        )


if __name__ == "__main__":
    main()

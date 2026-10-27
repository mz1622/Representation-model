#!/usr/bin/env python3
"""Representation and UMAP audit for dependency-mask v1 checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    checkpoint_metadata, extract_food_embeddings, load_dependency_model,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SEED = 20261027


def mapped_category(group):
    group = str(group).lower().strip()
    mapping = {
        "fruits and fruit juices": "Fruit", "fruits": "Fruit",
        "vegetables and vegetable products": "Vegetable", "vegetables": "Vegetable",
        "gourds": "Vegetable", "beef products": "Meat", "pork products": "Meat",
        "poultry products": "Meat", "lamb, veal, and game products": "Meat",
        "lamb, veal and game": "Meat", "finfish and shellfish products": "Fish",
        "aquatic foods": "Fish", "cereal grains and pasta": "Grain",
        "cereals, grains and pasta": "Grain", "breakfast cereals": "Grain",
        "cereals and cereal products": "Grain", "dairy and egg products": "Dairy_egg",
        "milk and milk products": "Dairy_egg", "eggs": "Dairy_egg",
        "legumes and legume products": "Legume", "pulses": "Legume",
        "soy": "Legume", "nut and seed products": "Nut_seed",
        "nuts and seeds": "Nut_seed", "nuts": "Nut_seed", "fats and oils": "Fat_oil",
    }
    return mapping.get(group, "")


def category_metrics(x, labels, train, validation):
    nn = NearestNeighbors(n_neighbors=20, metric="cosine", n_jobs=4).fit(x[train])
    neighbors = train[nn.kneighbors(x[validation], return_distance=False)]
    result = {}
    for k in (5, 10, 20):
        purity = (labels[neighbors[:, :k]] == labels[validation, None]).mean(1)
        result[f"purity_{k}"] = float(purity.mean())
        result[f"macro_purity_{k}"] = float(
            pd.Series(purity).groupby(labels[validation]).mean().mean()
        )
    scaler = StandardScaler().fit(x[train])
    classifier = LogisticRegression(
        max_iter=1_500, class_weight="balanced", random_state=SEED, n_jobs=4,
    )
    classifier.fit(scaler.transform(x[train]), labels[train])
    predicted = classifier.predict(scaler.transform(x[validation]))
    result["macro_f1"] = float(f1_score(labels[validation], predicted, average="macro"))
    result["balanced_accuracy"] = float(
        balanced_accuracy_score(labels[validation], predicted)
    )
    result["silhouette_cosine"] = float(silhouette_score(
        x[validation], labels[validation], metric="cosine",
        sample_size=min(2_000, len(validation)), random_state=SEED,
    ))
    return result


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.monotonic()
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    profiles = data.profiles.copy()
    labels = np.full(len(profiles), "", dtype=object)
    approved = profiles.source_key.isin(["usda_sr_legacy", "cnf", "foodb"])
    labels[approved] = profiles.loc[approved, "food_group"].map(mapped_category)
    train = data.train[labels[data.train] != ""]
    validation = data.validation[labels[data.validation] != ""]
    if not len(train) or not len(validation):
        raise ValueError("category mapping produced an empty partition")

    raw = np.where(data.observed, data.values, np.nan)
    medians = np.nanmedian(raw[data.train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)
    filled = np.where(np.isfinite(raw), raw, medians).astype(np.float32)
    scaler = StandardScaler().fit(filled[data.train])
    standardized = scaler.transform(filled).astype(np.float32)
    nutrients_mask = np.concatenate(
        (standardized, data.observed.astype(np.float32)), axis=1
    )
    missingness = data.observed.astype(np.float32)
    pca = PCA(n_components=50, random_state=SEED).fit(nutrients_mask[data.train])
    representations = {
        "Nutrients_mask": nutrients_mask,
        "Missingness_only": missingness,
        "PCA50": pca.transform(nutrients_mask).astype(np.float32),
    }
    checkpoints = {
        "V4": args.checkpoint_dir / "v4_best_validation.pt",
        "V5": args.checkpoint_dir / "v5_best_validation.pt",
        "V5_CCE": args.checkpoint_dir / "cce_best_validation.pt",
    }
    checkpoint_info = {}
    models = {}
    rows = np.arange(len(profiles), dtype=np.int64)
    for name, path in checkpoints.items():
        print(f"extracting {name} FOOD states", flush=True)
        model, checkpoint, _ = load_dependency_model(
            path, data, args.policy_manifest, device
        )
        representations[name] = extract_food_embeddings(
            model, data, rows, device, batch_size=128
        )
        np.save(args.output_dir / f"embedding_{name}.npy", representations[name])
        checkpoint_info[name] = checkpoint_metadata(path, checkpoint)
        models[name] = model

    result_rows = []
    for name, representation in representations.items():
        print(f"category metrics {name}", flush=True)
        result_rows.append({
            "method": name,
            **category_metrics(representation, labels, train, validation),
        })
    category = pd.DataFrame(result_rows)
    category.to_csv(args.output_dir / "category_metrics.csv", index=False)

    # How much of the apparent structure is source-specific coverage rather than food class?
    source_train = data.train
    source_validation = data.validation
    source_labels = profiles.source_key.astype(str).to_numpy()
    source_metrics = category_metrics(
        missingness, source_labels, source_train, source_validation
    )
    write_json(args.output_dir / "missingness_source_metrics.json", source_metrics)

    import umap
    selected = np.random.default_rng(SEED).choice(
        data.validation, min(4_000, len(data.validation)), replace=False
    )
    coordinates = umap.UMAP(
        n_neighbors=15, min_dist=0.15, metric="cosine",
        random_state=SEED, n_jobs=1,
    ).fit_transform(representations["V5_CCE"][selected])
    pd.DataFrame({
        "profile_index": selected, "x": coordinates[:, 0], "y": coordinates[:, 1],
        "category": labels[selected],
        "source": profiles.iloc[selected].source_key.to_numpy(),
        "observed_axes": data.observed[selected].sum(1),
    }).to_csv(args.output_dir / "umap_v5_cce.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    categories = sorted(set(labels[selected]) - {""})
    colors = {value: plt.get_cmap("tab10")(i) for i, value in enumerate(categories)}
    for value in categories:
        keep = labels[selected] == value
        axes[0].scatter(
            coordinates[keep, 0], coordinates[keep, 1], s=6,
            color=colors[value], alpha=0.68, label=value, rasterized=True,
        )
    axes[0].legend(fontsize=6, markerscale=2, loc="best")
    axes[0].set_title("Mapped food category")
    source = pd.Categorical(profiles.iloc[selected].source_key)
    axes[1].scatter(
        coordinates[:, 0], coordinates[:, 1], c=source.codes,
        cmap="tab20", s=5, alpha=0.7, rasterized=True,
    )
    axes[1].set_title("Database source")
    image = axes[2].scatter(
        coordinates[:, 0], coordinates[:, 1], c=data.observed[selected].sum(1),
        cmap="viridis", s=5, rasterized=True,
    )
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    axes[2].set_title("Observed-axis count")
    protein = np.flatnonzero(data.axes.canonical_name.str.lower().eq("protein, total"))
    if not len(protein):
        protein = np.flatnonzero(data.axes.canonical_name.str.lower().str.startswith("protein"))
    values = np.where(
        data.observed[selected, protein[0]],
        np.log1p(data.raw[selected, protein[0]]), np.nan,
    )
    image = axes[3].scatter(
        coordinates[:, 0], coordinates[:, 1], c=values,
        cmap="plasma", s=5, rasterized=True,
    )
    fig.colorbar(image, ax=axes[3], fraction=0.046)
    axes[3].set_title("log1p protein (g/100g)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Dependency-mask V5+CCE FOOD representation", fontsize=14)
    save_figure(args.output_dir / "umap_v5_cce.png")

    # UMAP seed stability measures neighbourhood overlap, not coordinate alignment.
    stability = []
    reference = None
    x = representations["V5_CCE"][selected]
    for seed in (SEED, SEED + 1, SEED + 2):
        coord = umap.UMAP(
            n_neighbors=15, min_dist=0.15, metric="cosine",
            random_state=seed, n_jobs=1,
        ).fit_transform(x)
        neighbor = NearestNeighbors(n_neighbors=11).fit(coord).kneighbors(
            coord, return_distance=False
        )[:, 1:]
        if reference is None:
            reference = neighbor
        overlap = np.mean([
            len(set(left) & set(right)) / 10 for left, right in zip(neighbor, reference)
        ])
        stability.append({"seed": seed, "neighbor_overlap10_vs_first": overlap})
    pd.DataFrame(stability).to_csv(
        args.output_dir / "umap_seed_stability.csv", index=False
    )

    # Robustness of the FOOD state to randomly hidden observed inputs.
    counts = data.observed[data.validation].sum(1)
    threshold = max(20, float(np.quantile(counts, 2 / 3)))
    robust_rows = data.validation[counts >= threshold]
    robust_rows = np.random.default_rng(SEED).choice(
        robust_rows, min(1_000, len(robust_rows)), replace=False
    )
    full = representations["V5_CCE"][robust_rows]
    full_neighbors = NearestNeighbors(n_neighbors=10, metric="cosine", n_jobs=4).fit(
        representations["V5_CCE"][train]
    ).kneighbors(full, return_distance=False)
    labelled_position = np.flatnonzero(labels[robust_rows] != "")
    robustness = []
    for fraction in (0.1, 0.3, 0.5):
        for repeat in range(5):
            visible = data.observed[robust_rows].copy()
            visible &= np.random.default_rng(SEED + 100 * repeat + int(fraction * 10)).random(
                visible.shape
            ) >= fraction
            embedding = extract_food_embeddings(
                models["V5_CCE"], data, robust_rows, device,
                batch_size=128, visible_override=visible,
            )
            cosine = (normalize(full) * normalize(embedding)).sum(1)
            neighbor = NearestNeighbors(
                n_neighbors=10, metric="cosine", n_jobs=4
            ).fit(representations["V5_CCE"][train]).kneighbors(
                embedding, return_distance=False
            )
            overlap = np.mean([
                len(set(left) & set(right)) / 10
                for left, right in zip(neighbor, full_neighbors)
            ])
            labelled_neighbors = train[neighbor[labelled_position]]
            purity = (
                labels[labelled_neighbors] == labels[robust_rows[labelled_position], None]
            ).mean(1)
            macro_purity = pd.Series(purity).groupby(
                labels[robust_rows[labelled_position]]
            ).mean().mean()
            robustness.append({
                "fraction_hidden": fraction, "repeat": repeat,
                "profiles": len(robust_rows),
                "cosine_to_full": float(cosine.mean()),
                "neighbor_overlap10": float(overlap),
                "macro_purity10": float(macro_purity),
            })
        print(f"robustness fraction {fraction} complete", flush=True)
    pd.DataFrame(robustness).to_csv(args.output_dir / "robustness.csv", index=False)

    manifest = {
        "status": "complete", "version": "dependency_mask_representation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "seed": SEED, "profiles": len(profiles),
        "train_profiles": len(data.train), "validation_profiles": len(data.validation),
        "category_train": len(train), "category_validation": len(validation),
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoints": checkpoint_info,
        "food_name_model_input": False, "source_model_input": False,
        "category_and_source_use": "evaluation labels only",
        "embedding_protocol": "first-pass FOOD state; all observed numerical tokens; empty target query",
        "umap_profiles": len(selected),
        "software": {"torch": torch.__version__, "umap": umap.__version__},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

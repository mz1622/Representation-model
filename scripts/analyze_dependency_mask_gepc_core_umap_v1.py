#!/usr/bin/env python3
"""Audit V5+GEPC FOOD embeddings on a predeclared homogeneous FoodEx2 subset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.foodex2_umap_v1 import (  # noqa: E402
    DAIRY, FATS, FISH, FRUIT, GRAINS, LEGUMES, MEAT, SUGAR, VEGETABLES,
    DISPLAY_NAMES, assign_foodex2_labels,
)
from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    extract_food_embeddings, load_dependency_model,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData  # noqa: E402


SEED = 20261109
CAP_PER_CLASS = 1_000
CORE_LEVEL1 = (
    GRAINS, VEGETABLES, LEGUMES, FRUIT, MEAT, FISH, DAIRY, FATS, SUGAR,
)
CORE_COLORS = {
    GRAINS: "#C49A00",
    VEGETABLES: "#2B8C3E",
    LEGUMES: "#7A9A01",
    FRUIT: "#E87027",
    MEAT: "#B23A48",
    FISH: "#247BA0",
    DAIRY: "#70B7D3",
    FATS: "#8C6BB1",
    SUGAR: "#D95F9D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / np.maximum(norm, 1e-12)).astype(np.float32)


def balanced_indices(labels: pd.Series) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected = []
    for category in CORE_LEVEL1:
        candidates = np.flatnonzero(labels.eq(category).to_numpy())
        if len(candidates) < CAP_PER_CLASS:
            raise ValueError(
                f"core category {category!r} has {len(candidates)} profiles; "
                f"expected at least {CAP_PER_CLASS}"
            )
        selected.append(rng.choice(candidates, CAP_PER_CLASS, replace=False))
    return np.sort(np.concatenate(selected))


def fit_umap(embedding: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=SEED,
        n_jobs=1,
        low_memory=True,
    ).fit_transform(normalize(embedding)).astype(np.float32)


def geometry_metrics(values: np.ndarray, labels: np.ndarray, method: str, space: str):
    neighbors = NearestNeighbors(
        n_neighbors=16, metric="euclidean", algorithm="auto"
    ).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    purity = (labels[neighbors] == labels[:, None]).mean(axis=1)
    per_class = pd.Series(purity).groupby(labels).mean()
    return {
        "method": method,
        "space": space,
        "profiles": int(len(values)),
        "classes": int(pd.Series(labels).nunique()),
        "neighbor_purity_15": float(purity.mean()),
        "macro_neighbor_purity_15": float(per_class.mean()),
        "silhouette_sample_10000": float(silhouette_score(
            values, labels, metric="euclidean", sample_size=10_000,
            random_state=SEED,
        )),
    }


def save_plot(points: pd.DataFrame, output_dir: Path) -> None:
    count = 3 if "dab_x" in points else 2
    fig, axes = plt.subplots(
        1, count, figsize=(20.0, 6.8) if count == 3 else (15.2, 6.8),
        facecolor="white",
    )
    methods = [
        (axes[0], "V5+CCE", "cce"),
        (axes[1], "V5+GEPC", "gepc"),
    ]
    if "dab_x" in points:
        methods.append((axes[2], "V5+GEPC+DAB", "dab"))
    for number, (axis, method, prefix) in enumerate(methods):
        rng = np.random.default_rng(SEED + number)
        for category in CORE_LEVEL1:
            subset = points[points.foodex2_level1.eq(category)]
            order = rng.permutation(len(subset))
            axis.scatter(
                subset[f"{prefix}_x"].to_numpy()[order],
                subset[f"{prefix}_y"].to_numpy()[order],
                s=3.2, alpha=0.66, linewidths=0, rasterized=True,
                color=CORE_COLORS[category],
            )
        axis.set_title(method, fontsize=14, loc="left")
        axis.set_xlabel("UMAP1", fontsize=10)
        axis.set_ylabel("UMAP2", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6,
               color=CORE_COLORS[category],
               label=f"{DISPLAY_NAMES[category]} ({CAP_PER_CLASS:,})")
        for category in CORE_LEVEL1
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.7,
               loc="center left", bbox_to_anchor=(0.80, 0.50))
    center = 0.40 if len(methods) == 2 else 0.42
    fig.suptitle("Balanced composition-homogeneous FoodEx2 core", fontsize=16, x=center)
    fig.text(
        center, 0.925,
        "9,000 profiles; labels select/display points only and never enter UMAP or model training",
        ha="center", fontsize=9.5, color="#5A6472",
    )
    fig.tight_layout(rect=(0, 0, 0.82 if len(methods) == 3 else 0.79, 0.90), w_pad=1.5)
    filename = (
        "umap_balanced_core_cce_gepc_dab.png"
        if count == 3 else "umap_balanced_core_cce_vs_gepc.png"
    )
    fig.savefig(output_dir / filename,
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cce-embedding", type=Path, required=True)
    parser.add_argument("--gepc-embedding", type=Path, required=True)
    parser.add_argument("--dab-checkpoint", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    profiles = data.profiles.reset_index(drop=True)
    labels = assign_foodex2_labels(profiles).reset_index(drop=True)
    cce = np.load(args.cce_embedding)
    gepc = np.load(args.gepc_embedding)
    dab = None
    if args.dab_checkpoint is not None:
        if args.policy_manifest is None:
            raise ValueError("--policy-manifest is required with --dab-checkpoint")
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        )
        model, checkpoint, _ = load_dependency_model(
            args.dab_checkpoint, data, args.policy_manifest, device
        )
        if checkpoint.get("stage") != "gepc_dab":
            raise ValueError("DAB checkpoint has the wrong stage")
        dab = extract_food_embeddings(
            model, data, np.arange(len(profiles), dtype=np.int64), device,
            batch_size=256, forward_pass=1,
        )
        np.save(args.output_dir / "embedding_v5_gepc_dab_all_profiles.npy", dab)
    expected = (len(profiles), 192)
    shapes = [cce.shape, gepc.shape] + ([] if dab is None else [dab.shape])
    if any(shape != expected for shape in shapes):
        raise ValueError(f"embedding shape mismatch: {shapes}, expected {expected}")

    core = labels.foodex2_level1.isin(CORE_LEVEL1)
    core_counts = labels.loc[core, "foodex2_level1"].value_counts()
    core_counts.rename_axis("foodex2_level1").rename("profiles").reset_index().to_csv(
        args.output_dir / "core_label_distribution.csv", index=False
    )
    selected = balanced_indices(labels.foodex2_level1)
    selected_labels = labels.iloc[selected].foodex2_level1.to_numpy()
    cce_core = cce[selected]
    gepc_core = gepc[selected]
    cce_xy = fit_umap(cce_core)
    gepc_xy = fit_umap(gepc_core)
    dab_core = None if dab is None else dab[selected]
    dab_xy = None if dab_core is None else fit_umap(dab_core)

    points = pd.DataFrame({
        "profile_index": selected,
        "partition": profiles.iloc[selected].partition.to_numpy(),
        "source_key": profiles.iloc[selected].source_key.astype(str).to_numpy(),
        "foodex2_level1": selected_labels,
        "display_category": labels.iloc[selected].display_category.to_numpy(),
        "label_provenance": labels.iloc[selected].label_provenance.to_numpy(),
        "cce_x": cce_xy[:, 0], "cce_y": cce_xy[:, 1],
        "gepc_x": gepc_xy[:, 0], "gepc_y": gepc_xy[:, 1],
    })
    if dab_xy is not None:
        points["dab_x"] = dab_xy[:, 0]
        points["dab_y"] = dab_xy[:, 1]
    points.to_csv(args.output_dir / "balanced_core_points.csv", index=False)
    save_plot(points, args.output_dir)

    rows = []
    methods = [
        ("V5+CCE", cce_core, cce_xy),
        ("V5+GEPC", gepc_core, gepc_xy),
    ]
    if dab_core is not None:
        methods.append(("V5+GEPC+DAB", dab_core, dab_xy))
    for method, embedding, coordinates in methods:
        rows.append(geometry_metrics(
            normalize(embedding), selected_labels, method, "original_192d_l2"
        ))
        rows.append(geometry_metrics(
            coordinates, selected_labels, method, "umap_2d"
        ))
        source = points.source_key.to_numpy()
        rows.append(geometry_metrics(
            normalize(embedding), source, method, "original_192d_l2_source_labels"
        ))
        rows.append(geometry_metrics(
            coordinates, source, method, "umap_2d_source_labels"
        ))
    pd.DataFrame(rows).to_csv(args.output_dir / "balanced_core_metrics.csv", index=False)
    points.groupby(["foodex2_level1", "label_provenance"]).size().rename(
        "profiles"
    ).reset_index().to_csv(args.output_dir / "balanced_core_provenance.csv", index=False)

    manifest = {
        "schema_version": "dependency_mask_gepc_core_umap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_scope": "nine predeclared composition-homogeneous FoodEx2 level-1 categories",
        "excluded_label_types": (
            "mixed dishes, seasonings/sauces, beverages, juices, special-purpose foods, "
            "young-population foods, additives, human milk, eggs, starchy roots, and unlabelled"
        ),
        "selection_uses_embedding_or_coordinates": False,
        "classes": [DISPLAY_NAMES[value] for value in CORE_LEVEL1],
        "cap_per_class": CAP_PER_CLASS,
        "profiles": int(len(selected)),
        "umap": {
            "l2_normalized": True, "n_neighbors": 15, "min_dist": 0.3,
            "metric": "euclidean", "seed": SEED,
        },
        "inputs": {
            "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
            "cce_embedding_sha256": sha256(args.cce_embedding),
            "gepc_embedding_sha256": sha256(args.gepc_embedding),
            "dab_checkpoint_sha256": (
                None if args.dab_checkpoint is None else sha256(args.dab_checkpoint)
            ),
        },
        "profile_level_coordinates_committed": False,
        "embeddings_committed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

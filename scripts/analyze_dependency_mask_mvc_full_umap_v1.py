#!/usr/bin/env python3
"""Render all development profiles with the UMAP workflow used by scGPT examples."""
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

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    extract_food_embeddings, load_dependency_model,
)
from foodcomp.foodex2_umap_v1 import assign_foodex2_labels  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData  # noqa: E402
from build_foodex2_umap_v1 import file_record  # noqa: E402
from build_foodex2_umap_v2_macro import CATEGORY_ORDER, COLORS  # noqa: E402
from plot_dependency_mask_umap_v2 import SOURCE_NAMES  # noqa: E402


SEED = 20261027
UNLABELLED = "#D8DCE2"
SOURCE_CMAP = "gist_ncar"


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def l2_normalize(values):
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-12)).astype(np.float32)


def fit_scgpt_style_umap(embedding):
    import umap
    # scGPT's official integration example L2-normalizes the CLS embedding,
    # constructs the default 15-neighbour graph, and uses min_dist=0.3.
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=SEED,
        n_jobs=1,
        low_memory=True,
    ).fit_transform(l2_normalize(embedding)).astype(np.float32)


def clean_axis(axis, title):
    axis.set_title(title, fontsize=13, loc="left")
    axis.set_xlabel("UMAP1", fontsize=10)
    axis.set_ylabel("UMAP2", fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def shuffled_scatter(axis, coordinates, color, rng, *, size=1.6, alpha=0.72):
    order = rng.permutation(len(coordinates))
    axis.scatter(
        coordinates[order, 0], coordinates[order, 1], c=np.asarray(color)[order],
        s=size, alpha=alpha, linewidths=0, rasterized=True,
    )


def save_category_plot(points, output_dir):
    color = points.macro_category.map(COLORS).fillna(UNLABELLED).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), facecolor="white")
    for number, (axis, method, columns) in enumerate((
        (axes[0], "V5+CCE", ["cce_x", "cce_y"]),
        (axes[1], "V5+MVC", ["mvc_x", "mvc_y"]),
    )):
        shuffled_scatter(
            axis, points[columns].to_numpy(np.float32), color,
            np.random.default_rng(SEED + number),
        )
        clean_axis(axis, method)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.5,
               color=COLORS[category],
               label=f"{category} ({points.macro_category.eq(category).sum():,})")
        for category in CATEGORY_ORDER
    ]
    handles.append(Line2D(
        [], [], marker="o", linestyle="", markersize=5.5, color=UNLABELLED,
        label=f"Unlabelled ({points.macro_category.eq('Unlabelled').sum():,})",
    ))
    fig.legend(handles=handles, frameon=False, fontsize=8.7, ncol=1,
               loc="center left", bbox_to_anchor=(0.82, 0.5), handletextpad=0.4)
    fig.suptitle("All profiles — FoodEx2 macro category", fontsize=16, x=0.41)
    fig.text(
        0.41, 0.93,
        "scGPT-style UMAP: L2-normalized [FOOD], 15-neighbour graph, min_dist=0.3, frame off",
        ha="center", fontsize=9.5, color="#5A6472",
    )
    fig.tight_layout(rect=(0, 0, 0.81, 0.91), w_pad=1.8)
    fig.savefig(output_dir / "umap_all_profiles_food_category.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def source_palette(sources):
    values = sorted(map(str, sources))
    missing = set(values) - set(SOURCE_NAMES)
    if missing:
        raise ValueError(f"missing source display names: {sorted(missing)}")
    cmap = plt.get_cmap(SOURCE_CMAP, len(values))
    return values, {value: cmap(i) for i, value in enumerate(values)}


def save_source_plot(points, output_dir):
    sources, palette = source_palette(points.source.unique())
    color = points.source.map(palette).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 7.4), facecolor="white")
    for number, (axis, method, columns) in enumerate((
        (axes[0], "V5+CCE", ["cce_x", "cce_y"]),
        (axes[1], "V5+MVC", ["mvc_x", "mvc_y"]),
    )):
        shuffled_scatter(
            axis, points[columns].to_numpy(np.float32), color,
            np.random.default_rng(SEED + 10 + number),
        )
        clean_axis(axis, method)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5,
               color=palette[source],
               label=f"{SOURCE_NAMES[source]} ({points.source.eq(source).sum():,})")
        for source in sources
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7.3, ncol=2,
               loc="center left", bbox_to_anchor=(0.79, 0.5),
               handletextpad=0.3, columnspacing=0.7)
    fig.suptitle("All profiles — database source", fontsize=16, x=0.395)
    fig.text(
        0.395, 0.93,
        "Same geometry as the food-category view; source is shown as the scGPT batch analogue",
        ha="center", fontsize=9.5, color="#5A6472",
    )
    fig.tight_layout(rect=(0, 0, 0.78, 0.91), w_pad=1.8)
    fig.savefig(output_dir / "umap_all_profiles_database_source.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({
        "source_key": sources,
        "source_name": [SOURCE_NAMES[source] for source in sources],
        "profiles": [int(points.source.eq(source).sum()) for source in sources],
    }).to_csv(output_dir / "database_sources.csv", index=False)


def structure_metrics(points):
    rows = []
    labelled = points.foodex2_level1.ne("").to_numpy()
    selections = (
        ("macro_food_category", labelled,
         points.loc[labelled, "macro_category"].to_numpy()),
        ("detailed_foodex2_category", labelled,
         points.loc[labelled, "display_category"].to_numpy()),
        ("database_source_all_profiles", np.ones(len(points), dtype=bool),
         points.source.to_numpy()),
    )
    for method, columns in (
        ("V5+CCE", ["cce_x", "cce_y"]),
        ("V5+MVC", ["mvc_x", "mvc_y"]),
    ):
        for label_type, keep, labels in selections:
            values = points.loc[keep, columns].to_numpy(np.float32)
            neighbors = NearestNeighbors(
                n_neighbors=16, algorithm="kd_tree", metric="euclidean"
            ).fit(values).kneighbors(values, return_distance=False)[:, 1:]
            purity = (labels[neighbors] == labels[:, None]).mean(axis=1)
            rows.append({
                "method": method,
                "label_type": label_type,
                "space": "full_profile_umap_2d",
                "profiles": int(len(values)),
                "classes": int(pd.Series(labels).nunique()),
                "neighbor_purity_15": float(purity.mean()),
                "macro_neighbor_purity_15": float(
                    pd.Series(purity).groupby(labels).mean().mean()
                ),
                "silhouette_sample_10000": float(silhouette_score(
                    values, labels, metric="euclidean",
                    sample_size=min(10_000, len(values)), random_state=SEED,
                )),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--mvc-checkpoint", type=Path, required=True)
    parser.add_argument("--cce-embedding", type=Path, required=True)
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
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    profiles = data.profiles.reset_index(drop=True).copy()
    rows = np.arange(len(profiles), dtype=np.int64)
    if len(rows) != 75_875:
        raise ValueError(f"expected 75,875 development profiles, found {len(rows):,}")

    model, checkpoint, _ = load_dependency_model(
        args.mvc_checkpoint, data, args.policy_manifest, device
    )
    mvc_embedding = extract_food_embeddings(
        model, data, rows, device, batch_size=256, forward_pass=1
    )
    cce_embedding = np.load(args.cce_embedding)
    if cce_embedding.shape != mvc_embedding.shape:
        raise ValueError("CCE and MVC full-profile embeddings are not aligned")
    np.save(args.output_dir / "embedding_v5_mvc_all_profiles.npy", mvc_embedding)

    cce_xy = fit_scgpt_style_umap(cce_embedding)
    mvc_xy = fit_scgpt_style_umap(mvc_embedding)
    labels = assign_foodex2_labels(profiles).reset_index(drop=True)
    points = pd.DataFrame({
        "profile_index": rows,
        "partition": profiles.partition.to_numpy(),
        "source": profiles.source_key.astype(str).to_numpy(),
        "cce_x": cce_xy[:, 0], "cce_y": cce_xy[:, 1],
        "mvc_x": mvc_xy[:, 0], "mvc_y": mvc_xy[:, 1],
    })
    points = pd.concat((points, labels), axis=1)
    points.to_csv(args.output_dir / "umap_all_profiles_points.csv", index=False)
    save_category_plot(points, args.output_dir)
    save_source_plot(points, args.output_dir)
    metrics = structure_metrics(points)
    metrics.to_csv(args.output_dir / "full_profile_umap_metrics.csv", index=False)
    manifest = {
        "schema_version": "dependency_mask_mvc_full_umap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": int(len(points)),
        "train_profiles": int(points.partition.eq("train").sum()),
        "validation_profiles": int(points.partition.eq("validation").sum()),
        "labelled_profiles": int(points.foodex2_level1.ne("").sum()),
        "target_query_tokens": 0,
        "representation": "first-forward FOOD state",
        "scgpt_alignment": {
            "reference": "official examples/finetune_integration.py",
            "l2_normalized_embedding": True,
            "n_neighbors": 15,
            "min_dist": 0.3,
            "frameon": False,
            "food_category_analogue": "celltype",
            "database_source_analogue": "batch",
            "implementation_note": (
                "umap-learn is called directly on the L2-normalized embedding; "
                "this reproduces the scGPT example settings without adding Scanpy"
            ),
        },
        "inputs": {
            "data_manifest": file_record(args.data_dir / "manifest.json"),
            "policy_manifest": file_record(args.policy_manifest),
            "mvc_checkpoint": file_record(args.mvc_checkpoint),
            "cce_embedding": file_record(args.cce_embedding),
        },
        "embedding_sha256": {
            "V5_CCE": array_sha256(np.asarray(cce_embedding, dtype=np.float32)),
            "V5_MVC": array_sha256(mvc_embedding),
        },
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "food_name_model_input": False,
        "source_model_input": False,
        "profile_level_coordinates_committed": False,
        "embeddings_committed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

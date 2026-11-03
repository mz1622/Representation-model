#!/usr/bin/env python3
"""Relabel and render the frozen dependency-mask V5+CCE UMAP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.foodex2_umap_v1 import assign_foodex2_labels  # noqa: E402
from plot_dependency_mask_umap_v2 import SOURCE_NAMES  # noqa: E402


SEED = 20261104
UNLABELLED = "#D7DCE2"
PALETTE = (
    "#4477AA", "#228833", "#CCBB44", "#EE6677", "#66CCEE", "#AA3377",
    "#BBBBBB", "#332288", "#44AA99", "#999933", "#882255", "#117733",
    "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#661100", "#6699CC",
    "#888888", "#994455", "#004488",
)


def file_record(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": path.name, "sha256": digest.hexdigest()}


def clean_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=13, weight="semibold", loc="left")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9, color="#56616F")
    axis.set_ylabel("UMAP 2", fontsize=9, color="#56616F")
    for spine in axis.spines.values():
        spine.set_visible(False)


def category_palette(points: pd.DataFrame) -> dict[str, str]:
    order = points.loc[points.display_category.ne("Unlabelled"), "display_category"].value_counts().index
    return {value: PALETTE[i % len(PALETTE)] for i, value in enumerate(order)}


def draw_categories(axis, points: pd.DataFrame, colors: dict[str, str], overlay=None) -> None:
    axis.scatter(points.x, points.y, s=7, c=UNLABELLED, alpha=0.28, linewidths=0, rasterized=True)
    frame = points if overlay is None else overlay
    for value, color in colors.items():
        keep = frame.display_category.eq(value)
        if keep.any():
            axis.scatter(frame.loc[keep, "x"], frame.loc[keep, "y"], s=10, c=color,
                         alpha=0.78, linewidths=0, rasterized=True)


def legend_handles(colors: dict[str, str], points: pd.DataFrame) -> list[Line2D]:
    handles = []
    for value, color in colors.items():
        n = int(points.display_category.eq(value).sum())
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=5,
                              color=color, label=f"{value} ({n:,})"))
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=5,
                          color=UNLABELLED, label=f"Unlabelled ({points.display_category.eq('Unlabelled').sum():,})"))
    return handles


def save_foodex2_views(points: pd.DataFrame, output_dir: Path) -> None:
    colors = category_palette(points)
    rng = np.random.default_rng(SEED)
    pieces = []
    for category, group in points.loc[points.display_category.ne("Unlabelled")].groupby("display_category"):
        count = min(140, len(group))
        pieces.append(group.iloc[rng.choice(len(group), count, replace=False)])
    balanced = pd.concat(pieces, ignore_index=True) if pieces else points.iloc[0:0]

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.5), facecolor="white")
    draw_categories(axes[0], points, colors)
    clean_axis(axes[0], "A  All validation profiles")
    draw_categories(axes[1], points, colors, overlay=balanced)
    clean_axis(axes[1], "B  Category-balanced overlay (max 140/category)")
    fig.legend(handles=legend_handles(colors, points), frameon=False, fontsize=8.2,
               ncol=2, loc="center left", bbox_to_anchor=(0.80, 0.50),
               handletextpad=0.35, columnspacing=0.9)
    coverage = points.display_category.ne("Unlabelled").mean()
    fig.suptitle("V5+CCE [FOOD] representation — FoodEx2 level-1 labels",
                 fontsize=16, weight="bold", x=0.40, y=0.98)
    fig.text(0.40, 0.935,
             f"Frozen UMAP geometry; {coverage:.1%} of the 4,000 points have an auditable category label",
             ha="center", fontsize=10, color="#56616F")
    fig.tight_layout(rect=(0, 0, 0.79, 0.92), w_pad=1.8)
    fig.savefig(output_dir / "umap_foodex2_all_balanced.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_provenance(points: pd.DataFrame, output_dir: Path) -> None:
    order = [
        "native_foodex2_level1", "source_code_crosswalk",
        "native_group_crosswalk", "exact_name_unanimous_transfer", "unlabelled",
    ]
    names = {
        "native_foodex2_level1": "Native FoodEx2 level 1",
        "source_code_crosswalk": "Official source-code crosswalk",
        "native_group_crosswalk": "Native group-name crosswalk",
        "exact_name_unanimous_transfer": "Unanimous exact-name transfer",
        "unlabelled": "Unlabelled",
    }
    colors = {
        "native_foodex2_level1": "#332288", "source_code_crosswalk": "#117733",
        "native_group_crosswalk": "#4477AA", "exact_name_unanimous_transfer": "#CC6677",
        "unlabelled": UNLABELLED,
    }
    fig, axis = plt.subplots(figsize=(10.5, 8), facecolor="white")
    for value in order[::-1]:
        keep = points.label_provenance.eq(value)
        axis.scatter(points.loc[keep, "x"], points.loc[keep, "y"],
                     s=9 if value != "unlabelled" else 7, c=colors[value],
                     alpha=0.72 if value != "unlabelled" else 0.25,
                     linewidths=0, rasterized=True,
                     label=f"{names[value]} ({keep.sum():,})")
    clean_axis(axis, "Label provenance")
    axis.legend(frameon=False, fontsize=9, markerscale=1.7, loc="upper left")
    fig.suptitle("FoodEx2 labelling audit", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "umap_label_provenance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_database(points: pd.DataFrame, profiles: pd.DataFrame, output_dir: Path) -> None:
    sources = sorted(points.source.astype(str).unique())
    missing_names = set(sources) - set(SOURCE_NAMES)
    if missing_names:
        raise ValueError(f"missing source display names: {sorted(missing_names)}")
    code = {source: f"D{i:02d}" for i, source in enumerate(sources, start=1)}
    counts = profiles.source_key.astype(str).value_counts()
    table = pd.DataFrame({
        "source_code": [code[value] for value in sources],
        "source_key": sources,
        "source_name": [SOURCE_NAMES[value] for value in sources],
        "umap_profiles": [int(points.source.eq(value).sum()) for value in sources],
        "corpus_profiles": [int(counts.get(value, 0)) for value in sources],
    })
    table.to_csv(output_dir / "database_sources.csv", index=False)

    cmap = plt.get_cmap("turbo", len(sources))
    fig, axis = plt.subplots(figsize=(12.8, 8.3), facecolor="white")
    handles = []
    for i, source in enumerate(sources):
        keep = points.source.eq(source)
        color = cmap(i)
        axis.scatter(points.loc[keep, "x"], points.loc[keep, "y"], s=8, color=color,
                     alpha=0.67, linewidths=0, rasterized=True)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=5,
                              color=color, label=f"{code[source]}  {SOURCE_NAMES[source]} ({keep.sum():,})"))
        if keep.sum() >= 12:
            centre = points.loc[keep, ["x", "y"]].median().to_numpy()
            axis.text(centre[0], centre[1], code[source], fontsize=7.5, weight="bold",
                      ha="center", va="center",
                      bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.76})
    clean_axis(axis, "Database source")
    axis.legend(handles=handles, frameon=False, fontsize=7.7, markerscale=1.4,
                ncol=2, loc="center left", bbox_to_anchor=(1.00, 0.50),
                handletextpad=0.3, columnspacing=0.8)
    fig.suptitle("V5+CCE [FOOD] representation — source audit",
                 fontsize=16, weight="bold", x=0.42, y=0.98)
    fig.tight_layout(rect=(0, 0, 0.83, 0.95))
    fig.savefig(output_dir / "umap_database_source.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def neighbourhood_metrics(x: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, float]:
    neighbours = NearestNeighbors(n_neighbors=16, metric=metric).fit(x).kneighbors(
        x, return_distance=False
    )[:, 1:]
    purity = (labels[neighbours] == labels[:, None]).mean(axis=1)
    macro_purity = pd.Series(purity).groupby(labels).mean().mean()
    frequencies = pd.Series(labels).value_counts(normalize=True)
    return {
        "silhouette": float(silhouette_score(x, labels, metric=metric)),
        "neighbor_purity_15": float(purity.mean()),
        "macro_neighbor_purity_15": float(macro_purity),
        "chance_neighbor_purity": float((frequencies ** 2).sum()),
    }


def save_structure_metrics(points: pd.DataFrame, embeddings: np.ndarray, output_dir: Path) -> None:
    rows = []
    labelled = points.foodex2_level1.ne("").to_numpy()
    selections = (
        ("foodex2_category", labelled, points.loc[labelled, "display_category"].to_numpy()),
        ("database_source_same_labelled_points", labelled, points.loc[labelled, "source"].to_numpy()),
        ("database_source_all_points", np.ones(len(points), dtype=bool), points.source.to_numpy()),
    )
    for label_type, keep, labels in selections:
        selected_embedding = embeddings[points.loc[keep, "profile_index"].to_numpy(dtype=np.int64)]
        selected_umap = points.loc[keep, ["x", "y"]].to_numpy(dtype=np.float32)
        for space, x, metric in (
            ("original_embedding", selected_embedding, "cosine"),
            ("umap_2d", selected_umap, "euclidean"),
        ):
            rows.append({
                "label_type": label_type,
                "space": space,
                "profiles": int(len(x)),
                "classes": int(pd.Series(labels).nunique()),
                **neighbourhood_metrics(x, labels, metric),
            })
    pd.DataFrame(rows).to_csv(output_dir / "umap_structure_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    coordinates = pd.read_csv(args.coordinates)
    profiles = pd.read_csv(args.profiles, low_memory=False)
    embeddings = np.load(args.embedding, mmap_mode="r")
    if len(embeddings) != len(profiles):
        raise ValueError("embedding/profile length mismatch")
    labels = assign_foodex2_labels(profiles)
    joined = coordinates.drop(columns=["category"], errors="ignore").merge(
        labels.reset_index(names="profile_index"), on="profile_index", how="left", validate="one_to_one"
    )
    if len(joined) != len(coordinates) or joined.display_category.isna().any():
        raise ValueError("coordinate/label join is incomplete")

    source_summary = (
        pd.concat([profiles[["source_key"]], labels], axis=1)
        .assign(labelled=lambda frame: frame.foodex2_level1.ne(""))
        .groupby(["source_key", "label_provenance", "foodex2_level1"], dropna=False)
        .size().rename("profiles").reset_index()
    )
    source_summary.to_csv(args.output_dir / "foodex2_source_label_audit.csv", index=False)
    distribution = (
        pd.concat([profiles[["partition"]], labels], axis=1)
        .groupby(["partition", "foodex2_level1", "display_category"], dropna=False)
        .size().rename("profiles").reset_index()
    )
    distribution.to_csv(args.output_dir / "foodex2_label_distribution.csv", index=False)
    coverage = (
        pd.concat([profiles[["source_key", "partition"]], labels], axis=1)
        .assign(labelled=lambda frame: frame.foodex2_level1.ne(""))
        .groupby(["source_key", "partition"], dropna=False)
        .agg(profiles=("labelled", "size"), labelled_profiles=("labelled", "sum"))
        .reset_index()
    )
    coverage["coverage"] = coverage.labelled_profiles / coverage.profiles
    coverage.to_csv(args.output_dir / "foodex2_label_coverage.csv", index=False)

    # Profile-level coordinates are useful locally for audit, but are excluded
    # from version control; only aggregates and rendered figures are published.
    joined.to_csv(args.output_dir / "umap_foodex2_points.csv", index=False)
    save_foodex2_views(joined, args.output_dir)
    save_provenance(joined, args.output_dir)
    save_database(joined, profiles, args.output_dir)
    save_structure_metrics(joined, embeddings, args.output_dir)

    manifest = {
        "schema_version": "foodex2_umap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinates": file_record(args.coordinates),
        "profiles": file_record(args.profiles),
        "embedding": file_record(args.embedding),
        "umap_points": int(len(joined)),
        "labelled_umap_points": int(joined.foodex2_level1.ne("").sum()),
        "labelled_umap_fraction": float(joined.foodex2_level1.ne("").mean()),
        "corpus_profiles": int(len(profiles)),
        "labelled_corpus_profiles": int(labels.foodex2_level1.ne("").sum()),
        "labelled_corpus_fraction": float(labels.foodex2_level1.ne("").mean()),
        "policy": {
            "taxonomy": "EFSA FoodEx2 Exposure Hierarchy level 1",
            "geometry": "frozen existing V5+CCE UMAP coordinates",
            "allowed_label_evidence": [
                "native FoodEx2 level-1 category",
                "source-published category code/name crosswalk",
                "unanimous exact-name-group transfer",
            ],
            "forbidden_label_evidence": [
                "food name", "nutrient values", "model embeddings", "UMAP coordinates", "cluster position",
            ],
            "unresolved_policy": "retain as Unlabelled",
        },
        "references": [
            "https://www.efsa.europa.eu/en/press/news/150504",
            "https://www.efsa.europa.eu/sites/default/files/2024-12/dietex-userguide.pdf",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render eight auditable macro food classes on the frozen V5+CCE UMAP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.foodex2_umap_v1 import MACRO_CATEGORY, assign_foodex2_labels  # noqa: E402
from build_foodex2_umap_v1 import file_record, neighbourhood_metrics  # noqa: E402


SEED = 20261104
UNLABELLED = "#D9DEE5"
CATEGORY_ORDER = (
    "Cereal & starchy staples",
    "Plant produce, legumes & nuts",
    "Fruit & juices",
    "Meat, fish & eggs",
    "Dairy & milk",
    "Sugars & fats",
    "Beverages",
    "Prepared & special-purpose foods",
)
COLORS = {
    "Cereal & starchy staples": "#4477AA",
    "Plant produce, legumes & nuts": "#228833",
    "Fruit & juices": "#CCBB44",
    "Meat, fish & eggs": "#EE6677",
    "Dairy & milk": "#66CCEE",
    "Sugars & fats": "#AA3377",
    "Beverages": "#332288",
    "Prepared & special-purpose foods": "#44AA99",
}


def clean_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=13, weight="semibold", loc="left")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1", fontsize=9, color="#56616F")
    axis.set_ylabel("UMAP 2", fontsize=9, color="#56616F")
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw(axis, context: pd.DataFrame, overlay: pd.DataFrame) -> None:
    axis.scatter(context.x, context.y, s=8, c=UNLABELLED, alpha=0.25,
                 linewidths=0, rasterized=True)
    for category in CATEGORY_ORDER:
        keep = overlay.macro_category.eq(category)
        axis.scatter(overlay.loc[keep, "x"], overlay.loc[keep, "y"],
                     s=12, c=COLORS[category], alpha=0.78,
                     linewidths=0, rasterized=True)


def save_plot(points: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    pieces = []
    for category in CATEGORY_ORDER:
        group = points.loc[points.macro_category.eq(category)]
        count = min(200, len(group))
        pieces.append(group.iloc[rng.choice(len(group), count, replace=False)])
    balanced = pd.concat(pieces, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), facecolor="white")
    draw(axes[0], points, points)
    clean_axis(axes[0], "A  All labelled profiles")
    draw(axes[1], points, balanced)
    clean_axis(axes[1], "B  Balanced overlay (max 200/category)")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6,
               color=COLORS[category],
               label=f"{category} ({points.macro_category.eq(category).sum():,})")
        for category in CATEGORY_ORDER
    ]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                          color=UNLABELLED,
                          label=f"Unlabelled ({points.macro_category.eq('Unlabelled').sum():,})"))
    fig.legend(handles=handles, frameon=False, fontsize=9.2, ncol=1,
               loc="center left", bbox_to_anchor=(0.82, 0.51), handletextpad=0.4)
    fig.suptitle("V5+CCE [FOOD] representation — 8 macro food categories",
                 fontsize=16, weight="bold", x=0.41, y=0.98)
    fig.text(0.41, 0.935,
             "Macro classes are fixed before viewing the UMAP; grey points lack an auditable FoodEx2 label",
             ha="center", fontsize=10, color="#56616F")
    fig.tight_layout(rect=(0, 0, 0.81, 0.92), w_pad=1.8)
    fig.savefig(output_dir / "umap_macro_categories_all_balanced.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metrics(points: pd.DataFrame, embeddings: np.ndarray, output_dir: Path) -> None:
    labelled = points.foodex2_level1.ne("").to_numpy()
    selected_embedding = embeddings[points.loc[labelled, "profile_index"].to_numpy(dtype=np.int64)]
    selected_umap = points.loc[labelled, ["x", "y"]].to_numpy(dtype=np.float32)
    label_sets = (
        ("macro_food_category", points.loc[labelled, "macro_category"].to_numpy()),
        ("detailed_foodex2_category", points.loc[labelled, "display_category"].to_numpy()),
        ("database_source_same_points", points.loc[labelled, "source"].to_numpy()),
    )
    rows = []
    for label_type, labels in label_sets:
        for space, x, metric in (
            ("original_embedding", selected_embedding, "cosine"),
            ("umap_2d", selected_umap, "euclidean"),
        ):
            rows.append({
                "label_type": label_type,
                "space": space,
                "profiles": int(len(labels)),
                "classes": int(pd.Series(labels).nunique()),
                **neighbourhood_metrics(x, labels, metric),
            })
    pd.DataFrame(rows).to_csv(output_dir / "macro_structure_metrics.csv", index=False)


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
    points = coordinates.drop(columns=["category"], errors="ignore").merge(
        labels.reset_index(names="profile_index"),
        on="profile_index", how="left", validate="one_to_one",
    )
    if len(points) != len(coordinates) or points.macro_category.isna().any():
        raise ValueError("coordinate/label join is incomplete")

    mapping = pd.DataFrame({
        "foodex2_level1": list(MACRO_CATEGORY),
        "macro_category": list(MACRO_CATEGORY.values()),
    }).sort_values(["macro_category", "foodex2_level1"])
    mapping.to_csv(args.output_dir / "macro_category_mapping.csv", index=False)
    distribution = (
        points.groupby("macro_category", dropna=False).size()
        .rename("umap_profiles").reset_index()
    )
    distribution.to_csv(args.output_dir / "macro_category_distribution.csv", index=False)
    save_plot(points, args.output_dir)
    save_metrics(points, embeddings, args.output_dir)

    manifest = {
        "schema_version": "foodex2_umap_v2_macro",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinates": file_record(args.coordinates),
        "profiles": file_record(args.profiles),
        "embedding": file_record(args.embedding),
        "umap_points": int(len(points)),
        "labelled_umap_points": int(points.foodex2_level1.ne("").sum()),
        "macro_categories": len(CATEGORY_ORDER),
        "merge_policy": "Fixed semantic merger of FoodEx2 level-1 categories; no embedding or UMAP evidence used",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render only food-category and database-source views of the frozen UMAP."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_NAMES = {
    "afcd": "Australian Food Composition Database",
    "anfood_2_0": "FAO/INFOODS AnFooD 2.0",
    "bangladesh_fct_2013": "Bangladesh Food Composition Table 2013",
    "biofoodcomp_4_0": "FAO/INFOODS BioFoodComp 4.0",
    "bls_4_0": "German Nutrient Database BLS 4.0",
    "ciqual": "ANSES-Ciqual",
    "cnf": "Canadian Nutrient File",
    "cofid": "UK CoFID",
    "efsa_eu_fcdb_2013": "EFSA EU Food Composition Database",
    "fndds": "USDA FNDDS",
    "foodb": "FooDB",
    "frida": "Frida FoodData, Denmark",
    "lesotho_fct_2006": "Lesotho Food Composition Table 2006",
    "mext_japan_2023": "Japan MEXT Food Composition Tables",
    "norway": "Norwegian Food Composition Database",
    "phyfoodcomp_1_0": "FAO/INFOODS/IZiNCG PhyFoodComp 1.0",
    "smiling_cambodia_2013": "SMILING Cambodia FCT 2013",
    "smiling_indonesia_2013": "SMILING Indonesia FCT 2013",
    "smiling_laos_2013": "SMILING Laos FCT 2013",
    "smiling_thailand_2013": "SMILING Thailand FCT 2013",
    "smiling_vietnam_2013": "SMILING Vietnam FCT 2013",
    "swiss_fcdb_7_1": "Swiss Food Composition Database 7.1",
    "usda_sr_legacy": "USDA Standard Reference Legacy",
    "wafct_2019": "FAO/INFOODS West Africa FCT 2019",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    points = pd.read_csv(args.coordinates)
    profiles = pd.read_csv(args.profiles, low_memory=False)
    corpus_counts = profiles.source_key.value_counts()
    sources = sorted(points.source.astype(str).unique())
    code = {source: f"D{i:02d}" for i, source in enumerate(sources, start=1)}
    table = pd.DataFrame({
        "source_code": [code[source] for source in sources],
        "source_key": sources,
        "source_name": [SOURCE_NAMES[source] for source in sources],
        "umap_profiles": [int(points.source.eq(source).sum()) for source in sources],
        "corpus_profiles": [int(corpus_counts.get(source, 0)) for source in sources],
    })
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError("source display-name registry does not match UMAP sources")
    table.to_csv(args.output_dir / "database_sources.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.2))
    category = points.category.fillna("").replace("", "Unmapped")
    categories = sorted(set(category) - {"Unmapped"})
    category_colors = {
        value: plt.get_cmap("tab10")(i % 10) for i, value in enumerate(categories)
    }
    axes[0].scatter(
        points.loc[category.eq("Unmapped"), "x"],
        points.loc[category.eq("Unmapped"), "y"],
        s=5, c="#D9D9D9", alpha=0.32, label="Unmapped", rasterized=True,
    )
    for value in categories:
        keep = category.eq(value)
        axes[0].scatter(
            points.loc[keep, "x"], points.loc[keep, "y"], s=7,
            color=category_colors[value], alpha=0.72, label=value, rasterized=True,
        )
    axes[0].legend(frameon=False, fontsize=8, markerscale=2, loc="best")
    axes[0].set_title("Food category", fontsize=14)

    cmap = plt.get_cmap("turbo", len(sources))
    for i, source in enumerate(sources):
        keep = points.source.eq(source)
        axes[1].scatter(
            points.loc[keep, "x"], points.loc[keep, "y"], s=6,
            color=cmap(i), alpha=0.65, label=code[source], rasterized=True,
        )
        if keep.sum() >= 10:
            centre = points.loc[keep, ["x", "y"]].median().to_numpy()
            axes[1].text(
                centre[0], centre[1], code[source], fontsize=7, weight="bold",
                ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
            )
    axes[1].legend(
        frameon=False, fontsize=7, markerscale=2, ncol=3,
        bbox_to_anchor=(0.5, -0.08), loc="upper center",
        columnspacing=1.0, handletextpad=0.3,
    )
    axes[1].set_title("Database source (codes listed in the report)", fontsize=14)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    fig.suptitle("Dependency-mask V5+CCE [FOOD] representation", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.output_dir / "umap_food_category_database.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

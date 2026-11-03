#!/usr/bin/env python3
"""Compare frozen V5+CCE and matched V5+MVC FOOD representations on one UMAP sample."""
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    extract_food_embeddings, load_dependency_model,
)
from foodcomp.foodex2_umap_v1 import assign_foodex2_labels  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData  # noqa: E402
from build_foodex2_umap_v1 import file_record, neighbourhood_metrics  # noqa: E402
from build_foodex2_umap_v2_macro import (  # noqa: E402
    CATEGORY_ORDER, COLORS, UNLABELLED, clean_axis,
)


UMAP_SEED = 20261027


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def draw(axis, points, x_column, y_column, title):
    axis.scatter(points[x_column], points[y_column], s=8, c=UNLABELLED,
                 alpha=0.25, linewidths=0, rasterized=True)
    for category in CATEGORY_ORDER:
        keep = points.macro_category.eq(category)
        axis.scatter(
            points.loc[keep, x_column], points.loc[keep, y_column],
            s=12, c=COLORS[category], alpha=0.78,
            linewidths=0, rasterized=True,
        )
    clean_axis(axis, title)


def save_plot(points, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), facecolor="white")
    draw(axes[0], points, "cce_x", "cce_y", "A  V5+CCE")
    draw(axes[1], points, "mvc_x", "mvc_y", "B  V5+MVC")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6,
               color=COLORS[category],
               label=f"{category} ({points.macro_category.eq(category).sum():,})")
        for category in CATEGORY_ORDER
    ]
    handles.append(Line2D(
        [], [], marker="o", linestyle="", markersize=6, color=UNLABELLED,
        label=f"Unlabelled ({points.macro_category.eq('Unlabelled').sum():,})",
    ))
    fig.legend(handles=handles, frameon=False, fontsize=9.2, ncol=1,
               loc="center left", bbox_to_anchor=(0.82, 0.51), handletextpad=0.4)
    fig.suptitle("Does MVC improve the [FOOD] representation?",
                 fontsize=16, weight="bold", x=0.41, y=0.98)
    fig.text(
        0.41, 0.935,
        "Same 4,000 profiles, FoodEx2 labels and UMAP settings; each embedding is projected independently",
        ha="center", fontsize=10, color="#56616F",
    )
    fig.tight_layout(rect=(0, 0, 0.81, 0.92), w_pad=1.8)
    fig.savefig(output_dir / "umap_v5_cce_vs_v5_mvc.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metrics(points, cce_embedding, mvc_embedding, output_dir):
    labelled = points.foodex2_level1.ne("").to_numpy()
    label_sets = (
        ("macro_food_category", points.loc[labelled, "macro_category"].to_numpy()),
        ("detailed_foodex2_category", points.loc[labelled, "display_category"].to_numpy()),
        ("database_source_same_points", points.loc[labelled, "source"].to_numpy()),
    )
    rows = []
    for method, embedding, columns in (
        ("V5+CCE", cce_embedding, ["cce_x", "cce_y"]),
        ("V5+MVC", mvc_embedding, ["mvc_x", "mvc_y"]),
    ):
        spaces = (
            ("original_embedding", embedding[labelled], "cosine"),
            ("umap_2d", points.loc[labelled, columns].to_numpy(np.float32), "euclidean"),
        )
        for label_type, labels in label_sets:
            for space, values, metric in spaces:
                rows.append({
                    "method": method,
                    "label_type": label_type,
                    "space": space,
                    "profiles": int(len(labels)),
                    "classes": int(pd.Series(labels).nunique()),
                    **neighbourhood_metrics(values, labels, metric),
                })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "representation_structure_metrics.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--mvc-checkpoint", type=Path, required=True)
    parser.add_argument("--cce-coordinates", type=Path, required=True)
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
    profiles = data.profiles.copy()
    cce_coordinates = pd.read_csv(args.cce_coordinates)
    selected = cce_coordinates.profile_index.to_numpy(dtype=np.int64)
    if len(selected) != 4_000 or len(np.unique(selected)) != len(selected):
        raise ValueError("unexpected frozen UMAP sample")

    model, checkpoint, _ = load_dependency_model(
        args.mvc_checkpoint, data, args.policy_manifest, device
    )
    mvc_embedding = extract_food_embeddings(
        model, data, selected, device, batch_size=128, forward_pass=1
    )
    np.save(args.output_dir / "embedding_v5_mvc_sample.npy", mvc_embedding)
    cce_all = np.load(args.cce_embedding, mmap_mode="r")
    if len(cce_all) != len(profiles):
        raise ValueError("CCE embedding/profile length mismatch")
    cce_embedding = np.asarray(cce_all[selected], dtype=np.float32)

    import umap
    mvc_coordinates = umap.UMAP(
        n_neighbors=15, min_dist=0.15, metric="cosine",
        random_state=UMAP_SEED, n_jobs=1,
    ).fit_transform(mvc_embedding)
    labels = assign_foodex2_labels(profiles).iloc[selected].reset_index(drop=True)
    points = pd.DataFrame({
        "profile_index": selected,
        "source": profiles.iloc[selected].source_key.to_numpy(),
        "cce_x": cce_coordinates.x.to_numpy(),
        "cce_y": cce_coordinates.y.to_numpy(),
        "mvc_x": mvc_coordinates[:, 0],
        "mvc_y": mvc_coordinates[:, 1],
    })
    points = pd.concat((points, labels), axis=1)
    points.to_csv(args.output_dir / "umap_v5_cce_vs_v5_mvc_points.csv", index=False)
    save_plot(points, args.output_dir)
    metrics = save_metrics(points, cce_embedding, mvc_embedding, args.output_dir)
    manifest = {
        "schema_version": "dependency_mask_mvc_umap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "profiles": len(selected),
        "labelled_profiles": int(points.foodex2_level1.ne("").sum()),
        "target_query_tokens_during_embedding_extraction": 0,
        "representation": "first-forward FOOD state",
        "umap": {
            "n_neighbors": 15, "min_dist": 0.15,
            "metric": "cosine", "random_state": UMAP_SEED,
        },
        "inputs": {
            "data_manifest": file_record(args.data_dir / "manifest.json"),
            "policy_manifest": file_record(args.policy_manifest),
            "mvc_checkpoint": file_record(args.mvc_checkpoint),
            "cce_coordinates": file_record(args.cce_coordinates),
            "cce_embedding": file_record(args.cce_embedding),
        },
        "derived_mvc_embedding_sha256": array_sha256(mvc_embedding),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "food_name_model_input": False,
        "source_model_input": False,
        "selection_warning": "UMAP and representation metrics were not used for checkpoint selection",
        "metric_rows": len(metrics),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

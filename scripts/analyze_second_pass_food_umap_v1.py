#!/usr/bin/env python3
"""Compare first- and second-forward FOOD states on one fixed UMAP sample."""
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


def draw(axis, points: pd.DataFrame, x_column: str, y_column: str, title: str) -> None:
    axis.scatter(points[x_column], points[y_column], s=8, c=UNLABELLED,
                 alpha=0.25, linewidths=0, rasterized=True)
    for category in CATEGORY_ORDER:
        keep = points.macro_category.eq(category)
        axis.scatter(points.loc[keep, x_column], points.loc[keep, y_column],
                     s=12, c=COLORS[category], alpha=0.78,
                     linewidths=0, rasterized=True)
    clean_axis(axis, title)


def save_comparison_plot(points: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), facecolor="white")
    draw(axes[0], points, "first_x", "first_y", "A  First-forward [FOOD]")
    draw(axes[1], points, "second_x", "second_y", "B  Second-forward [FOOD]")
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
    fig.suptitle("V5+CCE [FOOD] representation — first vs second forward",
                 fontsize=16, weight="bold", x=0.41, y=0.98)
    fig.text(0.41, 0.935,
             "Same 4,000 profiles, labels and UMAP settings; each pass is projected independently",
             ha="center", fontsize=10, color="#56616F")
    fig.tight_layout(rect=(0, 0, 0.81, 0.92), w_pad=1.8)
    fig.savefig(output_dir / "umap_first_vs_second_food.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    points: pd.DataFrame,
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
    output_dir: Path,
) -> None:
    labelled = points.foodex2_level1.ne("").to_numpy()
    label_sets = (
        ("macro_food_category", points.loc[labelled, "macro_category"].to_numpy()),
        ("detailed_foodex2_category", points.loc[labelled, "display_category"].to_numpy()),
        ("database_source_same_points", points.loc[labelled, "source"].to_numpy()),
    )
    rows = []
    for pass_name, embedding, xy_columns in (
        ("first", first_embedding, ["first_x", "first_y"]),
        ("second", second_embedding, ["second_x", "second_y"]),
    ):
        spaces = (
            ("original_embedding", embedding[labelled], "cosine"),
            ("umap_2d", points.loc[labelled, xy_columns].to_numpy(dtype=np.float32), "euclidean"),
        )
        for label_type, labels in label_sets:
            for space, values, metric in spaces:
                rows.append({
                    "forward_pass": pass_name,
                    "label_type": label_type,
                    "space": space,
                    "profiles": int(len(labels)),
                    "classes": int(pd.Series(labels).nunique()),
                    **neighbourhood_metrics(values, labels, metric),
                })
    pd.DataFrame(rows).to_csv(output_dir / "first_second_structure_metrics.csv", index=False)


def embedding_shift(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    cosine = np.sum(first * second, axis=1) / np.maximum(first_norm * second_norm, 1e-12)
    distance = np.linalg.norm(second - first, axis=1)
    ratio = second_norm / np.maximum(first_norm, 1e-12)
    result = {"profiles": int(len(first))}
    for name, values in (("cosine_similarity", cosine), ("l2_distance", distance), ("norm_ratio", ratio)):
        result.update({
            f"{name}_mean": float(np.mean(values)),
            f"{name}_median": float(np.median(values)),
            f"{name}_q05": float(np.quantile(values, 0.05)),
            f"{name}_q95": float(np.quantile(values, 0.95)),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-coordinates", type=Path, required=True)
    parser.add_argument("--first-embedding", type=Path, required=True)
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
    first_coordinates = pd.read_csv(args.first_coordinates)
    selected = first_coordinates.profile_index.to_numpy(dtype=np.int64)
    if len(selected) != 4_000 or len(np.unique(selected)) != len(selected):
        raise ValueError("unexpected frozen UMAP sample")

    model, checkpoint, _ = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, device
    )
    second_embedding = extract_food_embeddings(
        model, data, selected, device, batch_size=128, forward_pass=2
    )
    np.save(args.output_dir / "embedding_second_forward_sample.npy", second_embedding)
    first_all = np.load(args.first_embedding, mmap_mode="r")
    if len(first_all) != len(profiles):
        raise ValueError("first-pass embedding/profile length mismatch")
    first_embedding = np.asarray(first_all[selected], dtype=np.float32)

    import umap
    second_coordinates = umap.UMAP(
        n_neighbors=15, min_dist=0.15, metric="cosine",
        random_state=UMAP_SEED, n_jobs=1,
    ).fit_transform(second_embedding)
    labels = assign_foodex2_labels(profiles).iloc[selected].reset_index(drop=True)
    points = pd.DataFrame({
        "profile_index": selected,
        "source": profiles.iloc[selected].source_key.to_numpy(),
        "first_x": first_coordinates.x.to_numpy(),
        "first_y": first_coordinates.y.to_numpy(),
        "second_x": second_coordinates[:, 0],
        "second_y": second_coordinates[:, 1],
    })
    points = pd.concat((points, labels), axis=1)
    points.to_csv(args.output_dir / "umap_first_second_points.csv", index=False)

    save_comparison_plot(points, args.output_dir)
    save_metrics(points, first_embedding, second_embedding, args.output_dir)
    shift = embedding_shift(first_embedding, second_embedding)
    (args.output_dir / "embedding_shift.json").write_text(
        json.dumps(shift, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "second_forward_food_umap_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "profiles": len(selected),
        "target_query_tokens": 0,
        "first_representation": "forward_details(...)[1]",
        "second_representation": "forward_prompt_details(...)[3]",
        "second_prompt": "detached first-forward FOOD state",
        "umap": {
            "n_neighbors": 15, "min_dist": 0.15,
            "metric": "cosine", "random_state": UMAP_SEED,
        },
        "inputs": {
            "data_manifest": file_record(args.data_dir / "manifest.json"),
            "policy_manifest": file_record(args.policy_manifest),
            "checkpoint": file_record(args.checkpoint),
            "first_coordinates": file_record(args.first_coordinates),
            "first_embedding": file_record(args.first_embedding),
        },
        "derived_second_embedding_sha256": array_sha256(second_embedding),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "food_name_model_input": False,
        "source_model_input": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

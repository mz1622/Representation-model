#!/usr/bin/env python3
"""Run a projection-free foundation-representation evidence audit.

The audit compares pretrained FOOD states with raw numerical inputs, a
train-fitted dimension-matched PCA baseline, missingness alone, and random
embeddings.  Food-category geometry is evaluated only on held-out validation
profiles.  Linear probes and neighbour references are fitted on training rows.
Database source is treated as a nuisance label, and exact-name candidate groups
are used only for a conservative cross-database retrieval audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    extract_food_embeddings,
    load_dependency_model,
)
from foodcomp.foodex2_umap_v1 import (  # noqa: E402
    DAIRY,
    FATS,
    FISH,
    FRUIT,
    GRAINS,
    LEGUMES,
    MEAT,
    SUGAR,
    VEGETABLES,
    DISPLAY_NAMES,
    assign_foodex2_labels,
)
from foodcomp.foundation_representation_evidence_v1 import (  # noqa: E402
    balanced_panel_indices,
    cross_source_retrieval,
    multiclass_cosine_silhouette,
    neighbor_purity,
    one_vs_rest_silhouettes,
    pairwise_silhouettes,
    source_restricted_neighbor_purity,
)
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData  # noqa: E402


SEED = 20261110
CORE_LEVEL1 = (
    GRAINS,
    VEGETABLES,
    LEGUMES,
    FRUIT,
    MEAT,
    FISH,
    DAIRY,
    FATS,
    SUGAR,
)
METHOD_LABELS = {
    "V5_CCE_first": "V5+CCE first",
    "V5_CCE_second": "V5+CCE second",
    "V5_GEPC_first": "V5+GEPC first",
    "V5_GEPC_second": "V5+GEPC second",
    "Raw_values_mask": "Values + mask",
    "PCA192": "PCA-192",
    "Missingness_only": "Missingness only",
    "Random192": "Random-192",
}
COLORS = {
    "V5_CCE_first": "#17365D",
    "V5_CCE_second": "#4F81BD",
    "V5_GEPC_first": "#7A5195",
    "V5_GEPC_second": "#BC5090",
    "Raw_values_mask": "#59A14F",
    "PCA192": "#F28E2B",
    "Missingness_only": "#9C755F",
    "Random192": "#BAB0AC",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_embedding(path: Path, expected_rows: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.shape != (expected_rows, 192):
        raise ValueError(f"{path} has shape {values.shape}; expected {(expected_rows, 192)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def build_numeric_representations(data):
    observed_train = np.where(data.observed[data.train], data.values[data.train], np.nan)
    medians = np.nanmedian(observed_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    filled = np.where(data.observed, data.values, medians).astype(np.float32)
    value_scaler = StandardScaler().fit(filled[data.train])
    standardized = value_scaler.transform(filled).astype(np.float32)
    values_mask = np.concatenate(
        (standardized, data.observed.astype(np.float32)), axis=1
    )
    pca = PCA(
        n_components=192,
        svd_solver="randomized",
        random_state=SEED,
        iterated_power=7,
    ).fit(values_mask[data.train])
    return {
        "Raw_values_mask": values_mask,
        "PCA192": pca.transform(values_mask).astype(np.float32),
        "Missingness_only": data.observed.astype(np.float32),
    }, pca


def linear_probe(values, labels, train_rows, validation_rows):
    scaler = StandardScaler().fit(values[train_rows])
    classifier = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )
    classifier.fit(scaler.transform(values[train_rows]), labels[train_rows])
    predicted = classifier.predict(scaler.transform(values[validation_rows]))
    return {
        "linear_macro_f1": float(f1_score(
            labels[validation_rows], predicted, average="macro"
        )),
        "linear_balanced_accuracy": float(balanced_accuracy_score(
            labels[validation_rows], predicted
        )),
    }


def supported_source_rows(profiles, train_mask, validation_mask):
    train_counts = profiles.loc[train_mask, "source_key"].value_counts()
    validation_counts = profiles.loc[validation_mask, "source_key"].value_counts()
    sources = sorted(set(train_counts[train_counts >= 50].index) & set(
        validation_counts[validation_counts >= 20].index
    ))
    return (
        np.flatnonzero(train_mask & profiles.source_key.isin(sources).to_numpy()),
        np.flatnonzero(validation_mask & profiles.source_key.isin(sources).to_numpy()),
        sources,
    )


def conditional_silhouette_rows(values, labels, condition, *, min_per_label=10):
    rows = []
    for condition_value in sorted(pd.Series(condition).dropna().unique()):
        current = np.flatnonzero(condition == condition_value)
        counts = pd.Series(labels[current]).value_counts()
        retained = counts[counts >= min_per_label].index
        current = current[np.isin(labels[current], retained)]
        if len(retained) < 2:
            continue
        rows.append({
            "condition": condition_value,
            "profiles": len(current),
            "classes": len(retained),
            "silhouette_cosine": float(silhouette_score(
                values[current], labels[current], metric="cosine"
            )),
        })
    return pd.DataFrame(rows)


def save_category_plot(metrics: pd.DataFrame, output: Path):
    order = list(metrics.method)
    labels = [METHOD_LABELS[value] for value in order]
    colors = [COLORS[value] for value in order]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4), facecolor="white")
    columns = (
        ("silhouette_cosine", "Nine-class cosine\nsilhouette", None),
        ("one_vs_rest_silhouette", "Mean one-vs-rest\ncosine silhouette", None),
        ("macro_neighbor_purity_15", "Macro 15-NN purity", (0, 1)),
        ("linear_macro_f1", "Linear-probe macro-F1", (0, 1)),
    )
    x = np.arange(len(order))
    for axis, (column, title, limits) in zip(axes, columns):
        axis.bar(x, metrics[column], color=colors, width=0.78)
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Projection-free FoodEx2 representation audit — held-out validation profiles",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_ovr_plot(ovr: pd.DataFrame, output: Path):
    summary = ovr.groupby(["method", "category"], sort=False).silhouette_cosine.mean().unstack()
    summary = summary.loc[[method for method in METHOD_LABELS if method in summary.index], CORE_LEVEL1]
    fig, axis = plt.subplots(figsize=(14.5, 5.0), facecolor="white")
    image = axis.imshow(summary.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    axis.set_yticks(np.arange(len(summary)), [METHOD_LABELS[x] for x in summary.index])
    axis.set_xticks(
        np.arange(len(CORE_LEVEL1)),
        [DISPLAY_NAMES[x] for x in CORE_LEVEL1],
        rotation=35,
        ha="right",
    )
    axis.set_title("Balanced one-vs-rest cosine silhouette (20 negative resamples)")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Silhouette")
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_retrieval_plot(retrieval: pd.DataFrame, output: Path):
    plot = retrieval[~retrieval.method.str.startswith("Random192_repeat")].copy()
    order = [method for method in METHOD_LABELS if method in set(plot.method)]
    plot = plot.set_index("method").loc[order].reset_index()
    x = np.arange(len(plot))
    width = 0.26
    fig, axis = plt.subplots(figsize=(10.8, 4.8), facecolor="white")
    axis.bar(x - width, plot.top1_accuracy, width, label="Top-1", color="#17365D")
    axis.bar(x, plot.top5_accuracy, width, label="Top-5", color="#59A14F")
    axis.bar(x + width, plot.mean_reciprocal_rank, width, label="MRR", color="#F28E2B")
    axis.set_xticks(x, [METHOD_LABELS[value] for value in plot.method], rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Retrieval score")
    axis.set_title("Cross-database exact-name candidate retrieval — validation only")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--cce-embedding", type=Path, required=True)
    parser.add_argument("--gepc-embedding", type=Path, required=True)
    parser.add_argument("--cce-checkpoint", type=Path)
    parser.add_argument("--gepc-checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.random_repeats < 2:
        raise ValueError("--random-repeats must be at least two")
    args.output_dir.mkdir(parents=True)
    cache_dir = args.output_dir / "uncommitted_embedding_cache"
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir()
    started = time.monotonic()

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    profiles = data.profiles.reset_index(drop=True)
    foodex2 = assign_foodex2_labels(profiles).reset_index(drop=True)
    labels = foodex2.foodex2_level1.fillna("").astype(str).to_numpy()
    core = np.isin(labels, CORE_LEVEL1)
    train_mask = np.zeros(len(profiles), dtype=bool)
    validation_mask = np.zeros(len(profiles), dtype=bool)
    train_mask[data.train] = True
    validation_mask[data.validation] = True
    core_train = np.flatnonzero(core & train_mask)
    core_validation = np.flatnonzero(core & validation_mask)
    geometry_rows, per_class = balanced_panel_indices(
        labels,
        core & validation_mask,
        CORE_LEVEL1,
        cap_per_class=None,
        seed=SEED,
    )

    print("building train-fitted numeric baselines", flush=True)
    representations, pca = build_numeric_representations(data)
    representations["V5_CCE_first"] = check_embedding(args.cce_embedding, len(profiles))
    representations["V5_GEPC_first"] = check_embedding(args.gepc_embedding, len(profiles))

    checkpoint_hashes = {}
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    for method, checkpoint_path in (
        ("V5_CCE_second", args.cce_checkpoint),
        ("V5_GEPC_second", args.gepc_checkpoint),
    ):
        if checkpoint_path is None:
            continue
        cache_dir.mkdir(exist_ok=True)
        print(f"extracting {method} FOOD states on {device}", flush=True)
        model, checkpoint, _ = load_dependency_model(
            checkpoint_path, data, args.policy_manifest, device
        )
        values = extract_food_embeddings(
            model,
            data,
            np.arange(len(profiles), dtype=np.int64),
            device,
            batch_size=256,
            forward_pass=2,
        )
        cache_path = cache_dir / f"{method}.npy"
        np.save(cache_path, values)
        representations[method] = values
        checkpoint_hashes[method] = {
            "sha256": sha256(checkpoint_path),
            "stage": checkpoint.get("stage"),
            "epoch": checkpoint.get("epoch"),
        }

    random = np.random.default_rng(SEED).normal(
        size=(len(profiles), 192)
    ).astype(np.float32)
    representations["Random192"] = random
    method_order = [method for method in METHOD_LABELS if method in representations]

    category_rows = []
    ovr_frames = []
    pairwise_frames = []
    for method in method_order:
        print(f"category audit: {method}", flush=True)
        values = representations[method]
        micro, macro = neighbor_purity(
            values[core_train], labels[core_train],
            values[core_validation], labels[core_validation], k=15,
        )
        probe = linear_probe(values, labels, core_train, core_validation)
        cross_micro, cross_macro, cross_profiles, cross_sources = (
            source_restricted_neighbor_purity(
                values, labels, profiles.source_key.astype(str).to_numpy(),
                core_train, core_validation, cross_source=True, k=15,
            )
        )
        same_micro, same_macro, same_profiles, same_sources = (
            source_restricted_neighbor_purity(
                values, labels, profiles.source_key.astype(str).to_numpy(),
                core_train, core_validation, cross_source=False, k=15,
            )
        )
        ovr = one_vs_rest_silhouettes(
            values[geometry_rows], labels[geometry_rows], CORE_LEVEL1,
            repeats=20, seed=SEED,
        )
        ovr.insert(0, "method", method)
        ovr_frames.append(ovr)
        pairwise = pairwise_silhouettes(
            values[geometry_rows], labels[geometry_rows], CORE_LEVEL1
        )
        pairwise.insert(0, "method", method)
        pairwise_frames.append(pairwise)
        category_rows.append({
            "method": method,
            "display_name": METHOD_LABELS[method],
            "train_profiles": len(core_train),
            "validation_profiles": len(core_validation),
            "geometry_profiles": len(geometry_rows),
            "geometry_profiles_per_class": per_class,
            "dimensions": values.shape[1],
            "silhouette_cosine": multiclass_cosine_silhouette(
                values[geometry_rows], labels[geometry_rows]
            ),
            "one_vs_rest_silhouette": float(ovr.silhouette_cosine.mean()),
            "neighbor_purity_15": micro,
            "macro_neighbor_purity_15": macro,
            "cross_source_neighbor_purity_15": cross_micro,
            "cross_source_macro_neighbor_purity_15": cross_macro,
            "cross_source_evaluated_profiles": cross_profiles,
            "cross_source_evaluated_sources": cross_sources,
            "same_source_neighbor_purity_15": same_micro,
            "same_source_macro_neighbor_purity_15": same_macro,
            "same_source_evaluated_profiles": same_profiles,
            "same_source_evaluated_sources": same_sources,
            **probe,
        })
    category_metrics = pd.DataFrame(category_rows)
    category_metrics.to_csv(args.output_dir / "category_metrics.csv", index=False)
    ovr_all = pd.concat(ovr_frames, ignore_index=True)
    ovr_all.to_csv(args.output_dir / "category_one_vs_rest.csv", index=False)
    pd.concat(pairwise_frames, ignore_index=True).to_csv(
        args.output_dir / "category_pairwise.csv", index=False
    )

    print("random-embedding null distribution", flush=True)
    random_null = []
    for repeat in range(args.random_repeats):
        current = np.random.default_rng(SEED + 1000 + repeat).normal(
            size=(len(geometry_rows), 192)
        ).astype(np.float32)
        current_ovr = one_vs_rest_silhouettes(
            current, labels[geometry_rows], CORE_LEVEL1,
            repeats=1, seed=SEED + repeat,
        )
        random_null.append({
            "repeat": repeat,
            "silhouette_cosine": multiclass_cosine_silhouette(
                current, labels[geometry_rows]
            ),
            "one_vs_rest_silhouette": float(current_ovr.silhouette_cosine.mean()),
        })
    random_null = pd.DataFrame(random_null)
    random_null.to_csv(args.output_dir / "random_embedding_null.csv", index=False)

    print("database-source nuisance audit", flush=True)
    source_train, source_validation, retained_sources = supported_source_rows(
        profiles, core & train_mask, core & validation_mask
    )
    source_labels = profiles.source_key.astype(str).to_numpy()
    source_rows = []
    within_category_frames = []
    within_source_frames = []
    for method in method_order:
        values = representations[method]
        micro, macro = neighbor_purity(
            values[source_train], source_labels[source_train],
            values[source_validation], source_labels[source_validation], k=15,
        )
        probe = linear_probe(
            values, source_labels, source_train, source_validation
        )
        counts = pd.Series(source_labels[source_validation]).value_counts()
        geometry_source = source_validation[np.isin(
            source_labels[source_validation], counts[counts >= 20].index
        )]
        within_category = conditional_silhouette_rows(
            values[core_validation],
            source_labels[core_validation],
            labels[core_validation],
            min_per_label=10,
        )
        if len(within_category):
            within_category.insert(0, "method", method)
            within_category_frames.append(within_category)
        within_source = conditional_silhouette_rows(
            values[core_validation],
            labels[core_validation],
            source_labels[core_validation],
            min_per_label=10,
        )
        if len(within_source):
            within_source.insert(0, "method", method)
            within_source_frames.append(within_source)
        source_rows.append({
            "method": method,
            "display_name": METHOD_LABELS[method],
            "train_profiles": len(source_train),
            "validation_profiles": len(source_validation),
            "source_classes": len(retained_sources),
            "silhouette_profiles": len(geometry_source),
            "silhouette_source_classes": pd.Series(
                source_labels[geometry_source]
            ).nunique(),
            "silhouette_cosine": multiclass_cosine_silhouette(
                values[geometry_source], source_labels[geometry_source]
            ),
            "neighbor_purity_15": micro,
            "macro_neighbor_purity_15": macro,
            "within_category_source_silhouette": (
                np.nan if not len(within_category)
                else float(within_category.silhouette_cosine.mean())
            ),
            "within_source_category_silhouette": (
                np.nan if not len(within_source)
                else float(within_source.silhouette_cosine.mean())
            ),
            **probe,
        })
    pd.DataFrame(source_rows).to_csv(
        args.output_dir / "source_nuisance_metrics.csv", index=False
    )
    pd.concat(within_category_frames, ignore_index=True).to_csv(
        args.output_dir / "within_category_source_silhouette.csv", index=False
    )
    pd.concat(within_source_frames, ignore_index=True).to_csv(
        args.output_dir / "within_source_category_silhouette.csv", index=False
    )

    print("cross-database exact-name candidate retrieval", flush=True)
    group_counts = profiles.loc[validation_mask].groupby("exact_name_group_id").source_key.nunique()
    eligible_groups = set(group_counts[group_counts >= 2].index)
    retrieval_rows = np.flatnonzero(
        validation_mask & profiles.exact_name_group_id.isin(eligible_groups).to_numpy()
    )
    groups = profiles.iloc[retrieval_rows].exact_name_group_id.astype(str).to_numpy()
    sources = profiles.iloc[retrieval_rows].source_key.astype(str).to_numpy()
    retrieval_results = []
    for method in method_order:
        print(f"retrieval: {method}", flush=True)
        retrieval_results.append({
            "method": method,
            "display_name": METHOD_LABELS[method],
            **cross_source_retrieval(
                representations[method][retrieval_rows], groups, sources
            ),
        })
    for repeat in range(args.random_repeats):
        current = np.random.default_rng(SEED + 20_000 + repeat).normal(
            size=(len(retrieval_rows), 192)
        ).astype(np.float32)
        retrieval_results.append({
            "method": f"Random192_repeat_{repeat:02d}",
            "display_name": "Random-192 repeat",
            **cross_source_retrieval(current, groups, sources),
        })
    retrieval = pd.DataFrame(retrieval_results)
    retrieval.to_csv(args.output_dir / "cross_source_retrieval.csv", index=False)

    distributions = (
        pd.DataFrame({
            "partition": profiles.partition,
            "foodex2_level1": labels,
        })
        .query("foodex2_level1 in @CORE_LEVEL1")
        .groupby(["partition", "foodex2_level1"], sort=False)
        .size()
        .rename("profiles")
        .reset_index()
    )
    distributions.to_csv(args.output_dir / "core_category_distribution.csv", index=False)
    pd.DataFrame({
        "source_key": retained_sources,
        "train_profiles": [int(np.sum(source_labels[source_train] == source)) for source in retained_sources],
        "validation_profiles": [int(np.sum(source_labels[source_validation] == source)) for source in retained_sources],
    }).to_csv(args.output_dir / "source_probe_distribution.csv", index=False)

    save_category_plot(category_metrics, figures_dir / "category_evidence.png")
    save_ovr_plot(ovr_all, figures_dir / "category_one_vs_rest.png")
    save_retrieval_plot(retrieval, figures_dir / "cross_source_retrieval.png")

    random_retrieval = retrieval[retrieval.method.str.startswith("Random192_repeat")]
    manifest = {
        "status": "complete",
        "schema_version": "foundation_representation_evidence_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "seed": SEED,
        "device_for_second_pass_extraction": str(device),
        "profiles": len(profiles),
        "core_train_profiles": len(core_train),
        "core_validation_profiles": len(core_validation),
        "balanced_validation_geometry": {
            "profiles": len(geometry_rows),
            "profiles_per_class": per_class,
            "classes": [DISPLAY_NAMES[value] for value in CORE_LEVEL1],
        },
        "pca": {
            "fit_partition": "train only",
            "dimensions": 192,
            "input": "train-median-imputed normalized values, train-standardized, plus visibility mask",
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "random_repeats": args.random_repeats,
        "random_null_95_interval": {
            column: [
                float(random_null[column].quantile(0.025)),
                float(random_null[column].quantile(0.975)),
            ]
            for column in ("silhouette_cosine", "one_vs_rest_silhouette")
        },
        "random_retrieval_95_interval": {
            column: [
                float(random_retrieval[column].quantile(0.025)),
                float(random_retrieval[column].quantile(0.975)),
            ]
            for column in ("top1_accuracy", "top5_accuracy", "mean_reciprocal_rank")
        },
        "inputs": {
            "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
            "policy_manifest_sha256": sha256(args.policy_manifest),
            "cce_first_embedding_sha256": sha256(args.cce_embedding),
            "gepc_first_embedding_sha256": sha256(args.gepc_embedding),
            "second_pass_checkpoints": checkpoint_hashes,
        },
        "leakage_controls": {
            "category_geometry": "validation only",
            "linear_probe": "fit train; evaluate validation",
            "pca": "fit train; transform validation",
            "exact_name_retrieval": "validation only; candidates restricted to a different source",
            "food_names_model_input": False,
            "source_ids_model_input": False,
            "foodex2_labels_model_input": False,
        },
        "interpretation_guardrail": (
            "Exact-name groups are conservative candidate identities, not confirmed biological merges. "
            "Source separation is a nuisance/batch signal and is not positive semantic evidence."
        ),
        "large_artifacts_committed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()

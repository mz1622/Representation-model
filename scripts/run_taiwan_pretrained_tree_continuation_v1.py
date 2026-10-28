#!/usr/bin/env python3
"""Continue source-pretrained RF/XGBoost on aligned Taiwan few-shot subsets."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import (  # noqa: E402
    attach_taiwan_panel,
    load_dependency_model,
)
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    dense_features,
    make_baseline,
    score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import (  # noqa: E402
    inverse_values,
    numeric_only_training_view,
)
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


SHOTS = (8, 32, 128)
SEEDS = (20261027, 20261028, 20261029)
SPLIT_SEED = 20261026
TEST_PROFILES = 400
ADAPTATION_TREES = 20


def coverage_nested_order(pool, axes, observed, seed):
    """Match the coverage-first nested subsets used by all v2/v3 comparisons."""
    rng = np.random.default_rng(seed)
    remaining = np.asarray(pool, dtype=np.int64).copy()
    uncovered = np.ones(len(axes), dtype=bool)
    selected = []
    while uncovered.any():
        score = observed[np.ix_(remaining, axes)][:, uncovered].sum(1)
        best = score.max(initial=0)
        if best <= 0:
            raise ValueError("adaptation pool cannot cover all Taiwan axes")
        tied = np.flatnonzero(score == best)
        position = int(rng.choice(tied))
        row = int(remaining[position])
        selected.append(row)
        uncovered &= ~observed[row, axes]
        remaining = np.delete(remaining, position)
    if len(selected) > min(SHOTS):
        raise ValueError(
            f"axis coverage requires {len(selected)} profiles, more than 8 shots"
        )
    return (
        np.concatenate((np.asarray(selected, dtype=np.int64), rng.permutation(remaining))),
        len(selected),
    )


def row_hash(rows):
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def validate_inputs(args, computed_split):
    baseline_manifest = json.loads(
        (args.taiwan_only_baseline_dir / "manifest.json").read_text(encoding="utf-8")
    )
    neural_manifest = json.loads(
        (args.neural_finetune_dir / "manifest.json").read_text(encoding="utf-8")
    )
    zero_manifest = json.loads(
        (args.zero_shot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for label, manifest in (
        ("taiwan-only baseline", baseline_manifest),
        ("neural fine-tune", neural_manifest),
        ("zero-shot tree", zero_manifest),
    ):
        if manifest.get("status") != "complete":
            raise ValueError(f"{label} manifest is not complete")
    expected = {
        "shots": list(SHOTS),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES,
        "test_axes": 80,
    }
    for key, value in expected.items():
        if baseline_manifest.get(key) != value:
            raise ValueError(f"baseline {key} mismatch")
        if neural_manifest.get(key) != value:
            raise ValueError(f"neural fine-tune {key} mismatch")
    hashes = {
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
    }
    for key, value in hashes.items():
        if baseline_manifest.get(key) != value:
            raise ValueError(f"baseline {key} mismatch")
        if neural_manifest.get(key) != value:
            raise ValueError(f"neural fine-tune {key} mismatch")
    for key in ("data_sha256", "policy_manifest_sha256", "checkpoint_sha256"):
        if zero_manifest.get(key) != hashes[key]:
            raise ValueError(f"zero-shot {key} mismatch")

    saved_split = pd.read_csv(args.taiwan_only_baseline_dir / "profile_split.csv")[[
        "profile_index",
        "fewshot_partition",
    ]]
    saved_split = saved_split.sort_values("profile_index").reset_index(drop=True)
    computed_split = computed_split[[
        "profile_index",
        "fewshot_partition",
    ]].sort_values("profile_index").reset_index(drop=True)
    pd.testing.assert_frame_equal(saved_split, computed_split, check_dtype=False)
    return baseline_manifest, neural_manifest, zero_manifest, hashes


def source_fit(method, axis, data, policy, config):
    train = np.asarray(data.train, dtype=np.int64)
    train = train[data.observed[train, axis]]
    if not len(train):
        raise ValueError(f"source training has no labels for axis {axis}")
    target = np.full(len(train), axis, dtype=np.int64)
    x = dense_features(data, train, target, policy)
    y = data.values[train, axis]
    weight = data.weights[train, axis]
    model = make_baseline(method, axis, config)
    model.fit(x, y, sample_weight=weight / weight.mean())
    if method == "rf":
        model.n_jobs = 1
    return model, len(train)


def continue_rf(source, x, y, adaptation_trees, n_jobs):
    """Keep source trees and append Taiwan-only trees in one warm-start forest."""
    original = len(source.estimators_)
    adapted = copy.copy(source)
    adapted.estimators_ = list(source.estimators_)
    adapted.set_params(
        warm_start=True,
        n_estimators=original + adaptation_trees,
        n_jobs=n_jobs,
    )
    adapted.fit(x, y)
    if len(adapted.estimators_) != original + adaptation_trees:
        raise ValueError("RF continuation did not append the requested trees")
    if len(source.estimators_) != original:
        raise ValueError("RF continuation mutated the frozen source forest")
    adapted.n_jobs = 1
    return adapted


def continue_xgb(source, axis, x, y, adaptation_trees, config):
    """Continue the source booster for a fixed number of Taiwan boosting rounds."""
    seed = config.seed + int(axis)
    adapted = XGBRegressor(
        n_estimators=adaptation_trees,
        max_depth=config.xgb_depth,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=config.n_jobs,
    )
    original = source.get_booster().num_boosted_rounds()
    adapted.fit(x, y, xgb_model=source.get_booster())
    expected = original + adaptation_trees
    if adapted.get_booster().num_boosted_rounds() != expected:
        raise ValueError("XGBoost continuation round count mismatch")
    return adapted


def prediction_frame(rows, axis, normalized, scale):
    return pd.DataFrame(
        {
            "profile_index": rows,
            "axis_index": axis,
            "prediction": inverse_values(normalized, scale),
        }
    )


def summary_table(runs):
    metrics = [
        "macro_scaled_log_mae",
        "macro_scaled_log_rmse",
        "macro_raw_mae",
    ]
    return runs.groupby(["method", "shots"], as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sd": (metric, "std") for metric in metrics},
        min_axis_labels=("min_axis_labels", "min"),
        max_axis_labels=("max_axis_labels", "max"),
    )


def build_all_method_comparison(tree_summary, frozen_metrics, previous_comparison):
    metrics = [
        "macro_scaled_log_mae",
        "macro_scaled_log_rmse",
        "macro_raw_mae",
    ]
    columns = [
        "method",
        "shots",
        *[f"{metric}_mean" for metric in metrics],
        *[f"{metric}_sd" for metric in metrics],
    ]
    previous = previous_comparison[columns].copy()
    previous["method"] = previous.method.replace(
        {"RF": "RF Taiwan-only", "XGBoost": "XGBoost Taiwan-only"}
    )
    continued = tree_summary[columns].copy()
    frozen_rows = []
    for method in ("RF", "XGBoost"):
        values = frozen_metrics[method]
        for shots in SHOTS:
            frozen_rows.append(
                {
                    "method": f"{method} source frozen",
                    "shots": shots,
                    **{f"{metric}_mean": values[metric] for metric in metrics},
                    **{f"{metric}_sd": 0.0 for metric in metrics},
                }
            )
    result = pd.concat(
        [previous, continued, pd.DataFrame(frozen_rows)], ignore_index=True
    )
    return result.sort_values(["shots", "method"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zero-shot-dir", type=Path, required=True)
    parser.add_argument("--taiwan-only-baseline-dir", type=Path, required=True)
    parser.add_argument("--neural-finetune-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adaptation-trees", type=int, default=ADAPTATION_TREES)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.adaptation_trees < 1:
        raise ValueError("adaptation-trees must be positive")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, novel_rows, panel, axes = attach_taiwan_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    model, _, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, torch.device("cpu")
    )
    del model

    order = np.random.default_rng(SPLIT_SEED).permutation(novel_rows)
    test_rows = np.sort(order[:TEST_PROFILES])
    adaptation_pool = np.sort(order[TEST_PROFILES:])
    test_jobs = jobs[jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    if len(test_jobs) != 23_944 or test_jobs.axis_index.nunique() != 80:
        raise ValueError("unexpected fixed Taiwan test panel")
    split = panel[["profile_index"]].copy()
    split["fewshot_partition"] = np.where(
        split.profile_index.isin(test_rows),
        "fixed_test",
        np.where(
            split.profile_index.isin(adaptation_pool),
            "adaptation_pool",
            "excluded",
        ),
    )
    baseline_manifest, neural_manifest, zero_manifest, hashes = validate_inputs(
        args, split
    )
    subsets = {}
    subset_metadata = []
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            selected = np.sort(nested[:shots])
            subsets[(seed, shots)] = selected
            subset_metadata.append(
                {
                    "seed": seed,
                    "shots": shots,
                    "coverage_profiles": coverage_profiles,
                    "selected_profile_rows_sha256": row_hash(selected),
                }
            )

    expected = {
        "RF": pd.read_parquet(args.zero_shot_dir / "rf_predictions.parquet"),
        "XGBoost": pd.read_parquet(args.zero_shot_dir / "xgb_predictions.parquet"),
    }
    for method in expected:
        expected[method] = expected[method][
            expected[method].profile_index.isin(test_rows)
        ][["profile_index", "axis_index", "prediction"]]
        expected[method] = expected[method].sort_values(
            ["profile_index", "axis_index"]
        ).reset_index(drop=True)
        if len(expected[method]) != len(test_jobs):
            raise ValueError(f"stored {method} zero-shot panel mismatch")

    manifest = {
        "status": "running",
        "version": "taiwan_pretrained_tree_continuation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS),
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "test_profiles": TEST_PROFILES,
        "test_jobs": len(test_jobs),
        "test_axes": 80,
        "source_rf_trees": config.rf_trees,
        "source_xgb_rounds": config.xgb_trees,
        "adaptation_trees_or_rounds": args.adaptation_trees,
        "rf_continuation": (
            "retain all source trees and warm-start append Taiwan-only trees"
        ),
        "xgb_continuation": (
            "retain source booster and append Taiwan-only boosting rounds"
        ),
        "adaptation_model_selection": False,
        "subset_sampling": (
            "same deterministic coverage-first nested subsets as Taiwan-only trees and neural fine-tuning"
        ),
        "model_checkpoint_saved": False,
        "row_level_predictions_saved": False,
        **hashes,
        "zero_shot_manifest_sha256": digest(args.zero_shot_dir / "manifest.json"),
        "taiwan_only_baseline_manifest_sha256": digest(
            args.taiwan_only_baseline_dir / "manifest.json"
        ),
        "neural_finetune_manifest_sha256": digest(
            args.neural_finetune_dir / "manifest.json"
        ),
        "subset_hashes": subset_metadata,
        "completed_axes": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    source_parts = {"RF": [], "XGBoost": []}
    adapted_parts = {
        (method, seed, shots): []
        for method in ("RF source continued", "XGBoost source continued")
        for seed in SEEDS
        for shots in SHOTS
    }
    support = []
    sorted_axes = sorted(map(int, test_jobs.axis_index.unique()))
    for number, axis in enumerate(sorted_axes, start=1):
        part = test_jobs[test_jobs.axis_index.eq(axis)]
        test_axis_rows = part.profile_index.to_numpy(dtype=np.int64)
        x_test = dense_features(
            data,
            test_axis_rows,
            np.full(len(test_axis_rows), axis, dtype=np.int64),
            policy,
        )
        print(f"axis {number}/{len(sorted_axes)} index={axis}", flush=True)
        for short, label in (("rf", "RF"), ("xgb", "XGBoost")):
            source, source_labels = source_fit(short, axis, data, policy, config)
            source_normalized = source.predict(x_test)
            source_frame = prediction_frame(
                test_axis_rows, axis, source_normalized, data.scale[axis]
            )
            source_parts[label].append(source_frame)

            check = source_frame.merge(
                expected[label][expected[label].axis_index.eq(axis)],
                on=["profile_index", "axis_index"],
                suffixes=("_refit", "_stored"),
                validate="one_to_one",
            )
            maximum_difference = float(
                np.max(np.abs(check.prediction_refit - check.prediction_stored))
            )
            if maximum_difference > 1e-5:
                raise ValueError(
                    f"{label} refit differs from stored zero-shot predictions: "
                    f"{maximum_difference}"
                )
            support.append(
                {
                    "method": label,
                    "axis_index": axis,
                    "source_training_labels": source_labels,
                    "source_refit_max_abs_difference": maximum_difference,
                }
            )

            for seed in SEEDS:
                for shots in SHOTS:
                    selected = subsets[(seed, shots)]
                    labelled = selected[data.observed[selected, axis]]
                    if not len(labelled):
                        raise ValueError(
                            f"axis {axis} has no Taiwan labels for seed={seed}, shots={shots}"
                        )
                    x_local = dense_features(
                        data,
                        labelled,
                        np.full(len(labelled), axis, dtype=np.int64),
                        policy,
                    )
                    y_local = data.values[labelled, axis]
                    if short == "rf":
                        adapted = continue_rf(
                            source,
                            x_local,
                            y_local,
                            args.adaptation_trees,
                            config.n_jobs,
                        )
                        adapted_label = "RF source continued"
                    else:
                        adapted = continue_xgb(
                            source,
                            axis,
                            x_local,
                            y_local,
                            args.adaptation_trees,
                            config,
                        )
                        adapted_label = "XGBoost source continued"
                    normalized = adapted.predict(x_test)
                    adapted_parts[(adapted_label, seed, shots)].append(
                        prediction_frame(
                            test_axis_rows, axis, normalized, data.scale[axis]
                        )
                    )
                    del adapted
            del source
        manifest["completed_axes"].append(axis)
        write_json(args.output_dir / "manifest.json", manifest)

    source_metrics = {}
    axis_parts = []
    for method, pieces in source_parts.items():
        prediction = pd.concat(pieces, ignore_index=True)
        metrics, per_axis, _ = score_profile_axis_macro(data, test_jobs, prediction)
        source_metrics[method] = metrics
        per_axis.insert(0, "seed", 0)
        per_axis.insert(0, "shots", 0)
        per_axis.insert(0, "method", f"{method} source frozen")
        axis_parts.append(per_axis)

    records = []
    for (method, seed, shots), pieces in adapted_parts.items():
        prediction = pd.concat(pieces, ignore_index=True)
        metrics, per_axis, _ = score_profile_axis_macro(data, test_jobs, prediction)
        selected = subsets[(seed, shots)]
        counts = data.observed[np.ix_(selected, axes)].sum(axis=0)
        record = {
            "method": method,
            "shots": shots,
            "seed": seed,
            **metrics,
            "min_axis_labels": int(counts.min()),
            "max_axis_labels": int(counts.max()),
            "selected_profile_rows_sha256": row_hash(selected),
        }
        records.append(record)
        per_axis.insert(0, "seed", seed)
        per_axis.insert(0, "shots", shots)
        per_axis.insert(0, "method", method)
        axis_parts.append(per_axis)

    runs = pd.DataFrame(records)
    summary = summary_table(runs)
    previous_comparison = pd.read_csv(
        args.neural_finetune_dir / "comparison_summary.csv"
    )
    comparison = build_all_method_comparison(
        summary, source_metrics, previous_comparison
    )
    pd.DataFrame(
        [{"method": method, **values} for method, values in source_metrics.items()]
    ).to_csv(args.output_dir / "source_frozen_metrics.csv", index=False)
    runs.to_csv(args.output_dir / "continued_runs.csv", index=False)
    summary.to_csv(args.output_dir / "continued_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_all_methods.csv", index=False)
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    pd.DataFrame(support).to_csv(
        args.output_dir / "source_refit_support.csv", index=False
    )

    manifest.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        source_frozen_metrics=source_metrics,
        continued_summary=json.loads(summary.to_json(orient="records")),
        comparison_all_methods=json.loads(comparison.to_json(orient="records")),
        zero_shot_protocol_version=zero_manifest["version"],
        taiwan_only_protocol_version=baseline_manifest["version"],
        neural_finetune_protocol_version=neural_manifest["version"],
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["source_frozen_metrics"], indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

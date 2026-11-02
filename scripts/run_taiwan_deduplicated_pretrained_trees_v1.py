#!/usr/bin/env python3
"""Continue source-pretrained RF/XGBoost on deduplicated Taiwan few-shot data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_analysis_v1 import load_dependency_model  # noqa: E402
from foodcomp.dependency_mask_v1 import DependencyMaskPolicy  # noqa: E402
from foodcomp.research_dependency_mask_v1 import dense_features, score_profile_axis_macro  # noqa: E402
from foodcomp.research_no_foodname_v1 import numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402
from foodcomp.taiwan_deduplicated_v1 import (  # noqa: E402
    SEEDS, SHOTS, SPLIT_SEED, TEST_PROFILES, attach_clean_panel,
    coverage_nested_order, group_aware_split, row_hash,
)
from scripts.run_taiwan_pretrained_tree_continuation_v1 import (  # noqa: E402
    continue_rf, continue_xgb, prediction_frame, source_fit, summary_table,
)


ADAPTATION_TREES = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--overlap-ledger", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-zero-shot-dir", type=Path, required=True)
    parser.add_argument("--deduplicated-zero-shot-dir", type=Path, required=True)
    parser.add_argument("--taiwan-only-baseline-dir", type=Path, required=True)
    parser.add_argument("--neural-finetune-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adaptation-trees", type=int, default=ADAPTATION_TREES)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest.get("completed_axes", [])) != 80:
            raise ValueError("cannot finalize an incomplete pretrained-tree run")
        source_frame = pd.read_csv(args.output_dir / "source_frozen_metrics.csv")
        summary = pd.read_csv(args.output_dir / "continued_summary.csv")
        comparison = pd.read_csv(args.output_dir / "comparison_all_methods.csv")
        source_metrics = {
            row.method: {
                "axes": int(row.axes), "jobs": int(row.jobs),
                "macro_scaled_log_mae": float(row.macro_scaled_log_mae),
                "macro_scaled_log_rmse": float(row.macro_scaled_log_rmse),
                "macro_raw_mae": float(row.macro_raw_mae),
            }
            for row in source_frame.itertuples(index=False)
        }
        completed = datetime.now(timezone.utc)
        created = datetime.fromisoformat(manifest["created_utc"])
        manifest.update(
            status="complete", completed_utc=completed.isoformat(),
            elapsed_seconds=(completed - created).total_seconds(),
            source_frozen_metrics=source_metrics,
            continued_summary=json.loads(summary.to_json(orient="records")),
            comparison_all_methods=json.loads(comparison.to_json(orient="records")),
        )
        write_json(manifest_path, manifest)
        print(json.dumps(source_metrics, indent=2))
        print(summary.to_string(index=False))
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)

    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, clean_jobs, clean, _, axes = attach_clean_panel(
        data, args.audit_dir, args.overlap_ledger
    )
    test_rows, adaptation_pool = group_aware_split(clean)
    test_jobs = clean_jobs[clean_jobs.profile_index.isin(test_rows)].reset_index(drop=True)
    policy = DependencyMaskPolicy(data.axes)
    model, _, config = load_dependency_model(
        args.checkpoint, data, args.policy_manifest, torch.device("cpu")
    )
    del model

    baseline_manifest = json.loads(
        (args.taiwan_only_baseline_dir / "manifest.json").read_text(encoding="utf-8")
    )
    neural_manifest = json.loads(
        (args.neural_finetune_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for label, manifest in (("baseline", baseline_manifest), ("neural", neural_manifest)):
        if manifest.get("status") != "complete":
            raise ValueError(f"{label} run is incomplete")
        for key, value in {
            "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
            "test_profiles": TEST_PROFILES, "test_jobs": len(test_jobs),
            "overlap_ledger_sha256": digest(args.overlap_ledger),
        }.items():
            if manifest.get(key) != value:
                raise ValueError(f"{label} {key} mismatch")

    subsets, subset_metadata = {}, []
    for seed in SEEDS:
        nested, coverage_profiles = coverage_nested_order(
            adaptation_pool, axes, data.observed, seed
        )
        for shots in SHOTS:
            selected = np.sort(nested[:shots])
            subsets[(seed, shots)] = selected
            subset_metadata.append({
                "seed": seed, "shots": shots,
                "coverage_profiles": coverage_profiles,
                "selected_profile_rows_sha256": row_hash(selected),
            })

    expected = {
        "RF": pd.read_parquet(args.source_zero_shot_dir / "rf_predictions.parquet"),
        "XGBoost": pd.read_parquet(args.source_zero_shot_dir / "xgb_predictions.parquet"),
    }
    for method, prediction in expected.items():
        prediction = prediction[prediction.profile_index.isin(test_rows)][
            ["profile_index", "axis_index", "prediction"]
        ].sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
        if len(prediction) != len(test_jobs):
            raise ValueError(f"stored {method} zero-shot panel mismatch")
        expected[method] = prediction

    manifest = {
        "status": "running", "version": "taiwan_deduplicated_pretrained_trees_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shots": list(SHOTS), "seeds": list(SEEDS), "split_seed": SPLIT_SEED,
        "test_profiles": len(test_rows), "test_jobs": len(test_jobs),
        "test_axes": len(axes), "adaptation_pool_profiles": len(adaptation_pool),
        "identity_group_aware_split": True,
        "source_rf_trees": config.rf_trees,
        "source_xgb_rounds": config.xgb_trees,
        "adaptation_trees_or_rounds": args.adaptation_trees,
        "rf_continuation": "retain 200 source trees and append 20 Taiwan-only trees",
        "xgb_continuation": "retain 300 source rounds and append 20 Taiwan-only rounds",
        "adaptation_model_selection": False,
        "model_checkpoint_saved": False, "row_predictions_saved": False,
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "overlap_ledger_sha256": digest(args.overlap_ledger),
        "policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "deduplicated_zero_shot_manifest_sha256": digest(
            args.deduplicated_zero_shot_dir / "manifest.json"
        ),
        "baseline_manifest_sha256": digest(args.taiwan_only_baseline_dir / "manifest.json"),
        "neural_manifest_sha256": digest(args.neural_finetune_dir / "manifest.json"),
        "subset_hashes": subset_metadata, "completed_axes": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)

    source_parts = {"RF": [], "XGBoost": []}
    adapted_parts = {
        (method, seed, shots): []
        for method in ("RF source continued", "XGBoost source continued")
        for seed in SEEDS for shots in SHOTS
    }
    support = []
    sorted_axes = sorted(map(int, test_jobs.axis_index.unique()))
    for number, axis in enumerate(sorted_axes, start=1):
        part = test_jobs[test_jobs.axis_index.eq(axis)]
        rows = part.profile_index.to_numpy(np.int64)
        x_test = dense_features(
            data, rows, np.full(len(rows), axis, dtype=np.int64), policy
        )
        print(f"axis {number}/{len(sorted_axes)} index={axis}", flush=True)
        for short, label in (("rf", "RF"), ("xgb", "XGBoost")):
            source, source_labels = source_fit(short, axis, data, policy, config)
            source_frame = prediction_frame(
                rows, axis, source.predict(x_test), data.scale[axis]
            )
            source_parts[label].append(source_frame)
            check = source_frame.merge(
                expected[label][expected[label].axis_index.eq(axis)],
                on=["profile_index", "axis_index"], suffixes=("_refit", "_stored"),
                validate="one_to_one",
            )
            maximum = float(np.max(np.abs(check.prediction_refit - check.prediction_stored)))
            if maximum > 1e-5:
                raise ValueError(f"{label} source refit mismatch: {maximum}")
            support.append({
                "method": label, "axis_index": axis,
                "source_training_labels": source_labels,
                "source_refit_max_abs_difference": maximum,
            })
            for seed in SEEDS:
                for shots in SHOTS:
                    selected = subsets[(seed, shots)]
                    labelled = selected[data.observed[selected, axis]]
                    x_local = dense_features(
                        data, labelled, np.full(len(labelled), axis, dtype=np.int64), policy
                    )
                    y_local = data.values[labelled, axis]
                    if short == "rf":
                        adapted = continue_rf(
                            source, x_local, y_local, args.adaptation_trees, config.n_jobs
                        )
                        adapted_label = "RF source continued"
                    else:
                        adapted = continue_xgb(
                            source, axis, x_local, y_local, args.adaptation_trees, config
                        )
                        adapted_label = "XGBoost source continued"
                    adapted_parts[(adapted_label, seed, shots)].append(
                        prediction_frame(rows, axis, adapted.predict(x_test), data.scale[axis])
                    )
                    del adapted
            del source
        manifest["completed_axes"].append(axis)
        write_json(args.output_dir / "manifest.json", manifest)

    source_metrics, axis_parts = {}, []
    for method, pieces in source_parts.items():
        metrics, per_axis, _ = score_profile_axis_macro(
            data, test_jobs, pd.concat(pieces, ignore_index=True)
        )
        source_metrics[method] = metrics
        per_axis.insert(0, "seed", 0)
        per_axis.insert(0, "shots", 0)
        per_axis.insert(0, "method", f"{method} source frozen")
        axis_parts.append(per_axis)

    records = []
    for (method, seed, shots), pieces in adapted_parts.items():
        metrics, per_axis, _ = score_profile_axis_macro(
            data, test_jobs, pd.concat(pieces, ignore_index=True)
        )
        counts = data.observed[np.ix_(subsets[(seed, shots)], axes)].sum(0)
        records.append({
            "method": method, "shots": shots, "seed": seed, **metrics,
            "min_axis_labels": int(counts.min()), "max_axis_labels": int(counts.max()),
            "selected_profile_rows_sha256": row_hash(subsets[(seed, shots)]),
        })
        per_axis.insert(0, "seed", seed)
        per_axis.insert(0, "shots", shots)
        per_axis.insert(0, "method", method)
        axis_parts.append(per_axis)
    runs = pd.DataFrame(records)
    summary = summary_table(runs)

    comparison = pd.read_csv(args.neural_finetune_dir / "comparison_summary.csv")
    metrics = ["macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"]
    frozen_rows = [{
        "method": f"{method} source frozen", "shots": shots,
        **{f"{metric}_mean": values[metric] for metric in metrics},
        **{f"{metric}_sd": 0.0 for metric in metrics},
        "min_axis_labels": 0, "max_axis_labels": 0,
    } for method, values in source_metrics.items() for shots in SHOTS]
    comparison = pd.concat((
        comparison, summary, pd.DataFrame(frozen_rows)
    ), ignore_index=True).sort_values(["shots", "method"])

    pd.DataFrame([
        {"method": method, **values} for method, values in source_metrics.items()
    ]).to_csv(args.output_dir / "source_frozen_metrics.csv", index=False)
    runs.to_csv(args.output_dir / "continued_runs.csv", index=False)
    summary.to_csv(args.output_dir / "continued_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_all_methods.csv", index=False)
    pd.concat(axis_parts, ignore_index=True).to_csv(
        args.output_dir / "axis_metrics.csv", index=False
    )
    pd.DataFrame(support).to_csv(args.output_dir / "source_refit_support.csv", index=False)
    manifest.update(
        status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,
        source_frozen_metrics=source_metrics,
        continued_summary=json.loads(summary.to_json(orient="records")),
        comparison_all_methods=json.loads(comparison.to_json(orient="records")),
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(source_metrics, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

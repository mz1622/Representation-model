"""Shared-panel baseline screen. Does not authorize confirmatory claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData, VERSION, weighted_median, score_predictions, write_json, digest
from foodcomp.research_text import prepare_names

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/processed"/VERSION)
    parser.add_argument("--view", choices=["inclusive", "quarantined"], default="quarantined")
    parser.add_argument("--method", choices=["median", "name_knn", "rf", "xgb"], required=True)
    parser.add_argument("--mode", choices=["completion", "name_only"], default="completion")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--leaf-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--xgb-objective", choices=["reg:squarederror", "reg:absoluteerror"], default="reg:squarederror")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--text-components", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    data = ResearchData(args.data_dir, args.view)
    manifest = {"status": "running", "run_kind": "exploratory single-seed baseline",
                "configuration": vars(args), "data_manifest_sha256": digest(args.data_dir/"manifest.json"),
                "protocol_sha256": data.manifest["protocol_sha256"],
                "jobs_sha256": data.manifest["artifact_hashes"][f"{args.view}_validation_jobs.parquet"],
                "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "code_hashes": {p: digest(ROOT/p) for p in ["src/foodcomp/research_r0.py", "src/foodcomp/research_text.py", "scripts/run_foodnutrigpt_v9_r0_baseline.py"]},
                "training_row_cap": None, "complete_test_opened": False,
                "scientific_claim_allowed": False}
    write_json(args.output_dir/"run_manifest.json", manifest)
    try:
        if args.method == "median":
            text = None
        else:
            text, cache = prepare_names(data, ROOT, args.text_components)
            manifest["name_cache"] = str(cache)
            manifest["name_cache_manifest_sha256"] = digest(cache/"manifest.json")
        predictions, fitting = [], []
        with threadpool_limits(limits=args.n_jobs):
            for count, axis in enumerate(data.targets, 1):
                train = data.train[data.observed[data.train, axis]]
                jobs = data.jobs[data.jobs.axis_index.eq(axis)]
                rows = jobs.profile_index.to_numpy()
                if not len(train):
                    raise ValueError(f"No training labels for axis {axis}")
                weights = data.weights[train, axis]
                started_axis = time.monotonic()
                if args.method == "median":
                    result = np.full(len(rows), weighted_median(data.raw[train, axis], weights))
                elif args.method == "name_knn":
                    knn = NearestNeighbors(n_neighbors=min(10, len(train)), n_jobs=args.n_jobs)
                    knn.fit(text[train])
                    distances, indexes = knn.kneighbors(text[rows])
                    neighbor_rows = train[indexes]
                    w = 1/(distances+1e-3)*data.weights[neighbor_rows, axis]
                    predicted_t = np.sum(data.values[neighbor_rows, axis]*w, axis=1)/w.sum(axis=1)
                    result = data.scale[axis]*np.expm1(predicted_t)
                else:
                    x_train = data.dense_features(train, text, axis, mode=args.mode)
                    x_valid = data.dense_features(rows, text, axis, mode=args.mode)
                    if args.method == "rf":
                        model = RandomForestRegressor(n_estimators=args.trees, max_depth=args.max_depth or None,
                            min_samples_leaf=args.leaf_size, max_features=.5, random_state=args.seed+int(axis), n_jobs=args.n_jobs)
                    else:
                        model = XGBRegressor(n_estimators=args.trees, max_depth=args.max_depth,
                            learning_rate=args.learning_rate, min_child_weight=5, subsample=.8,
                            colsample_bytree=.8, reg_lambda=1., tree_method="hist",
                            objective=args.xgb_objective, random_state=args.seed+int(axis), n_jobs=args.n_jobs)
                    model.fit(x_train, data.values[train, axis], sample_weight=weights/weights.mean())
                    predicted_t = np.maximum(model.predict(x_valid).astype(float), 0)
                    result = data.scale[axis]*np.expm1(predicted_t)
                if not np.isfinite(result).all():
                    raise ValueError("Nonfinite predictions.")
                predictions.append(pd.DataFrame({"profile_index": rows, "axis_index": int(axis), "prediction": result}))
                fitting.append({"axis_index": int(axis), "train_profiles": len(train), "validation_jobs": len(rows),
                                "fit_predict_seconds": time.monotonic()-started_axis})
                if count % 10 == 0 or count == len(data.targets):
                    print(f"{args.method}/{args.mode} axis {count}/{len(data.targets)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
        pred = pd.concat(predictions, ignore_index=True)
        summary, per_axis, candidates = score_predictions(data, pred)
        pred.to_parquet(args.output_dir/"predictions.parquet", index=False)
        per_axis.to_csv(args.output_dir/"axis_metrics.csv", index=False)
        candidates.to_parquet(args.output_dir/"candidate_errors.parquet", index=False)
        pd.DataFrame(fitting).to_csv(args.output_dir/"fitting_cost.csv", index=False)
        write_json(args.output_dir/"metrics.json", summary)
        manifest.update(status="complete", elapsed_seconds=time.monotonic()-started,
                        nutrition_primary=summary["nutrition"]["scaled_log_mae"],
                        unresolved_evidence=data.manifest["root_cause_status"])
        write_json(args.output_dir/"run_manifest.json", manifest)
        (args.output_dir/"README.md").write_text(
            f"# R0 {args.method} / {args.mode} / {args.view}\n\n"
            "Hypothesis: a shared label/context contract permits a comparable baseline screen.\n\n"
            f"Nutrition primary: {summary['nutrition']['scaled_log_mae']:.8f}; "
            f"legacy log MAE: {summary['nutrition']['log_mae']:.8f}.\n\n"
            "All eligible training rows were used; this is one fixed configuration and one seed, not a tuned or confirmed baseline. "
            "No causal model claim is justified. Unit provenance remains unresolved. Frozen test stayed closed. "
            "Configuration, source-code hashes, data/mask hashes, cost and per-axis results accompany this report.\n",
            encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/"run_manifest.json", manifest)
        raise

if __name__ == "__main__":
    main()

"""Rescore one registered tree configuration across all three seeds, without ensembling."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_confirmation import SEEDS, seed_average_errors


def validate_manifests(manifests, method, registered, fingerprints):
    """Reject partial, duplicate or changed runs before loading any predictions."""
    if len(manifests) != len(SEEDS):
        raise ValueError("Exactly three registered seeds required.")
    seen = set()
    config = code = None
    for manifest in manifests:
        if manifest["status"] != "complete" or manifest.get("complete_test_opened", True):
            raise ValueError("Require completed runs with the test closed.")
        if manifest.get("training_row_cap", "missing") is not None:
            raise ValueError("All eligible training rows are required.")
        if any(manifest.get(key) != value for key, value in fingerprints.items()):
            raise ValueError("Data, protocol, task or name input fingerprint changed.")
        settings = dict(manifest["configuration"])
        seed = settings.pop("seed")
        settings.pop("output_dir")
        if seed not in SEEDS or seed in seen:
            raise ValueError("Unregistered or duplicate seed.")
        seen.add(seed)
        if settings.get("method") != method or settings.get("mode") != "completion":
            raise ValueError("Incorrect method or task.")
        expected = next(r for r in registered if r["method"] == method and r["seed"] == seed)
        if any(settings.get(k) != v for k, v in expected.items() if k not in {"name", "seed"}):
            raise ValueError("Configuration differs from preregistered tree.")
        if config is None:
            config, code = settings, manifest["code_hashes"]
        elif config != settings or code != manifest["code_hashes"]:
            raise ValueError("Configuration or implementation changed across seeds.")
    return config, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rf", "xgb"], required=True)
    parser.add_argument("--run-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    receipt = {"status": "incomplete", "method": args.method, "complete_test_opened": False,
               "neural_improvement_confirmed": False, "script_sha256": digest(Path(__file__))}
    try:
        data = ResearchData(ROOT / "data/processed" / VERSION)
        config_file = ROOT / "experiments/foodnutrigpt_v9_research/baseline_confirmation_v1/config.json"
        registry = json.loads(config_file.read_text(encoding="utf-8"))
        manifests = [json.loads((p / "run_manifest.json").read_text(encoding="utf-8")) for p in args.run_dirs]
        cache = Path(manifests[0]["name_cache"]) / "manifest.json"
        fingerprints = {"data_manifest_sha256": digest(data.root / "manifest.json"),
                        "protocol_sha256": digest(data.root / "protocol.json"),
                        "jobs_sha256": digest(data.root / f"{data.view}_validation_jobs.parquet"),
                        "name_cache_manifest_sha256": digest(cache)}
        configuration, code = validate_manifests(manifests, args.method, registry["registered_candidates"], fingerprints)
        for path, hashed in code.items():
            if digest(ROOT / path) != hashed:
                raise ValueError("Local training implementation differs from frozen runs; use its recorded checkout.")
        scores = {}; groups = {}; all_axes = []; sources = []; runs = []; counts = None
        for folder, manifest in zip(args.run_dirs, manifests):
            seed = manifest["configuration"]["seed"]
            predictions_file = folder / "predictions.parquet"
            predictions = pd.read_parquet(predictions_file)
            score, axes, grouped = score_predictions(data, predictions)
            if score != json.loads((folder / "metrics.json").read_text(encoding="utf-8")):
                raise ValueError("Saved metrics differ from evaluator recomputation.")
            cost = pd.read_csv(folder / "fitting_cost.csv").sort_values("axis_index").reset_index(drop=True)
            current = cost[["axis_index", "train_profiles", "validation_jobs"]]
            np.testing.assert_array_equal(current.axis_index, data.targets)
            expected_train = data.observed[data.train][:, data.targets].sum(axis=0)
            np.testing.assert_array_equal(current.train_profiles, expected_train)
            np.testing.assert_array_equal(current.validation_jobs, data.jobs.groupby("axis_index").size().reindex(data.targets))
            if counts is None:
                counts = current
            else:
                pd.testing.assert_frame_equal(counts, current)
            scores[seed], groups[seed] = score, grouped
            axes.insert(0, "seed", seed); all_axes.append(axes)
            jobs = data.jobs.merge(predictions, on=["profile_index", "axis_index"], validate="one_to_one")
            jobs = jobs.merge(data.profiles[["profile_index", "exact_name_group_id", "source_key"]],
                              on="profile_index", validate="many_to_one")
            cells = jobs.groupby(["exact_name_group_id", "axis_index", "source_key"], as_index=False).agg(
                target=("target", "median"), prediction=("prediction", "median"))
            cells = cells.merge(data.axes[["axis_index", "loss_group"]], on="axis_index", validate="many_to_one")
            cells = cells[cells.loss_group.eq("nutrition")].copy()
            scales = data.scale[cells.axis_index.to_numpy()]
            cells["scaled_log_mae"] = np.abs(np.log1p(cells.prediction / scales) - np.log1p(cells.target / scales))
            cells["log_mae"] = np.abs(np.log1p(cells.prediction) - np.log1p(cells.target))
            if not np.isfinite(cells[["scaled_log_mae", "log_mae"]].to_numpy()).all():
                raise ValueError("Nonfinite source diagnostic.")
            for source, frame in cells.groupby("source_key"):
                by_axis = frame.groupby("axis_index")[["scaled_log_mae", "log_mae"]].mean()
                sources.append({"seed": seed, "source": source, "supported_nutrition_axes": len(by_axis),
                                "candidate_groups": frame.exact_name_group_id.nunique(), **by_axis.mean().to_dict()})
            runs.append({"seed": seed, "directory": str(folder), "code_commit": manifest["code_commit"],
                         "manifest_sha256": digest(folder / "run_manifest.json"),
                         "predictions_sha256": digest(predictions_file), "elapsed_seconds": manifest["elapsed_seconds"],
                         "scores": score})
        summaries = {}
        for group in ["nutrition", "food_metabolome", "all"]:
            eligible = data.axes[data.axes.loss_eligible]
            if group != "all":
                eligible = eligible[eligible.loss_group.eq(group)]
            _, check = seed_average_errors(groups, eligible.axis_index.to_numpy())
            summary = {}
            for metric in scores[SEEDS[0]][group]:
                if metric == "axes":
                    continue
                values = np.array([scores[s][group][metric] for s in SEEDS], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("Undefined or nonfinite aggregate seed metric.")
                summary[metric] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
                                   "min": float(values.min()), "max": float(values.max())}
            for metric, expected in check["metrics"].items():
                np.testing.assert_allclose(summary[metric]["mean"], expected["mean"], rtol=1e-12)
            summaries[group] = {"axes": len(eligible), "metrics": summary}
        pd.concat(all_axes).to_csv(out / "all_seed_axis_metrics.csv", index=False)
        pd.DataFrame(sources).to_csv(out / "all_seed_source_metrics.csv", index=False)
        receipt.update(status="complete", seeds=list(SEEDS), runs=sorted(runs, key=lambda r: r["seed"]),
                       configuration=configuration, training_code_hashes=code, fingerprints=fingerprints,
                       registry_sha256=digest(config_file), group_summaries=summaries,
                       all_training_counts_verified=True,
                       aggregation="Score each fitted model, then average its errors; no averaged predictions, best-seed choice or ensemble.",
                       uncertainty="Sample SD across three seeds only. No food-group confidence interval or model superiority claim is made here.",
                       scope="Fixed selected tree stability on quarantined validation only. Source diagnostics have differing axis support; provenance unresolved.")
        lines = [f"# {args.method.upper()} fixed-configuration seed stability", "",
                 "| Seed | Nutrition primary | Legacy log-MAE | Seconds |", "|---|---:|---:|---:|"]
        for run in receipt["runs"]:
            metrics = run["scores"]["nutrition"]
            lines.append(f"| {run['seed']} | {metrics['scaled_log_mae']:.9f} | {metrics['log_mae']:.9f} | {run['elapsed_seconds']:.2f} |")
        for metric in ["scaled_log_mae", "log_mae"]:
            result = summaries["nutrition"]["metrics"][metric]
            lines.extend(["", f"{metric}: {result['mean']:.9f} ± {result['sample_std']:.9f} (seed sample SD)."])
        lines.extend(["", receipt["aggregation"], "", receipt["uncertainty"], "", receipt["scope"],
                      "", "No neural improvement or foundation-model claim. Test remains closed."])
        (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out / "summary.json", receipt)
        raise
    write_json(out / "summary.json", receipt)
    print({"method": args.method, "status": receipt["status"], "nutrition": summaries["nutrition"]})


if __name__ == "__main__":
    main()

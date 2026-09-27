"""Partition the existing model gap without changing primary metric denominators."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from foodcomp.research_r0 import ResearchData, VERSION, digest, score_predictions, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-dir", type=Path, default=ROOT / "output/v9_r2/mlp60_mae_width512")
    parser.add_argument("--tree-dir", type=Path, default=ROOT / "output/v9_r1/xgb800d10")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        data = ResearchData(ROOT / "data/processed" / VERSION)
        near_path = ROOT / "reports/v9_r0_sources_v1/near_name_candidates.csv"
        near = pd.read_csv(near_path)
        if not near.decision.eq("candidate_only_requires_identity_review_no_merge").all():
            raise ValueError("Unexpected identity-review policy in the fixed candidate list.")
        validation = data.profiles.iloc[data.validation]
        training = data.profiles.iloc[data.train]
        if not near.validation_profile_id.isin(validation.profile_id).all() or not near.train_profile_id.isin(training.profile_id).all():
            raise ValueError("Near-name review list contains a profile outside its registered partition.")
        flagged = set(validation.loc[validation.original_name.isin(set(near.validation_name)), "exact_name_group_id"])
        keys = ["exact_name_group_id", "axis_index"]
        metrics = ["scaled_log_mae", "log_mae"]
        scored = {}
        for label, folder, filename in [
            ("neural", args.neural_dir, "completion_predictions.parquet"),
            ("tree", args.tree_dir, "predictions.parquet"),
        ]:
            manifest = json.loads((folder / "run_manifest.json").read_text())
            if manifest["status"] != "complete" or manifest.get("test_opened", False):
                raise ValueError("Completed, test-closed model predictions required.")
            predictions = pd.read_parquet(folder / filename)
            aggregate, _, cells = score_predictions(data, predictions)
            scored[label] = (aggregate, cells)
            result[f"{label}_prediction_sha256"] = digest(folder / filename)
            result[f"{label}_run_manifest_sha256"] = digest(folder / "run_manifest.json")
        joined = scored["neural"][1][keys + metrics].merge(
            scored["tree"][1][keys + metrics], on=keys, how="outer", validate="one_to_one",
            suffixes=("_neural", "_tree"), indicator=True,
        )
        if not joined._merge.eq("both").all():
            raise ValueError("Model predictions cover different candidate/axis cells.")
        nutrition = data.axes.loc[data.axes.loss_group.eq("nutrition") & data.axes.axis_index.isin(data.targets), "axis_index"].to_numpy()
        joined = joined[joined.axis_index.isin(nutrition)].copy()
        if len(nutrition) != 142 or set(joined.axis_index) != set(nutrition):
            raise ValueError("Primary-axis coverage changed.")
        values = joined[[f"{metric}_{model}" for metric in metrics for model in ["neural", "tree"]]]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Non-finite observed errors.")
        joined["near_name_review_candidate"] = joined.exact_name_group_id.isin(flagged)
        denominators = joined.groupby("axis_index").size().reindex(nutrition)
        rows, partitions = [], {}
        for flag in [False, True]:
            subset = joined[joined.near_name_review_candidate.eq(flag)]
            supports = subset.groupby("axis_index").size().reindex(nutrition, fill_value=0)
            label = "flagged_for_review" if flag else "not_flagged_by_this_list"
            partitions[label] = {
                "candidate_groups": int(subset.exact_name_group_id.nunique()),
                "candidate_axis_cells": len(subset),
                "supported_axes": int(supports.gt(0).sum()),
                "primary_axis_count_in_denominator": len(nutrition),
                "macro_weight_mass": float((supports / denominators).mean()),
                "metrics": {},
            }
            for metric in metrics:
                columns = [f"{metric}_{model}" for model in ["neural", "tree"]]
                contributions = subset.groupby("axis_index")[columns].sum().reindex(nutrition, fill_value=0).div(denominators, axis=0)
                contributions["difference"] = contributions[columns[0]] - contributions[columns[1]]
                partitions[label]["metrics"][metric] = {
                    "neural_contribution": float(contributions[columns[0]].mean()),
                    "tree_contribution": float(contributions[columns[1]].mean()),
                    "neural_minus_tree_contribution": float(contributions.difference.mean()),
                }
                for axis, row in contributions.iterrows():
                    rows.append({"partition": label, "metric": metric, "axis_index": int(axis),
                                 "partition_support": int(supports.loc[axis]), "full_axis_support": int(denominators.loc[axis]),
                                 "neural_contribution": row[columns[0]], "tree_contribution": row[columns[1]],
                                 "neural_minus_tree_contribution": row.difference})
        originals = {}
        for metric in metrics:
            originals[metric] = {model: scored[model][0]["nutrition"][metric] for model in ["neural", "tree"]}
            originals[metric]["neural_minus_tree"] = originals[metric]["neural"] - originals[metric]["tree"]
            for model in ["neural", "tree"]:
                total = sum(part["metrics"][metric][f"{model}_contribution"] for part in partitions.values())
                np.testing.assert_allclose(total, originals[metric][model], rtol=0, atol=1e-12)
            total_gap = sum(part["metrics"][metric]["neural_minus_tree_contribution"] for part in partitions.values())
            np.testing.assert_allclose(total_gap, originals[metric]["neural_minus_tree"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(sum(p["macro_weight_mass"] for p in partitions.values()), 1, rtol=0, atol=1e-12)
        if sum(p["candidate_groups"] for p in partitions.values()) != joined.exact_name_group_id.nunique():
            raise ValueError("Review partitions overlap or omit a candidate group.")
        pd.DataFrame(rows).merge(data.axes[["axis_index", "canonical_name"]], on="axis_index", validate="many_to_one").to_csv(args.output_dir / "axis_contributions.csv", index=False)
        result.update(status="complete", originals=originals, partitions=partitions,
                      reconstruction_absolute_tolerance=1e-12,
                      all_partitions_reconstruct_both_original_metrics=True,
                      data_sha256=digest(data.root / "manifest.json"), near_name_list_sha256=digest(near_path),
                      near_name_review_rows=len(near), nutrition_candidate_groups=int(joined.exact_name_group_id.nunique()),
                      elapsed_seconds=time.monotonic() - started,
                      scope="Descriptive partition of fixed fitted-model errors, not a changed metric or confidence interval. Flagged names are unconfirmed identity-review candidates; absence of a flag does not establish absence of aliases. No rows removed, no identity merge, no leakage or source-copying conclusion.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / "summary.json", result)
        raise
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps({key: result[key] for key in ["status", "originals", "partitions"]}, indent=2))


if __name__ == "__main__":
    main()

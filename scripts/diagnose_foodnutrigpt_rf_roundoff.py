"""Investigate a strict RF replay failure without changing its acceptance threshold."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, _forest
from threadpoolctl import threadpool_limits
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_text import prepare_names


def state_hash(model):
    h = hashlib.sha256()
    for tree in model.estimators_:
        state = tree.tree_.__getstate__()
        for key in ["max_depth", "node_count"]:
            h.update(str(state[key]).encode())
        # Structured array padding is not semantic; hash fields independently.
        for key in state["nodes"].dtype.names:
            h.update(np.ascontiguousarray(state["nodes"][key]).tobytes())
        h.update(state["values"].tobytes())
    return h.hexdigest()


def main():
    out = ROOT / "reports/v9_rf_roundoff_diagnosis_v1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    record = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        original = ROOT / "output/v9_r1/rf400_unlimited_leaf1"
        replay = ROOT / "output/v9_baseline_confirmation_v1/rf_seed20260922"
        strict = ROOT / "reports/v9_baseline_rf_seed22_replay_v1/audit.json"
        prior = json.loads(strict.read_text())
        if prior["status"] != "failed" or prior["message"] != "Replay predictions differ; investigate before attributing to seed variation.":
            raise ValueError("This diagnostic requires the preserved prediction mismatch audit.")
        data = ResearchData(ROOT / "data/processed" / VERSION)
        old = pd.read_parquet(original / "predictions.parquet")
        new = pd.read_parquet(replay / "predictions.parquet")
        joined = old.merge(new, on=["profile_index", "axis_index"], suffixes=("_old", "_new"), validate="one_to_one")
        if len(joined) != len(old) or len(joined) != len(new):
            raise ValueError("Prediction coverage differs.")
        a, b = joined.prediction_old.to_numpy(), joined.prediction_new.to_numpy()
        if not np.isfinite(np.column_stack([a, b])).all() or np.any(a < 0) or np.any(b < 0):
            raise ValueError("Invalid replay values.")
        delta = np.abs(a - b)
        changed = joined.loc[delta > 0, ["axis_index"]].copy()
        changed["absolute_difference"] = delta[delta > 0]
        changed["ulp_difference"] = delta[delta > 0] / np.spacing(np.maximum(a, b)[delta > 0])
        differences = changed.groupby("axis_index").agg(changed=("absolute_difference", "size"),
            max_absolute_difference=("absolute_difference", "max"), max_ulp_difference=("ulp_difference", "max"))
        costs = pd.read_csv(original / "fitting_cost.csv")
        differences = differences.merge(costs, on="axis_index").sort_values(["fit_predict_seconds", "axis_index"])
        # Predefined resource-saving selection: fastest affected axis, not largest effect.
        axis = int(differences.iloc[0].axis_index)
        text, cache = prepare_names(data, ROOT)
        train = data.train[data.observed[data.train, axis]]
        rows = data.jobs.loc[data.jobs.axis_index.eq(axis), "profile_index"].to_numpy()
        xtrain = data.dense_features(train, text, axis); xvalid = data.dense_features(rows, text, axis)
        weights = data.weights[train, axis]; y = data.values[train, axis]
        params = dict(n_estimators=400, max_depth=None, min_samples_leaf=1, max_features=.5,
                      random_state=20260922 + axis, n_jobs=6)
        with threadpool_limits(limits=6):
            model = RandomForestRegressor(**params).fit(xtrain, y, sample_weight=weights / weights.mean())
            first_hash = state_hash(model)
            parallel = np.stack([model.predict(xvalid) for _ in range(12)])
            model.n_jobs = 1
            serial = np.stack([model.predict(xvalid) for _ in range(3)])
            np.testing.assert_array_equal(serial[0], serial[1]); np.testing.assert_array_equal(serial[0], serial[2])
            if state_hash(model) != first_hash:
                raise AssertionError("Prediction changed fitted tree parameters.")
            refit = RandomForestRegressor(**params).fit(xtrain, y, sample_weight=weights / weights.mean())
            second_hash = state_hash(refit)
            if second_hash != first_hash:
                raise AssertionError("Repeated same-seed fit changes tree states on this axis.")
            refit.n_jobs = 1
            np.testing.assert_array_equal(serial[0], refit.predict(xvalid))
            individual = np.stack([tree.predict(xvalid) for tree in model.estimators_])
            rng = np.random.default_rng(20260922)
            orders = [np.arange(400), np.arange(400)[::-1]] + [rng.permutation(400) for _ in range(10)]
            sums = []
            for order in orders:
                total = np.zeros(len(rows), dtype=np.float64)
                for index in order:
                    total += individual[index]
                sums.append(total / 400)
            sums = np.stack(sums)
        np.testing.assert_array_equal(sums[0], serial[0])
        oldscore, _, _ = score_predictions(data, old); newscore, _, _ = score_predictions(data, new)
        if oldscore != newscore:
            raise AssertionError("Aggregate metrics differ; requires additional impact analysis.")
        transformed = np.abs(np.log1p(a / data.scale[joined.axis_index]) - np.log1p(b / data.scale[joined.axis_index]))
        source_text = inspect.getsource(RandomForestRegressor.predict) + "\n" + inspect.getsource(_forest._accumulate_prediction)
        (out / "installed_forest_prediction_source.py.txt").write_text(source_text, encoding="utf-8")
        differences.to_csv(out / "affected_axes.csv", index=False)
        np.savez_compressed(out / "axis_prediction_probe.npz", axis=axis, rows=rows, parallel=parallel,
                            serial=serial, addition_orders=sums)
        record.update(status="complete", strict_audit_sha256=digest(strict), strict_bitwise_replay_passed=False,
            prediction_count=len(a), changed_prediction_count=int((delta > 0).sum()),
            max_absolute_difference=float(delta.max()), max_ulp_difference=float(changed.ulp_difference.max()),
            max_transformed_difference=float(transformed.max()), explicit_zero_status_identical=bool(np.array_equal(a == 0, b == 0)),
            all_recomputed_aggregate_metrics_bitwise_equal=True, axis=axis, train_profiles=len(train), validation_jobs=len(rows),
            axis_selection="Fastest previously fitted axis with any observed replay discrepancy.",
            probe_fit_count=2, probe_parameters=params, identical_refit_tree_state_hash=first_hash,
            serial_prediction_repeats_identical=True,
            parallel_repeat_changed_cells=int(np.count_nonzero(parallel != parallel[0])),
            parallel_repeat_max_difference=float(np.max(np.abs(parallel - parallel[0]))),
            parallel_vs_serial_max_difference=float(np.max(np.abs(parallel - serial[0]))),
            changed_addition_order_max_difference=float(np.max(np.abs(sums - sums[0]))),
            sklearn_version=sklearn.__version__, installed_forest_module_sha256=digest(Path(inspect.getfile(_forest))),
            data_sha256=digest(data.root / "manifest.json"), name_cache_sha256=digest(cache / "manifest.json"),
            elapsed_seconds=time.monotonic() - started,
            scope="One real affected axis, two same-seed fits, fixed trees under parallel/serial and permuted summation. No baseline hyperparameter change or new screening candidate.",
            causal_limit="Local library code and this controlled probe can establish accumulation-order roundoff on this axis. Original full forests were not saved, so this is not bitwise proof about all 187 old forest states; keep the strict failure.")
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out / "summary.json", record)
        raise
    write_json(out / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

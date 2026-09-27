"""Actual training-only R8 tree preflight; diagnostics are not screened candidates."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_completion_input import completion_view
from foodcomp.research_tree_prediction import make_tree, deterministic_inference, predict_raw, parameter_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    receipt = {'status': 'incomplete', 'complete_test_opened': False, 'script_sha256': digest(Path(__file__))}
    try:
        data = ResearchData(ROOT / 'data/processed' / VERSION)
        nutrition = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
        axes = nutrition[[0, len(nutrition) // 2, -1]]
        records = []
        for dims in [32, 128]:
            text, _, _ = completion_view(data, ROOT, dims)
            for axis in axes:
                train = data.train[data.observed[data.train, axis]]
                train = np.random.default_rng(20260922 + int(axis)).permutation(train)[:256]
                query = train[:64]
                x = data.dense_features(train, text, axis)
                q = data.dense_features(query, text, axis)
                name = np.concatenate([text[query], np.zeros((len(query), 504), np.float32)], axis=1)
                np.testing.assert_array_equal(name, data.dense_features(query, text, axis, mode='name_only'))
                family = data.families == data.families[axis]
                where = np.ix_(query, np.flatnonzero(family))
                previous_values = data.values[where].copy(); previous_observed = data.observed[where].copy()
                try:
                    data.values[where] = 999
                    data.observed[where] = ~previous_observed
                    np.testing.assert_array_equal(q, data.dense_features(query, text, axis))
                    np.testing.assert_array_equal(name, data.dense_features(query, text, axis, mode='name_only'))
                finally:
                    data.values[where] = previous_values; data.observed[where] = previous_observed
                weight = data.weights[train, axis]; weight = weight / weight.mean()
                for kind in ['rf', 'xgb']:
                    cfg = {'kind': kind, 'trees': 24, 'max_depth': 10, 'leaf_size': 1, 'max_features': .5}
                    model = make_tree(cfg, seed=20260922, axis=axis, n_jobs=2)
                    reference = (RandomForestRegressor(n_estimators=24, max_depth=None, min_samples_leaf=1,
                        max_features=.5, random_state=20260922 + int(axis), n_jobs=2) if kind == 'rf' else
                        XGBRegressor(n_estimators=24, max_depth=10, learning_rate=.03, min_child_weight=5,
                            subsample=.8, colsample_bytree=.8, reg_lambda=1., tree_method='hist',
                            objective='reg:squarederror', random_state=20260922 + int(axis), n_jobs=2))
                    assert parameter_record(model) == parameter_record(reference)
                    for net in [model, reference]:
                        net.fit(x, data.values[train, axis], sample_weight=weight); deterministic_inference(net)
                    loaded = pickle.loads(pickle.dumps(model, protocol=5))
                    for features in [q, name]:
                        raw = predict_raw(model, features, data.scale[axis])
                        expected = data.scale[axis] * np.expm1(np.maximum(reference.predict(features).astype(float), 0))
                        np.testing.assert_array_equal(raw, expected)
                        np.testing.assert_array_equal(predict_raw(model, features[::-1], data.scale[axis])[::-1], raw)
                        np.testing.assert_array_equal(predict_raw(model, features[:7], data.scale[axis]), raw[:7])
                        np.testing.assert_array_equal(predict_raw(loaded, features, data.scale[axis]), raw)
                    records.append({'kind': kind, 'dimensions': dims, 'axis_index': int(axis),
                        'diagnostic_train_profiles': len(train), 'diagnostic_trees': 24,
                        'train_rows_sha256': fingerprint_array(train), 'features_sha256': fingerprint_array(x),
                        'parameters': parameter_record(model), 'historical_recipe_serial_prediction_exact': True,
                        'all_query_orders_and_pickle_reload_exact': True})
        receipt.update(status='complete', records=records, both632feature_views=True,
            hidden_family_values_and_presence_do_not_change_features=True,
            name_only_features_ignore_all_numeric_values=True,
            scope='Training-only numerical preflight: three axes, at most256 training rows and24 trees. No validation metrics, candidate selection or formal small-data baseline claim.')
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / 'verification.json', receipt)
        raise
    write_json(args.output_dir / 'verification.json', receipt)
    print(receipt['status'], len(records), 'training-only tree fixtures')


if __name__ == '__main__':
    main()

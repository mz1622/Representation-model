"""Registered full-data R8 trees; three tasks share each fitted per-axis model."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_completion_input import completion_view, require_r7_completion, execution_contract
from foodcomp.research_tree_prediction import make_tree, deterministic_inference, predict_raw, parameter_record
from foodcomp.research_alignment import SCORING, CORRECT
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--seed', type=int, choices=[20260922, 20260923, 20260924], default=20260922)
    parser.add_argument('--n-jobs', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(ROOT / 'experiments/foodnutrigpt_v9_research/r8/config.json')
    selected = [c for c in protocol['candidates'] if c['name'] == args.candidate and c.get('kind') in {'rf', 'xgb'}]
    if len(selected) != 1:
        raise ValueError('Select exactly one preregistered tree configuration.')
    config = selected[0]
    require_r7_completion(ROOT)
    frozen, frozen_path = execution_contract(ROOT)
    if selected != [c for c in frozen['registered_candidates'] if c['name'] == args.candidate]:
        raise ValueError('Registered hyperparameters changed after execution freeze.')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic(); torch.set_num_threads(4)
    snapshot = args.output_dir / 'code_snapshot'; snapshot.mkdir()
    for path in frozen['code_hashes']:
        shutil.copyfile(ROOT / path, snapshot / Path(path).name)
    manifest = {'status': 'running', 'version': 'V9-R8', 'kind': config['kind'], 'configuration': config,
        'args': vars(args), 'seed': args.seed, 'code_commit': frozen['code_commit'],
        'workspace_commit_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_hashes': frozen['code_hashes'], 'execution_contract_sha256': digest(frozen_path),
        'environment': frozen['environment'], 'training_row_cap': None, 'fit_n_jobs': args.n_jobs,
        'rf_prediction_n_jobs': 1, 'complete_test_opened': False, 'scientific_confirmation': False,
        'task_contract': 'Same per-axis fitted model: family-hidden completion, all-numeric-hidden name-only, then predicted name profiles for retrieval. No candidate measured nutrition.',
        'model_storage': 'Per-axis in-memory pickle replay; discard fitted trees after all predictions. Saved candidate matrices cannot predict arbitrary new names.'}
    write_json(args.output_dir / 'run_manifest.json', manifest)
    try:
        data = ResearchData(ROOT / 'data/processed' / VERSION)
        text, cache, panel_root = completion_view(data, ROOT, config['active_dimensions'])
        assert text.shape[1] == 128 and len(data.axes) == 252
        names, first, inverse = np.unique(data.profiles.original_name.astype(str).to_numpy(), return_index=True, return_inverse=True)
        np.testing.assert_array_equal(text, text[first][inverse])
        candidates = np.concatenate([text[first], np.zeros((len(first), 504), np.float32)], axis=1)
        assert candidates.shape == (49913, 632) and not candidates[:, 128:].any()
        axes = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
        columns = {int(axis): col for col, axis in enumerate(axes)}
        candidate_raw = np.empty((len(names), len(axes)), dtype=np.float64)
        probe = np.random.default_rng(20260922).choice(len(names), 256, replace=False)
        write_json(args.output_dir / 'candidate_names.json', names.tolist())
        assert digest(args.output_dir / 'candidate_names.json') == 'e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
        manifest.update(data_hash=digest(data.root / 'manifest.json'), panel_hash=digest(panel_root / 'manifest.json'),
            panel_root=str(panel_root), name_cache=str(cache), name_cache_hash=digest(cache / 'manifest.json'),
            input_slots=632, active_name_dimensions=config['active_dimensions'],
            candidate_features_sha256=fingerprint_array(candidates), candidate_features_have_no_numeric_context=True)
        receipts = []; predictions = {'completion': [], 'name_only': []}
        with threadpool_limits(limits=args.n_jobs):
            for count, axis in enumerate(data.targets, 1):
                axis_start = time.monotonic()
                train = data.train[data.observed[data.train, axis]]
                rows = data.jobs.loc[data.jobs.axis_index.eq(axis), 'profile_index'].to_numpy()
                x_train = data.dense_features(train, text, axis)
                visible = data.observed[train] & (data.families != data.families[axis])[None, :]
                expected = np.concatenate([text[train], np.where(visible, data.values[train], 0), visible.astype(np.float32)], axis=1)
                np.testing.assert_array_equal(x_train, expected)
                weights = data.weights[train, axis]
                sample_weight = weights / weights.mean()
                model = make_tree(config, seed=args.seed, axis=axis, n_jobs=args.n_jobs)
                fit_params = parameter_record(model)
                model.fit(x_train, data.values[train, axis], sample_weight=sample_weight)
                fit_seconds = time.monotonic() - axis_start
                deterministic_inference(model)
                xs = {'completion': data.dense_features(rows, text, axis), 'name_only': candidates[inverse[rows]]}
                values = {}
                for mode, x in xs.items():
                    values[mode] = predict_raw(model, x, data.scale[axis])
                    np.testing.assert_array_equal(predict_raw(model, x[::-1], data.scale[axis])[::-1], values[mode])
                    np.testing.assert_array_equal(predict_raw(model, x[:32], data.scale[axis]), values[mode][:32])
                    predictions[mode].append(pd.DataFrame({'profile_index': rows, 'axis_index': int(axis), 'prediction': values[mode]}))
                blob = pickle.dumps(model, protocol=5)
                restored = pickle.loads(blob)
                for mode, x in xs.items():
                    np.testing.assert_array_equal(predict_raw(restored, x, data.scale[axis]), values[mode])
                receipt = {'axis_index': int(axis), 'train_profiles': len(train), 'validation_jobs': len(rows),
                    'train_rows_sha256': fingerprint_array(train), 'training_features_sha256': fingerprint_array(x_train),
                    'training_values_sha256': fingerprint_array(data.values[train, axis]),
                    'training_weights_sha256': fingerprint_array(weights), 'fitting_weights_sha256': fingerprint_array(sample_weight),
                    'tree_family_features_exact': True, 'fit_parameters': fit_params,
                    'reverse_and_subset_both_tasks_exact': True, 'save_reload_both_full_validation_tasks_exact': True,
                    'serialized_model_sha256': hashlib.sha256(blob).hexdigest(), 'serialized_model_bytes': len(blob),
                    'fit_seconds': fit_seconds}
                if int(axis) in columns:
                    raw = predict_raw(model, candidates, data.scale[axis])
                    np.testing.assert_array_equal(raw[inverse[rows]], values['name_only'])
                    np.testing.assert_array_equal(predict_raw(model, candidates[probe], data.scale[axis]), raw[probe])
                    np.testing.assert_array_equal(predict_raw(restored, candidates[probe], data.scale[axis]), raw[probe])
                    candidate_raw[:, columns[int(axis)]] = raw
                    receipt.update(candidate_predictions_sha256=fingerprint_array(raw),
                        candidate_all_validation_names_exact=True, candidate_subbatch_reload256_exact=True)
                receipt['fit_and_all_predictions_seconds'] = time.monotonic() - axis_start
                receipts.append(receipt)
                write_json(args.output_dir / 'axis_fitting_manifest.json', receipts)
                manifest.update(phase='axis_fit_and_three_task_predictions', axes_completed=count, elapsed_seconds=time.monotonic()-started)
                write_json(args.output_dir / 'run_manifest.json', manifest)
                if count % 10 == 0 or count == len(data.targets):
                    print(f"{args.candidate} axis{count}/187; {time.monotonic()-started:.1f}s", flush=True)
                del model, restored, blob, x_train, expected, xs
        scores = {}
        for mode, pieces in predictions.items():
            frame = pd.concat(pieces, ignore_index=True)
            score, by_axis, cells = score_predictions(data, frame); scores[mode] = score
            frame.to_parquet(args.output_dir / f'{mode}_predictions.parquet', index=False)
            by_axis.to_csv(args.output_dir / f'{mode}_axis_metrics.csv', index=False)
            cells.to_parquet(args.output_dir / f'{mode}_candidate_errors.parquet', index=False)
        write_json(args.output_dir / 'metrics.json', scores)
        np.save(args.output_dir / 'candidate_predicted_raw.npy', candidate_raw)
        scaled = np.log1p(candidate_raw / data.scale[axes]).astype(np.float32)
        if not np.isfinite(scaled).all(): raise FloatingPointError('Nonfinite candidate matrix.')
        np.save(args.output_dir / 'candidate_scaled.npy', scaled)
        manifest.update(phase='retrieval', nutrition_evaluation_complete=True, elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / 'run_manifest.json', manifest)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        retrieval_dir = args.output_dir / 'retrieval'; retrieval_dir.mkdir()
        shutil.copyfile(args.output_dir / 'candidate_names.json', retrieval_dir / 'candidate_names.json')
        ranks, metrics = evaluate_candidates(data, names.tolist(), scaled, axes, device)
        ranks.to_parquet(retrieval_dir / 'ranks.parquet', index=False)
        replay, other = evaluate_candidates(data, names.tolist(), np.load(args.output_dir / 'candidate_scaled.npy'), axes, device)
        pd.testing.assert_frame_equal(ranks, replay, check_exact=True); assert metrics == other
        write_json(retrieval_dir / 'metrics.json', {'metrics': metrics, 'candidate_count': len(names),
            'candidate_sha256': digest(args.output_dir / 'candidate_names.json'), 'query_profiles': ranks.groupby('visible_fraction').size().to_dict(),
            'data_sha256': manifest['data_hash'], 'name_cache_sha256': manifest['name_cache_hash'],
            'scoring': SCORING, 'correct_answers': CORRECT, 'method': args.candidate + '_same_fitted_completion_models',
            'complete_test_opened': False, 'scientific_claim_allowed': False})
        manifest.update(status='complete', phase='complete', elapsed_seconds=time.monotonic()-started,
            retrieval_directory='retrieval',
            candidate_vectors_sha256=digest(args.output_dir / 'candidate_scaled.npy'),
            all187_both_tasks_reverse_subset_reload_exact=True,
            all142_candidate_name_validation_subbatch_reload_exact=True, all19089_ranks_reload_exact=True)
        write_json(args.output_dir / 'run_manifest.json', manifest)
        print('Completion', scores['completion']['nutrition'], 'Name-only', scores['name_only']['nutrition'], flush=True)
        print('Retrieval', metrics, flush=True)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / 'run_manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()

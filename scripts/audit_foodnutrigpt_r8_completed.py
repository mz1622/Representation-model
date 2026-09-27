"""R8 complete-run audit: full task receipts, scores, inputs and retrieval replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_r1 import FamilyPanel, fingerprint_array
from foodcomp.research_completion_input import completion_view, execution_contract
from foodcomp.research_neural import evaluate
from foodcomp.research_inference import NutritionModel
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def check_identity(run, manifest, frozen, contract_path, data):
    assert manifest['status'] == 'complete'
    assert not manifest.get('test_opened', manifest.get('complete_test_opened', True))
    assert manifest['code_commit'] == frozen['code_commit']
    hashes = {path.replace('\\', '/'): value for path, value in manifest['code_hashes'].items()}
    assert hashes == frozen['code_hashes']
    assert manifest['execution_contract_sha256'] == digest(contract_path)
    assert manifest['data_hash'] == digest(data.root / 'manifest.json')
    for path, expected in hashes.items():
        assert digest(run / 'code_snapshot' / Path(path).name) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['mlp_pair', 'tree'], required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.kind == 'tree' and args.run is None: raise ValueError('Tree audit requires a completed run.')
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True); torch.set_num_threads(4)
    receipt = {'status': 'incomplete', 'kind': args.kind, 'complete_test_opened': False, 'script_sha256': digest(Path(__file__))}
    try:
        frozen, contract_path = execution_contract(ROOT)
        data = ResearchData(ROOT / 'data/processed' / VERSION)
        records, manifests, histories = [], [], []
        runs = [ROOT / f'output/v9_r8/mlp512_name{dim}' for dim in [32, 128]] if args.kind == 'mlp_pair' else [args.run.resolve()]
        for run in runs:
            m = read(run / 'run_manifest.json'); check_identity(run, m, frozen, contract_path, data)
            dims = m['active_name_dimensions']
            text, cache, panel_root = completion_view(data, ROOT, dims)
            assert m['name_cache_hash'] == digest(cache / 'manifest.json') and m['panel_hash'] == digest(panel_root / 'manifest.json')
            scores = read(run / 'metrics.json')
            frames = {task: pd.read_parquet(run / f'{task}_predictions.parquet') for task in ['completion', 'name_only']}
            for task, frame in frames.items(): assert score_predictions(data, frame)[0] == scores[task]
            record = {'run': str(run.relative_to(ROOT)), 'manifest_sha256': digest(run / 'run_manifest.json'),
                'both_prediction_scores_recomputed_exact': True, 'active_name_dimensions': dims}
            if args.kind == 'mlp_pair':
                assert m['epoch_completed'] == 60 and m['parameter_count'] == 717052
                assert digest(run / 'best_model.pt') == m['checkpoint_hash']
                wrapper = NutritionModel(run / 'best_model.pt')
                np.testing.assert_array_equal(wrapper._cached_text, text)
                panel = FamilyPanel(data, text, panel_root, 'cpu')
                targets = data.observed[panel.rows] & panel.masks.numpy()[panel.family_ids] & panel.eligible.numpy()
                assert int(targets.sum()) == 1828536
                h = pd.read_csv(run / 'history.csv', float_precision='round_trip')
                np.testing.assert_array_equal(h.epoch, np.arange(1, 61))
                assert np.isfinite(h.select_dtypes('number')).all().all()
                parameter = torch.nn.Parameter(torch.zeros(1))
                opt = torch.optim.AdamW([parameter], lr=.001, weight_decay=.0001)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=.00001)
                for row in h.itertuples():
                    assert row.training_tasks == 337048 and row.observed_target_cells == 1828536
                    assert row.training_order_sha256 == fingerprint_array(np.random.default_rng(m['seed'] + row.epoch).permutation(len(panel.rows)))
                    assert row.learning_rate == opt.param_groups[0]['lr']; opt.step(); schedule.step()
                assert m['best_epoch'] == int(h.loc[h.validation_primary.idxmin(), 'epoch'])
                for task in frames:
                    replay = evaluate(wrapper.model, data, text, wrapper.device, task)
                    pd.testing.assert_frame_equal(replay, frames[task], check_exact=True)
                retrieval = ROOT / f'output/v9_r8/retrieval_mlp512_name{dims}'
                replay_dir = args.output_dir / f'retrieval_replay_name{dims}'
                subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py'),
                    '--checkpoint', str(run / 'best_model.pt'), '--output-dir', str(replay_dir.resolve())],
                    cwd=ROOT, check=True, capture_output=True, text=True)
                pd.testing.assert_frame_equal(pd.read_parquet(replay_dir / 'ranks.parquet'),
                    pd.read_parquet(retrieval / 'ranks.parquet'), check_exact=True)
                for key in ['metrics','candidate_sha256','checkpoint_sha256','data_sha256','scoring','correct_answers']:
                    assert read(replay_dir / 'metrics.json')[key] == read(retrieval / 'metrics.json')[key]
                record.update(all60_orders_exposures_lr_exact=True, both323809_model_predictions_reloaded_exact=True,
                    all19089_model_retrieval_ranks_reloaded_exact=True, checkpoint_sha256=m['checkpoint_hash'])
                histories.append(h)
            else:
                fits = read(run / 'axis_fitting_manifest.json')
                assert [f['axis_index'] for f in fits] == data.targets.tolist()
                for f in fits:
                    axis = f['axis_index']; train = data.train[data.observed[data.train, axis]]
                    weights = data.weights[train, axis]
                    assert f['train_rows_sha256'] == fingerprint_array(train)
                    assert f['training_features_sha256'] == fingerprint_array(data.dense_features(train, text, axis))
                    assert f['training_values_sha256'] == fingerprint_array(data.values[train, axis])
                    assert f['training_weights_sha256'] == fingerprint_array(weights)
                    assert f['fitting_weights_sha256'] == fingerprint_array(weights / weights.mean())
                    assert f['reverse_and_subset_both_tasks_exact'] and f['save_reload_both_full_validation_tasks_exact']
                    if data.axes.loss_group.iloc[axis] == 'nutrition':
                        assert f['candidate_all_validation_names_exact'] and f['candidate_subbatch_reload256_exact']
                axes = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
                raw = np.load(run / 'candidate_predicted_raw.npy'); scaled = np.load(run / 'candidate_scaled.npy')
                assert raw.shape == (49913, 142) and np.isfinite(raw).all() and (raw >= 0).all()
                np.testing.assert_array_equal(scaled, np.log1p(raw / data.scale[axes]).astype(np.float32))
                assert digest(run / 'candidate_scaled.npy') == m['candidate_vectors_sha256']
                names = read(run / 'candidate_names.json'); lookup = {name: i for i, name in enumerate(names)}
                columns = {int(axis): i for i, axis in enumerate(axes)}
                prediction = frames['name_only']; prediction = prediction[prediction.axis_index.isin(axes)]
                rows = [lookup[str(data.profiles.original_name.iloc[r])] for r in prediction.profile_index]
                cols = [columns[int(a)] for a in prediction.axis_index]
                np.testing.assert_array_equal(raw[rows, cols], prediction.prediction.to_numpy())
                ranks, rs = evaluate_candidates(data, names, scaled, axes, 'cuda' if torch.cuda.is_available() else 'cpu')
                retrieval = run / 'retrieval'
                pd.testing.assert_frame_equal(ranks, pd.read_parquet(retrieval / 'ranks.parquet'), check_exact=True)
                assert rs == read(retrieval / 'metrics.json')['metrics']
                assert digest(retrieval / 'candidate_names.json') == digest(run / 'candidate_names.json')
                record.update(all187_fit_rows_features_labels_weights_verified=True,
                    all317616_name_predictions_match_candidate_names_exact=True, all19089_saved_matrix_ranks_replayed_exact=True,
                    model_replay_scope='Runtime in-memory pickle checks were required for every axis. Full forests are discarded; this audit verifies saved data/receipts/matrix, not a second full refit.')
            records.append(record); manifests.append(m)
        if args.kind == 'mlp_pair':
            for field in ['code_commit','code_hashes','execution_contract_sha256','data_hash','initial_state_sha256','parameter_count','seed']:
                assert manifests[0][field] == manifests[1][field], field
            for field in ['epoch','training_tasks','observed_target_cells','training_order_sha256','learning_rate']:
                np.testing.assert_array_equal(histories[0][field], histories[1][field])
        receipt.update(status='complete', records=records, execution_contract_sha256=digest(contract_path),
            scope='Fixed R8 input/configuration and complete saved-output verification. Does not establish superiority, seed stability, label validity or external transfer.')
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / 'verification.json', receipt)
        raise
    write_json(args.output_dir / 'verification.json', receipt)
    print(receipt['status'], args.kind, [r['run'] for r in records])


if __name__ == '__main__':
    main()

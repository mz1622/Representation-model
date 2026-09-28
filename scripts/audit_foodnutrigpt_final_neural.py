"""Audit all three fixed neural seeds against actual R8 inputs, trajectories and predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_foodnutrigpt_r8_completed import check_identity
from foodcomp.research_completion_input import completion_view, execution_contract
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_r1 import FamilyPanel, fingerprint_array
from foodcomp.research_neural import evaluate
from foodcomp.research_inference import NutritionModel
from run_foodnutrigpt_final_neural_repeats import recipe


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replication-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    queue_path = args.replication_dir / 'queue_manifest.json'
    queue = read(queue_path)
    if queue['status'] != 'complete' or [r['seed'] for r in queue['runs']] != [20260923, 20260924]:
        raise ValueError('Both new fixed-seed runs and retrieval must be complete.')
    plan = ROOT / 'experiments/foodnutrigpt_v9_research/final_report_v1/config.json'
    if digest(plan) != queue['plan_sha256']:
        raise ValueError('Replication plan changed.')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    receipt = {'status': 'running', 'complete_test_opened': False,
        'scope': 'Three neural seeds, full trajectories and independent three-task replay; no tree or final-report closure.',
        'script_sha256': digest(Path(__file__)), 'plan_sha256': digest(plan),
        'queue_manifest_sha256': digest(queue_path), 'records': []}
    receipt_path = args.output_dir / 'verification.json'
    write_json(receipt_path, receipt)
    try:
        frozen, contract_path = execution_contract(ROOT)
        torch.set_num_threads(4)
        data = ResearchData(ROOT / 'data/processed' / VERSION)
        text, cache, panel_root = completion_view(data, ROOT, 128)
        panel = FamilyPanel(data, text, panel_root, 'cpu')
        targets = data.observed[panel.rows] & panel.masks.numpy()[panel.family_ids] & panel.eligible.numpy()
        assert int(targets.sum()) == 1828536 and len(panel.rows) == 337048
        first = ROOT / 'output/v9_r8/mlp512_name128'
        parent = read(first / 'run_manifest.json')
        assert digest(first / 'run_manifest.json') == queue['seed22_manifest_sha256']
        fixed = recipe(parent)
        initial_states = []
        for seed in [20260922, 20260923, 20260924]:
            run = first if seed == 20260922 else args.replication_dir / f'mlp_seed{seed}'
            retrieval = ROOT / 'output/v9_r8/retrieval_mlp512_name128' if seed == 20260922 else run / 'retrieval'
            manifest = read(run / 'run_manifest.json')
            check_identity(run.resolve(), manifest, frozen, contract_path, data)
            assert manifest['seed'] == seed and recipe(manifest) == fixed
            assert manifest['active_name_dimensions'] == 128 and manifest['input_slots'] == 632
            assert manifest['epoch_completed'] == 60 and manifest['parameter_count'] == 717052
            assert manifest['name_cache_hash'] == digest(cache / 'manifest.json')
            assert manifest['panel_hash'] == digest(panel_root / 'manifest.json')
            assert digest(run / 'best_model.pt') == manifest['checkpoint_hash']
            if seed != 20260922:
                queued = [item for item in queue['runs'] if item['seed'] == seed]
                assert len(queued) == 1 and queued[0]['manifest_sha256'] == digest(run / 'run_manifest.json')
                assert queued[0]['checkpoint_sha256'] == manifest['checkpoint_hash']
                assert queued[0]['retrieval_metrics_sha256'] == digest(retrieval / 'metrics.json')
            history = pd.read_csv(run / 'history.csv', float_precision='round_trip')
            np.testing.assert_array_equal(history.epoch, np.arange(1, 61))
            assert np.isfinite(history.select_dtypes('number')).all().all()
            opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=.00001)
            for row in history.itertuples():
                assert row.training_tasks == 337048 and row.observed_target_cells == 1828536
                assert row.training_order_sha256 == fingerprint_array(np.random.default_rng(seed + row.epoch).permutation(len(panel.rows)))
                assert row.learning_rate == opt.param_groups[0]['lr']
                opt.step()
                scheduler.step()
            assert manifest['best_epoch'] == int(history.loc[history.validation_primary.idxmin(), 'epoch'])
            wrapper = NutritionModel(run / 'best_model.pt')
            np.testing.assert_array_equal(wrapper._cached_text, text)
            metrics = read(run / 'metrics.json')
            prediction_hashes = {}
            for task in ['completion', 'name_only']:
                saved = pd.read_parquet(run / f'{task}_predictions.parquet')
                assert len(saved) == 323809
                scores, by_axis, groups = score_predictions(data, saved)
                assert scores == metrics[task]
                replay = evaluate(wrapper.model, data, text, wrapper.device, task)
                pd.testing.assert_frame_equal(replay, saved, check_exact=True)
                by_axis.to_csv(args.output_dir / f'seed{seed}_{task}_axis_metrics.csv', index=False)
                prediction_hashes[task] = digest(run / f'{task}_predictions.parquet')
            replay_dir = args.output_dir / f'retrieval_replay_seed{seed}'
            command = [sys.executable, 'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py',
                '--checkpoint', str(run / 'best_model.pt'), '--output-dir', str(replay_dir)]
            with (args.output_dir / f'retrieval_replay_seed{seed}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            ranks = pd.read_parquet(retrieval / 'ranks.parquet')
            assert len(ranks) == 19089
            pd.testing.assert_frame_equal(pd.read_parquet(replay_dir / 'ranks.parquet'), ranks, check_exact=True)
            original_metrics = read(retrieval / 'metrics.json')
            replay_metrics = read(replay_dir / 'metrics.json')
            for key in ['metrics', 'candidate_sha256', 'candidate_count', 'checkpoint_sha256',
                        'data_sha256', 'scoring', 'correct_answers', 'complete_test_opened']:
                assert replay_metrics[key] == original_metrics[key], key
            assert not original_metrics['complete_test_opened'] and original_metrics['candidate_count'] == 49913
            assert original_metrics['candidate_sha256'] == 'e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
            receipt['records'].append({'seed': seed, 'run': str(run), 'retrieval_directory': str(retrieval),
                'manifest_sha256': digest(run / 'run_manifest.json'), 'checkpoint_sha256': manifest['checkpoint_hash'],
                'prediction_sha256': prediction_hashes, 'metrics_sha256': digest(run / 'metrics.json'),
                'retrieval_metrics_sha256': digest(retrieval / 'metrics.json'),
                'retrieval_ranks_sha256': digest(retrieval / 'ranks.parquet'),
                'best_epoch': manifest['best_epoch'], 'initial_state_sha256': manifest['initial_state_sha256'],
                'all60_orders_exposures_lr_exact': True, 'both323809_predictions_reloaded_exact': True,
                'all19089_retrieval_ranks_reloaded_exact': True, 'metrics': metrics,
                'training_seconds': manifest['elapsed_seconds']})
            initial_states.append(manifest['initial_state_sha256'])
            write_json(receipt_path, receipt)
            del wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        assert len(set(initial_states)) == 3
        receipt.update(status='complete', elapsed_seconds=time.monotonic() - started,
            all_three_fixed_recipe_seeds_audited=True, distinct_seed_initial_states=True)
        write_json(receipt_path, receipt)
        print(json.dumps({'status': 'complete', 'seeds': [r['seed'] for r in receipt['records']]}))
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started)
        write_json(receipt_path, receipt)
        raise


if __name__ == '__main__':
    main()


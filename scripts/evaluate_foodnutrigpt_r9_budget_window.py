"""Evaluate the preregistered 20-epoch selection window after full R9 training."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_budget_window import validate_budget_snapshot
from foodcomp.research_transformer_r9 import frozen_inputs, TransformerNutritionModel
from foodcomp.research_neural import evaluate
from foodcomp.research_r0 import digest, score_predictions, write_json
from foodcomp.research_alignment import SCORING, CORRECT
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Preserve earlier evaluation attempts.')
    frozen, freeze_path = frozen_inputs(ROOT)
    parent_path = args.run / 'run_manifest.json'
    parent = json.loads(parent_path.read_text())
    if parent['status'] != 'complete':
        raise ValueError('Wait for the full registered parent run to complete before budget evaluation.')
    audit_path = args.audit_dir / 'verification.json'
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'complete' or audit['complete_test_opened']
            or audit['manifest_sha256'] != digest(parent_path)
            or audit['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Independent completed-parent audit is required.')
    for name, expected in parent['code_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Parent executable changed: ' + name)
    history_path = args.run / 'history.csv'
    checkpoint = args.run / 'best_through_epoch_020.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    window = validate_budget_snapshot(pd.read_csv(history_path, float_precision='round_trip'), parent, saved, 20)
    final = torch.load(args.run / 'best_through_epoch_060.pt', map_location='cpu', weights_only=True)
    validate_budget_snapshot(pd.read_csv(history_path, float_precision='round_trip'), parent, final, 60)
    if digest(args.run / 'best_through_epoch_060.pt') != parent['checkpoint_sha256']:
        raise ValueError('Final budget snapshot differs from the audited selected checkpoint.')
    del saved, final
    torch.set_num_threads(4)
    model = TransformerNutritionModel(checkpoint, ROOT)
    data = model.data
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    inputs = {str(p): digest(p) for p in [parent_path, audit_path, history_path, checkpoint, freeze_path]}
    manifest = {'status': 'running', 'version': 'V9-R9-budget-window', 'training_performed': False,
        'parent_run': str(args.run), 'checkpoint': str(checkpoint), 'window': window,
        'input_hashes': inputs, 'complete_test_opened': False, 'data_modified': False, 'baseline_refit': False,
        'script_sha256': digest(Path(__file__)), 'helper_sha256': digest(ROOT / 'src/foodcomp/research_budget_window.py')}
    write_json(args.output_dir / 'manifest.json', manifest)
    try:
        scores = {}
        for task in ['completion', 'name_only']:
            predictions = evaluate(model.model, data, model._cached_text, model.device, task)
            score, axes, groups = score_predictions(data, predictions)
            if task == 'completion' and score['nutrition']['scaled_log_mae'] != window['validation_primary']:
                raise ValueError('The window checkpoint does not reproduce its recorded primary metric.')
            predictions.to_parquet(args.output_dir / f'{task}_predictions.parquet', index=False)
            axes.to_csv(args.output_dir / f'{task}_axis_metrics.csv', index=False)
            groups.to_parquet(args.output_dir / f'{task}_candidate_errors.parquet', index=False)
            scores[task] = score
        write_json(args.output_dir / 'metrics.json', scores)
        names = sorted(set(data.profiles.original_name.astype(str)))
        axes = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
        raw = model.candidate_profiles(names, target_axes=axes)
        scaled = np.log1p(raw / data.scale[axes]).astype(np.float32)
        np.save(args.output_dir / 'candidate_scaled.npy', scaled)
        ret = args.output_dir / 'retrieval'
        ret.mkdir()
        write_json(ret / 'candidate_names.json', names)
        ranks, metrics = evaluate_candidates(data, names, scaled, axes, str(model.device))
        ranks.to_parquet(ret / 'ranks.parquet', index=False)
        replay, other = evaluate_candidates(data, names, np.load(args.output_dir / 'candidate_scaled.npy'), axes, str(model.device))
        pd.testing.assert_frame_equal(ranks, replay, check_exact=True)
        if metrics != other:
            raise ValueError('Saved candidate matrix cannot replay retrieval.')
        write_json(ret / 'metrics.json', {'metrics': metrics, 'candidate_count': len(names),
            'query_profiles': ranks.groupby('visible_fraction').size().to_dict(),
            'candidate_sha256': digest(ret / 'candidate_names.json'), 'checkpoint_sha256': digest(checkpoint),
            'data_sha256': frozen['data_hash'], 'name_cache_sha256': frozen['name_cache_hash'],
            'scoring': SCORING, 'correct_answers': CORRECT, 'method': 'R9_transformer_best_through_epoch20',
            'complete_test_opened': False})
        for path, expected in inputs.items():
            if digest(Path(path)) != expected:
                raise ValueError('Evaluation input changed: ' + path)
        frozen_inputs(ROOT)
        manifest.update(status='complete', elapsed_seconds=time.monotonic() - started,
            primary_history_score_reproduced_exact=True,
            retrieval_saved_matrix_replayed_exact=True,
            prediction_sha256={task: digest(args.output_dir / f'{task}_predictions.parquet') for task in scores})
        write_json(args.output_dir / 'manifest.json', manifest)
        print(json.dumps({'status': 'complete', 'window': window, 'scores': scores}))
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()

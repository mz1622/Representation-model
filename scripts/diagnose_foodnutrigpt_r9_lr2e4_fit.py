"""After completed learning-rate analysis, compare source-free training fit without optimization."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_neural import evaluate
from foodcomp.research_r0 import digest, score_predictions, write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_transformer_r9 import frozen_inputs
from foodcomp.research_transformer_r9 import TransformerNutritionModel
from foodcomp.research_transformer_r9_lr2e4 import verify_bindings

ANALYSIS = ROOT / 'reports/v9_r9_lr2e4_analysis_v1/summary.json'
PARENT_FIT = ROOT / 'reports/v9_r9_first_group_fit_v1/summary.json'
RUN = ROOT / 'output/v9_r9_methods/tf192_mae_lr2e4_60'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    required = {ANALYSIS: 'complete_registered_lr2e4_analysis', PARENT_FIT: 'complete',
                RUN / 'run_manifest.json': 'complete'}
    pending = [str(path.relative_to(ROOT)) for path, state in required.items()
               if not path.exists() or read(path).get('status') != state]
    if args.check_only:
        print(json.dumps({'ready': not pending, 'pending': pending, 'training_performed': False,
                          'output_written': False, 'model_forward_performed': False}))
        return
    if pending:
        raise ValueError('Wait for complete learning-rate training, audit, comparisons and analysis')
    if args.output_dir is None or args.local_dir is None:
        parser.error('Both --output-dir and --local-dir are required')
    for path in [args.output_dir, args.local_dir]:
        if path.exists():
            raise FileExistsError(path)
    if not args.local_dir.resolve().is_relative_to((ROOT / 'data/local/research_diagnostics').resolve()):
        raise ValueError('Numeric predictions must stay in ignored local diagnostics')
    analysis, old = read(ANALYSIS), read(PARENT_FIT)
    verify_bindings(ROOT, analysis['input_hashes'])
    verify_bindings(ROOT, old['input_hashes'])
    frozen, freeze_path = frozen_inputs(ROOT)
    parent = [r for r in old['models'] if r['run'] == 'tf192_mae_lr3e4_60']
    if len(parent) != 1:
        raise ValueError('Unique existing MAE training-fit control required')
    parent = parent[0]
    parent_path = ROOT / Path(parent['prediction_path'].replace('\\', '/'))
    if (not parent_path.resolve().is_relative_to((ROOT / 'data/local/research_diagnostics').resolve())
            or digest(parent_path) != parent['prediction_sha256']
            or parent['checkpoint_sha256'] != analysis['records']['mae']['manifest']['checkpoint_sha256']):
        raise ValueError('Existing MAE training predictions are not the declared control')
    manifest = read(RUN / 'run_manifest.json')
    if manifest != analysis['records']['lr2e4']['manifest']:
        raise ValueError('learning-rate analysis no longer matches the completed model')
    if any(analysis[k] or old[k] for k in ['training_performed', 'data_modified', 'baseline_refit', 'complete_test_opened']):
        raise ValueError('Prior diagnostic scope differs')
    checkpoint = RUN / 'best_model.pt'
    if digest(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('learning-rate checkpoint changed')
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in
              [ANALYSIS, PARENT_FIT, parent_path, checkpoint, RUN / 'run_manifest.json', freeze_path]}
    torch.set_num_threads(3)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True)
    args.local_dir.mkdir(parents=True)
    status = {'status': 'running', 'training_performed': False, 'optimization_performed': False,
              'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
              'script_sha256': digest(Path(__file__))}
    write_json(args.output_dir / 'status.json', status)
    try:
        model = TransformerNutritionModel(checkpoint, ROOT)
        train = copy.copy(model.data)
        local_rows, local_axes = np.where(train.observed[train.train][:, train.targets])
        rows, axes = train.train[local_rows], train.targets[local_axes]
        if not train.profiles.iloc[rows].partition.eq('train').all():
            raise ValueError('Non-training row entered training-fit panel')
        train.jobs = pd.DataFrame({'profile_index': rows, 'axis_index': axes,
                                  'target': train.raw[rows, axes], 'mask_family': train.families[axes]})
        query_hash = fingerprint_array(np.stack([rows, axes], axis=1))
        if (len(train.jobs) != 1828536 or train.jobs.duplicated(['profile_index', 'axis_index']).any()
                or query_hash != parent['training_query_panel_sha256']):
            raise ValueError('Training-fit query panel differs from MAE control')
        old_predictions = pd.read_parquet(parent_path)
        old_metrics, old_axes, _ = score_predictions(train, old_predictions)
        if old_metrics != parent['training']:
            raise ValueError('Saved MAE training fit did not rescore exactly')
        del old_predictions
        predictions = evaluate(model.model, train, model._cached_text, model.device)
        metrics, axis_metrics, _ = score_predictions(train, predictions)
        local_predictions = args.local_dir / 'lr2e4_training_predictions.parquet'
        predictions.to_parquet(local_predictions, index=False)
        reloaded = pd.read_parquet(local_predictions)
        pd.testing.assert_frame_equal(predictions, reloaded, check_exact=True)
        del predictions, reloaded
        outputs = []
        for method, partition, frame in [
            ('mae', 'train', old_axes), ('lr2e4', 'train', axis_metrics),
            ('mae', 'validation', pd.read_csv(ROOT / analysis['records']['mae']['run'] / 'completion_axis_metrics.csv')),
            ('lr2e4', 'validation', pd.read_csv(RUN / 'completion_axis_metrics.csv'))]:
            outputs.append(frame.assign(method=method, partition=partition))
        pd.concat(outputs, ignore_index=True).to_csv(args.output_dir / 'per_axis_fit.csv', index=False)
        validations = {role: analysis['records'][role]['metrics']['completion'] for role in ['mae', 'lr2e4']}
        if validations['mae'] != parent['validation']:
            raise ValueError('MAE validation control differs from parent fit record')
        summary = {'status': 'complete', 'data_sha256': frozen['data_hash'],
            'source_free_training_fit': {'mae': old_metrics, 'lr2e4': metrics},
            'source_free_validation': validations,
            'training_primary_lr2e4_minus_mae': metrics['nutrition']['scaled_log_mae'] - old_metrics['nutrition']['scaled_log_mae'],
            'validation_primary_lr2e4_minus_mae': validations['lr2e4']['nutrition']['scaled_log_mae'] - validations['mae']['nutrition']['scaled_log_mae'],
            'training_job_count': len(train.jobs), 'training_query_panel_sha256': query_hash,
            'existing_mae_predictions_rescored_exact': True, 'new_lr2e4_predictions_disk_reloaded_exact': True,
            'prediction_path': str(local_predictions), 'prediction_sha256': digest(local_predictions),
            'input_hashes': hashes, 'script_sha256': digest(Path(__file__)),
            'device': str(model.device), 'elapsed_seconds': time.monotonic()-started,
            'training_performed': False, 'optimization_performed': False, 'data_modified': False,
            'baseline_refit': False, 'complete_test_opened': False, 'goal_achieved': False,
            'scope': 'Complete training-family source-free inference. Only an in-memory job table is replaced; data files and validation jobs remain unchanged. Reuses/rescores audited-parent training predictions without refitting. Train and validation have different foods/support, and checkpoints were selected on validation. Fit differences alone do not prove overfitting or uniquely separate capacity, objective, calibration or optimization mechanisms.'}
        verify_bindings(ROOT, hashes)
        frozen_inputs(ROOT)
        write_json(args.output_dir / 'summary.json', summary)
        status.update(status='complete', elapsed_seconds=time.monotonic()-started,
                      summary_sha256=digest(args.output_dir / 'summary.json'))
        write_json(args.output_dir / 'status.json', status)
        print(json.dumps({key: summary[key] for key in ['status', 'training_primary_lr2e4_minus_mae', 'validation_primary_lr2e4_minus_mae']}))
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

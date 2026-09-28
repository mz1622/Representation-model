"""Full training-family reconstruction of audited R9 checkpoints, with no refitting."""
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
from foodcomp.research_transformer_r9 import TransformerNutritionModel, frozen_inputs
from foodcomp.research_neural import evaluate
from foodcomp.research_r0 import digest, score_predictions, write_json
from foodcomp.research_r1 import fingerprint_array

CANDIDATES = ['tf192_mae_lr1e4_60', 'tf192_mae_lr3e4_60']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    analysis = ROOT / 'reports/v9_r9_first_group_v1/summary.json'
    ready = analysis.exists() and read(analysis).get('status') == 'first_registered_group_analysis_complete'
    if args.check_only:
        print(json.dumps({'ready': ready, 'required_analysis': str(analysis), 'output_written': False,
                          'optimization_performed': False}))
        return
    if not ready:
        raise ValueError('Wait for both completed runs, audits and first-group comparisons.')
    for path in [args.output_dir, args.local_dir]:
        if path.exists():
            raise FileExistsError(path)
    if not args.local_dir.resolve().is_relative_to((ROOT / 'data/local/research_diagnostics').resolve()):
        raise ValueError('Numeric predictions must remain in the ignored local diagnostics directory.')
    frozen, freeze_path = frozen_inputs(ROOT)
    group = read(analysis)
    if group['complete_test_opened'] or group['baseline_refit'] or group['data_modified']:
        raise ValueError('Unexpected parent analysis scope.')
    torch.set_num_threads(3)
    args.output_dir.mkdir(parents=True)
    args.local_dir.mkdir(parents=True)
    started = time.monotonic()
    status = {'status': 'running', 'training_performed': False, 'optimization_performed': False,
        'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
        'script_sha256': digest(Path(__file__)), 'parent_analysis_sha256': digest(analysis),
        'freeze_sha256': digest(freeze_path)}
    write_json(args.output_dir / 'status.json', status)
    try:
        records, per_axis, input_hashes = [], [], {str(analysis.relative_to(ROOT)): digest(analysis)}
        for name in CANDIDATES:
            run = ROOT / 'output/v9_r9' / name
            checkpoint = run / 'best_model.pt'
            manifest = read(run / 'run_manifest.json')
            audit_path = ROOT / 'reports' / f'v9_r9_{name}_audit_v1/verification.json'
            audit = read(audit_path)
            if (manifest['status'] != 'complete' or audit['status'] != 'complete'
                    or audit['manifest_sha256'] != digest(run / 'run_manifest.json')
                    or digest(checkpoint) != manifest['checkpoint_sha256']
                    or manifest != group['records'][name]['manifest']):
                raise ValueError('Audited checkpoint identity changed: ' + name)
            for path in [checkpoint, run / 'run_manifest.json', audit_path,
                         run / 'metrics.json', run / 'completion_axis_metrics.csv']:
                input_hashes[str(path.relative_to(ROOT))] = digest(path)
            model = TransformerNutritionModel(checkpoint, ROOT)
            # Only the in-memory evaluation jobs change. No research-view files,
            # validation jobs, fitted scales, cached names or training masks are rewritten.
            train = copy.copy(model.data)
            local_rows, local_axes = np.where(train.observed[train.train][:, train.targets])
            rows, axes = train.train[local_rows], train.targets[local_axes]
            if not train.profiles.iloc[rows].partition.eq('train').all():
                raise ValueError('A non-training profile entered the training-fit diagnostic.')
            train.jobs = pd.DataFrame({'profile_index': rows, 'axis_index': axes,
                'target': train.raw[rows, axes], 'mask_family': train.families[axes]})
            if len(train.jobs) != 1828536 or train.jobs.duplicated(['profile_index', 'axis_index']).any():
                raise ValueError('Unexpected training target support.')
            before = digest(checkpoint)
            run_started = time.monotonic()
            pred = evaluate(model.model, train, model._cached_text, model.device)
            training, train_axes, _ = score_predictions(train, pred)
            local = args.local_dir / name
            local.mkdir()
            pred.to_parquet(local / 'completion_predictions.parquet', index=False)
            train_axes.to_csv(args.output_dir / f'{name}_training_axis_metrics.csv', index=False)
            validation = read(run / 'metrics.json')['completion']
            validation_axes = pd.read_csv(run / 'completion_axis_metrics.csv')
            if validation != group['records'][name]['budget60']['metrics']['completion']:
                raise ValueError('Saved validation metrics changed.')
            selected = ['axis_index', 'scaled_log_mae', 'log_mae', 'positive_scaled_log_mae',
                        'zero_scaled_log_mae', 'candidate_support']
            axis = train_axes[selected].merge(validation_axes[selected], on='axis_index',
                suffixes=('_train', '_validation'), validate='one_to_one')
            axis = axis.merge(train.axes[['axis_index', 'canonical_name', 'loss_group']],
                              on='axis_index', validate='one_to_one')
            axis['run'] = name
            per_axis.append(axis)
            if digest(checkpoint) != before:
                raise ValueError('Checkpoint changed during inference.')
            record = {'run': name, 'kind': 'transformer_direct', 'checkpoint_sha256': before,
                'training': training, 'validation': validation, 'training_job_count': len(train.jobs),
                'training_query_panel_sha256': fingerprint_array(np.stack([rows, axes], axis=1)),
                'prediction_path': str(local / 'completion_predictions.parquet'),
                'prediction_sha256': digest(local / 'completion_predictions.parquet'),
                'data_sha256': frozen['data_hash'], 'elapsed_seconds': time.monotonic() - run_started}
            records.append(record)
            print(json.dumps({'run': name, 'train_primary': training['nutrition']['scaled_log_mae'],
                              'validation_primary': validation['nutrition']['scaled_log_mae']}), flush=True)
            del model, train, pred
        for path, expected in input_hashes.items():
            if digest(ROOT / path) != expected:
                raise ValueError('Diagnostic input changed: ' + path)
        frozen_inputs(ROOT)
        pd.concat(per_axis, ignore_index=True).to_csv(args.output_dir / 'per_axis_fit.csv', index=False)
        write_json(args.output_dir / 'summary.json', {'status': 'complete', 'models': records,
            'input_hashes': input_hashes, 'script_sha256': digest(Path(__file__)),
            'complete_test_opened': False, 'training_performed': False, 'optimization_performed': False,
            'data_modified': False, 'baseline_refit': False,
            'scope': 'Full training-family reconstruction in inference mode with the same source-free scorer. '
                     'Only a shallow in-memory evaluation view is created; frozen data and validation tasks are unchanged. '
                     'Validation metrics are reused from independently audited saved predictions. '
                     'Training/validation differ in foods and support; the gap alone does not prove overfitting, '
                     'identify a unique bottleneck, or establish that capacity cannot help. '
                     'No calibrated-source prediction is presented as an inference capability.'})
        status.update(status='complete', elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / 'status.json', status)
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

"""Compare the two completed registered R9 learning rates; do not start new training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json

CANDIDATES = ['tf192_mae_lr1e4_60', 'tf192_mae_lr3e4_60']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paths(name):
    return {'run': ROOT / 'output/v9_r9' / name,
            'audit': ROOT / 'reports' / f'v9_r9_{name}_audit_v1',
            'budget': ROOT / 'output/v9_r9_budget' / f'{name}_through20'}


def requirements():
    required = {}
    for name in CANDIDATES:
        p = paths(name)
        required[str(p['run'] / 'run_manifest.json')] = 'complete'
        required[str(p['audit'] / 'verification.json')] = 'complete'
        required[str(p['budget'] / 'manifest.json')] = 'complete'
        for ref, tasks in [('rf32', ['completion', 'name_only', 'retrieval']),
                           ('xgb32', ['completion', 'name_only', 'retrieval']),
                           ('knn32', ['name_only', 'retrieval'])]:
            for task in tasks:
                required[str(ROOT / 'reports' / f'v9_r9_{name}_vs_{ref}_{task}_v1/summary.json')] = None
        for task in ['completion', 'name_only', 'retrieval']:
            required[str(ROOT / 'reports' / f'v9_r9_{name}_budget20_vs60_{task}_v1/summary.json')] = None
    pending = []
    for item, status in required.items():
        path = Path(item)
        if not path.exists():
            pending.append({'path': str(path.relative_to(ROOT)), 'reason': 'absent'})
        elif status is not None and read(path).get('status') != status:
            pending.append({'path': str(path.relative_to(ROOT)), 'reason': read(path).get('status', 'missing_status')})
    return required, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    required, pending = requirements()
    if args.check_only:
        print(json.dumps({'ready': not pending, 'pending': pending,
                          'training_performed': False, 'output_written': False}))
        return
    if args.output_dir is None:
        parser.error('--output-dir is required unless --check-only is used.')
    if pending:
        raise ValueError('Complete the registered runs, audits, references and budget analyses first.')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    inputs = {str(Path(p).relative_to(ROOT)): digest(Path(p)) for p in required}
    freeze_path = ROOT / 'reports/v9_r9_freeze_v1/manifest.json'
    freeze = read(freeze_path)
    inputs[str(freeze_path.relative_to(ROOT))] = digest(freeze_path)
    manifests, records, histories = {}, {}, {}
    for name in CANDIDATES:
        p = paths(name)
        manifest = read(p['run'] / 'run_manifest.json')
        audit = read(p['audit'] / 'verification.json')
        window = read(p['budget'] / 'manifest.json')
        if (manifest['complete_test_opened'] or manifest['baseline_refit'] or manifest['data_modified']
                or manifest['freeze_sha256'] != digest(freeze_path)
                or audit['manifest_sha256'] != digest(p['run'] / 'run_manifest.json')
                or audit['freeze_sha256'] != digest(freeze_path)):
            raise ValueError('Parent identity or closed-test contract changed.')
        for item, expected in window['input_hashes'].items():
            file = Path(item)
            file = file if file.is_absolute() else ROOT / file
            if digest(file) != expected:
                raise ValueError('Budget analysis is stale: ' + item)
        if window['training_performed'] or window['complete_test_opened'] or window['baseline_refit'] or window['data_modified']:
            raise ValueError('Unexpected budget analysis scope.')
        history_path = p['run'] / 'history.csv'
        history = pd.read_csv(history_path, float_precision='round_trip')
        if not np.array_equal(history.epoch, np.arange(1, 61)) or not np.isfinite(history.select_dtypes('number')).all().all():
            raise ValueError('Expected the complete finite60-epoch trajectory.')
        inputs[str(history_path.relative_to(ROOT))] = digest(history_path)
        manifests[name], histories[name] = manifest, history
        records[name] = {'run': str(p['run'].relative_to(ROOT)), 'manifest': manifest,
            'budget20': {'window': window['window'], 'metrics': read(p['budget'] / 'metrics.json'),
                         'retrieval': read(p['budget'] / 'retrieval/metrics.json')},
            'budget60': {'selected_epoch': manifest['best_epoch'], 'metrics': read(p['run'] / 'metrics.json'),
                         'retrieval': read(p['run'] / 'retrieval/metrics.json')},
            'curve_diagnostics': {'first_primary': float(history.validation_primary.iloc[0]),
                'epoch20_primary': float(history.validation_primary.iloc[19]),
                'epoch60_primary': float(history.validation_primary.iloc[-1]),
                'best_through20_primary': float(history.validation_primary.iloc[:20].min()),
                'best_through60_primary': float(history.validation_primary.min()),
                'first10_mean_clip_fraction': float(history.gradient_clip_fraction.iloc[:10].mean()),
                'last10_mean_clip_fraction': float(history.gradient_clip_fraction.iloc[-10:].mean()),
                'epoch1_training_objective': float(history.train_loss.iloc[0]),
                'epoch60_training_objective': float(history.train_loss.iloc[-1]),
                'first20_recorded_seconds': float(history.elapsed_seconds.iloc[19]),
                'full60_recorded_seconds': float(history.elapsed_seconds.iloc[-1])}}
        for base, files in [(p['run'], ['metrics.json', 'retrieval/metrics.json']),
                            (p['budget'], ['metrics.json', 'retrieval/metrics.json'])]:
            for filename in files:
                file = base / filename
                inputs[str(file.relative_to(ROOT))] = digest(file)
        if records[name]['budget60']['metrics']['completion']['nutrition']['scaled_log_mae'] != manifest['best_primary']:
            raise ValueError('Selected score differs from recorded model selection.')

    a, b = [manifests[name] for name in CANDIDATES]
    for key in ['seed', 'initial_state_sha256', 'parameter_count', 'trainable_parameter_count',
                'training_tasks', 'observed_target_cells', 'code_hashes', 'config_sha256',
                'data_hash', 'panel_hash', 'name_cache_hash', 'functional_sha256']:
        if a[key] != b[key]:
            raise ValueError('Learning-rate control differs in ' + key)
    specs = [{k: v for k, v in m['spec'].items() if k not in ['name', 'learning_rate']} for m in [a, b]]
    if specs[0] != specs[1] or [a['spec']['learning_rate'], b['spec']['learning_rate']] != [1e-4, 3e-4]:
        raise ValueError('This is not the registered learning-rate-only contrast.')
    for key in ['training_order_sha256', 'training_tasks', 'observed_target_cells']:
        np.testing.assert_array_equal(histories[CANDIDATES[0]][key], histories[CANDIDATES[1]][key])

    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    status = {'status': 'running', 'stage': 'direct_learning_rate_comparisons',
              'training_performed': False, 'complete_test_opened': False,
              'baseline_refit': False, 'data_modified': False, 'script_sha256': digest(Path(__file__))}
    write_json(args.output_dir / 'status.json', status)
    try:
        direct = {}
        for task in ['completion', 'name_only', 'retrieval']:
            out = args.output_dir / f'lr1e4_vs_lr3e4_{task}'
            if task == 'retrieval':
                script = ROOT / 'scripts/compare_foodnutrigpt_research_retrieval.py'
                cmd = ['--baseline-dir', str(paths(CANDIDATES[0])['run'] / 'retrieval'),
                       '--candidate-dir', str(paths(CANDIDATES[1])['run'] / 'retrieval')]
            else:
                script = ROOT / 'scripts/compare_foodnutrigpt_research_predictions.py'
                cmd = ['--baseline', str(paths(CANDIDATES[0])['run'] / f'{task}_predictions.parquet'),
                       '--candidate', str(paths(CANDIDATES[1])['run'] / f'{task}_predictions.parquet'), '--task', task]
            inputs[str(script.relative_to(ROOT))] = digest(script)
            subprocess.run([sys.executable, str(script), *cmd, '--output-dir', str(out)], cwd=ROOT, check=True)
            direct[task] = read(out / 'summary.json')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [('train_loss', 'Training objective (187 axes + calibration)'),
                  ('validation_primary', 'Validation scaled-log MAE (142 nutrition axes)'),
                  ('validation_positive_mae', 'Conditional positive validation MAE'),
                  ('gradient_clip_fraction', 'Fraction of steps with clipping')]
        for ax, (key, label) in zip(axes.flat, panels):
            for name, color in zip(CANDIDATES, ['#1969a7', '#d45521']):
                hist = histories[name]
                ax.plot(hist.epoch, hist[key], label=f"lr={manifests[name]['spec']['learning_rate']:g}", color=color)
                ax.axvline(manifests[name]['best_epoch'], color=color, alpha=.25, linestyle=':')
            if key == 'validation_primary':
                for name, color in [(freeze['primary_rf'], '#444444'), (freeze['matched_xgb'], '#777777')]:
                    ax.axhline(freeze['baselines'][name]['metrics']['completion']['nutrition']['scaled_log_mae'],
                               color=color, linestyle='--', label=name)
            ax.set(xlabel='Epoch', ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('R9 registered learning-rate contrast; exploratory single seed20260922')
        fig.tight_layout()
        fig.savefig(args.output_dir / 'learning_curves.png', dpi=160)
        plt.close(fig)
        lower = min(CANDIDATES, key=lambda name: manifests[name]['best_primary'])
        references = {str(Path(p).relative_to(ROOT)): read(Path(p)) for p, s in required.items() if s is None}
        payload = {'status': 'first_registered_group_analysis_complete', 'records': records,
            'learning_rate_only_controls_verified': True, 'direct_learning_rate_comparisons': direct,
            'reference_and_budget_comparisons': references,
            'exploratory_lower_primary_configuration': lower,
            'tie_rule': 'registered order; each run selects earliest strict primary minimum',
            'input_hashes': inputs, 'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
            'baseline_refit': False, 'data_modified': False, 'scientific_confirmation': False,
            'goal_achieved': False, 'training_performed': False,
            'scope': 'Two registered single-seed optimization settings with identical initialization, data and task orders. Intervals condition on validation-selected checkpoints; no adjustment for selection uncertainty. Nested20/60 windows combine training budget and additional selection opportunities. This is a completed stage, not the whole R9 round or a replicated Transformer win.'}
        for item, expected in inputs.items():
            if digest(ROOT / item) != expected:
                raise ValueError('Analysis input changed: ' + item)
        write_json(args.output_dir / 'summary.json', payload)
        status.update(status='complete', stage='first_group_analysis_ready_for_research_interpretation',
                      elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / 'status.json', status)
        print(json.dumps({'status': status['status'], 'exploratory_lower_primary_configuration': lower,
                          'scientific_confirmation': False, 'goal_achieved': False}))
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        write_json(args.output_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    main()

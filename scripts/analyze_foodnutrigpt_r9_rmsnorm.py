"""Analyze the registered encoder RMSNorm contrast after the evidence queue finishes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_transformer_r9 import frozen_inputs
from foodcomp.research_transformer_r9_rmsnorm import load_method, verify_bindings

PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/rmsnorm_v1/config_v2.json'
RUNS = {'mae': 'output/v9_r9/tf192_mae_lr3e4_60',
        'rmsnorm': 'output/v9_r9_methods/tf192_mae_rmsnorm_lr3e4_60'}
AUDITS = {'mae': 'reports/v9_r9_tf192_mae_lr3e4_60_audit_v1/verification.json',
          'rmsnorm': 'reports/v9_r9_rmsnorm_60_audit_v1/verification.json'}
REFERENCES = {'mae_parent': RUNS['mae'], 'rf32': 'output/v9_r8/rf400leaf1half_name32',
              'xgb32': 'output/v9_r8/xgb800d10_name32', 'knn32': 'output/v9_r7/exact_name_knn32'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def requirements():
    expected = {PLAN: 'registered',
                ROOT / 'reports/v9_r9_rmsnorm_60_audit_v1/method_verification.json':
                'complete_registered_rmsnorm_method_audit'}
    for role, run in RUNS.items():
        expected[ROOT / run / 'run_manifest.json'] = 'complete'
        expected[ROOT / AUDITS[role]] = 'complete'
        for name in ['history.csv', 'metrics.json', 'retrieval/metrics.json']:
            expected[ROOT / run / name] = None
    for reference in REFERENCES:
        tasks = ['name_only', 'retrieval'] if reference == 'knn32' else ['completion', 'name_only', 'retrieval']
        for task in tasks:
            expected[ROOT / f'reports/v9_r9_rmsnorm_vs_{reference}_{task}_v1/summary.json'] = None
    for run in ['v9_r9_tf192_lr3e4_rf32_gap_v1', 'v9_r9_rmsnorm_rf32_gap_v1']:
        for name in ['summary.json', 'axis_signed_contributions.csv', 'gap_decomposition.csv', 'fixed_denominator_partitions.csv']:
            expected[ROOT / 'reports' / run / name] = None
    pending = []
    for path, status in expected.items():
        reason = 'absent' if not path.exists() else None
        if reason is None and status and read(path).get('status') != status:
            reason = read(path).get('status', 'missing_status')
        if reason:
            pending.append({'path': path.relative_to(ROOT).as_posix(), 'reason': reason})
    return expected, pending


def screening_gates(comparison):
    primary = comparison['paired_intervals']['scaled_log_mae']
    legacy = comparison['paired_intervals']['log_mae']
    values = [primary['relative_improvement'], *primary['relative_improvement_95_interval'],
              legacy['relative_improvement']]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite screening evidence')
    return {'primary_point_improves_over_same_seed_mae': primary['relative_improvement'] > 0,
            'paired_interval_supports_improvement': primary['relative_improvement_95_interval'][0] > 0,
            'legacy_regression_at_most_2percent_vs_mae': legacy['relative_improvement'] >= -.02}


def partition_change(mae, rmsnorm, expected_difference):
    """Subtract complete fixed-denominator partitions, retaining the common RF as a check."""
    keys = ['stratification', 'stratum']
    joined = mae.merge(rmsnorm, on=keys, how='outer', validate='one_to_one',
                       suffixes=('_mae', '_rmsnorm'), indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Different diagnostic stratum coverage')
    metadata = ['source_cells', 'candidate_groups', 'supported_axes', 'macro_weight_mass']
    for name in metadata + ['tree_under_contribution', 'tree_over_contribution', 'tree_mae_contribution']:
        np.testing.assert_allclose(joined[name + '_mae'], joined[name + '_rmsnorm'], atol=1e-12, rtol=0)
    result = joined[keys].copy()
    for name in metadata:
        result[name] = joined[name + '_mae']
    for direction in ['under', 'over', 'mae']:
        result['rmsnorm_minus_mae_' + direction] = (joined['neural_' + direction + '_contribution_rmsnorm']
                                               - joined['neural_' + direction + '_contribution_mae'])
    if not np.isfinite(result.select_dtypes('number')).all().all():
        raise ValueError('Nonfinite diagnostic contribution')
    np.testing.assert_allclose(result.rmsnorm_minus_mae_under + result.rmsnorm_minus_mae_over,
                               result.rmsnorm_minus_mae_mae, atol=1e-12, rtol=0)
    for _, group in result.groupby('stratification'):
        np.testing.assert_allclose(group.rmsnorm_minus_mae_mae.sum(), expected_difference, atol=1e-12, rtol=0)
        np.testing.assert_allclose(group.macro_weight_mass.sum(), 1., atol=1e-12, rtol=0)
    labels = result.loc[result.stratification == 'label_stratum'].set_index('stratum')
    decomposition = {'positive_under': float(labels.loc['positive', 'rmsnorm_minus_mae_under']),
                     'positive_over': float(labels.loc['positive', 'rmsnorm_minus_mae_over']),
                     'explicit_zero': float(labels.loc['explicit_zero', 'rmsnorm_minus_mae_mae'])}
    np.testing.assert_allclose(sum(decomposition.values()), expected_difference, atol=1e-12, rtol=0)
    return result, decomposition


def verify_method_control(candidate, parent, spec):
    if candidate['spec'] != spec or candidate['kind'] != 'transformer_rmsnorm':
        raise ValueError('Actual candidate disagrees with registration')
    for key in ['seed', 'training_tasks', 'observed_target_cells', 'data_hash',
                'panel_hash', 'name_cache_hash']:
        if candidate[key] != parent[key]:
            raise ValueError('Single-factor control differs: ' + key)
    if (candidate['numerical_recipe_changes'] != ['encoder_normalization']
            or candidate['shared_initial_tensors_match_parent'] is not True):
        raise ValueError('Require only encoder normalization change and identical parent initialization')
    expected = dict(parent['spec'], name=spec['name'], architecture='encoder_rmsnorm_v1')
    if spec != expected or parent['spec']['dropout'] != .15:
        raise ValueError('Encoder normalization must be the only architectural change')
    for key in ['parameter_count', 'trainable_parameter_count']:
        if candidate[key] != parent[key] - 1344:
            raise ValueError('Unexpected normalization parameter delta')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    required, pending = requirements()
    if args.check_only:
        print(json.dumps({'ready': not pending, 'pending': pending, 'output_written': False,
                          'training_performed': False}))
        return
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError('A new output directory is required')
    if pending:
        raise ValueError('Wait for complete training, independent replay,11 comparisons and both gap diagnoses')
    frozen, freeze_path = frozen_inputs(ROOT)
    plan, parent, spec, bindings = load_method(ROOT, PLAN, frozen, freeze_path)
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in required}
    hashes.update(bindings)
    records, histories, comparisons = {}, {}, {}
    for role, run in RUNS.items():
        root = ROOT / run
        manifest = read(root / 'run_manifest.json')
        audit = read(ROOT / AUDITS[role])
        if (manifest['freeze_sha256'] != digest(freeze_path)
                or audit['manifest_sha256'] != digest(root / 'run_manifest.json')
                or audit['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or not audit['all_epoch_orders_exposures_and_schedule_verified']
                or not audit['both323809_predictions_public_loader_replayed_exact']
                or not audit['all_candidate_vectors_and19089_ranks_replayed_exact']):
            raise ValueError('Complete matching replay required: ' + role)
        if any(manifest[k] for k in ['complete_test_opened', 'baseline_refit', 'data_modified']):
            raise ValueError('Method scope changed')
        hashes[(root / 'best_model.pt').relative_to(ROOT).as_posix()] = manifest['checkpoint_sha256']
        verify_bindings(ROOT, manifest['code_hashes'])
        h = pd.read_csv(root / 'history.csv', float_precision='round_trip')
        if not np.array_equal(h.epoch, np.arange(1, 61)) or not np.isfinite(h.select_dtypes('number')).all().all():
            raise ValueError('Full finite60-epoch trajectory required')
        if int(h.loc[h.validation_primary.idxmin(), 'epoch']) != manifest['best_epoch']:
            raise ValueError('Checkpoint does not implement earliest strict primary minimum')
        metrics = read(root / 'metrics.json')
        np.testing.assert_allclose(metrics['completion']['nutrition']['scaled_log_mae'],
                                   h.validation_primary.min(), atol=1e-12, rtol=0)
        records[role] = {'run': run, 'manifest': manifest, 'metrics': metrics,
                         'retrieval': read(root / 'retrieval/metrics.json'),
                         'first10_clip_fraction': float(h.gradient_clip_fraction.iloc[:10].mean()),
                         'last10_clip_fraction': float(h.gradient_clip_fraction.iloc[-10:].mean()),
                         'first_training_objective': float(h.train_loss.iloc[0]),
                         'last_training_objective': float(h.train_loss.iloc[-1])}
        histories[role] = h
    candidate = records['rmsnorm']['manifest']
    if records['mae']['manifest'] != parent:
        raise ValueError('Actual parent disagrees with registration')
    verify_method_control(candidate, parent, spec)
    method_audit = read(ROOT / 'reports/v9_r9_rmsnorm_60_audit_v1/method_verification.json')
    if (method_audit['config_sha256'] != digest(PLAN)
            or method_audit['manifest_sha256'] != digest(ROOT / RUNS['rmsnorm'] / 'run_manifest.json')
            or method_audit['full_replay_sha256'] != digest(ROOT / AUDITS['rmsnorm'])
            or not method_audit['shared_parent_initial_tensors_verified_in_preflight']
            or not method_audit['same_all60_orders_and_exposures']
            or not method_audit['both_saved_checkpoint_architecture_configs_verified']
            or not method_audit['all60_learning_rates_exactly_match_parent']
            or method_audit['sole_numerical_recipe_change'] != 'seven_encoder_layernorms_to_rmsnorm'):
        raise ValueError('Matching complete RMSNorm method audit required')
    for key in ['training_order_sha256', 'training_tasks', 'observed_target_cells']:
        np.testing.assert_array_equal(histories['mae'][key], histories['rmsnorm'][key])
    np.testing.assert_array_equal(histories['rmsnorm'].learning_rate, histories['mae'].learning_rate)
    for reference, baseline in REFERENCES.items():
        comparisons[reference] = {}
        tasks = ['name_only', 'retrieval'] if reference == 'knn32' else ['completion', 'name_only', 'retrieval']
        for task in tasks:
            result = read(ROOT / f'reports/v9_r9_rmsnorm_vs_{reference}_{task}_v1/summary.json')
            if result['complete_test_opened']:
                raise ValueError('Test must remain closed')
            if task == 'retrieval':
                base_dir = ROOT / baseline if reference == 'knn32' else ROOT / baseline / 'retrieval'
                for name, directory in [('baseline', base_dir), ('candidate', ROOT / RUNS['rmsnorm'] / 'retrieval')]:
                    for key, filename in [('ranks', 'ranks.parquet'), ('metrics', 'metrics.json')]:
                        file = directory / filename
                        expected = result['hashes'][name][key]
                        if file.relative_to(ROOT).as_posix() in hashes and hashes[file.relative_to(ROOT).as_posix()] != expected:
                            raise ValueError('Inconsistent retrieval comparison identity')
                        hashes[file.relative_to(ROOT).as_posix()] = expected
            if task != 'retrieval':
                baseline_file = 'nutrition_predictions.parquet' if reference == 'knn32' else task + '_predictions.parquet'
                for file, expected in [(ROOT / baseline / baseline_file, result['baseline_sha256']),
                                       (ROOT / RUNS['rmsnorm'] / (task + '_predictions.parquet'), result['candidate_sha256'])]:
                    hashes[file.relative_to(ROOT).as_posix()] = expected
                if result['scores']['candidate'] != records['rmsnorm']['metrics'][task]:
                    raise ValueError('Comparison candidate scores disagree')
            comparisons[reference][task] = result
    gates = screening_gates(comparisons['mae_parent']['completion'])
    gap_paths = {'mae': ROOT / 'reports/v9_r9_tf192_lr3e4_rf32_gap_v1',
                 'rmsnorm': ROOT / 'reports/v9_r9_rmsnorm_rf32_gap_v1'}
    for role, path in gap_paths.items():
        gap = read(path / 'summary.json')
        if (gap['complete_test_opened'] or gap['neural'] != records[role]['metrics']['completion']
                or gap['tree_prediction_hash'] != comparisons['rf32']['completion']['baseline_sha256']
                or gap['neural_prediction_hash'] != comparisons['mae_parent']['completion'][
                    'baseline_sha256' if role == 'mae' else 'candidate_sha256']):
            raise ValueError('Gap decomposition identity mismatch')
        hashes[(ROOT / RUNS[role] / 'completion_predictions.parquet').relative_to(ROOT).as_posix()] = gap['neural_prediction_hash']
    primary_difference = (records['rmsnorm']['metrics']['completion']['nutrition']['scaled_log_mae']
                          - records['mae']['metrics']['completion']['nutrition']['scaled_log_mae'])
    partitions, decomposition = partition_change(
        pd.read_csv(gap_paths['mae'] / 'fixed_denominator_partitions.csv'),
        pd.read_csv(gap_paths['rmsnorm'] / 'fixed_denominator_partitions.csv'), primary_difference)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    status = {'status': 'running', 'training_performed': False, 'data_modified': False,
              'baseline_refit': False, 'complete_test_opened': False, 'goal_achieved': False}
    write_json(args.output_dir / 'status.json', status)
    try:
        rows = []
        for role, record in records.items():
            for task, subsets in record['metrics'].items():
                for subset in ['nutrition', 'food_metabolome', 'all']:
                    rows.append({'method': role, 'task': task, 'subset': subset, **subsets[subset]})
        pd.DataFrame(rows).to_csv(args.output_dir / 'metrics.csv', index=False)
        partitions.to_csv(args.output_dir / 'rmsnorm_minus_mae_partitions.csv', index=False)
        curve = pd.concat([h.assign(method=role) for role, h in histories.items()], ignore_index=True)
        curve.to_csv(args.output_dir / 'all_epochs.csv', index=False)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        panels = [('validation_primary', 'Validation primary:142 nutrition axes'),
                  ('validation_legacy_log_mae', 'Legacy nutrition log-MAE'),
                  ('validation_positive_mae', 'Conditional positive nutrition MAE'),
                  ('validation_zero_mae', 'Conditional explicit-zero nutrition MAE'),
                  ('gradient_clip_fraction', 'Fraction of training steps clipped'),
                  ('mean_preclip_gradient_norm', 'Mean gradient norm before clipping')]
        metrics = ['scaled_log_mae', 'log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']
        for index, (ax, (key, title)) in enumerate(zip(axes.flat, panels)):
            for role, color in [('mae', '#2468a8'), ('rmsnorm', '#ce5724')]:
                h = histories[role]
                ax.plot(h.epoch, h[key], label='MAE control' if role == 'mae' else 'MAE, encoder RMSNorm', color=color)
                row = h.loc[h.epoch == records[role]['manifest']['best_epoch']].iloc[0]
                ax.scatter([row.epoch], [row[key]], marker='*', s=100, color=color)
            if index < 4:
                for reference, label, style in [(frozen['primary_rf'], 'Frozen RF', '--'),
                                                 (frozen['matched_xgb'], 'Frozen XGB', ':')]:
                    value = frozen['baselines'][reference]['metrics']['completion']['nutrition'][metrics[index]]
                    ax.axhline(value, color='#555555', linestyle=style, label=label)
            if index == 5:
                ax.set_yscale('log')
            ax.set(title=title, xlabel='Epoch')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('R9 encoder RMSNorm: shared initialization and fixed 60-epoch recipe')
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(args.output_dir / ('learning_curves.' + extension), dpi=160)
        plt.close(fig)
        verify_bindings(ROOT, hashes)
        frozen_inputs(ROOT)
        summary = {'status': 'complete_registered_rmsnorm_analysis', 'records': records,
            'comparisons': comparisons, 'single_factor_controls_verified': True, 'screening_gates': gates,
            'fixed_denominator_rmsnorm_minus_mae': decomposition,
            'eligible_for_separately_registered_seed_confirmation': False,
            'structural_screen_passed': all(gates.values()), 'new_seed_runs_prohibited_by_current_user': True,
            'seed_confirmation_performed': False, 'scientific_confirmation': False,
            'visual_review_complete': False, 'report_complete': False, 'goal_achieved': False,
            'input_hashes': hashes, 'script_sha256': digest(Path(__file__)),
            'figure_hashes': {name: digest(args.output_dir / name) for name in ['learning_curves.png', 'learning_curves.svg']},
            'training_performed': False, 'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
            'scope': 'Fixed-seed encoder LayerNorm-to-RMSNorm structural contrast. Width, heads, depth, dropout, MAE, optimizer and schedule remain frozen. Seven norm biases removed (1344 parameters); all shared initialization matches. Explicit attention/FFN bridge is preflight-verified. Food-group intervals condition on fitted models, exclude seed-population, selection and label uncertainty. No extra seed trials are authorized; screening is not foundation-model confirmation.'}
        write_json(args.output_dir / 'summary.json', summary)
        status.update(status='complete', summary_sha256=digest(args.output_dir / 'summary.json'),
                      elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / 'status.json', status)
        print(json.dumps({'status': 'complete', 'screening_gates': gates, 'visual_review_complete': False}))
    except Exception as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    main()


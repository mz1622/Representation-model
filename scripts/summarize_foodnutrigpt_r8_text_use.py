"""Export aggregate-only results of the ten registered R8 name interventions."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', type=Path,
        default=ROOT / 'data/local/research_diagnostics/v9_r8_text_use_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.diagnostic_dir / 'manifest.json'
    manifest = read(manifest_path)
    assert manifest['status'] == 'complete' and manifest['completed_conditions'] == 10
    assert not manifest['complete_test_opened']
    for flag in ['original_predictions_replayed_exact',
                 'labels_weights_observations_and_checkpoints_unchanged',
                 'permutation_row_order_and_duplicate_name_invariance']:
        assert manifest[flag]
    plan = ROOT / 'experiments/foodnutrigpt_v9_research/r8/TEXT_DIAGNOSTIC_PLAN.md'
    assert digest(plan) == manifest['plan_sha256']
    for file, expected in manifest['code_hashes'].items():
        assert digest(args.diagnostic_dir / 'code_snapshot' / Path(file.replace('\\', '/')).name) == expected
    expected_conditions = {(dim, mode, task) for dim in [32, 128]
        for mode in (['zero_name', 'permute_name'] + (['drop_extra96'] if dim == 128 else []))
        for task in ['completion', 'name_only']}
    assert len(manifest['records']) == len(expected_conditions)
    assert {(r['dimensions'], r['intervention'], r['task']) for r in manifest['records']} == expected_conditions
    rows, public_records = [], []
    args.output_dir.mkdir(parents=True)
    for record in manifest['records']:
        dim, mode, task = record['dimensions'], record['intervention'], record['task']
        folder = args.diagnostic_dir / f'name{dim}_{mode}'
        comparison_path = folder / f'{task}_comparison' / 'summary.json'
        prediction = folder / f'{task}_predictions.parquet'
        assert digest(comparison_path) == record['comparison_sha256']
        assert digest(prediction) == record['prediction_sha256']
        comparison = read(comparison_path)
        assert comparison == record['comparison']
        original = ROOT / f'output/v9_r8/mlp512_name{dim}'
        assert digest(original / 'best_model.pt') == record['checkpoint_sha256']
        assert digest(original / f'{task}_predictions.parquet') == comparison['baseline_sha256']
        assert comparison['scores']['baseline'] == read(original / 'metrics.json')[task]
        ci = comparison['paired_intervals']['scaled_log_mae']
        assert ci['group_count'] == 7344 and ci['repeats'] == 1000 and ci['valid_resamples'] == 1000
        candidate = comparison['scores']['candidate']
        row = {'active_name_dimensions': dim, 'intervention': mode, 'task': task,
            'baseline_primary': ci['baseline'], 'candidate_primary': ci['candidate'],
            'relative_error_change': -ci['relative_improvement'],
            'relative_error_change_95_low': -ci['relative_improvement_95_interval'][1],
            'relative_error_change_95_high': -ci['relative_improvement_95_interval'][0],
            'legacy_log_mae': candidate['nutrition']['log_mae'],
            'raw_mae': candidate['nutrition']['raw_mae'],
            'positive_scaled_log_mae': candidate['nutrition']['positive_scaled_log_mae'],
            'zero_scaled_log_mae': candidate['nutrition']['zero_scaled_log_mae'],
            'metabolome45_scaled_log_mae': candidate['food_metabolome']['scaled_log_mae'],
            'all187_scaled_log_mae': candidate['all']['scaled_log_mae'],
            **comparison['primary_change_decomposition'],
            **{key: record[key] for key in ['axes_point_worse', 'axes_interval_worse',
                'axes_interval_better', 'sources_point_worse', 'source_count']}}
        assert np.isfinite([v for v in row.values() if isinstance(v, (int, float))]).all()
        rows.append(row)
        export = args.output_dir / f'name{dim}_{mode}_{task}'
        export.mkdir()
        aggregate_hashes = {}
        for name in ['axis_paired_intervals.csv', 'baseline_axis_metrics.csv',
                     'candidate_axis_metrics.csv', 'source_metrics.csv']:
            source = comparison_path.parent / name
            shutil.copyfile(source, export / name)
            aggregate_hashes[name] = digest(source)
        public_records.append({'dimensions': dim, 'intervention': mode, 'task': task,
            'scores': comparison['scores'], 'paired_intervals': comparison['paired_intervals'],
            'primary_change_decomposition': comparison['primary_change_decomposition'],
            'checkpoint_sha256': record['checkpoint_sha256'],
            'prediction_sha256': record['prediction_sha256'],
            'comparison_sha256': record['comparison_sha256'], 'aggregate_sha256': aggregate_hashes})
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / 'conditions.csv', index=False)
    labels = ['32: all name coordinates zero', '32: permuted names',
              '128: all name coordinates zero', '128: permuted names',
              '128: extra 96 coordinates zero']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout='constrained')
    for ax, task in zip(axs, ['completion', 'name_only']):
        frame = table[table.task.eq(task)]
        y = np.arange(len(frame))
        center = frame.relative_error_change.to_numpy() * 100
        lower = frame.relative_error_change_95_low.to_numpy() * 100
        upper = frame.relative_error_change_95_high.to_numpy() * 100
        for pos, point, low, high in zip(y, center, lower, upper):
            ax.errorbar(point, pos, xerr=[[point - low], [high - point]], fmt='o',
                        color='#aa483e' if point > 0 else '#15775f', capsize=4)
        ax.axvline(0, color='grey', lw=1)
        ax.set_yticks(y, labels)
        ax.set(title='Completion' if task == 'completion' else 'Name only',
               xlabel='Relative change in 142-axis error (%)\nPositive = worse; negative = better')
        ax.grid(axis='x', alpha=.2)
        ax.margins(x=.15)
    axs[0].invert_yaxis()
    fig.suptitle('R8 fixed-checkpoint name interventions (no retraining)\n'
                 '95% paired food-group intervals; seed and selection uncertainty excluded', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(args.output_dir / f'name_interventions.{ext}', dpi=180)
    plt.close(fig)
    summary = {'status': 'complete', 'stage': 'post_fit_diagnostic_only',
        'diagnostic_manifest_sha256': digest(manifest_path),
        'diagnostic_code_commit': manifest['code_commit'], 'plan_sha256': manifest['plan_sha256'],
        'diagnostic_code_hashes': manifest['code_hashes'],
        'fitted_model_code_commit': manifest['fitted_model_code_commit'],
        'execution_contract_sha256': manifest['execution_contract_sha256'],
        'elapsed_diagnostic_seconds': manifest['elapsed_seconds'], 'conditions': rows, 'records': public_records,
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'model_improvement_accepted': False, 'scientific_confirmation': False,
        'raw_case_details_exported': False, 'retrieval_reevaluated': False,
        'scope': 'All ten registered interventions. Corruptions may be out of distribution; '
                 'effects on a fixed fitted predictor are not effects of deleting features before retraining. '
                 'No epoch, model or benchmark input was changed after seeing these results.'}
    write_json(args.output_dir / 'summary.json', summary)
    print(summary['status'], len(rows), 'conditions; aggregate-only export')


if __name__ == '__main__':
    main()

"""Explain completed three-seed variation from aggregate metrics only, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260922, 20260923, 20260924]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    stats = ROOT / 'reports/v9_r9_three_seed_confirmation_v1'
    paths = [stats / 'summary.json', stats / 'axis_metrics_by_seed.csv',
             stats / 'axis_paired_intervals.csv',
             ROOT / 'reports/v9_r9_tf192_lr3e4_rf32_gap_v1/axis_comparison.csv']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    summary = json.loads(paths[0].read_text(encoding='utf-8'))
    if summary['status'] != 'complete_fixed_reference_statistics':
        raise ValueError('Complete three-seed evidence required')
    metrics = pd.read_csv(paths[1])
    metrics = metrics.loc[(metrics.task == 'completion') & (metrics.loss_group == 'nutrition')]
    if not np.isfinite(metrics[['scaled_log_mae', 'log_mae']].to_numpy()).all():
        raise ValueError('Nonfinite primary or legacy error')
    neural = metrics.loc[metrics.method == 'transformer']
    if set(neural.seed.astype(int)) != set(SEEDS):
        raise ValueError('Unexpected seeds')
    axis = neural.pivot(index='axis_index', columns='seed', values='scaled_log_mae')
    axis.columns = ['seed' + str(int(seed)) for seed in axis.columns]
    axis['neural_mean'] = axis.mean(axis=1)
    axis['neural_sd'] = axis.iloc[:, :3].std(axis=1, ddof=1)
    for method in ['rf', 'xgb']:
        reference = metrics.loc[metrics.method == method].set_index('axis_index')
        if not reference.index.is_unique:
            raise ValueError('Duplicate reference axis')
        axis[method] = reference.scaled_log_mae
    if len(axis) != 142 or not np.isfinite(axis.to_numpy()).all():
        raise ValueError('Expected exactly142 complete nutrition axes')
    metadata = pd.read_csv(paths[3])
    columns = ['axis_index', 'canonical_name', 'mask_family', 'training_candidates',
               'positive_train_candidate_support', 'support_bin']
    axis = axis.reset_index().merge(metadata[columns], on='axis_index', validate='one_to_one')
    support = metrics.loc[metrics.method == 'rf', ['axis_index', 'candidate_support']]
    axis = axis.merge(support, on='axis_index', validate='one_to_one')
    axis['mean_rf_gap_contribution'] = (axis.neural_mean - axis.rf) / 142
    axis['seed24_minus_seed22_contribution'] = (axis.seed20260924 - axis.seed20260922) / 142
    axis['seed23_minus_seed22_contribution'] = (axis.seed20260923 - axis.seed20260922) / 142
    axis['all_seeds_better_rf'] = (axis[['seed'+str(s) for s in SEEDS]].lt(axis.rf, axis=0)).all(axis=1)
    axis['all_seeds_worse_rf'] = (axis[['seed'+str(s) for s in SEEDS]].gt(axis.rf, axis=0)).all(axis=1)
    cis = pd.read_csv(paths[2])
    cis = cis.loc[(cis.task == 'completion') & (cis.baseline == 'rf')]
    if len(cis) != 142:
        raise ValueError('Expected142 primary paired intervals')
    axis = axis.merge(cis[['axis_index', 'difference_95_low', 'difference_95_high',
                           'sparse_support_below_30', 'resamples_without_axis_support']],
                      on='axis_index', validate='one_to_one')
    axis['seed_range'] = axis[['seed'+str(s) for s in SEEDS]].max(axis=1) - axis[['seed'+str(s) for s in SEEDS]].min(axis=1)
    contribution_columns = ['mean_rf_gap_contribution', 'seed24_minus_seed22_contribution',
                            'seed23_minus_seed22_contribution']
    grouped = []
    for key in ['support_bin', 'mask_family']:
        groups = axis.groupby(key, observed=True)
        table = groups[contribution_columns].sum()
        table['axes'] = groups.size()
        table['all_seeds_better_rf'] = groups.all_seeds_better_rf.sum()
        table['all_seeds_worse_rf'] = groups.all_seeds_worse_rf.sum()
        table['partition'] = key
        table.index.name = 'level'
        for col in contribution_columns:
            np.testing.assert_allclose(table[col].sum(), axis[col].sum(), atol=1e-12, rtol=0)
        grouped.append(table.reset_index())
    result = {
        'status': 'complete', 'seed_primary': {str(s): float(axis['seed'+str(s)].mean()) for s in SEEDS},
        'mean_primary': float(axis.neural_mean.mean()), 'rf_primary': float(axis.rf.mean()),
        'mean_gap_to_rf': float(axis.mean_rf_gap_contribution.sum()),
        'seed24_minus_seed22': float(axis.seed24_minus_seed22_contribution.sum()),
        'all_seeds_better_rf_axes': int(axis.all_seeds_better_rf.sum()),
        'all_seeds_worse_rf_axes': int(axis.all_seeds_worse_rf.sum()),
        'mean_better_rf_axes': int((axis.neural_mean < axis.rf).sum()),
        'sparse_validation_axes_below30': int(axis.sparse_support_below_30.sum()),
        'input_hashes': hashes, 'script_sha256': digest(Path(__file__)),
        'scope': 'Descriptive decomposition of previously selected checkpoints. Axis intervals are exploratory, unadjusted for multiple comparisons. Each primary contribution uses the fixed142-axis denominator. No unique causal mechanism is established.',
        'complete_test_opened': False, 'data_modified': False, 'baseline_refit': False,
        'training_performed': False,
    }
    # Independent reconstruction of the known aggregate and individual selected checkpoints.
    expected = [0.18355327847469688, 0.18708334056598974, 0.1969165951810234]
    np.testing.assert_allclose(list(result['seed_primary'].values()), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['mean_primary'], 0.18918440474057, atol=1e-12, rtol=0)
    for path in paths:
        if digest(path) != hashes[path.relative_to(ROOT).as_posix()]:
            raise ValueError('Input changed during aggregate diagnosis')
    args.output_dir.mkdir(parents=True)
    axis.to_csv(args.output_dir / 'axis_variation.csv', index=False)
    pd.concat(grouped, ignore_index=True).to_csv(args.output_dir / 'partition_contributions.csv', index=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(pd.concat(grouped, ignore_index=True).to_string(index=False))
    print(axis.nlargest(12, 'seed24_minus_seed22_contribution')[['canonical_name', 'training_candidates', 'candidate_support', *contribution_columns]].to_string(index=False))


if __name__ == '__main__':
    main()

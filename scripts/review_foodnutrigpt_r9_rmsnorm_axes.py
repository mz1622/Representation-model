"""Summarize existing paired axis evidence without model execution or new scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paths = {'mae': ROOT / 'reports/v9_r9_tf192_lr3e4_rf32_gap_v1/axis_signed_contributions.csv',
         'rmsnorm': ROOT / 'reports/v9_r9_rmsnorm_rf32_gap_v1/axis_signed_contributions.csv',
         'paired': ROOT / 'reports/v9_r9_rmsnorm_vs_mae_parent_completion_v1/axis_paired_intervals.csv',
         'analysis': ROOT / 'reports/v9_r9_rmsnorm_analysis_v1/summary.json'}
paths['script'] = Path(__file__)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def ready():
    analysis = paths['analysis']
    if not analysis.exists():
        return ['completed RMSNorm analysis absent']
    result = json.loads(analysis.read_text(encoding='utf-8'))
    if result.get('status') != 'complete_registered_rmsnorm_analysis':
        return ['completed RMSNorm analysis required']
    return [p.relative_to(ROOT).as_posix() for p in paths.values() if not p.exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    pending = ready()
    if args.check_only:
        print(json.dumps({'ready': not pending, 'pending': pending, 'output_written': False, 'model_forward_performed': False}))
        return
    if pending:
        raise ValueError('Wait for completed analysis: ' + ', '.join(pending))
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths.values()}
    a = pd.read_csv(paths['mae'], float_precision='round_trip')
    b = pd.read_csv(paths['rmsnorm'], float_precision='round_trip')
    keys = ['axis_index', 'canonical_name', 'mask_family', 'training_candidates', 'macro_weight']
    j = a.merge(b, on=keys, how='outer', validate='one_to_one', suffixes=('_mae', '_rmsnorm'), indicator=True)
    if not j._merge.eq('both').all() or len(j) != 142:
        raise ValueError('Changed common nutrition-axis coverage.')
    for key in ['tree_under_contribution', 'tree_over_contribution', 'tree_mae_contribution']:
        np.testing.assert_array_equal(j[key + '_mae'], j[key + '_rmsnorm'])
    out = j[keys].copy()
    for direction in ['under', 'over', 'mae']:
        out['rmsnorm_minus_mae_' + direction] = j['neural_' + direction + '_contribution_rmsnorm'] - j['neural_' + direction + '_contribution_mae']
    out['mae_minus_rf'] = j['mae_gap_contribution_mae']
    out['rmsnorm_minus_rf'] = j['mae_gap_contribution_rmsnorm']
    paired = pd.read_csv(paths['paired'], float_precision='round_trip')
    paired = paired.loc[paired.loss_group.eq('nutrition')].drop(columns=['canonical_name', 'loss_group'])
    out = out.merge(paired, on='axis_index', validate='one_to_one')
    if len(out) != 142 or not np.isfinite(out.select_dtypes('number')).all().all():
        raise ValueError('Incomplete or nonfinite axis contrast.')
    np.testing.assert_allclose(out.candidate_minus_baseline * out.macro_weight, out.rmsnorm_minus_mae_mae, atol=1e-12, rtol=0)
    analysis = json.loads(paths['analysis'].read_text(encoding='utf-8'))
    if analysis['status'] != 'complete_registered_rmsnorm_analysis' or analysis['complete_test_opened']:
        raise ValueError('Require completed registered analysis with test closed.')
    delta = sum(analysis['fixed_denominator_rmsnorm_minus_mae'].values())
    np.testing.assert_allclose(out.rmsnorm_minus_mae_mae.sum(), delta, atol=1e-12, rtol=0)
    result = {'status': 'complete_aggregate_axis_contrast', 'axis_count': len(out),
              'primary_rmsnorm_minus_mae': delta, 'axes_point_better_than_mae': int(out.rmsnorm_minus_mae_mae.lt(0).sum()),
              'axes_unadjusted_interval_supports_improvement': int(out.difference_95_high.lt(0).sum()),
              'axes_unadjusted_interval_supports_regression': int(out.difference_95_low.gt(0).sum()),
              'sparse_validation_axes_below30': int(out.sparse_support_below_30.sum()),
              'rmsnorm_axes_better_than_rf': int(out.rmsnorm_minus_rf.lt(0).sum()),
              'amino_acid_axes_rmsnorm_worse_than_mae': int(out.loc[out.mask_family.eq('amino_acid'), 'rmsnorm_minus_mae_mae'].gt(0).sum()),
              'input_hashes': hashes, 'training_performed': False, 'model_forward_performed': False,
              'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
              'scope': 'Subtraction of existing complete same-panel contributions and existing food-group intervals. Exploratory axis inspection without multiplicity correction or new selection. Axis/source differences cannot establish a unique mechanism.'}
    directory = ROOT / 'reports/v9_r9_rmsnorm_axis_changes_v1'
    if directory.exists():
        raise FileExistsError(directory)
    for path, expected in hashes.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Input changed during axis contrast: ' + path)
    directory.mkdir()
    out.to_csv(directory / 'axis_changes.csv', index=False)
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'axis_count', 'axes_point_better_than_mae',
        'axes_unadjusted_interval_supports_improvement', 'axes_unadjusted_interval_supports_regression']}))
    print(out.sort_values('rmsnorm_minus_mae_mae')[['canonical_name','training_candidates','candidate_support','rmsnorm_minus_mae_mae','difference_95_low','difference_95_high']].iloc[np.r_[0:7, -7:0]].to_string(index=False))


if __name__ == '__main__':
    main()

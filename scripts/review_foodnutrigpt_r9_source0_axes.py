"""Summarize existing paired axis evidence without model execution or new scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paths = {'mae': ROOT / 'reports/v9_r9_tf192_lr3e4_rf32_gap_v1/axis_signed_contributions.csv',
         'source0': ROOT / 'reports/v9_r9_source0_rf32_gap_v1/axis_signed_contributions.csv',
         'paired': ROOT / 'reports/v9_r9_source0_vs_mae_parent_completion_v1/axis_paired_intervals.csv',
         'analysis': ROOT / 'reports/v9_r9_source0_analysis_v1/summary.json'}
paths['script'] = Path(__file__)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths.values()}
a = pd.read_csv(paths['mae'], float_precision='round_trip')
b = pd.read_csv(paths['source0'], float_precision='round_trip')
keys = ['axis_index', 'canonical_name', 'mask_family', 'training_candidates', 'macro_weight']
j = a.merge(b, on=keys, how='outer', validate='one_to_one', suffixes=('_mae', '_source0'), indicator=True)
if not j._merge.eq('both').all() or len(j) != 142:
    raise ValueError('Changed common nutrition-axis coverage.')
for key in ['tree_under_contribution', 'tree_over_contribution', 'tree_mae_contribution']:
    np.testing.assert_array_equal(j[key + '_mae'], j[key + '_source0'])
out = j[keys].copy()
for direction in ['under', 'over', 'mae']:
    out['source0_minus_mae_' + direction] = j['neural_' + direction + '_contribution_source0'] - j['neural_' + direction + '_contribution_mae']
out['mae_minus_rf'] = j['mae_gap_contribution_mae']
out['source0_minus_rf'] = j['mae_gap_contribution_source0']
paired = pd.read_csv(paths['paired'], float_precision='round_trip')
paired = paired.loc[paired.loss_group.eq('nutrition')].drop(columns=['canonical_name', 'loss_group'])
out = out.merge(paired, on='axis_index', validate='one_to_one')
if len(out) != 142 or not np.isfinite(out.select_dtypes('number')).all().all():
    raise ValueError('Incomplete or nonfinite axis contrast.')
np.testing.assert_allclose(out.candidate_minus_baseline * out.macro_weight, out.source0_minus_mae_mae, atol=1e-12, rtol=0)
analysis = json.loads(paths['analysis'].read_text(encoding='utf-8'))
if analysis['status'] != 'complete_registered_source0_analysis' or analysis['complete_test_opened']:
    raise ValueError('Require completed registered analysis with test closed.')
delta = sum(analysis['fixed_denominator_source0_minus_mae'].values())
np.testing.assert_allclose(out.source0_minus_mae_mae.sum(), delta, atol=1e-12, rtol=0)
result = {'status': 'complete_aggregate_axis_contrast', 'axis_count': len(out),
          'primary_source0_minus_mae': delta, 'axes_point_better_than_mae': int(out.source0_minus_mae_mae.lt(0).sum()),
          'axes_unadjusted_interval_supports_improvement': int(out.difference_95_high.lt(0).sum()),
          'axes_unadjusted_interval_supports_regression': int(out.difference_95_low.gt(0).sum()),
          'sparse_validation_axes_below30': int(out.sparse_support_below_30.sum()),
          'source0_axes_better_than_rf': int(out.source0_minus_rf.lt(0).sum()),
          'amino_acid_axes_source0_worse_than_mae': int(out.loc[out.mask_family.eq('amino_acid'), 'source0_minus_mae_mae'].gt(0).sum()),
          'input_hashes': hashes, 'training_performed': False, 'model_forward_performed': False,
          'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
          'scope': 'Subtraction of existing complete same-panel contributions and existing food-group intervals. Exploratory axis inspection without multiplicity correction or new selection. Axis/source differences cannot establish a unique mechanism.'}
directory = ROOT / 'reports/v9_r9_source0_axis_changes_v1'
if directory.exists():
    raise FileExistsError(directory)
for path, expected in hashes.items():
    if sha(ROOT / path) != expected:
        raise ValueError('Input changed during axis contrast: ' + path)
directory.mkdir()
out.to_csv(directory / 'axis_changes.csv', index=False)
(directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
print(out.sort_values('source0_minus_mae_mae')[['canonical_name','training_candidates','candidate_support','source0_minus_mae_mae','difference_95_low','difference_95_high']].iloc[np.r_[0:7, -7:0]].to_string(index=False))

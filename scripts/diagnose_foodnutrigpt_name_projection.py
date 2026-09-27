"""R7 post-prediction gap strata and local-only examples; no refitting or selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, score_predictions, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    private_root = (ROOT / 'data/local/research_diagnostics').resolve()
    out = args.output_dir.resolve()
    if not out.is_relative_to(private_root):
        raise ValueError('Original-value examples must remain under data/local/research_diagnostics.')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    axes = data.axes.copy()
    axes['training_candidates'] = [data.profiles.iloc[data.train[data.observed[data.train, a]]]
        .exact_name_group_id.nunique() for a in axes.axis_index]
    axes['support_bin'] = pd.cut(axes.training_candidates, [-1, 99, 999, np.inf],
        labels=['below_100', '100_to_999', 'at_least_1000'])
    axes['scale_bin'] = np.where(axes.research_scale < .001,
        'below_1mg_typical', 'at_least_1mg_typical')
    frames, per_axis, hashes = {}, {}, {}
    for kind in ['knn', 'mlp']:
        for dims in [32, 128]:
            name = f'{kind}{dims}'
            filename = 'nutrition_predictions.parquet' if kind == 'knn' else 'name_only_predictions.parquet'
            path = ROOT / f'output/v9_r7/exact_name_{name}' / filename
            frames[name] = pd.read_parquet(path)
            per_axis[name] = score_predictions(data, frames[name])[1]
            hashes[name] = digest(path)
    pairs = {}
    for base, candidate in [('knn32', 'knn128'), ('mlp32', 'mlp128'), ('knn128', 'mlp128')]:
        label = f'{candidate}_vs_{base}'
        a = per_axis[base][['axis_index', 'scaled_log_mae']].merge(
            per_axis[candidate][['axis_index', 'scaled_log_mae']], on='axis_index',
            suffixes=('_baseline', '_candidate'), validate='one_to_one').merge(
                axes, on='axis_index', validate='one_to_one')
        a = a[a.loss_group.eq('nutrition')].copy()
        assert len(a) == 142
        a['gap'] = a.scaled_log_mae_candidate - a.scaled_log_mae_baseline
        strata = []
        for column in ['mask_family', 'support_bin', 'scale_bin']:
            for value, group in a.groupby(column, observed=True):
                strata.append({'stratification': column, 'stratum': str(value), 'axes': len(group),
                    'baseline_primary': group.scaled_log_mae_baseline.mean(),
                    'candidate_primary': group.scaled_log_mae_candidate.mean(),
                    'gap_macro_contribution': group.gap.sum() / 142,
                    'candidate_better_axes': int((group.gap < 0).sum())})
        a.to_csv(out / f'{label}_axis.csv', index=False)
        keys = ['profile_index', 'axis_index']
        joined = data.jobs.merge(frames[base].rename(columns={'prediction': 'baseline_prediction'}),
            on=keys, validate='one_to_one').merge(frames[candidate].rename(
                columns={'prediction': 'candidate_prediction'}), on=keys, validate='one_to_one')
        joined = joined[joined.axis_index.isin(a.axis_index)].copy()
        scale = data.scale[joined.axis_index.to_numpy()]
        for role in ['baseline', 'candidate']:
            joined[role + '_error'] = np.abs(np.log1p(joined[role + '_prediction'] / scale)
                - np.log1p(joined.target / scale))
        joined['gap'] = joined.candidate_error - joined.baseline_error
        cases = pd.concat([joined.nsmallest(20, 'gap').assign(selection='largest_improvement'),
            joined.nlargest(20, 'gap').assign(selection='largest_regression')])
        cases = cases.merge(data.profiles[['profile_index', 'original_name', 'source_key',
            'exact_name_group_id']], on='profile_index', validate='many_to_one').merge(
                axes[['axis_index', 'canonical_name']], on='axis_index', validate='many_to_one')
        cases.to_csv(out / f'{label}_local_cases.csv', index=False)
        pairs[label] = {'baseline': base, 'candidate': candidate, 'primary_gap': float(a.gap.mean()),
            'decomposition': strata, 'baseline_sha256': hashes[base], 'candidate_sha256': hashes[candidate],
            'case_count': len(cases), 'case_scope': 'Selected profile-cell extremes, not typical examples or causal evidence.'}
    result = {'status': 'complete', 'comparisons': pairs, 'data_sha256': digest(data.root / 'manifest.json'),
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'scope': 'Post-prediction descriptive strata only; no models, tasks, metrics or selection modified. '
            'Original food names and values remain in this ignored local directory.'}
    write_json(out / 'summary.json', result)
    for name, item in pairs.items():
        print(name, item['primary_gap'])
        print([s for s in item['decomposition'] if s['stratification'] == 'support_bin'])


if __name__ == '__main__':
    main()

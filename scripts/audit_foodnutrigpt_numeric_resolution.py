"""Training-only numeric-encoding feasibility audit; no model fitting or validation scoring."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import VERSION, cell_weights, digest, write_json


def weighted_knots(values, weights, bins):
    """Deduplicated inverse weighted ECDF at j/bins, including observed endpoints."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or bins not in (8, 16, 32):
        raise ValueError('Invalid knot inputs.')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Observed values and strictly positive weights must be finite.')
    if not len(values):
        return np.array([], dtype=np.float64)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    unique = sorted_values[starts]
    mass = np.add.reduceat(weights[order], starts)
    cumulative = np.cumsum(mass, dtype=np.float64)
    if not np.isfinite(cumulative).all() or cumulative[-1] <= 0:
        raise ValueError('Invalid cumulative mass.')
    index = np.searchsorted(cumulative / cumulative[-1], np.linspace(0, 1, bins + 1), side='left')
    knots = np.unique(unique[np.minimum(index, len(unique) - 1)])
    assert knots[0] == unique[0] and knots[-1] == unique[-1]
    assert len(knots) <= bins + 1 and (np.diff(knots) > 0).all()
    return knots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, required=True)
    args = parser.parse_args()
    local = args.local_dir.resolve()
    if not local.is_relative_to((ROOT / 'data/local/research_diagnostics').resolve()):
        raise ValueError('Numeric knot values must stay in ignored local data.')
    if args.output_dir.exists() or local.exists():
        raise FileExistsError('Audit output already exists; use a new version.')
    start = time.monotonic()
    # Analytic fixtures: zeros retain probability mass; constant axes have no interval.
    np.testing.assert_array_equal(weighted_knots([0, 0, 1, 2], [3, 3, 1, 1], 8), [0, 1, 2])
    np.testing.assert_array_equal(weighted_knots([2, 2], [1, 1], 16), [2])
    assert len(weighted_knots([], [], 8)) == 0
    for bad_values, bad_weights in [([np.nan], [1]), ([1], [np.inf]), ([1], [0])]:
        try:
            weighted_knots(bad_values, bad_weights, 8)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite/invalid observed inputs were accepted.')
    data_root = ROOT / 'data/processed' / VERSION
    manifest = json.loads((data_root / 'manifest.json').read_text(encoding='utf-8'))
    hashes = {'manifest.json': digest(data_root / 'manifest.json')}
    for name in ['canonical_cells.parquet', 'axes.csv', 'profiles.csv.gz']:
        hashes[name] = digest(data_root / name)
        assert hashes[name] == manifest['artifact_hashes'][name]
    axes = pd.read_csv(data_root / 'axes.csv')
    profiles = pd.read_csv(data_root / 'profiles.csv.gz', usecols=['profile_id', 'partition'])
    train_ids = set(profiles.loc[profiles.partition.eq('train'), 'profile_id'])
    columns = ['profile_id', 'axis_index', 'value', 'partition', 'source_key', 'exact_name_group_id', 'quarantined']
    # Predicate filtering avoids loading validation numeric rows or any validation-job table.
    cells = pd.read_parquet(data_root / 'canonical_cells.parquet', columns=columns,
                           filters=[('partition', '==', 'train'), ('quarantined', '==', False)])
    assert cells.partition.eq('train').all() and not cells.quarantined.any()
    assert set(cells.profile_id).issubset(train_ids) and len(train_ids) == 64700
    assert not cells.duplicated(['profile_id', 'axis_index']).any()
    if not np.isfinite(cells.value).all() or cells.value.lt(0).any():
        raise ValueError('Invalid observed training value.')
    cells['weight'] = cell_weights(cells)
    scale = axes.set_index('axis_index').research_scale
    cells['transformed'] = np.log1p(cells.value.to_numpy() / scale.loc[cells.axis_index].to_numpy()).astype(np.float32)
    if not np.isfinite(cells.transformed).all():
        raise FloatingPointError('Nonfinite frozen-input-space value.')
    positive = cells[cells.value.gt(0)].copy()
    positive['weight'] = cell_weights(positive)
    rows, private = [], {}
    for policy, frame in [('all_observed', cells), ('positive_only', positive)]:
        grouped = {int(a): group for a, group in frame.groupby('axis_index')}
        for axis in axes.itertuples(index=False):
            group = grouped.get(int(axis.axis_index), frame.iloc[:0])
            groups = group.exact_name_group_id.nunique()
            mass = float(group.weight.sum())
            if not np.isclose(mass, groups, atol=1e-10, rtol=1e-12):
                raise AssertionError('Candidate/source-balanced weight mass changed.')
            zero_mass = float(group.loc[group.value.eq(0), 'weight'].sum())
            for bins in [8, 16, 32]:
                knots = weighted_knots(group.transformed.to_numpy(), group.weight.to_numpy(), bins)
                positive_intervals = max(0, int(np.sum(knots > 0)) - 1)
                rows.append({'axis_index': int(axis.axis_index), 'canonical_name': axis.canonical_name,
                    'loss_group': axis.loss_group, 'training_role': axis.training_role,
                    'loss_eligible': bool(axis.loss_eligible), 'policy': policy, 'requested_intervals': bins,
                    'observed_training_cells': len(group), 'training_candidate_groups': groups,
                    'unique_input_values': int(group.transformed.nunique()), 'unique_positive_input_values': int(group.loc[group.value.gt(0), 'transformed'].nunique()),
                    'source_balanced_zero_fraction': zero_mass / mass if mass else None,
                    'unique_knots': len(knots), 'effective_intervals': max(0, len(knots) - 1),
                    'intervals_between_strictly_positive_knots': positive_intervals,
                    'requires_constant_or_empty_fallback': len(knots) < 2})
                private[f'{policy}/axis{axis.axis_index}/bins{bins}'] = knots.tolist()
    table = pd.DataFrame(rows)
    summaries = []
    for (policy, bins, role), group in table.groupby(['policy', 'requested_intervals', 'training_role']):
        summaries.append({'policy': policy, 'requested_intervals': int(bins), 'training_role': role,
            'axes': len(group), 'axes_below_requested_intervals': int(group.effective_intervals.lt(bins).sum()),
            'axes_constant_or_empty': int(group.requires_constant_or_empty_fallback.sum()),
            'effective_intervals_min': int(group.effective_intervals.min()),
            'effective_intervals_median': float(group.effective_intervals.median()),
            'effective_intervals_max': int(group.effective_intervals.max())})
    local.mkdir(parents=True)
    write_json(local / 'candidate_knots.json', private)
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / 'axis_resolution.csv', index=False)
    summary = {'status': 'complete_training_only_feasibility_no_model', 'data_sha256': hashes['manifest.json'],
        'artifact_hashes': hashes, 'script_sha256': digest(Path(__file__)),
        'plan_sha256': digest(ROOT / 'experiments/foodnutrigpt_v9_research/NEXT_NUMERIC_ENCODING.md'),
        'observed_train_cells': len(cells), 'positive_train_cells': len(positive),
        'training_profiles': len(train_ids), 'axes': len(axes), 'candidate_policies': ['all_observed', 'positive_only'],
        'requested_intervals': [8, 16, 32], 'summaries': summaries,
        'knots_sha256': digest(local / 'candidate_knots.json'), 'knots_directory': str(local),
        'elapsed_seconds': time.monotonic() - start, 'analytic_fixtures_and_invalid_input_rejection_passed': True,
        'validation_numeric_rows_loaded': False, 'validation_scored': False, 'complete_test_opened': False,
        'models_trained': 0, 'model_improvement_accepted': False,
        'scope': 'Train-only source-balanced ECDF resolution in frozen float32 scaled-log input space; no encoding deployed, no model result, no R9 preregistration. Positive-only policy recomputes weights on positive cells, as train positive scales do. Repeated observations stay profile medians; no cross-source labels pooled. Duplicate knots are removed without jitter.'}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['artifact_hashes', 'summaries']}, ensure_ascii=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()

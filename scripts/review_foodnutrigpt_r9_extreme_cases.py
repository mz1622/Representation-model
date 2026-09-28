"""Check existing parent-run extreme cases and export aggregate counts only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    view = ROOT / 'data/processed/foodnutrigpt_v9_r0_v1'
    diagnostic = ROOT / 'reports/v9_r9_tf192_lr3e4_rf32_gap_v1'
    case_path = ROOT / 'data/local/research_diagnostics/v9_r9_tf192_lr3e4_rf32_gap_v1/local_success_failure_cases.csv'
    paths = [view/'manifest.json', diagnostic/'summary.json', case_path,
             diagnostic/'axis_signed_contributions.csv', Path(__file__)]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    data_manifest, parent = read(view/'manifest.json'), read(diagnostic/'summary.json')
    if digest(view/'manifest.json') != parent['data_hash'] or parent['complete_test_opened']:
        raise ValueError('Changed parent diagnostic identity.')
    for name in ['axes.csv', 'profiles.csv.gz', 'quarantined_validation_jobs.parquet']:
        path = view/name
        if digest(path) != data_manifest['artifact_hashes'][name]:
            raise ValueError('Changed frozen validation input: ' + name)
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    axes = pd.read_csv(view/'axes.csv')
    profiles = pd.read_csv(view/'profiles.csv.gz')
    if set(profiles.partition) != {'train', 'validation'}:
        raise ValueError('Unexpected research-view partitions.')
    eligible = axes[axes.loss_eligible & axes.loss_group.eq('nutrition')]
    if len(eligible) != 142:
        raise ValueError('Nutrition axis coverage changed.')
    keys = ['profile_index', 'axis_index']
    joined = pd.read_parquet(view/'quarantined_validation_jobs.parquet')
    for role, relative, hash_key in [
        ('neural', 'output/v9_r9/tf192_mae_lr3e4_60/completion_predictions.parquet', 'neural_prediction_hash'),
        ('tree', 'output/v9_r8/rf400leaf1half_name32/completion_predictions.parquet', 'tree_prediction_hash'),
    ]:
        path = ROOT/relative
        if digest(path) != parent[hash_key]:
            raise ValueError('Changed saved predictions: ' + role)
        hashes[relative] = digest(path)
        prediction = pd.read_parquet(path)
        joined = joined.merge(prediction[keys+['prediction']].rename(columns={'prediction': role}),
                              on=keys, validate='one_to_one')
    if len(joined) != 323809:
        raise ValueError('Validation task count changed.')
    scale = axes.set_index('axis_index').research_scale.loc[joined.axis_index].to_numpy()
    for role in ['neural', 'tree']:
        joined[role+'_error'] = np.abs(np.log1p(joined[role]/scale)-np.log1p(joined.target/scale))
    joined['gap'] = joined.neural_error-joined.tree_error
    if not np.isfinite(joined[['target', 'neural', 'tree', 'neural_error', 'tree_error', 'gap']]).all().all():
        raise ValueError('Nonfinite prediction or error.')
    nutrition = joined[joined.axis_index.isin(eligible.axis_index)]
    expected = pd.concat([nutrition.nsmallest(20, 'gap'), nutrition.nlargest(20, 'gap')], ignore_index=True)
    cases = pd.read_csv(case_path, float_precision='round_trip')
    if len(cases) != 40 or cases.duplicated(keys).any():
        raise ValueError('Invalid existing case set.')
    pd.testing.assert_frame_equal(cases[keys], expected[keys], check_dtype=False)
    for name in ['target', 'neural', 'tree', 'neural_error', 'tree_error', 'gap']:
        values = cases[name].to_numpy(dtype=expected[name].dtype)
        np.testing.assert_allclose(values, expected[name].to_numpy(), rtol=1e-12, atol=1e-12)
    metadata = profiles[['profile_index', 'profile_id', 'source_key', 'exact_name_group_id', 'original_name', 'partition']]
    matched = expected[keys].merge(metadata, on='profile_index', validate='many_to_one')
    if not matched.partition.eq('validation').all():
        raise ValueError('Case outside validation.')
    for name in ['profile_id', 'source_key', 'exact_name_group_id', 'original_name']:
        if not cases[name].equals(matched[name]):
            raise ValueError('Case metadata changed: ' + name)
    cases['direction'] = np.where(cases.gap < 0, 'transformer_better', 'transformer_worse')
    cases['label_stratum'] = np.where(cases.target > 0, 'positive', 'explicit_zero')
    counts = []
    for direction, group in cases.groupby('direction'):
        counts.append({'direction': direction, 'profile_axis_rows': len(group),
            'unique_food_groups': int(group.exact_name_group_id.nunique()),
            'unique_group_axis_pairs': len(group[['exact_name_group_id', 'axis_index']].drop_duplicates()),
            'positive_rows': int(group.target.gt(0).sum()), 'explicit_zero_rows': int(group.target.eq(0).sum()),
            'source_counts': group.source_key.value_counts().to_dict(),
            'axis_counts': group.canonical_name.value_counts().to_dict()})
    cholesterol = cases[(cases.direction == 'transformer_worse') & cases.canonical_name.eq('Cholesterol')]
    annotation = {'rows': len(cholesterol), 'all_foodb': bool(cholesterol.source_key.eq('foodb').all()),
        'all_positive': bool(cholesterol.target.gt(0).all()),
        'all_neural_below_retained_label': bool(cholesterol.neural.lt(cholesterol.target).all()),
        'rows_neural_at_least_500_fold_below_retained_label': int((cholesterol.neural*500 <= cholesterol.target).sum()),
        'unit_error_confirmed': False, 'label_correctness_confirmed': False}
    signed = pd.read_csv(diagnostic/'axis_signed_contributions.csv', float_precision='round_trip')
    signed = signed[signed.axis_index.isin(cases.axis_index.unique())]
    result = {'status': 'complete_existing_parent_case_selection_and_value_check',
        'seed': 20260922, 'case_rows': len(cases), 'all_40_rows_content_review_scope': 'Two extreme tails, not a representative sample.',
        'case_selection': '20 smallest and 20 largest profile-axis scaled-log absolute-error differences: Transformer minus fixed RF. Reconstructed from frozen validation jobs and saved predictions.',
        'case_keys_targets_predictions_errors_and_metadata_verified': True,
        'numeric_comparison_rtol': 1e-12, 'numeric_comparison_atol': 1e-12,
        'visible_context_and_near_name_columns_recomputed': False,
        'counts': counts, 'worst_tail_cholesterol_pattern': annotation,
        'full_panel_axis_contributions_not_case_sample_contributions': signed.to_dict('records'),
        'input_hashes': hashes, 'data_modified': False, 'baseline_refit': False,
        'model_forward_performed': False, 'new_training_performed': False,
        'complete_test_opened': False, 'new_model_selection_performed': False,
        'individual_food_names_targets_and_predictions_exported': False,
        'scope': 'Post-hoc descriptive review of seed22 only. Profile-level extremes are not source/group/axis-weighted contributions or independent samples. Possible label-scale problems remain hypotheses without upstream records; no relabeling, exclusion, new scoring protocol, or causal attribution follows from these cases.'}
    for relative, expected_hash in hashes.items():
        if digest(ROOT/relative) != expected_hash:
            raise ValueError('Input changed during review: ' + relative)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'counts': counts, 'cholesterol_pattern': annotation}))


if __name__ == '__main__':
    main()

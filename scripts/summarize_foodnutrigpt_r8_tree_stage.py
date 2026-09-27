"""Record one audited R8 tree and all three-task comparisons; no final tree selection."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--comparison-tag', required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    run = ROOT / 'output/v9_r8' / args.candidate
    manifest = read(run / 'run_manifest.json')
    audit = read(args.audit_dir / 'verification.json')
    assert manifest['status'] == audit['status'] == 'complete'
    assert not manifest['complete_test_opened'] and not audit['complete_test_opened']
    assert len(audit['records']) == 1
    assert audit['records'][0]['manifest_sha256'] == digest(run / 'run_manifest.json')
    assert audit['records'][0]['all187_registered_fitting_parameters_and_counts_verified']
    dim = manifest['active_name_dimensions']
    comparisons, counts, input_hashes = {}, {}, {}
    paths = [f'v9_r8_mlp{dim}_vs_{args.comparison_tag}_{task}_v1'
             for task in ['completion', 'name_only', 'retrieval']]
    paths += [f'v9_r8_{args.comparison_tag}_{task}_vs_knn_v1' for task in ['name', 'retrieval']]
    for name in paths:
        folder = ROOT / 'reports' / name
        comparison = read(folder / 'summary.json')
        assert not comparison['complete_test_opened'] and not comparison['confirmation']
        assert comparison['data_sha256'] == manifest['data_hash']
        if 'baseline_path' in comparison:
            for role in ['baseline', 'candidate']:
                assert digest(Path(comparison[role + '_path'])) == comparison[role + '_sha256']
            axis = pd.read_csv(folder / 'axis_paired_intervals.csv')
            axis = axis[axis.loss_group.eq('nutrition')]
            source = pd.read_csv(folder / 'source_metrics.csv').pivot(
                index='source', columns='role', values='scaled_log_mae')
            counts[name] = {'candidate_axes_point_worse': int((axis.candidate_minus_baseline > 0).sum()),
                'candidate_axes_interval_worse': int((axis.difference_95_low > 0).sum()),
                'candidate_axes_interval_better': int((axis.difference_95_high < 0).sum()),
                'candidate_sources_point_worse': int((source.candidate > source.baseline).sum()),
                'source_count': len(source)}
        else:
            for role in ['baseline', 'candidate']:
                for file in ['ranks', 'metrics']:
                    suffix = '.parquet' if file == 'ranks' else '.json'
                    assert digest(ROOT / comparison[role] / (file + suffix)) == comparison['hashes'][role][file]
        comparisons[name] = comparison
        input_hashes[str((folder / 'summary.json').relative_to(ROOT))] = digest(folder / 'summary.json')
    payload = {'status': 'one_tree_stage_complete_full_r8_pending', 'candidate': args.candidate,
        'manifest': manifest, 'manifest_sha256': digest(run / 'run_manifest.json'),
        'audit': audit, 'audit_sha256': digest(args.audit_dir / 'verification.json'),
        'metrics': read(run / 'metrics.json'), 'retrieval': read(run / 'retrieval/metrics.json'),
        'same_input_mlp_metrics': read(ROOT / f'output/v9_r8/mlp512_name{dim}/metrics.json'),
        'same_input_mlp_retrieval': read(ROOT / f'output/v9_r8/retrieval_mlp512_name{dim}/metrics.json'),
        'same_name_view_knn_metrics': read(ROOT / f'output/v9_r7/exact_name_knn{dim}/nutrition_metrics.json'),
        'same_name_view_knn_retrieval': read(ROOT / f'output/v9_r7/exact_name_knn{dim}/metrics.json'),
        'comparisons': comparisons, 'counts': counts, 'input_hashes': input_hashes,
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'scientific_confirmation': False, 'model_improvement_accepted': False,
        'final_tree_selected': False,
        'scope': 'A completed registered configuration, not a completed tuning budget or accepted neural improvement.'}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'summary.json', payload)
    print(payload['status'], args.candidate, counts)


if __name__ == '__main__':
    main()

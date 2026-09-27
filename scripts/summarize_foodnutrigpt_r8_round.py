"""Collect all twelve R8 configurations only after complete runs and independent post-audits."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_completion_input import execution_contract
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_r8_summary import select_registered_references

TREE_TAGS = {
    'xgb800d10_name32': 'xgb32', 'xgb800d10_name128': 'xgb128d10',
    'rf400leaf1half_name32': 'rf32leaf1half', 'rf400leaf1half_name128': 'rf128leaf1half',
    'xgb800d6_name128': 'xgb128d6', 'xgb800d14_name128': 'xgb128d14',
    'rf400leaf3half_name128': 'rf128leaf3half', 'rf400leaf1all_name128': 'rf128leaf1all',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def closed(metadata):
    fields = [key for key in ['test_opened', 'complete_test_opened'] if key in metadata]
    if not fields or any(metadata[key] for key in fields):
        raise ValueError('Explicit test-closed evidence is required.')


def path_for(item):
    return ROOT / item.get('reused_reference', f"output/v9_r8/{item['name']}")


def candidate_requirements(item):
    if item.get('kind') == 'mlp':
        return [ROOT / 'reports/v9_r8_mlp_stage_v2/summary.json',
                ROOT / 'reports/v9_r8_mlp_completed_audit_v2/verification.json']
    if 'reused_reference' in item:
        return [ROOT / 'reports/v9_r7_knn_completed_audit_v1/verification.json']
    tag = TREE_TAGS[item['name']]
    return [ROOT / f'reports/v9_r8_{tag}_completed_audit_v1/verification.json',
            ROOT / f'reports/v9_r8_{tag}_stage_v1/summary.json']


def requirements(registered):
    required = list(dict.fromkeys(path for item in registered for path in candidate_requirements(item)))
    for method in ['xgb_d10', 'rf_leaf1half']:
        for task in ['completion', 'name_only', 'retrieval']:
            required.append(ROOT / f'reports/v9_r8_{method}_128_vs32_{task}_v1/summary.json')
    return required


def verify_comparison(path):
    result = read(path)
    closed(result)
    if 'baseline_path' in result:
        for role in ['baseline', 'candidate']:
            assert digest(ROOT / result[role + '_path']) == result[role + '_sha256']
    else:
        for role in ['baseline', 'candidate']:
            for name, suffix in [('ranks', '.parquet'), ('metrics', '.json')]:
                assert digest(ROOT / result[role] / (name + suffix)) == result['hashes'][role][name]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.check_only and args.output_dir is None:
        parser.error('--output-dir is required unless --check-only is used.')
    contract, contract_path = execution_contract(ROOT)
    config_path = ROOT / 'experiments/foodnutrigpt_v9_research/r8/config.json'
    config = read(config_path)
    registered = config['candidates']
    assert registered == contract['registered_candidates'] and len(registered) == config['max_candidates'] == 12
    pending = []
    for item in registered:
        path = path_for(item) / 'run_manifest.json'
        if not path.exists():
            pending.append({'configuration': item['name'], 'manifest_state': 'absent'})
            continue
        try:
            state = read(path)['status']
        except json.JSONDecodeError:
            pending.append({'configuration': item['name'], 'manifest_state': 'transient_unreadable'})
            continue
        if state != 'complete':
            pending.append({'configuration': item['name'], 'manifest_state': state})
    missing = [str(path.relative_to(ROOT)) for path in requirements(registered) if not path.exists()]
    if not args.check_only and (pending or missing):
        raise RuntimeError(f'R8 is incomplete; no selection or output written: {pending}; missing={missing}')
    if not args.check_only and args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    registry = read(ROOT / config['name_cache_registry'])
    records, tree_stages, artifact_hashes = {}, {}, {}
    retrieval_identity = None
    for item in registered:
        name = item['name']
        if args.check_only and (name in {p['configuration'] for p in pending}
                or any(not path.exists() for path in candidate_requirements(item))):
            continue
        run = path_for(item)
        manifest = read(run / 'run_manifest.json')
        closed(manifest)
        assert manifest['status'] == 'complete' and manifest['data_hash'] == config['data_sha256']
        kind = item.get('kind', 'name_knn')
        dimension = item.get('active_dimensions', manifest.get('active_dimensions'))
        assert dimension in [32, 128]
        assert manifest['name_cache_hash'] == registry['manifest_sha256'][str(dimension)]
        if kind == 'name_knn':
            audit_path = ROOT / 'reports/v9_r7_knn_completed_audit_v1/verification.json'
            scores = read(run / 'nutrition_metrics.json')
            metrics = {'completion': scores, 'name_only': scores}
            retrieval_dir = run
            assert digest(run / 'nutrition_predictions.parquet') == manifest['nutrition_prediction_sha256']
        else:
            assert manifest['seed'] == config['seed'] and manifest['code_commit'] == contract['code_commit']
            assert manifest['execution_contract_sha256'] == digest(contract_path)
            assert manifest['active_name_dimensions'] == dimension and manifest['input_slots'] == 632
            metrics = read(run / 'metrics.json')
            if kind == 'mlp':
                audit_path = ROOT / 'reports/v9_r8_mlp_completed_audit_v2/verification.json'
                retrieval_dir = ROOT / f'output/v9_r8/retrieval_mlp512_name{dimension}'
                assert manifest['epoch_completed'] == 60 and manifest['parameter_count'] == 717052
                assert digest(run / 'best_model.pt') == manifest['checkpoint_hash']
            else:
                assert manifest['configuration'] == item and manifest['axes_completed'] == 187
                assert manifest['training_row_cap'] is None
                tag = TREE_TAGS[name]
                audit_path = ROOT / f'reports/v9_r8_{tag}_completed_audit_v1/verification.json'
                stage_path = ROOT / f'reports/v9_r8_{tag}_stage_v1/summary.json'
                stage = read(stage_path)
                assert stage['manifest_sha256'] == digest(run / 'run_manifest.json') and stage['metrics'] == metrics
                for relpath, expected in stage['input_hashes'].items():
                    path = ROOT / relpath
                    assert digest(path) == expected
                    assert verify_comparison(path) == stage['comparisons'][path.parent.name]
                tree_stages[name] = stage
                artifact_hashes[str(stage_path.relative_to(ROOT))] = digest(stage_path)
                retrieval_dir = run / 'retrieval'
        audit = read(audit_path)
        assert audit['status'] == 'complete'
        closed(audit)
        matching = [r for r in audit['records'] if Path(r['run'].replace('\\', '/')).as_posix() == run.relative_to(ROOT).as_posix()]
        assert len(matching) == 1 and matching[0]['manifest_sha256'] == digest(run / 'run_manifest.json')
        if kind in ['rf', 'xgb']:
            assert matching[0]['all187_registered_fitting_parameters_and_counts_verified']
            assert tree_stages[name]['audit_sha256'] == digest(audit_path)
        retrieval = read(retrieval_dir / 'metrics.json')
        closed(retrieval)
        assert digest(retrieval_dir / 'candidate_names.json') == retrieval['candidate_sha256']
        identity = {key: retrieval[key] for key in ['candidate_sha256', 'candidate_count', 'query_profiles',
                    'data_sha256', 'scoring', 'correct_answers']}
        if retrieval_identity is None:
            retrieval_identity = identity
        assert identity == retrieval_identity and identity['candidate_count'] == 49913
        for task, score in metrics.items():
            closed(score)
            assert score['nutrition']['axes'] == 142 and score['food_metabolome']['axes'] == 45 and score['all']['axes'] == 187
        artifact_hashes[str(audit_path.relative_to(ROOT))] = digest(audit_path)
        artifact_hashes[str((run / 'run_manifest.json').relative_to(ROOT))] = digest(run / 'run_manifest.json')
        records[name] = {'registered_configuration': item, 'status': 'complete', 'kind': kind,
            'active_name_dimensions': dimension, 'directory': str(run.relative_to(ROOT)),
            'manifest': manifest, 'manifest_sha256': digest(run / 'run_manifest.json'),
            'metrics': metrics, 'retrieval': retrieval, 'complete_test_opened': False}
    if args.check_only:
        print(json.dumps({'all_artifacts_present_and_manifests_complete': not pending and not missing,
            'completed_record_identity_and_post_hash_checks_passed': list(records),
            'pending_configurations': pending, 'missing_post_artifacts': missing,
            'scope': 'Read-only check of available completed records plus missing-artifact inventory. '
                     'Not process liveness, full numerical re-audit, baseline selection or completion certificate.'}))
        return
    selection = select_registered_references(registered, records)
    paired = {}
    for method in ['xgb_d10', 'rf_leaf1half']:
        for task in ['completion', 'name_only', 'retrieval']:
            path = ROOT / f'reports/v9_r8_{method}_128_vs32_{task}_v1/summary.json'
            paired[f'{method}_{task}'] = verify_comparison(path)
            artifact_hashes[str(path.relative_to(ROOT))] = digest(path)
    stronger = selection['stronger_same_input_tree']
    comparison = tree_stages[stronger]['comparisons'][f"v9_r8_mlp128_vs_{TREE_TAGS[stronger]}_completion_v1"]
    interval = comparison['paired_intervals']['scaled_log_mae']
    assert interval['baseline'] == records[stronger]['metrics']['completion']['nutrition']['scaled_log_mae']
    assert interval['candidate'] == records[selection['matched_neural']]['metrics']['completion']['nutrition']['scaled_log_mae']
    selection['selected_pair_intervals'] = comparison['paired_intervals']
    selection['conditional_food_group_interval_supports_gain'] = interval['relative_improvement_95_interval'][0] > 0
    rows = []
    for name, record in records.items():
        for task in ['completion', 'name_only']:
            metric = record['metrics'][task]
            rows.append({'candidate': name, 'kind': record['kind'], 'active_name_dimensions': record['active_name_dimensions'],
                'task': task, **metric['nutrition'],
                'metabolome45_scaled_log_mae': metric['food_metabolome']['scaled_log_mae'],
                'all187_scaled_log_mae': metric['all']['scaled_log_mae']})
    result = {'version': 'V9-R8', 'status': 'all_registered_outputs_audited_report_and_decision_pending',
        'registered_count': 12, 'new_training_runs': 10, 'reused_references': 2, 'records': records,
        'selection': selection, 'fixed_configuration_dimension_comparisons': paired,
        'tree_stages': tree_stages, 'artifact_hashes': artifact_hashes,
        'config_sha256_at_summary': digest(config_path), 'execution_contract_sha256': digest(contract_path),
        'script_sha256': digest(Path(__file__)),
        'selection_code_sha256': digest(ROOT / 'src/foodcomp/research_r8_summary.py'),
        'new_training_recorded_seconds': sum(r['manifest']['elapsed_seconds'] for r in records.values() if r['kind'] != 'name_knn'),
        'complete_test_opened': False, 'scientific_confirmation': False, 'model_improvement_accepted': False,
        'scope': 'Complete registered single-seed evidence collection, not final research-report closure or goal completion. '
                 'All RF/XGB candidates retained; strongest128 reference uses completion primary only. '
                 'Concurrent recorded wall times are not exclusive hardware time. Label provenance and external transfer remain unresolved.'}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'summary.json', result)
    pd.DataFrame(rows).to_csv(args.output_dir / 'nutrition_metrics.csv', index=False)
    retrieval_rows = [{'candidate': name, **metric} for name, record in records.items() for metric in record['retrieval']['metrics']]
    pd.DataFrame(retrieval_rows).to_csv(args.output_dir / 'retrieval_metrics.csv', index=False)
    print(json.dumps({'status': result['status'], 'selection': selection}))


if __name__ == '__main__':
    main()

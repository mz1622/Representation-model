"""Repeat both selected same-input R8 trees after complete search, regardless of neural superiority."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_completion_input import execution_contract
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_r8_summary import select_registered_references


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / 'experiments/foodnutrigpt_v9_research/final_report_v1/config.json'
    plan = read(plan_path)
    if plan['seeds'] != [20260922, 20260923, 20260924] or not plan['supersedes_superiority_completion_gate']:
        raise ValueError('Expected revised final-report replication plan.')
    frozen, contract_path = execution_contract(ROOT)
    summary = read(args.summary)
    if (summary['status'] != 'all_registered_outputs_audited_report_and_decision_pending'
            or summary['registered_count'] != 12 or summary['complete_test_opened']):
        raise ValueError('Complete audited R8 search is required before tree repetition.')
    if summary['execution_contract_sha256'] != digest(contract_path):
        raise ValueError('R8 execution identity changed.')
    for relative, expected in summary['artifact_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Stale R8 artifact: ' + relative)
    selection = select_registered_references(frozen['registered_candidates'], summary['records'])
    if selection['selected_by_method'] != summary['selection']['selected_by_method']:
        raise ValueError('Selected tree configuration differs from complete registered search.')
    if args.output_dir.exists() or args.audit_dir.exists():
        raise FileExistsError('Use new output and audit directories; never overwrite a partial queue.')
    args.output_dir.mkdir(parents=True)
    args.audit_dir.mkdir(parents=True)
    started = time.monotonic()
    receipt = {'status': 'running', 'purpose': 'fixed_selected_tree_seed_reproducibility',
        'plan_sha256': digest(plan_path), 'r8_summary_sha256': digest(args.summary),
        'execution_contract_sha256': digest(contract_path), 'script_sha256': digest(Path(__file__)),
        'audit_script_sha256': digest(ROOT / 'scripts/audit_foodnutrigpt_r8_completed.py'),
        'workspace_commit_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'selected_by_method': selection['selected_by_method'], 'seeds': plan['seeds'],
        'seed22_parents': {}, 'runs': [], 'commands': [], 'complete_test_opened': False,
        'superiority_is_not_an_execution_gate': True, 'final_report_complete': False}
    for kind, candidate in selection['selected_by_method'].items():
        parent = ROOT / summary['records'][candidate]['directory']
        manifest = read(parent / 'run_manifest.json')
        assert manifest['status'] == 'complete' and manifest['seed'] == 20260922
        assert manifest['kind'] == kind and manifest['active_name_dimensions'] == 128
        assert digest(parent / 'run_manifest.json') == summary['records'][candidate]['manifest_sha256']
        receipt['seed22_parents'][kind] = {'directory': str(parent),
            'manifest_sha256': digest(parent / 'run_manifest.json'), 'configuration': manifest['configuration']}
    status_path = args.output_dir / 'queue_manifest.json'
    write_json(status_path, receipt)
    try:
        for kind in ['xgb', 'rf']:
            candidate = selection['selected_by_method'][kind]
            parent = read(Path(receipt['seed22_parents'][kind]['directory']) / 'run_manifest.json')
            for seed in plan['seeds'][1:]:
                if digest(plan_path) != receipt['plan_sha256']:
                    raise ValueError('Replication plan changed during execution.')
                execution_contract(ROOT)
                run = args.output_dir / f'{kind}_seed{seed}'
                audit_dir = args.audit_dir / f'{kind}_seed{seed}'
                command = [sys.executable, 'scripts/run_foodnutrigpt_v9_r8_trees.py',
                    '--candidate', candidate, '--seed', str(seed), '--n-jobs', '4', '--output-dir', str(run)]
                receipt.update(stage=f'training_{kind}_seed{seed}', elapsed_seconds=time.monotonic() - started)
                receipt['commands'].append(command)
                write_json(status_path, receipt)
                with (args.output_dir / f'{kind}_seed{seed}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
                manifest = read(run / 'run_manifest.json')
                assert manifest['status'] == 'complete' and manifest['seed'] == seed
                for field in ['configuration', 'code_commit', 'code_hashes', 'execution_contract_sha256',
                              'data_hash', 'panel_hash', 'name_cache_hash', 'input_slots',
                              'active_name_dimensions', 'fit_n_jobs', 'rf_prediction_n_jobs', 'training_row_cap']:
                    assert manifest[field] == parent[field], field
                if digest(ROOT / 'scripts/audit_foodnutrigpt_r8_completed.py') != receipt['audit_script_sha256']:
                    raise ValueError('Tree audit implementation changed during repetitions.')
                command = [sys.executable, 'scripts/audit_foodnutrigpt_r8_completed.py',
                    '--kind', 'tree', '--run', str(run), '--output-dir', str(audit_dir)]
                receipt.update(stage=f'auditing_{kind}_seed{seed}')
                receipt['commands'].append(command)
                write_json(status_path, receipt)
                with (args.output_dir / f'{kind}_seed{seed}_audit.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
                audit = read(audit_dir / 'verification.json')
                assert audit['status'] == 'complete' and not audit['complete_test_opened']
                assert len(audit['records']) == 1
                assert audit['records'][0]['manifest_sha256'] == digest(run / 'run_manifest.json')
                receipt['runs'].append({'kind': kind, 'seed': seed, 'candidate': candidate, 'directory': str(run),
                    'manifest_sha256': digest(run / 'run_manifest.json'), 'audit_directory': str(audit_dir),
                    'audit_sha256': digest(audit_dir / 'verification.json')})
                write_json(status_path, receipt)
        receipt.update(status='complete', stage='all_four_tree_repeats_and_independent_audits_complete',
            elapsed_seconds=time.monotonic() - started)
        write_json(status_path, receipt)
        print(json.dumps({'status': 'complete', 'selected_by_method': selection['selected_by_method']}))
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started)
        write_json(status_path, receipt)
        raise


if __name__ == '__main__':
    main()


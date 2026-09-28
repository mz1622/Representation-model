"""Repeat the fixed R8 neural recipe; no new configuration selection or acceptance."""
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


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def recipe(manifest):
    return {k: v for k, v in manifest['args'].items() if k not in {'seed', 'output_dir'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / 'experiments/foodnutrigpt_v9_research/final_report_v1/config.json'
    plan_hash = digest(plan_path)
    plan = read(plan_path)
    if plan['neural_candidate'] != 'mlp512_name128' or plan['seeds'] != [20260922, 20260923, 20260924]:
        raise ValueError('Final-report neural recipe or registered seeds changed.')
    frozen, contract_path = execution_contract(ROOT)
    parent = ROOT / 'output/v9_r8/mlp512_name128'
    parent_manifest = read(parent / 'run_manifest.json')
    if (parent_manifest['status'] != 'complete' or parent_manifest['seed'] != 20260922
            or parent_manifest['test_opened'] or parent_manifest['code_commit'] != frozen['code_commit']
            or parent_manifest['execution_contract_sha256'] != digest(contract_path)):
        raise ValueError('Completed frozen seed22 parent required.')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    receipt = {'status': 'running', 'purpose': 'fixed_configuration_seed_reproducibility',
        'plan_sha256': plan_hash, 'script_sha256': digest(Path(__file__)),
        'workspace_commit_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seed22_parent': str(parent.relative_to(ROOT)),
        'seed22_manifest_sha256': digest(parent / 'run_manifest.json'),
        'execution_contract_sha256': digest(contract_path), 'fixed_recipe': recipe(parent_manifest),
        'runs': [], 'commands': [], 'complete_test_opened': False,
        'independent_numerical_audit_complete': False, 'final_report_complete': False}
    status_path = args.output_dir / 'queue_manifest.json'
    write_json(status_path, receipt)
    try:
        for seed in plan['seeds'][1:]:
            if digest(plan_path) != plan_hash:
                raise ValueError('Final-report plan changed during registered repeats.')
            execution_contract(ROOT)
            run = args.output_dir / f'mlp_seed{seed}'
            settings = dict(parent_manifest['args'], seed=seed, output_dir=str(run))
            command = [sys.executable, 'scripts/train_foodnutrigpt_v9_r8_completion.py']
            for key, value in settings.items():
                flag = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        command.append(flag)
                else:
                    command.extend([flag, str(value)])
            receipt.update(stage=f'training_seed{seed}', elapsed_seconds=time.monotonic() - started)
            receipt['commands'].append(command)
            write_json(status_path, receipt)
            with (args.output_dir / f'train_seed{seed}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            manifest = read(run / 'run_manifest.json')
            if manifest['status'] != 'complete' or recipe(manifest) != recipe(parent_manifest) or manifest['seed'] != seed:
                raise ValueError('Repeated neural run did not preserve the registered recipe.')
            for field in ['code_commit', 'data_hash', 'panel_hash', 'name_cache_hash', 'execution_contract_sha256', 'parameter_count']:
                if manifest[field] != parent_manifest[field]:
                    raise ValueError('Parent identity differs: ' + field)
            normalize = lambda hashes: {key.replace('\\', '/'): value for key, value in hashes.items()}
            if normalize(manifest['code_hashes']) != normalize(parent_manifest['code_hashes']):
                raise ValueError('Training implementation changed across seeds.')
            command = [sys.executable, 'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py',
                '--checkpoint', str(run / 'best_model.pt'), '--output-dir', str(run / 'retrieval')]
            receipt.update(stage=f'retrieval_seed{seed}')
            receipt['commands'].append(command)
            write_json(status_path, receipt)
            with (args.output_dir / f'retrieval_seed{seed}.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            retrieval = read(run / 'retrieval/metrics.json')
            if retrieval['complete_test_opened'] or retrieval['candidate_count'] != 49913:
                raise ValueError('Retrieval protocol changed.')
            receipt['runs'].append({'seed': seed, 'directory': str(run),
                'manifest_sha256': digest(run / 'run_manifest.json'),
                'checkpoint_sha256': digest(run / 'best_model.pt'),
                'retrieval_metrics_sha256': digest(run / 'retrieval/metrics.json')})
            write_json(status_path, receipt)
        receipt.update(status='complete', stage='both_new_seeds_three_tasks_ready_for_independent_audit',
            elapsed_seconds=time.monotonic() - started)
        write_json(status_path, receipt)
        print(json.dumps({'status': receipt['status'], 'runs': receipt['runs']}), flush=True)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - started)
        write_json(status_path, receipt)
        raise


if __name__ == '__main__':
    main()

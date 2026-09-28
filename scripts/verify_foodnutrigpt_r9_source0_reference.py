"""Run source-weight functional checks under a deterministic gradient reference.

This process-only setting does not alter the production trainer or its recipe.
The original functional script and registration remain byte-for-byte unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    diagnostic = ROOT/'reports/v9_r9_source0_gradient_reference_v1/summary.json'
    record = json.loads(diagnostic.read_text(encoding='utf-8'))
    deterministic = [row for row in record['records'] if row['deterministic_algorithms']]
    if {row['device'] for row in deterministic} != {'cpu','cuda'} or any(
            row['same_objective_repeat_differences'] or row['source_weight_contrast_differences']
            for row in deterministic):
        raise ValueError('Require exact CPU/GPU deterministic reference gradients.')
    stochastic = [row for row in record['records'] if not row['deterministic_algorithms']]
    if len(stochastic) != 1 or not stochastic[0]['same_objective_repeat_differences']:
        raise ValueError('Require observed nondeterministic same-objective repeat differences.')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.use_deterministic_algorithms(True)
    runpy.run_path(str(ROOT/'scripts/verify_foodnutrigpt_r9_source0_functional.py'),run_name='__main__')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'status':'complete_deterministic_functional_reference',
        'functional_sha256':sha(args.output_dir/'verification.json'),
        'gradient_reference_sha256':sha(diagnostic), 'wrapper_sha256':sha(Path(__file__)),
        'deterministic_algorithms':True, 'cublas_workspace_config':':4096:8',
        'exact_checks_loosened':False, 'production_trainer_changed':False,
        'formal_training_started':False, 'data_modified':False, 'baseline_refit':False,
        'complete_test_opened':False,
        'scope':'Exact initial shared-gradient checks and the 50-step functional exercise ran in this deterministic reference process. Ordinary CUDA reduction order differed even for repeated identical objectives; no tolerance was widened. Formal training retains the parent backend settings and independently repeats its small-batch check.'}
    (args.output_dir/'reference_execution.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'reference_status':receipt['status']}))


if __name__ == '__main__':
    main()

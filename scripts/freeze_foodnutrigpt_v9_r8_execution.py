"""Freeze committed executable bytes and audited inputs before any R8 formal run."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json, VERSION
from foodcomp.research_completion_input import EXECUTION_FILES, require_r7_completion


def main():
    require_r7_completion(ROOT)
    destination = ROOT / 'reports/v9_r8_execution_contract_v1'
    if destination.exists(): raise FileExistsError(destination)
    # All executable files must be tracked and identical to their committed state.
    subprocess.run(['git', 'ls-files', '--error-unmatch', *EXECUTION_FILES], cwd=ROOT, check=True, capture_output=True)
    for command in [['git', 'diff', '--quiet', '--', *EXECUTION_FILES],
                    ['git', 'diff', '--cached', '--quiet', '--', *EXECUTION_FILES]]:
        subprocess.run(command, cwd=ROOT, check=True)
    inputs = [f'data/processed/{VERSION}/manifest.json',
        'reports/v9_r7_name_cache_v1/verification.json',
        'reports/v9_r8_input_panels_v2/verification.json',
        'reports/v9_r8_input_functional_v2/verification.json',
        'reports/v9_r8_tree_functional_v1/verification.json']
    for name in inputs[1:]:
        record = json.loads((ROOT / name).read_text(encoding='utf-8'))
        assert record['status'] == 'complete', name
    registry = json.loads((ROOT / inputs[1]).read_text(encoding='utf-8'))
    for dims in [32, 128]:
        inputs.extend([f'data/processed/foodnutrigpt_v9_r8_tasks_name{dims}_v2/manifest.json',
            str((Path(registry['paths'][str(dims)]) / 'manifest.json').as_posix())])
    protocol = json.loads((ROOT / 'experiments/foodnutrigpt_v9_research/r8/config.json').read_text(encoding='utf-8'))
    destination.mkdir()
    write_json(destination / 'manifest.json', {'status': 'frozen',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_hashes': {name: digest(ROOT / name) for name in EXECUTION_FILES},
        'input_hashes': {name: digest(ROOT / name) for name in inputs},
        'registered_candidates': protocol['candidates'], 'registered_mlp': protocol['mlp'],
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
            'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'complete_test_opened': False,
        'scope': 'Fitted code bytes and inputs are frozen. Each run also records workspace HEAD; later documentation/audit commits do not alter the implementation revision.'})
    print(destination)


if __name__ == '__main__':
    main()

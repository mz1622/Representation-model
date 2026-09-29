"""Run the registered real-data functional check in an isolated deterministic diagnostic process.

These settings do not apply to formal training. The initial nondeterministic
gradient comparison failure is preserved separately rather than retried for luck.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json

def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    path=ROOT/'scripts/verify_foodnutrigpt_r9_lr2e4_functional.py'
    spec=importlib.util.spec_from_file_location('registered_lr2e4_functional',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    output=Path(sys.argv[sys.argv.index('--output-dir')+1])
    record_path=output/'verification.json'
    record=json.loads(record_path.read_text(encoding='utf-8'))
    record['implementation_hashes'][Path(__file__).relative_to(ROOT).as_posix()]=digest(Path(__file__))
    record['diagnostic_backend']={
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),
        'memory_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp':torch.backends.cuda.math_sdp_enabled(),
        'scope':'Isolated functional diagnostic only; formal training retains parent nondeterministic backend and no CUBLAS environment override.'}
    record['initial_production_backend_exact_gradient_check_failed']=True
    record['failed_diagnostic_record']='reports/v9_r9_lr2e4_functional_failure_v1/record.json'
    record['failed_diagnostic_sha256']=digest(ROOT/record['failed_diagnostic_record'])
    record['scope']='Training-row implementation and learnability under isolated deterministic diagnostic backend, not production bitwise trajectory equality or validation performance. Initial production-backend backward check had tiny unequal gradients and is preserved. Formal method is unchanged.'
    write_json(record_path,record)
    print(json.dumps({'status':'complete_deterministic_diagnostic_only','formal_backend_changed':False,
        'small_batch_overfit':record['small_batch_overfit'],'verification_sha256':digest(record_path)}))

if __name__=='__main__':main()

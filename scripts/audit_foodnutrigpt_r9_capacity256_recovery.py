"""Audit the preserved48-epoch prefix and complete the original capacity replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_transformer_r9_capacity256 import verify_bindings
from foodcomp.research_transformer_r9_capacity256_recovery import load_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/recovery1/config.json'
    plan, original, prefix, _, hashes = load_recovery(ROOT, config)
    manifest_path = args.run/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (args.run.resolve() != (ROOT/plan['output_run']).resolve()
            or manifest['status'] != 'complete' or manifest['epoch_completed'] != 60
            or manifest['spec'] != original['spec']
            or manifest['original_manifest_sha256'] != digest(ROOT/plan['source_run']/'run_manifest.json')
            or manifest['recovery_config_sha256'] != digest(config)
            or manifest['original_completed_epochs'] != 48 or manifest['resumed_first_epoch'] != 49
            or manifest['numerical_recipe_changed_by_recovery']
            or not manifest['recovery_model_optimizer_scheduler_and_rng_loaded_exact']):
        raise ValueError('Complete unmodified candidate6 recovery required')
    verify_bindings(ROOT, manifest['recovery_input_hashes'])
    history = pd.read_csv(args.run/'history.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(prefix, history.iloc[:48].reset_index(drop=True), check_exact=True)
    if digest(args.run/'best_through_epoch_020.pt') != digest(ROOT/plan['source_run']/'best_through_epoch_020.pt'):
        raise ValueError('Original epoch20 checkpoint changed')
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/audit_foodnutrigpt_r9_capacity256.py'),
        '--run', str(args.run), '--output-dir', str(args.output_dir)], cwd=ROOT, check=True)
    verify_bindings(ROOT, hashes)
    write_json(args.output_dir/'recovery_verification.json', {
        'status': 'complete_registered_capacity256_recovery_audit',
        'run': str(args.run), 'manifest_sha256': digest(manifest_path),
        'recovery_config_sha256': digest(config),
        'method_audit_sha256': digest(args.output_dir/'method_verification.json'),
        'full_replay_sha256': digest(args.output_dir/'verification.json'),
        'original48_epoch_history_exact': True, 'original_files_unchanged': True,
        'original20_epoch_snapshot_exact': True, 'full60_epoch_original_schedule_verified': True,
        'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
        'input_hashes': hashes, 'script_sha256': digest(Path(__file__)),
        'scope': 'Operational recovery of saved epoch48; full original audit reused. Saved-state and prefix identities are verified. No uninterrupted GPU counterfactual trajectory or unknown lost compute cost is asserted.'})


if __name__ == '__main__':
    main()

"""Audit the declared encoder RMSNorm change, then reuse full independent R9 replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_transformer_r9 import frozen_inputs
from foodcomp.research_transformer_r9_rmsnorm import load_method, verify_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    frozen,freeze_path=frozen_inputs(ROOT)
    config_path=ROOT/'experiments/foodnutrigpt_v9_research/r9/rmsnorm_v1/config_v2.json'
    plan,parent,spec,bindings=load_method(ROOT,config_path,frozen,freeze_path)
    manifest_path=args.run/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['status']!='complete' or manifest['candidate']!=plan['candidate']
            or manifest['spec']!=spec or manifest['seed']!=plan['seed']
            or manifest['epoch_completed']!=60 or manifest['numerical_recipe_changes']!=['encoder_normalization']):
        raise ValueError('Completed registered rmsnorm recipe required')
    for key in ['training_tasks','observed_target_cells','data_hash','panel_hash','name_cache_hash']:
        if manifest[key]!=parent[key]:raise ValueError('Changed control: '+key)
    functional=json.loads((ROOT/'reports/v9_r9_rmsnorm_functional_v1/verification.json').read_text(encoding='utf-8'))
    if manifest['initial_state_sha256']!=functional['initial_state_sha256']:
        raise ValueError('Structural initialization differs from preflight')
    if manifest['parameter_count']!=parent['parameter_count']-1344:
        raise ValueError('Unexpected structural parameter delta')
    history=pd.read_csv(args.run/'history.csv',float_precision='round_trip')
    old=pd.read_csv(ROOT/plan['parent_run']/'history.csv',float_precision='round_trip')
    for key in ['epoch','training_tasks','observed_target_cells','training_order_sha256']:
        np.testing.assert_array_equal(history[key].to_numpy(),old[key].to_numpy())
    np.testing.assert_array_equal(history.learning_rate, old.learning_rate)
    for name in ['best_model.pt','latest_training_state.pt']:
        saved=torch.load(args.run/name,map_location='cpu',weights_only=True)
        if (saved['spec']!=spec or saved['config']['dropout']!=.15
                or saved['kind']!='transformer_rmsnorm'):
            raise ValueError('Checkpoint architecture differs from registered intervention')
        for key,tensor in saved['model_state'].items():
            if not torch.isfinite(tensor).all():raise FloatingPointError(key)
    verify_bindings(ROOT,manifest['method_evidence_hashes'])
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/audit_foodnutrigpt_r9_rmsnorm_replay.py'),
                    '--run',str(args.run),'--output-dir',str(args.output_dir)],cwd=ROOT,check=True)
    verify_bindings(ROOT,bindings)
    frozen_inputs(ROOT)
    write_json(args.output_dir/'method_verification.json',{
        'status':'complete_registered_rmsnorm_method_audit','config_sha256':digest(config_path),
        'manifest_sha256':digest(manifest_path),'full_replay_sha256':digest(args.output_dir/'verification.json'),
        'script_sha256':digest(Path(__file__)),
        'shared_parent_initial_tensors_verified_in_preflight':True,
        'same_all60_orders_and_exposures':True,
        'both_saved_checkpoint_architecture_configs_verified':True,
        'all60_learning_rates_exactly_match_parent':True,
        'sole_numerical_recipe_change':'seven_encoder_layernorms_to_rmsnorm',
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False})
    print(json.dumps({'status':'complete_registered_rmsnorm_method_audit'}))


if __name__=='__main__':main()


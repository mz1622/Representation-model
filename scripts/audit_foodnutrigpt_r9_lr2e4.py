"""Audit the declared learning-rate change, then reuse full independent R9 replay."""
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
from foodcomp.research_transformer_r9_lr2e4 import load_method, verify_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    frozen,freeze_path=frozen_inputs(ROOT)
    config_path=ROOT/'experiments/foodnutrigpt_v9_research/r9/lr2e4_v1/config.json'
    plan,parent,spec,bindings=load_method(ROOT,config_path,frozen,freeze_path)
    manifest_path=args.run/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['status']!='complete' or manifest['candidate']!=plan['candidate']
            or manifest['spec']!=spec or manifest['seed']!=plan['seed']
            or manifest['epoch_completed']!=60 or manifest['numerical_recipe_changes']!=['learning_rate']):
        raise ValueError('Completed registered lr2e4 recipe required')
    for key in ['initial_state_sha256','parameter_count','trainable_parameter_count',
                'training_tasks','observed_target_cells','data_hash','panel_hash','name_cache_hash']:
        if manifest[key]!=parent[key]:raise ValueError('Changed control: '+key)
    history=pd.read_csv(args.run/'history.csv',float_precision='round_trip')
    old=pd.read_csv(ROOT/plan['parent_run']/'history.csv',float_precision='round_trip')
    for key in ['epoch','training_tasks','observed_target_cells','training_order_sha256']:
        np.testing.assert_array_equal(history[key].to_numpy(),old[key].to_numpy())
    np.testing.assert_allclose(history.learning_rate, (2 / 3) * old.learning_rate, rtol=1e-12, atol=1e-15)
    for name in ['best_model.pt','latest_training_state.pt']:
        saved=torch.load(args.run/name,map_location='cpu',weights_only=True)
        for key,tensor in saved['model_state'].items():
            if not torch.isfinite(tensor).all():raise FloatingPointError(key)
    verify_bindings(ROOT,manifest['method_evidence_hashes'])
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/audit_foodnutrigpt_r9_completed_utf8.py'),
                    '--run',str(args.run),'--output-dir',str(args.output_dir)],cwd=ROOT,check=True)
    verify_bindings(ROOT,bindings)
    frozen_inputs(ROOT)
    write_json(args.output_dir/'method_verification.json',{
        'status':'complete_registered_lr2e4_method_audit','config_sha256':digest(config_path),
        'manifest_sha256':digest(manifest_path),'full_replay_sha256':digest(args.output_dir/'verification.json'),
        'script_sha256':digest(Path(__file__)),
        'same_parent_initial_state_and_parameter_count':True,
        'same_all60_orders_and_exposures':True,
        'all60_learning_rates_two_thirds_parent_within_float_tolerance':True,
        'sole_numerical_recipe_change':'learning_rate3e-4_to2e-4',
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False})
    print(json.dumps({'status':'complete_registered_lr2e4_method_audit'}))


if __name__=='__main__':main()

"""Verify the registered PDF capacity group before full independent R9 replay."""
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
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_transformer_r9 import frozen_inputs,make_transformer
from foodcomp.research_transformer_r9_capacity256 import load_method,verify_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    frozen,fp=frozen_inputs(ROOT)
    config_path=ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/config.json'
    plan,parent,spec,bindings=load_method(ROOT,config_path,frozen,fp)
    manifest_path=args.run/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if (manifest['status']!='complete' or manifest['candidate']!=plan['candidate'] or manifest['spec']!=spec
            or manifest['seed']!=plan['seed'] or manifest['epoch_completed']!=60
            or manifest['numerical_recipe_changes']!=['d_model','n_heads','feedforward_dim']
            or manifest['same_seed_parent_initialization_exact'] is not False
            or manifest['parent_initial_state_sha256']!=parent['initial_state_sha256']):
        raise ValueError('Complete registered capacity group required')
    functional=json.loads((ROOT/'reports/v9_r9_capacity256_functional_v1/verification.json').read_text())
    for key in ['initial_state_sha256','parameter_count','trainable_parameter_count']:
        if manifest[key]!=functional[key]:raise ValueError('Changed verified capacity model: '+key)
    for key in ['training_tasks','observed_target_cells','data_hash','panel_hash','name_cache_hash']:
        if manifest[key]!=parent[key]:raise ValueError('Changed control: '+key)
    data=ResearchData(ROOT/'data/processed'/VERSION)
    torch.manual_seed(spec['seed']);initial,_=make_transformer(data,128,spec)
    if state_fingerprint(initial)!=manifest['initial_state_sha256']:raise ValueError('Initial capacity fingerprint not reproduced')
    del initial,data
    history=pd.read_csv(args.run/'history.csv',float_precision='round_trip')
    old=pd.read_csv(ROOT/plan['parent_run']/'history.csv',float_precision='round_trip')
    for key in ['epoch','training_tasks','observed_target_cells','training_order_sha256','learning_rate']:
        np.testing.assert_array_equal(history[key].to_numpy(),old[key].to_numpy())
    verify_bindings(ROOT,manifest['method_evidence_hashes'])
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/audit_foodnutrigpt_r9_completed_utf8.py'),
        '--run',str(args.run),'--output-dir',str(args.output_dir)],cwd=ROOT,check=True)
    verify_bindings(ROOT,bindings);frozen_inputs(ROOT)
    write_json(args.output_dir/'method_verification.json',{
        'status':'complete_registered_capacity256_method_audit','config_sha256':digest(config_path),
        'manifest_sha256':digest(manifest_path),'full_replay_sha256':digest(args.output_dir/'verification.json'),
        'script_sha256':digest(Path(__file__)),'initial_capacity_fingerprint_reconstructed_exact':True,
        'parameter_counts_match_preflight':True,'same_all60_orders_exposures_and_learning_rates':True,
        'sole_intervention_group':['d_model192_to256','n_heads6_to8','feedforward768_to1024'],
        'same_seed_parent_initialization_exact':False,'data_modified':False,'baseline_refit':False,'complete_test_opened':False})
    print(json.dumps({'status':'complete_registered_capacity256_method_audit'}))


if __name__=='__main__':main()

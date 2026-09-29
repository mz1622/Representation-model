"""Estimate runtime from interleaved training-only throughput, without validation scoring."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import FamilyPanel,fingerprint_array
from foodcomp.research_completion_input import completion_view
from foodcomp.research_transformer_r9 import frozen_inputs,make_transformer,transformer_loss
from foodcomp.research_transformer_r9_capacity256 import load_method,verify_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():raise RuntimeError('Runtime estimate must use the actual GPU')
    torch.set_num_threads(4)
    frozen,freeze_path=frozen_inputs(ROOT)
    config_path=ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/config.json'
    plan,parent,spec,bindings=load_method(ROOT,config_path,frozen,freeze_path)
    functional_path=ROOT/'reports/v9_r9_capacity256_functional_v1/verification.json'
    functional=json.loads(functional_path.read_text())
    if functional['status']!='complete' or functional['plan_sha256']!=digest(config_path):
        raise ValueError('Verified current capacity implementation required')
    verify_bindings(ROOT,functional['implementation_hashes'])
    data=ResearchData(ROOT/'data/processed'/VERSION)
    text,_,panel_root=completion_view(data,ROOT,32)
    device=torch.device('cuda');panel=FamilyPanel(data,text,panel_root,device)
    warmup,steps=8,96
    order=np.random.default_rng(spec['seed']+1).permutation(len(panel.rows))[:(warmup+steps)*spec['batch_size']]
    assert data.profiles.iloc[panel.rows[order]].partition.eq('train').all()
    def run(role,recipe,repetition):
        torch.manual_seed(spec['seed'])
        model,config=make_transformer(data,128,recipe);model.to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
        def step(index):
            indices=order[index*recipe['batch_size']:(index+1)*recipe['batch_size']]
            batch=panel.batch(indices);optimizer.zero_grad(set_to_none=True)
            loss=transformer_loss(model,batch,config,'mae');loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite timing-step gradient')
            optimizer.step()
            # Include the same scalar synchronizations and exposure accounting as formal training.
            return float(loss.detach()),int(batch['target'].sum()),float(norm),int(norm>recipe['gradient_clip'])
        for index in range(warmup):step(index)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        exposure=0
        for index in range(warmup,warmup+steps):exposure+=step(index)[1]
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        result={'role':role,'repetition':repetition,'timed_steps':steps,'elapsed_seconds':elapsed,
            'seconds_per_step':elapsed/steps,'observed_targets_timed':exposure,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        return result
    records=[]
    for repetition,roles in [(1,['parent','capacity256']),(2,['capacity256','parent'])]:
        for role in roles:records.append(run(role,parent['spec'] if role=='parent' else spec,repetition))
    medians={role:float(np.median([row['seconds_per_step'] for row in records if row['role']==role]))
             for role in ['parent','capacity256']}
    ratio=medians['capacity256']/medians['parent']
    recent_path=ROOT/'output/v9_r9_methods/tf192_mae_source0_lr3e4_60/run_manifest.json'
    recent=json.loads(recent_path.read_text())
    if recent['status']!='complete' or recent['epoch_completed']!=60:raise ValueError('Complete recent timing control required')
    base_seconds=recent['elapsed_seconds']
    estimate_seconds=base_seconds*ratio*1.10
    delay_minutes=int(math.ceil(estimate_seconds/60)+10)
    verify_bindings(ROOT,bindings);frozen_inputs(ROOT)
    args.output_dir.mkdir(parents=True)
    result={'status':'complete_training_only_runtime_estimate','records':records,'median_seconds_per_step':medians,
        'capacity_to_parent_step_ratio':ratio,'reference_run_elapsed_seconds':base_seconds,
        'estimated_training_seconds':estimate_seconds,'first_followup_delay_minutes':delay_minutes,
        'safety_factor':1.10,'followup_processing_allowance_minutes':10,
        'training_order_sha256':fingerprint_array(order),'warmup_steps_per_round':warmup,
        'formal_training_started':False,'candidate_validation_evaluated':False,'optimizer_steps_are_discarded_diagnostic':True,
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False,
        'input_hashes':{**bindings,functional_path.relative_to(ROOT).as_posix():digest(functional_path),
            recent_path.relative_to(ROOT).as_posix():digest(recent_path)},'script_sha256':digest(Path(__file__)),
        'scope':'Two interleaved repeats per width, identical104 training batches,8 warmup/96 timed steps at batch64. No validation scoring, checkpoint selection or formal candidate training. Recent60-epoch wall time multiplied by measured step ratio and10percent margin is a planning estimate; evaluation, thermal and CPU costs may scale differently.'}
    write_json(args.output_dir/'summary.json',result)
    print(json.dumps({key:result[key] for key in ['status','capacity_to_parent_step_ratio','estimated_training_seconds','first_followup_delay_minutes']}))


if __name__=='__main__':main()

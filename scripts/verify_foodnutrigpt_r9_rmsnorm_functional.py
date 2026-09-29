"""Training-row RMSNorm checks and runtime estimate; no validation model selection."""
import argparse
import ast
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import FamilyPanel
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_completion_input import completion_view
from foodcomp.research_transformer_r9 import frozen_inputs, make_transformer as control, transformer_loss
from foodcomp.research_transformer_r9_drop25 import dropout_sites
from foodcomp.research_transformer_r9_rmsnorm import (
    PLAN, make_transformer, replace_encoder_norms, norm_sites, load_method,
    verify_bindings, RMSNormNutritionModel)


def loop(path):
    nodes=[n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
           if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='offset']
    assert len(nodes)==1
    return ast.dump(nodes[0],include_attributes=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    local=ROOT/'data/local/research_diagnostics/v9_r9_rmsnorm_functional_v1'
    if args.output_dir.exists() or local.exists(): raise FileExistsError('Preserve prior preflight')
    frozen,freeze_path=frozen_inputs(ROOT)
    plan,parent,spec,bindings=load_method(ROOT,ROOT/PLAN,frozen,freeze_path)
    assert loop(ROOT/'scripts/train_foodnutrigpt_v9_r9_transformer.py')==loop(ROOT/'scripts/train_foodnutrigpt_v9_r9_rmsnorm.py')
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=ResearchData(ROOT/'data/processed'/VERSION)
    text,cache,panel_root=completion_view(data,ROOT,32)
    assert text.shape[1]==128 and (text[:,32:]==0).all()
    panel=FamilyPanel(data,text,panel_root,device)
    torch.manual_seed(spec['seed'])
    old,conf=control(data,128,parent['spec'])
    rng=torch.get_rng_state()
    assert state_fingerprint(old)==parent['initial_state_sha256']
    torch.manual_seed(spec['seed'])
    new,config=make_transformer(data,128,spec)
    assert torch.equal(rng,torch.get_rng_state())
    initial_hash=state_fingerprint(new)
    for key,value in new.state_dict().items():
        assert torch.equal(value,old.state_dict()[key]),key
    removed=sorted(set(old.state_dict())-set(new.state_dict()))
    assert len(removed)==7 and all('norm' in name and name.endswith('.bias') for name in removed)
    assert sum(p.numel() for p in old.parameters())-sum(p.numel() for p in new.parameters())==1344
    assert dropout_sites(old)==dropout_sites(new)
    sites=norm_sites(new)
    assert len([s for s in sites.values() if s['type']=='RMSNorm'])==7
    assert sites['text_projection.0']['type']=='LayerNorm'
    bridge=replace_encoder_norms(copy.deepcopy(old),rms=False)
    old.to(device).eval(); new.to(device).eval(); bridge.to(device).eval()
    batch=panel.batch(np.arange(64))
    with torch.no_grad():
        baseline=old(batch)['amount_normalized']
        bridge_output=bridge(batch)['amount_normalized']
        # Default GPU fused LayerNorm and explicit implementation have distinct
        # rounding. Record it, then compare both with the same unfused path;
        # restore the original backend before all RMSNorm checks and training.
        backend_fastpath=torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            reference_explicit=old(batch)['amount_normalized']
            bridge_explicit=bridge(batch)['amount_normalized']
            torch.testing.assert_close(reference_explicit,bridge_explicit,rtol=0,atol=0)
        finally:
            torch.backends.mha.set_fastpath_enabled(backend_fastpath)
        first_output=new(batch)['amount_normalized']
        assert not torch.equal(first_output,baseline)
        changed=copy.deepcopy(batch)
        changed['value'][batch['masked']]=123456
        changed['source'][:]=0
        changed['target']=~changed['target']
        assert torch.equal(first_output,new(changed)['amount_normalized'])
    bridge_max=float((baseline-bridge_output).abs().max())
    del bridge
    args.output_dir.mkdir(parents=True); local.mkdir(parents=True)
    saved={'version':'V9-R9','kind':'transformer_rmsnorm','spec':spec,'config':asdict(config),
        'model_state':new.state_dict(),'text_dim':128,'seed':spec['seed'],
        'data_root':data.root.relative_to(ROOT).as_posix(),'view':data.view,
        'name_cache':cache.relative_to(ROOT).as_posix(),'data_hash':frozen['data_hash'],
        'name_cache_hash':frozen['name_cache_hash']}
    path=local/'functional_checkpoint.pt'
    torch.save(saved,path)
    loaded=RMSNormNutritionModel(path,ROOT,str(device))
    with torch.no_grad():
        assert torch.equal(first_output,loaded.model(batch)['amount_normalized'])
    row=int(data.train[0]); unseen=data.targets[~data.observed[row,data.targets]][:3]
    assert len(unseen)
    predicted=loaded.predict(str(data.profiles.original_name.iloc[row]),{},unseen.tolist())
    assert len(predicted)==len(unseen) and np.isfinite(list(predicted.values())).all()
    names=data.profiles.original_name.iloc[data.train[:8]].astype(str).tolist()
    assert np.isfinite(loaded.candidate_profiles(names,target_axes=data.targets)).all()
    del loaded
    before=float(transformer_loss(new,batch,config,'mae').detach())
    opt=torch.optim.AdamW(new.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    new.train()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss=transformer_loss(new,batch,config,'mae'); loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(new.parameters(),1.))
        opt.step()
    new.eval()
    after=float(transformer_loss(new,batch,config,'mae').detach())
    assert after<before
    saved['model_state']=new.state_dict()
    learned=local/'learned_functional_checkpoint.pt'; torch.save(saved,learned)
    loaded=RMSNormNutritionModel(learned,ROOT,str(device))
    with torch.no_grad():
        expected=new(batch)['amount_normalized']
        assert torch.equal(expected,loaded.model(batch)['amount_normalized'])
        assert torch.equal(expected,new(changed)['amount_normalized'])
    assert torch.count_nonzero(new.source_amount_residual.weight)>0
    del loaded
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    def timed(model,settings):
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        times=[]
        for block in range(4):
            sync(); started=time.perf_counter()
            for _ in range(20):
                optimizer.zero_grad(set_to_none=True)
                loss=transformer_loss(model,batch,settings,'mae'); loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                optimizer.step()
            sync()
            if block: times.append((time.perf_counter()-started)/20)
        return times
    timings={'parent':timed(old,conf),'rmsnorm':timed(new,config)}
    ratio=statistics.median(timings['rmsnorm'])/statistics.median(timings['parent'])
    historic=statistics.median(r['elapsed_seconds'] for r in plan['runtime_basis'])
    estimate=max(historic*max(1.,ratio),statistics.median(timings['rmsnorm'])*math.ceil(len(panel.rows)/64)*60)*1.15
    frozen_inputs(ROOT); verify_bindings(ROOT,bindings)
    files=['src/foodcomp/research_transformer_r9_rmsnorm.py','scripts/train_foodnutrigpt_v9_r9_rmsnorm.py',
           'scripts/verify_foodnutrigpt_r9_rmsnorm_functional.py']
    result={'status':'complete','plan_sha256':digest(ROOT/PLAN),'freeze_sha256':digest(freeze_path),
        'implementation_hashes':{name:digest(ROOT/name) for name in files},
        'initial_state_sha256':initial_hash,'parent_initial_state_sha256':parent['initial_state_sha256'],
        'shared_initial_tensors_exact':True,'constructor_rng_exact':True,'removed_keys':removed,
        'parameter_count':sum(p.numel() for p in new.parameters()),
        'trainable_parameter_count':sum(p.numel() for p in new.parameters() if p.requires_grad),
        'seven_encoder_norms_replaced':True,'norm_sites':sites,'unchanged_dropout_sites':dropout_sites(new),
        'default_fused_vs_explicit_layernorm_max_abs_difference':bridge_max,
        'matched_unfused_layernorm_bridge_exact':True,'formal_default_mha_backend_restored':True,
        'hidden_label_target_and_source_changes_forward_exact':True,'learned_source_independence_exact':True,
        'initial_and_learned_public_reload_exact':True,'unobserved_axes_predictable':True,
        'name_only_candidate_profiles_finite':True,'inner_training_loop_ast_unchanged':True,
        'small_batch_overfit':{'before':before,'after':after,'steps':50,'learning_rate':spec['learning_rate']},
        'benchmark_step_seconds':timings,'benchmark_ratio':ratio,'estimated_training_seconds':estimate,
        'followup_delay_minutes':math.ceil(estimate/60)+15,'device':str(device),
        'current_candidate_validation_evaluated':False,'data_modified':False,'baseline_refit':False,
        'complete_test_opened':False,'scope':'Training samples only. Benchmark timing is approximate, not a performance result.'}
    write_json(args.output_dir/'verification.json',result)
    print(json.dumps({k:result[k] for k in ['status','small_batch_overfit','benchmark_ratio','followup_delay_minutes']}))


if __name__=='__main__':main()

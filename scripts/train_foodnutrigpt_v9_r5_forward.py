"""Extend the original name-only baseline with recorded exposure, preserving its loss."""
import argparse
from dataclasses import asdict
import copy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,score_predictions,write_json,digest
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model,training_batch,loss,evaluate,OUTPUT_QUERY_POLICY
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_forward_loss import forward_loss


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epochs',type=int,choices=[8,60],default=60)
    p.add_argument('--seed',type=int,choices=[20260922,20260923,20260924],default=20260922)
    p.add_argument('--objective',choices=['smooth_l1','mae'],default='smooth_l1')
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.epochs==8 and args.seed!=20260922:raise ValueError('Only original seed registered for compatibility replay.')
    if args.epochs==8 and args.objective!='smooth_l1':raise ValueError('Only original loss registered for8-epoch replay.')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);started=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    files=[Path(__file__),ROOT/'src/foodcomp/research_neural.py',ROOT/'src/foodcomp/research_r0.py',
        ROOT/'src/foodcomp/research_text.py',ROOT/'src/foodcomp/research_inference.py',
        ROOT/'scripts/train_global_foodnutrigpt_v8_single_stage.py',ROOT/'scripts/train_global_foodnutrigpt_v9_source_calibrated.py',
        ROOT/'src/foodcomp/research_forward_loss.py']
    snapshot=args.output_dir/'code_snapshot';snapshot.mkdir()
    for f in files:shutil.copyfile(f,snapshot/f.name)
    manifest={'status':'running','version':'V9-R5','kind':'name_mlp','seed':args.seed,'args':vars(args),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(f.relative_to(ROOT)):digest(f) for f in files},'complete_test_opened':False,
        'output_query_policy':OUTPUT_QUERY_POLICY,'confirmation_completed':False,
        'selection':'minimum fixed142nutrition validation scaled-log MAE; earliest strict minimum; retrieval evaluated only after selection',
        'training_protocol':f'Original R0 name_mlp width256; all observed187targets; per-batch axis-normalized {args.objective}; batch256 AdamW lr.001 wd.0001 clip1; cosine horizon=epochs eta_min1e-5',
        'scope':'Original forward inputs, row sampling and normalization. SmoothL1 default is original; MAE changes only pointwise loss relative to60-epoch parent. Retrieval is not used to select checkpoint.',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda}}
    write_json(args.output_dir/'run_manifest.json',manifest)
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION,'quarantined');text,cache=prepare_names(data,ROOT)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model,config=make_model(data,text.shape[1],'name_mlp');model=model.to(device)
        manifest['environment']['device']=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu'
        active=data.train[data.observed[data.train][:,data.targets].any(1)]
        targets=data.observed[active][:,data.targets]
        manifest.update(data_hash=digest(data.root/'manifest.json'),protocol_hash=data.manifest['protocol_sha256'],
            name_cache_hash=digest(cache/'manifest.json'),parameter_count=sum(x.numel() for x in model.parameters()),
            initial_state_sha256=state_fingerprint(model),training_profiles=len(active),observed_target_cells=int(targets.sum()),
            training_rows_sha256=fingerprint_array(active),training_target_values_sha256=fingerprint_array(data.values[active][:,data.targets]),
            training_target_mask_sha256=fingerprint_array(targets),training_weights_sha256=fingerprint_array(data.weights[active][:,data.targets]))
        # Original50-step fixture, including its no-clipping smoke optimizer.
        initial=copy.deepcopy(model.state_dict());cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
        rng=np.random.default_rng(args.seed);rows=rng.choice(active,size=min(32,len(active)),replace=False)
        b=training_batch(data,text,rows,device,rng,'name_mlp')
        with torch.no_grad():
            original=model(b)['amount_normalized'];altered={k:v.clone() for k,v in b.items()}
            altered['value'].fill_(123.);altered['masked'].fill_(False)
            assert torch.equal(original,model(altered)['amount_normalized'])
        smoke=torch.optim.AdamW(model.parameters(),lr=.001);model.train();before=float(forward_loss(model,b,config,args.objective).detach())
        for _ in range(50):
            smoke.zero_grad(set_to_none=True);ll=forward_loss(model,b,config,args.objective);ll.backward();smoke.step()
        model.eval();after=float(forward_loss(model,b,config,args.objective).detach())
        if not after<before:raise AssertionError('Overfit check failed.')
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng:
            for x,y in zip(gpu_rng,torch.cuda.get_rng_state_all()):assert torch.equal(x,y)
        model.load_state_dict(initial);assert state_fingerprint(model)==manifest['initial_state_sha256']
        manifest['overfit']={'before':before,'after':after,'steps':50,'weights_restored_rng_unchanged':True,'numeric_context_mutation_prediction_exact':True}
        del initial,smoke,b,altered,original
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.00001)
        best=float('inf');history=[]
        def checkpoint(epoch):
            return {'model_state':model.state_dict(),'kind':'name_mlp','config':asdict(config),'text_dim':text.shape[1],
                'data_root':str(data.root),'view':'quarantined','name_cache':str(cache),'best_epoch':epoch,'seed':args.seed,
                'data_hash':manifest['data_hash'],'name_cache_hash':manifest['name_cache_hash'],'args':{'mlp_width':256}}
        for epoch in range(1,args.epochs+1):
            rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(active);model.train();total=0.;batches=0;seen=0;norms=0.;clipped=0
            for start in range(0,len(order),256):
                rows=order[start:start+256];b=training_batch(data,text,rows,device,rng,'name_mlp',.3,0.)
                opt.zero_grad(set_to_none=True);ll=forward_loss(model,b,config,args.objective);ll.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient.')
                opt.step();total+=float(ll.detach());batches+=1;seen+=len(rows);norms+=float(norm);clipped+=int(norm>1)
            pred=evaluate(model,data,text,device);metrics,_,_=score_predictions(data,pred);primary=metrics['nutrition']['scaled_log_mae']
            history.append({'epoch':epoch,'train_loss':total/batches,'validation_primary':primary,
                'validation_nutrition_legacy_log_mae':metrics['nutrition']['log_mae'],'learning_rate':opt.param_groups[0]['lr'],
                'training_profiles':seen,'training_order_sha256':fingerprint_array(order),'observed_target_cells':manifest['observed_target_cells'],
                'mean_preclip_gradient_norm':norms/batches,'gradient_clip_fraction':clipped/batches,'elapsed_seconds':time.monotonic()-started})
            pd.DataFrame(history).to_csv(args.output_dir/'history.csv',index=False)
            if primary<best:
                best=primary;torch.save(checkpoint(epoch),args.output_dir/'best_model.pt');manifest['best_epoch']=epoch
            if epoch in [8,20,60]:shutil.copyfile(args.output_dir/'best_model.pt',args.output_dir/f'best_through_epoch_{epoch:03d}.pt')
            scheduler.step();state=checkpoint(epoch);state.update(optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            torch.save(state,args.output_dir/'latest_training_state.pt')
            manifest.update(epoch_completed=epoch,best_validation_primary=best,elapsed_seconds=time.monotonic()-started);write_json(args.output_dir/'run_manifest.json',manifest)
            print(f'name_mlp epoch {epoch}/{args.epochs}: train {total/batches:.6f}; primary {primary:.6f}; elapsed {time.monotonic()-started:.1f}s',flush=True)
        selected=torch.load(args.output_dir/'best_model.pt',map_location=device,weights_only=True);model.load_state_dict(selected['model_state']);results={}
        for mode in ['completion','name_only']:
            pred=evaluate(model,data,text,device,mode);metrics,axes,candidates=score_predictions(data,pred)
            pred.to_parquet(args.output_dir/f'{mode}_predictions.parquet',index=False);axes.to_csv(args.output_dir/f'{mode}_axis_metrics.csv',index=False)
            candidates.to_parquet(args.output_dir/f'{mode}_candidate_errors.parquet',index=False);results[mode]=metrics
        assert results['completion']['nutrition']['scaled_log_mae']==best
        write_json(args.output_dir/'metrics.json',results)
        manifest.update(status='complete',checkpoint_hash=digest(args.output_dir/'best_model.pt'),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);print({k:v['nutrition'] for k,v in results.items()})
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);raise


if __name__=='__main__':main()

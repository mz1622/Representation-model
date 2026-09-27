"""Matched MSE/contrastive mapping into a frozen name-only PCA32 space."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import AlignmentPanel,NumericNameMapper,mapping_loss,evaluate_mapping,selection_score,save_evaluation,state_fingerprint
from foodcomp.research_alignment_views import partial_view_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objective',choices=['mse','contrastive'],required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--width',type=int,default=512);p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--temperature',type=float,default=.07);p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--partial-view-probability',type=float,default=0.)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if (args.epochs,args.batch_size,args.width,args.learning_rate,args.temperature)!=(60,256,512,.001,.07):raise ValueError('Unregistered mapping configuration.')
    if args.seed not in [20260922,20260923,20260924]:raise ValueError('Unregistered seed.')
    if args.partial_view_probability not in [0.,.5] or (args.partial_view_probability and args.objective!='contrastive'):raise ValueError('Unregistered partial-view configuration.')
    args.output_dir.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files=[Path(__file__),ROOT/'src/foodcomp/research_alignment.py',ROOT/'src/foodcomp/research_alignment_inference.py',ROOT/'src/foodcomp/research_r0.py',ROOT/'src/foodcomp/research_text.py']
    files.append(ROOT/'src/foodcomp/research_alignment_views.py')
    snapshot=args.output_dir/'code_snapshot';snapshot.mkdir()
    for file in files:shutil.copyfile(file,snapshot/file.name)
    manifest={'status':'running','version':'V9-R5','kind':'mlp','args':vars(args),'seed':args.seed,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(file.relative_to(ROOT)):digest(file) for file in files},'test_opened':False,'confirmation_allowed':False,
        'selection':'equal mean of full/30pct food-group MRR; strict maximum; earliest tie',
        'scope':'Independent nutrition-to-name mapping. No nutrition completion or name-only nutrition head. All nutrition inputs142; candidate name vectors fixed, no candidate measured profiles.'}
    write_json(args.output_dir/'run_manifest.json',manifest)
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);panel=AlignmentPanel(data,ROOT,device)
        write_json(args.output_dir/'panel_manifest.json',panel.manifest)
        manifest.update(data_hash=panel.manifest['data_sha256'],name_cache_hash=panel.manifest['name_cache_sha256'],panel_hash=digest(args.output_dir/'panel_manifest.json'))
        model=NumericNameMapper(width=args.width).to(device)
        features=torch.as_tensor(panel.features,device=device);target=torch.as_tensor(panel.targets,device=device)
        working_features=features
        name_ids=torch.as_tensor(panel.name_ids,dtype=torch.long,device=device)
        group_ids=torch.as_tensor(panel.group_ids,dtype=torch.long,device=device)
        weights=torch.as_tensor(panel.weights,dtype=torch.float32,device=device)
        n=len(features);weight_sum=float(panel.weights.sum())
        manifest.update(parameter_count=sum(x.numel() for x in model.parameters()),initial_state_sha256=state_fingerprint(model))
        def loss_on(ix):
            return mapping_loss(model(working_features[ix]),target[ix],name_ids[ix],weights[ix],objective=args.objective,
                population_size=n,weight_sum=weight_sum,temperature=args.temperature,group_ids=group_ids[ix])
        initial=copy.deepcopy(model.state_dict());cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
        ix=torch.as_tensor(np.random.default_rng(args.seed).choice(n,32,replace=False),device=device)
        if args.partial_view_probability:
            smoke_features,_,_=partial_view_features(panel.features,args.partial_view_probability,args.seed,0)
            working_features=torch.as_tensor(smoke_features,device=device)
            manifest['training_context_intervention']={'assignment_probability':.5,'axis_keep_probability':.3,'minimum_visible':3,
                'rng':'NumPy SeedSequence([seed,epoch,5105])','original_views_per_profile':1,
                'scope':'Training only. Randomly restore previously observed axes to minimum3. Same names,weights,order,negative groups andvalidation. No new values or profiles.'}
        smoke=torch.optim.AdamW(model.parameters(),lr=.001)
        before=float(loss_on(ix).detach())
        for _ in range(50):
            smoke.zero_grad(set_to_none=True);loss=loss_on(ix);loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite smoke gradient.')
            smoke.step()
        after=float(loss_on(ix).detach())
        if not after<before:raise AssertionError('Train-only mapping fixture did not overfit.')
        manifest['overfit']={'before':before,'after':after,'steps':50,'weights_and_rng_restored':True}
        model.load_state_dict(initial);torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        assert state_fingerprint(model)==manifest['initial_state_sha256']
        working_features=features
        del smoke,initial,ix
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
        best=-float('inf');history=[]
        def checkpoint(epoch):
            return {'model_state':model.state_dict(),'kind':'mlp','best_epoch':epoch,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'data_hash':manifest['data_hash'],'name_cache_hash':manifest['name_cache_hash'],'panel_hash':manifest['panel_hash']}
        def predict(x):return model(torch.as_tensor(x,device=device))
        for epoch in range(1,args.epochs+1):
            view_record={}
            if args.partial_view_probability:
                augmented,visible,assigned=partial_view_features(panel.features,args.partial_view_probability,args.seed,epoch)
                working_features=torch.as_tensor(augmented,device=device)
                original_visible=panel.features[:,142:].astype(bool)
                view_record={'partial_assigned_profiles':int(assigned.sum()),'partial_changed_profiles':int(np.any(visible!=original_visible,axis=1).sum()),
                    'original_visible_cells':int(original_visible.sum()),'training_visible_cells':int(visible.sum()),
                    'training_view_mask_sha256':fingerprint_array(visible),'partial_assignment_sha256':fingerprint_array(assigned)}
            order=np.random.default_rng(args.seed+epoch).permutation(n);model.train()
            total=0.;seen=0;norm_sum=0.;clipped=0;steps=0
            for start in range(0,n,args.batch_size):
                ids=order[start:start+args.batch_size];ix=torch.as_tensor(ids,device=device)
                optimizer.zero_grad(set_to_none=True);loss=loss_on(ix);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite training gradient.')
                optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
                norm_sum+=float(norm);clipped+=int(norm>1);steps+=1
            model.eval();ranks,scores=evaluate_mapping(predict,panel);selection=selection_score(scores)
            record={'epoch':epoch,'train_loss':total/seen,'selection_mean_mrr':selection,'training_profiles':seen,
                'training_order_sha256':fingerprint_array(order),'learning_rate':optimizer.param_groups[0]['lr'],
                'mean_preclip_gradient_norm':norm_sum/steps,'gradient_clip_fraction':clipped/steps,'elapsed_seconds':time.monotonic()-started}
            record.update(view_record)
            for score in scores:
                for key in ['mrr','recall_at_1','recall_at_5','recall_at_10']:record[f"validation_{score['visible_fraction']}_{key}"]=score[key]
            history.append(record);pd.DataFrame(history).to_csv(args.output_dir/'history.csv',index=False)
            if selection>best:
                best=selection;torch.save(checkpoint(epoch),args.output_dir/'best_model.pt');manifest['best_epoch']=epoch
            if epoch in [8,20,60]:shutil.copyfile(args.output_dir/'best_model.pt',args.output_dir/f'best_through_epoch_{epoch:03d}.pt')
            scheduler.step()
            state=checkpoint(epoch);state.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            torch.save(state,args.output_dir/'latest_training_state.pt')
            manifest.update(epoch_completed=epoch,best_selection_mean_mrr=best,elapsed_seconds=time.monotonic()-started)
            write_json(args.output_dir/'run_manifest.json',manifest)
            print(f'{args.objective} epoch {epoch}/60: loss {total/seen:.6f}; meanMRR {selection:.6f}; elapsed {time.monotonic()-started:.1f}s',flush=True)
        selected=torch.load(args.output_dir/'best_model.pt',map_location=device,weights_only=True);model.load_state_dict(selected['model_state']);model.eval()
        ranks,scores=evaluate_mapping(predict,panel)
        if selection_score(scores)!=best:raise AssertionError('Reloaded best selection differs.')
        save_evaluation(args.output_dir,panel,ranks,scores,method='numeric_to_name_'+args.objective,
            extra={'checkpoint_sha256':digest(args.output_dir/'best_model.pt'),'elapsed_seconds':time.monotonic()-started})
        manifest.update(status='complete',elapsed_seconds=time.monotonic()-started,checkpoint_hash=digest(args.output_dir/'best_model.pt'))
        write_json(args.output_dir/'run_manifest.json',manifest)
        print(scores,flush=True)
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);raise


if __name__=='__main__':main()

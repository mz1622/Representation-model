"""R1 controlled training on exhaustive shared family tasks with fixed LR horizon."""
import argparse
from dataclasses import asdict
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model,evaluate
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,model_loss

def main(version="V9-R1",protocol_change=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["mlp","v9"],required=True)
    p.add_argument("--epochs",type=int,default=20);p.add_argument("--schedule-epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=256);p.add_argument("--learning-rate",type=float,default=.001)
    p.add_argument("--amount-weight",type=float,default=1);p.add_argument("--source-weight",type=float,default=1)
    p.add_argument("--objective",choices=["hurdle","smooth_l1","mae"],default="hurdle")
    p.add_argument("--mlp-width",type=int,default=256)
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if not 0<args.epochs<=args.schedule_epochs:raise ValueError("Fixed schedule horizon must cover positive epochs.")
    if args.kind=="v9" and args.objective!="hurdle":raise ValueError("Direct-head Transformer is a separate R2 experiment.")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    start=time.monotonic();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=ResearchData(ROOT/"data/processed"/VERSION)
    text,cache=prepare_names(data,ROOT)
    panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,device)
    if digest(cache/"manifest.json")!=panel.manifest["name_cache_hash"]:raise ValueError("Text/task fingerprint mismatch.")
    model,config=make_model(data,text.shape[1],args.kind,amount_weight=args.amount_weight,source_weight=args.source_weight,mlp_width=args.mlp_width)
    model.to(device)
    files=[Path(__file__),ROOT/"src/foodcomp/research_r1.py",ROOT/"src/foodcomp/research_neural.py",ROOT/"src/foodcomp/research_r0.py"]
    snapshot=args.output_dir/"code_snapshot";snapshot.mkdir()
    for f in files:(snapshot/f.name).write_bytes(f.read_bytes())
    manifest={"status":"running","version":version,"args":vars(args),"seed":args.seed,
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "code_hashes":{str(f.relative_to(ROOT)):digest(f) for f in files},
        "data_hash":digest(data.root/"manifest.json"),"panel_hash":digest(panel.root/"manifest.json"),
        "name_cache_hash":digest(cache/"manifest.json"),"test_opened":False,"confirmation_allowed":False,
        "protocol_change":protocol_change or "R0->R1 aligns complete single-family training tasks and global axis/source loss weights; R0 architecture attribution is not allowed.",
        "selection":"fixed full validation nutrition primary metric; fixed learning-rate schedule horizon",
        "data_limitation":"R0 quarantine view retained; FooDB provenance remains unresolved; conditional benchmark research only"}
    write_json(args.output_dir/"run_manifest.json",manifest)
    try:
        # Verify overfitting, then restore both weights and RNG before all controlled arms.
        initial=copy.deepcopy(model.state_dict());cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None
        b=panel.batch(np.random.default_rng(args.seed).choice(len(panel.rows),32,replace=False))
        smoke=torch.optim.AdamW(model.parameters(),lr=.001)
        model.eval();before=float(model_loss(model,b,args.kind,config,args.objective).detach());model.train()
        for _ in range(50):
            smoke.zero_grad(set_to_none=True);loss=model_loss(model,b,args.kind,config,args.objective);loss.backward();smoke.step()
        model.eval();after=float(model_loss(model,b,args.kind,config,args.objective).detach())
        if not after<before:raise AssertionError("Train-only small batch failed to overfit.")
        manifest["overfit"]={"before":before,"after":after,"steps":50,"weights_and_rng_restored":True}
        model.load_state_dict(initial);torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        del smoke,initial,b
        opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.schedule_epochs,eta_min=args.learning_rate*.01)
        best=float("inf");history=[]
        def checkpoint(epoch):
            return {"model_state":model.state_dict(),"kind":args.kind,"config":asdict(config),"text_dim":text.shape[1],
                "data_root":str(data.root),"view":data.view,"name_cache":str(cache),"best_epoch":epoch,"seed":args.seed,
                "data_hash":manifest["data_hash"],"name_cache_hash":manifest["name_cache_hash"],"panel_hash":manifest["panel_hash"],"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        for epoch in range(1,args.epochs+1):
            order=np.random.default_rng(args.seed+epoch).permutation(len(panel.rows));model.train();weighted=0.;seen=0
            for start_batch in range(0,len(order),args.batch_size):
                ix=order[start_batch:start_batch+args.batch_size];batch=panel.batch(ix)
                opt.zero_grad(set_to_none=True);loss=model_loss(model,batch,args.kind,config,args.objective);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("Nonfinite gradient.")
                opt.step();weighted+=float(loss.detach())*len(ix);seen+=len(ix)
            pred=evaluate(model,data,text,device)
            metrics,_,_=score_predictions(data,pred);primary=metrics["nutrition"]["scaled_log_mae"]
            history.append({"epoch":epoch,"train_loss":weighted/seen,"validation_primary":primary,
                "validation_legacy_log_mae":metrics["nutrition"]["log_mae"],"learning_rate":opt.param_groups[0]["lr"],
                "training_tasks":seen,"elapsed_seconds":time.monotonic()-start})
            pd.DataFrame(history).to_csv(args.output_dir/"history.csv",index=False)
            if primary<best:
                best=primary;torch.save(checkpoint(epoch),args.output_dir/"best_model.pt")
            if epoch in {8,20,args.epochs}:
                selected=torch.load(args.output_dir/"best_model.pt",map_location="cpu",weights_only=True)
                torch.save(selected,args.output_dir/f"best_through_epoch_{epoch:03d}.pt")
            scheduler.step()
            state=checkpoint(epoch);state.update(optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [])
            torch.save(state,args.output_dir/"latest_training_state.pt")
            manifest.update(epoch_completed=epoch,elapsed_seconds=time.monotonic()-start,best_primary=best)
            write_json(args.output_dir/"run_manifest.json",manifest)
            print(f"{args.kind} epoch {epoch}/{args.epochs}: loss {weighted/seen:.5f}; primary {primary:.6f}; elapsed {time.monotonic()-start:.1f}s",flush=True)
        selected=torch.load(args.output_dir/"best_model.pt",map_location=device,weights_only=True);model.load_state_dict(selected["model_state"])
        results={}
        for mode in ["completion","name_only"]:
            pred=evaluate(model,data,text,device,mode);metric,axes,groups=score_predictions(data,pred)
            pred.to_parquet(args.output_dir/f"{mode}_predictions.parquet",index=False)
            axes.to_csv(args.output_dir/f"{mode}_axis_metrics.csv",index=False)
            groups.to_parquet(args.output_dir/f"{mode}_candidate_errors.parquet",index=False);results[mode]=metric
        write_json(args.output_dir/"metrics.json",results)
        manifest.update(status="complete",elapsed_seconds=time.monotonic()-start,best_epoch=selected["best_epoch"],checkpoint_hash=digest(args.output_dir/"best_model.pt"))
        write_json(args.output_dir/"run_manifest.json",manifest)
        (args.output_dir/"README.md").write_text(f"# {version} {args.kind}\n\nShared exhaustive family-task training; full-data source/axis weighting; fixed LR horizon.\n\nCompletion primary: {results['completion']['nutrition']['scaled_log_mae']:.8f}; name-only: {results['name_only']['nutrition']['scaled_log_mae']:.8f}.\n\nChange: {manifest['protocol_change']}\n\nSingle-seed exploration. Every epoch and all configuration/code hashes are retained. Frozen test not evaluated.\n",encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start)
        write_json(args.output_dir/"run_manifest.json",manifest);raise

if __name__=="__main__":main()

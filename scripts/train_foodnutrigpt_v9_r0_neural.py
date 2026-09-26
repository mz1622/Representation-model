"""Train controlled R0 neural baselines on the shared versioned protocol."""
import argparse
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,score_predictions,write_json,digest
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model,training_batch,loss,evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["mlp","name_mlp","numeric_mlp","v9","v8_optimized"],required=True)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--amount-weight",type=float,default=1.)
    p.add_argument("--source-weight",type=float,default=1.)
    p.add_argument("--mask-ratio",type=float,default=.3)
    p.add_argument("--text-only-probability",type=float,default=0.)
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--view",choices=["inclusive","quarantined"],default="quarantined")
    p.add_argument("--overfit-steps",type=int,default=50)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.epochs<1 or args.batch_size<1:raise ValueError("Positive epochs/batch size required.")
    for rate in [args.mask_ratio,args.text_only_probability]:
        if not 0<=rate<=1:raise ValueError("Mask and modality probabilities must lie in [0,1].")
    if args.source_weight<0 or args.amount_weight<0:raise ValueError("Nonnegative loss weights required.")
    args.output_dir.mkdir(parents=True)
    start=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    data=ResearchData(ROOT/"data/processed"/VERSION,args.view)
    text,cache=prepare_names(data,ROOT)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,config=make_model(data,text.shape[1],args.kind,amount_weight=args.amount_weight,source_weight=args.source_weight)
    model=model.to(device)
    manifest={"status":"running","run_kind":"exploratory; not historical metric reproduction",
        "hypothesis":"Controlled neural model learns the shared raw-cell/axis-query contract.",
        "parent":"frozen V8/V9 architectures rebased onto the new R0 protocol",
        "causal_warning":"Protocol, input text and target transform differ from historical runs; do not attribute historical score differences to architecture.",
        "args":vars(args),"data_hash":digest(data.root/"manifest.json"),"protocol_hash":data.manifest["protocol_sha256"],
        "name_cache":str(cache),"name_cache_hash":digest(cache/"manifest.json"),
        "code_hashes":{name:digest(ROOT/name) for name in ["src/foodcomp/research_neural.py","src/foodcomp/research_r0.py","scripts/train_foodnutrigpt_v9_r0_neural.py"]},
        "complete_test_opened":False,"scientific_claim_allowed":False}
    write_json(args.output_dir/"run_manifest.json",manifest)
    try:
        active_train=data.train[data.observed[data.train][:,data.targets].any(1)]
        # Overfit a train-only batch, then restore initialization before the real run.
        if args.overfit_steps:
            initial=copy.deepcopy(model.state_dict())
            rng=np.random.default_rng(args.seed)
            rows=rng.choice(active_train,size=min(32,len(active_train)),replace=False)
            b=training_batch(data,text,rows,device,rng,args.kind)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
            model.train();before=float(loss(model,b,args.kind,config).detach())
            for _ in range(args.overfit_steps):
                opt.zero_grad(set_to_none=True);ll=loss(model,b,args.kind,config);ll.backward();opt.step()
            model.eval();after=float(loss(model,b,args.kind,config).detach())
            if not after<before:raise AssertionError("Small-batch overfit check failed.")
            model.load_state_dict(initial)
            manifest["overfit_check"]={"before":before,"after":after,"steps":args.overfit_steps,"initialization_restored":True}
            print("Overfit check:",manifest["overfit_check"],flush=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.learning_rate*.01)
        best=float("inf");history=[]
        for epoch in range(1,args.epochs+1):
            rng=np.random.default_rng(args.seed+epoch)
            order=rng.permutation(active_train)
            model.train();total=0.;batches=0
            for offset in range(0,len(order),args.batch_size):
                rows=order[offset:offset+args.batch_size]
                b=training_batch(data,text,rows,device,rng,args.kind,args.mask_ratio,args.text_only_probability)
                opt.zero_grad(set_to_none=True)
                ll=loss(model,b,args.kind,config);ll.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("Nonfinite gradient.")
                opt.step();total+=float(ll.detach());batches+=1
            pred=evaluate(model,data,text,device)
            metrics,_,_=score_predictions(data,pred)
            primary=metrics["nutrition"]["scaled_log_mae"]
            history.append({"epoch":epoch,"train_loss":total/batches,"validation_primary":primary,
                            "validation_nutrition_legacy_log_mae":metrics["nutrition"]["log_mae"],
                            "learning_rate":opt.param_groups[0]["lr"]})
            print(f"{args.kind} epoch {epoch}/{args.epochs}: train {total/batches:.5f}, primary {primary:.5f}, elapsed {time.monotonic()-start:.1f}s",flush=True)
            if primary<best:
                best=primary
                torch.save({"model_state":model.state_dict(),"kind":args.kind,"config":asdict(config),
                            "text_dim":text.shape[1],"data_root":str(data.root),"view":args.view,
                            "name_cache":str(cache),"best_epoch":epoch,"seed":args.seed,
                            "data_hash":manifest["data_hash"],"name_cache_hash":manifest["name_cache_hash"]},
                           args.output_dir/"best_model.pt")
            scheduler.step()
        checkpoint=torch.load(args.output_dir/"best_model.pt",map_location=device,weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        all_results={}
        for mode in ["completion","name_only"]:
            pred=evaluate(model,data,text,device,mode)
            metrics,axes,candidates=score_predictions(data,pred)
            pred.to_parquet(args.output_dir/f"{mode}_predictions.parquet",index=False)
            axes.to_csv(args.output_dir/f"{mode}_axis_metrics.csv",index=False)
            candidates.to_parquet(args.output_dir/f"{mode}_candidate_errors.parquet",index=False)
            all_results[mode]=metrics
        pd.DataFrame(history).to_csv(args.output_dir/"history.csv",index=False)
        write_json(args.output_dir/"metrics.json",all_results)
        manifest.update(status="complete",elapsed_seconds=time.monotonic()-start,best_epoch=checkpoint["best_epoch"],
            checkpoint_sha256=digest(args.output_dir/"best_model.pt"))
        write_json(args.output_dir/"run_manifest.json",manifest)
        (args.output_dir/"README.md").write_text(f"# R0 {args.kind}\n\n"
            "Controlled shared-protocol baseline. Input: original food name only plus family-hidden observed values; fixed query axes independent of target availability. "
            "Training uses only observed target cells; missing is never a zero label. "
            "Best checkpoint selected by the complete shared validation panel primary metric.\n\n"
            f"Completion primary: {all_results['completion']['nutrition']['scaled_log_mae']:.8f}; "
            f"name-only primary: {all_results['name_only']['nutrition']['scaled_log_mae']:.8f}.\n\n"
            "This is a single-seed exploratory run. It is neither a reproduction of historical scores nor evidence of superiority to tuned RF/XGBoost. "
            "Configuration, hashes, controls and errors are in run_manifest.json; every epoch is in history.csv. Test stayed closed.\n",encoding="utf-8")
        print(json.dumps({m:r["nutrition"] for m,r in all_results.items()},indent=2))
    except Exception as e:
        manifest.update(status="failed",error_type=type(e).__name__,error=str(e),elapsed_seconds=time.monotonic()-start)
        write_json(args.output_dir/"run_manifest.json",manifest)
        raise

if __name__=="__main__":main()

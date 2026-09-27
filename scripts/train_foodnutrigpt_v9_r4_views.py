"""Registered R4 two-view consistency; same supervised tasks and inference model."""
import argparse
import copy
from dataclasses import asdict
import platform
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json, score_predictions
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model, evaluate, OUTPUT_QUERY_POLICY
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, fingerprint_array
from foodcomp.research_views import extra_view_masks, two_view_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--consistency-weight",type=float,required=True)
    p.add_argument("--consistency-centering",choices=["none","joint_batch"],default="none")
    p.add_argument("--view-drop-probability",type=float,default=.3)
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--output-dir",type=Path,required=True)
    p.set_defaults(kind="mlp",objective="mae",mlp_width=512,mlp_normalization="layer_norm",
        mlp_task_heads="shared",mlp_text_conditioning="none",mlp_query_residual=False,
        name_only_probability=0.,epochs=60,schedule_epochs=60,batch_size=256,learning_rate=.001)
    args=p.parse_args()
    args.output_query_policy=OUTPUT_QUERY_POLICY
    if not np.isfinite(args.consistency_weight) or args.consistency_weight<0:
        raise ValueError("Finite nonnegative consistency weight required.")
    if not np.isfinite(args.view_drop_probability) or not 0<=args.view_drop_probability<=1:
        raise ValueError("Finite view-drop probability in [0,1] required.")
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    start=time.monotonic();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=ResearchData(ROOT/"data/processed"/VERSION)
    text,cache=prepare_names(data,ROOT)
    panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,device)
    if digest(cache/"manifest.json")!=panel.manifest["name_cache_hash"]:
        raise ValueError("Text/task fingerprint mismatch.")
    model,config=make_model(data,text.shape[1],"mlp",mlp_width=512)
    model.to(device)
    files=[Path(__file__)]+[ROOT/f"src/foodcomp/{name}.py" for name in
        ["research_views","research_r1","research_neural","research_r0","research_text","research_inference"]]
    snapshot=args.output_dir/"code_snapshot";snapshot.mkdir()
    for f in files:(snapshot/f.name).write_bytes(f.read_bytes())
    manifest={"status":"running","version":"V9-R4","args":vars(args),"seed":args.seed,
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "code_hashes":{str(f.relative_to(ROOT)):digest(f) for f in files},
        "data_hash":digest(data.root/"manifest.json"),"panel_hash":digest(panel.root/"manifest.json"),
        "name_cache_hash":digest(cache/"manifest.json"),"test_opened":False,"confirmation_allowed":False,
        "parameter_count":sum(v.numel() for v in model.parameters()),
        "selection":"142-axis fixed full validation primary; earliest strict minimum; no early stopping",
        "protocol_change":"Matched two-view computation on original MLP512. A is unchanged common family task; B only hides additional visible axes, never restores targets. A alone receives original MAE187 supervision; loss=MAE(A)+lambda*axis/source-weighted normalized-representation distance(A,B), with gradients on both sides even when lambda=0. No new labels, features, parameters or inference changes. Lambda0 versus0.1 is the registered comparison; overall gradient/clipping changes are part of this intervention.",
        "view_assignment":"Independent NumPy SeedSequence([seed,epoch,4104]), float32 uniforms on all fixed task IDs by252 axes; same masks for both arms",
        "consistency_weighting":"q_task=sum_axis(target*cell_weight/train_axis_total); C=sum_task(q_task*0.5*||unit(hA)-unit(hB)||^2)/187; unbiased uniform-task minibatches",
        "data_limitation":"Immutable R0 quarantine view; FooDB raw provenance unresolved; conditional internal research only"}
    manifest["inverse_numerics"]="Finite old outputs preserved bitwise; only nonfinite intermediate decoded values recomputed in float64, cast to original dtype, and required finite. No cap or skipped query. Shared across neural methods."
    if args.consistency_centering == "joint_batch":
        manifest["protocol_change"] = "Only centre A/B representations by their shared unweighted minibatch mean before normalization in C; mean and both sides receive gradients. Original supervised head, inputs, model and inference unchanged. Same coefficient0.1 and task/mask/schedule as the origin-based reference. Gradient magnitude, clipping and batch-composition effects are part of this intervention; no collapse-prevention guarantee."
        manifest["consistency_weighting"] = "Original q_task and T/(B*187) factor, but distance uses the jointly centred current minibatch. This estimates the random-minibatch regularizer, not an unbiased full-panel centred-distance objective."
    write_json(args.output_dir/"run_manifest.json",manifest)
    write_json(args.output_dir/"environment.json",{"python":sys.version,"platform":platform.platform(),
        "numpy":np.__version__,"torch":str(torch.__version__),"cuda":torch.version.cuda,
        "device":str(device),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
        "cpu_threads":torch.get_num_threads()})
    try:
        initial=copy.deepcopy(model.state_dict());cpu_rng=torch.get_rng_state()
        gpu_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None
        b=panel.batch(np.random.default_rng(args.seed).choice(len(panel.rows),32,replace=False))
        drop=extra_view_masks(32,len(data.axes),args.view_drop_probability,args.seed,0)
        smoke=torch.optim.AdamW(model.parameters(),lr=.001)
        model.eval();before=float(two_view_loss(model,b,drop,args.consistency_weight,args.consistency_centering)[0].detach())
        model.train()
        for _ in range(50):
            smoke.zero_grad(set_to_none=True)
            loss,_=two_view_loss(model,b,drop,args.consistency_weight,args.consistency_centering)
            loss.backward();smoke.step()
        model.eval();after=float(two_view_loss(model,b,drop,args.consistency_weight,args.consistency_centering)[0].detach())
        if not after<before:raise AssertionError("Train-only two-view small batch failed to overfit.")
        manifest["overfit"]={"before":before,"after":after,"steps":50,"weights_and_rng_restored":True}
        model.load_state_dict(initial);torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        del smoke,initial,b,drop
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=.00001)
        best=float("inf");history=[]
        def checkpoint(epoch):
            return {"model_state":model.state_dict(),"kind":"mlp","config":asdict(config),
                "text_dim":text.shape[1],"data_root":str(data.root),"view":data.view,"name_cache":str(cache),
                "best_epoch":epoch,"seed":args.seed,"data_hash":manifest["data_hash"],
                "name_cache_hash":manifest["name_cache_hash"],"panel_hash":manifest["panel_hash"],
                "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        for epoch in range(1,61):
            order=np.random.default_rng(args.seed+epoch).permutation(len(panel.rows))
            masks=extra_view_masks(len(panel.rows),len(data.axes),args.view_drop_probability,args.seed,epoch)
            mask_hash=fingerprint_array(masks)
            model.train();seen=0;visible=0;removed=0;steps=0;clipped=0;norm_total=0.
            accum=np.zeros(5,np.float64);centred_accum=np.zeros(2,np.float64)
            for offset in range(0,len(order),args.batch_size):
                ix=order[offset:offset+args.batch_size];batch=panel.batch(ix)
                opt.zero_grad(set_to_none=True)
                loss,parts=two_view_loss(model,batch,masks[ix],args.consistency_weight,args.consistency_centering)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("Nonfinite gradient.")
                opt.step()
                stats=torch.stack([loss.detach(),parts["supervised"],parts["consistency"],
                    parts["mean_representation_norm"],parts["unit_batch_std"],
                    parts["visible_cells"],parts["removed_visible_cells"],norm,
                    parts["centred_representation_norm"],parts["centred_unit_batch_std"]]).detach().cpu().numpy()
                accum+=stats[:5].astype(np.float64)*len(ix);seen+=len(ix)
                centred_accum+=stats[8:10].astype(np.float64)*len(ix)
                visible+=int(stats[5]);removed+=int(stats[6]);norm_total+=float(stats[7])
                steps+=1;clipped+=int(stats[7]>1.)
            pred=evaluate(model,data,text,device)
            metrics,_,_=score_predictions(data,pred);primary=metrics["nutrition"]["scaled_log_mae"]
            means=accum/seen
            history.append({"epoch":epoch,"train_loss":means[0],"train_supervised_mae":means[1],
                "train_consistency_unweighted_coefficient":means[2],
                "mean_representation_norm":means[3],"mean_unit_batch_std":means[4],
                "mean_centred_representation_norm":centred_accum[0]/seen,
                "mean_centred_unit_batch_std":centred_accum[1]/seen,
                "mean_preclip_gradient_norm":norm_total/steps,"gradient_clip_fraction":clipped/steps,
                "training_tasks":seen,"visible_cells_A":visible,"removed_visible_cells_B":removed,
                "view_mask_sha256":mask_hash,"validation_primary":primary,
                "validation_legacy_log_mae":metrics["nutrition"]["log_mae"],
                "learning_rate":opt.param_groups[0]["lr"],"elapsed_seconds":time.monotonic()-start})
            pd.DataFrame(history).to_csv(args.output_dir/"history.csv",index=False)
            if primary<best:
                best=primary;torch.save(checkpoint(epoch),args.output_dir/"best_model.pt")
            if epoch in {8,20,60}:
                selected=torch.load(args.output_dir/"best_model.pt",map_location="cpu",weights_only=True)
                torch.save(selected,args.output_dir/f"best_through_epoch_{epoch:03d}.pt")
            scheduler.step()
            state=checkpoint(epoch);state.update(optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else [])
            torch.save(state,args.output_dir/"latest_training_state.pt")
            manifest.update(epoch_completed=epoch,elapsed_seconds=time.monotonic()-start,best_primary=best)
            write_json(args.output_dir/"run_manifest.json",manifest)
            print(f"views lambda={args.consistency_weight:g} epoch {epoch}/60: MAE {means[1]:.5f}; C {means[2]:.5f}; primary {primary:.6f}; elapsed {time.monotonic()-start:.1f}s",flush=True)
        selected=torch.load(args.output_dir/"best_model.pt",map_location=device,weights_only=True)
        model.load_state_dict(selected["model_state"])
        results={}
        for mode in ["completion","name_only"]:
            pred=evaluate(model,data,text,device,mode);metric,axes,groups=score_predictions(data,pred)
            pred.to_parquet(args.output_dir/f"{mode}_predictions.parquet",index=False)
            axes.to_csv(args.output_dir/f"{mode}_axis_metrics.csv",index=False)
            groups.to_parquet(args.output_dir/f"{mode}_candidate_errors.parquet",index=False)
            results[mode]=metric
        write_json(args.output_dir/"metrics.json",results)
        manifest.update(status="complete",elapsed_seconds=time.monotonic()-start,
            best_epoch=selected["best_epoch"],checkpoint_hash=digest(args.output_dir/"best_model.pt"))
        write_json(args.output_dir/"run_manifest.json",manifest)
        (args.output_dir/"README.md").write_text(
            f"# V9-R4 two-view consistency, lambda={args.consistency_weight}\n\n"
            f"Completion: {results['completion']['nutrition']['scaled_log_mae']:.8f}; name-only: {results['name_only']['nutrition']['scaled_log_mae']:.8f}.\n\n"
            "One completion-selected checkpoint, fixed60-epoch schedule. All curves/configuration/code hashes retained. "
            "Single-seed exploration; test closed. See version report for all tasks and causal interpretation.\n",encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start)
        write_json(args.output_dir/"run_manifest.json",manifest);raise


if __name__=="__main__":main()

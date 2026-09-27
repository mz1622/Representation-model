"""Replay the failed centred-view epoch for diagnosis, never resume model selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, fingerprint_array
from foodcomp.research_neural import make_model, batch_from_arrays, evaluate
from foodcomp.research_views import extra_view_masks, two_view_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest=json.loads((args.run_dir/"run_manifest.json").read_text())
    if manifest["status"]!="failed" or manifest["epoch_completed"] not in {16,17} or manifest["error"]!="Nonfinite prediction.":
        raise ValueError("Expected a retained epoch17/18 centred-view failure.")
    epoch=manifest["epoch_completed"]+1
    stable_inverse="inverse_numerics" in manifest
    for relative,sha in manifest["code_hashes"].items():
        if digest(ROOT/relative)!=sha:raise ValueError(f"Failed-run source changed: {relative}")
    result={"status":"incomplete","diagnostic_only":True,"candidate_resumed":False,
        "complete_test_opened":False,"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "script_sha256":digest(Path(__file__)),"failed_manifest_sha256":digest(args.run_dir/"run_manifest.json"),
        "start_checkpoint_sha256":digest(args.run_dir/"latest_training_state.pt")}
    started=time.monotonic()
    try:
        torch.set_num_threads(4);device=torch.device("cuda")
        saved=torch.load(args.run_dir/"latest_training_state.pt",map_location="cpu",weights_only=True)
        if saved["best_epoch"]!=epoch-1:raise ValueError("Wrong replay starting epoch.")
        data=ResearchData(ROOT/"data/processed"/VERSION);text,cache=prepare_names(data,ROOT)
        panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,device)
        for actual,expected in [(digest(data.root/"manifest.json"),manifest["data_hash"]),
                                (digest(cache/"manifest.json"),manifest["name_cache_hash"]),
                                (digest(panel.root/"manifest.json"),manifest["panel_hash"])]:
            if actual!=expected:raise ValueError("Replay data/cache/panel fingerprint mismatch.")
        model,_=make_model(data,text.shape[1],"mlp",mlp_width=512)
        model.to(device);model.load_state_dict(saved["model_state"])
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        opt.load_state_dict(saved["optimizer"])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=.00001)
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"])
        order=np.random.default_rng(20260922+epoch).permutation(len(panel.rows))
        masks=extra_view_masks(len(panel.rows),len(data.axes),.3,20260922,epoch)
        model.train();losses=[];norms=[];supervised=[];consistency=[]
        for start in range(0,len(order),256):
            ix=order[start:start+256];batch=panel.batch(ix)
            opt.zero_grad(set_to_none=True)
            loss,parts=two_view_loss(model,batch,masks[ix],.1,"joint_batch")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError("Nonfinite replay training gradient.")
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
            supervised.append(float(parts["supervised"]));consistency.append(float(parts["consistency"]))
        result.update(replayed_epoch=epoch,stable_inverse=stable_inverse,training_tasks=len(order),optimizer_steps=len(losses),
            view_mask_sha256=fingerprint_array(masks),learning_rate=opt.param_groups[0]["lr"],
            all_training_losses_gradients_finite=True,all_parameters_finite=all(torch.isfinite(x).all().item() for x in model.parameters()),
            max_training_loss=max(losses),max_preclip_gradient_norm=max(norms),
            gradient_clip_fraction=float(np.mean(np.array(norms)>1.)),
            mean_batch_supervised=float(np.mean(supervised)),mean_batch_consistency=float(np.mean(consistency)))
        try:
            evaluate(model,data,text,device)
        except FloatingPointError as error:
            if str(error)!="Nonfinite prediction.":raise
            result["original_guard_failure_reproduced"]=True
        else:
            raise AssertionError("Saved-state diagnostic replay did not reproduce the original failure.")
        model.eval();offending=[];amount_min=float("inf");amount_max=-float("inf");nonfinite_amounts=0
        with torch.no_grad():
            for family,jobs in data.jobs.groupby("mask_family"):
                rows=np.sort(jobs.profile_index.unique());visible=data.observed[rows].copy()
                visible[:,data.families==family]=False
                targets=set(zip(jobs.profile_index.astype(int),jobs.axis_index.astype(int)))
                for start in range(0,len(rows),128):
                    rr=rows[start:start+128]
                    batch=batch_from_arrays(data.values[rr],visible[start:start+128],text[rr],device)
                    amount=model(batch)["amount_normalized"]
                    nonfinite_amounts+=int((~torch.isfinite(amount)).sum())
                    if not torch.isfinite(amount).all():raise FloatingPointError("Nonfinite pre-inverse model output.")
                    amount_min=min(amount_min,float(amount.min()));amount_max=max(amount_max,float(amount.max()))
                    scale_tensor=torch.as_tensor(data.scale,device=device,dtype=amount.dtype)
                    raw=torch.expm1(amount.clamp_min(0))*scale_tensor
                    if stable_inverse:
                        wide=torch.expm1(amount.double().clamp_min(0))*scale_tensor.double()
                        raw=torch.where(~torch.isfinite(raw),wide.to(raw.dtype),raw)
                    bad=torch.nonzero(~torch.isfinite(raw),as_tuple=False).cpu().numpy()
                    for i,a in bad:
                        u=float(amount[i,a]);scale=float(data.scale[a]);diagnostic_raw64=float(np.expm1(max(0.,u))*scale)
                        offending.append({"profile_index":int(rr[i]),"axis_index":int(a),"mask_family":str(family),
                            "amount_normalized":u,"scale":scale,"raw_float64_diagnostic":diagnostic_raw64,
                            "is_scored_job":(int(rr[i]),int(a)) in targets,"loss_eligible":bool(data.axes.iloc[a].loss_eligible),
                            "axis_loss_group":str(data.axes.iloc[a].loss_group),
                            "final_float32_representable":bool(np.isfinite(diagnostic_raw64) and diagnostic_raw64<=float(np.finfo(np.float32).max))})
        if not offending:raise AssertionError("No overflow located after reproduced failure.")
        frame=pd.DataFrame(offending)
        frame.to_parquet(args.output_dir/"overflow_cells_private.parquet",index=False)
        torch.save({"model_state":model.state_dict(),"diagnostic_only":True,"replayed_epoch":epoch},args.output_dir/f"replayed_epoch{epoch}_diagnostic_not_candidate.pt")
        result.update(status="complete_failure_reproduced",preinverse_outputs_all_finite=nonfinite_amounts==0,
            preinverse_min=amount_min,preinverse_max=amount_max,overflow_cells=len(frame),
            overflow_unique_profiles=int(frame.profile_index.nunique()),overflow_axes=sorted(frame.axis_index.unique().tolist()),
            overflow_loss_groups=frame.axis_loss_group.value_counts().to_dict(),
            overflow_supervised_axes_cells=int(frame.loss_eligible.sum()),overflow_scored_jobs=int(frame.is_scored_job.sum()),
            inverse_float64_all_finite=bool(np.isfinite(frame.raw_float64_diagnostic).all()),
            final_float32_representable_cells=int(frame.final_float32_representable.sum()),
            maximum_float64_final_value=float(frame.raw_float64_diagnostic.max()),
            overflow_preinverse_range=[float(frame.amount_normalized.min()),float(frame.amount_normalized.max())],
            scope="Diagnostic replay from saved optimizer/scheduler/RNG with unchanged source and masks; original failed epoch was not saved, so no bitwise claim about that lost state. Float64 values only diagnose inverse overflow, never replace benchmark scores or resume selection. Failed candidate remains failed.",
            elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        result.update(status="diagnostic_failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/"summary.json",result);raise
    write_json(args.output_dir/"summary.json",result)
    print(json.dumps({k:result[k] for k in ["status","all_parameters_finite","preinverse_outputs_all_finite",
        "preinverse_max","overflow_cells","overflow_supervised_axes_cells","overflow_scored_jobs","overflow_loss_groups","inverse_float64_all_finite"]},indent=2))


if __name__=="__main__":main()

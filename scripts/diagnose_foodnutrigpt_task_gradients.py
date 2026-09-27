"""Same-target train-only completion/name-only gradients, without model updates."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_r0 import digest,write_json
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,panel_loss
from foodcomp.research_task_mix import remove_numeric_context
from foodcomp.research_gradient_geometry import gradient_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,nargs="+",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--batches",type=int,default=32);p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--seed",type=int,default=20260922);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.batches<1 or args.batch_size<1:raise ValueError("Positive diagnostic sizes required.")
    if len(set(x.parent.name for x in args.checkpoint))!=len(args.checkpoint):raise ValueError("Unique checkpoint names required.")
    args.output_dir.mkdir(parents=True);torch.set_num_threads(2)
    records=[];models=[];identity=None
    for checkpoint in args.checkpoint:
        started=time.monotonic();wrapper=NutritionModel(checkpoint)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model,data=wrapper.model,wrapper.data
        if wrapper.kind!="mlp" or saved["args"]["objective"]!="mae":raise ValueError("Diagnostic requires a direct MAE MLP.")
        panel=FamilyPanel(data,wrapper._cached_text,ROOT/"data/processed"/PANEL_VERSION,wrapper.device)
        current=(saved["data_hash"],saved["panel_hash"],saved["name_cache_hash"])
        if digest(panel.root/"manifest.json")!=saved["panel_hash"] or (identity is not None and current!=identity):
            raise ValueError("Different training data/task/name inputs.")
        identity=current
        count=min(args.batches*args.batch_size,len(panel.rows))
        indices=np.random.default_rng(args.seed).choice(len(panel.rows),count,replace=False)
        if not np.isin(panel.rows[indices],data.train).all():raise AssertionError("Non-training diagnostic task.")
        np.save(args.output_dir/f"{checkpoint.parent.name}_task_indices.npy",indices)
        items=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
        parameters=[param for _,param in items]
        scopes={"encoder":[i for i,(name,_) in enumerate(items) if name.startswith("encoder.")],
                "head":[i for i,(name,_) in enumerate(items) if name.startswith("head.")],"all":list(range(len(items)))}
        if not scopes["encoder"] or not scopes["head"] or sorted(scopes["encoder"]+scopes["head"])!=scopes["all"]:
            raise ValueError("Unexpected shared encoder/head parameters.")
        initial={k:v.clone() for k,v in model.state_dict().items()};model.eval()
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if wrapper.device.type=="cuda" else []
        total={view:[torch.zeros_like(p,dtype=torch.float64) for p in parameters] for view in ["completion","name_only"]}
        for start in range(0,count,args.batch_size):
            selected=indices[start:start+args.batch_size];batch=panel.batch(selected)
            name_batch=remove_numeric_context(batch,np.ones(len(selected),bool))
            for key in batch:
                if key!="masked" and name_batch[key] is not batch[key]:raise AssertionError("Task view changed more than numeric context.")
            gradients={};losses={}
            for view,inputs in [("completion",batch),("name_only",name_batch)]:
                value=panel_loss(model(inputs),inputs,objective="mae")
                gradients[view]=torch.autograd.grad(value,parameters,allow_unused=False)
                losses[view]=float(value.detach())
                for accumulator,g in zip(total[view],gradients[view]):accumulator.add_(g.double(),alpha=len(selected)/count)
            for scope,positions in scopes.items():
                stats=gradient_geometry([gradients["completion"][i] for i in positions],[gradients["name_only"][i] for i in positions])
                records.append({"run":checkpoint.parent.name,"batch":start//args.batch_size,"scope":scope,
                    "tasks":len(selected),"observed_targets":int(batch["target"].sum()),
                    **{key+"_loss":value for key,value in losses.items()},**stats})
        for key,value in initial.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        if any(p.grad is not None for p in parameters):raise AssertionError("Diagnostic populated parameter .grad.")
        torch.testing.assert_close(cpu_rng,torch.get_rng_state(),rtol=0,atol=0)
        for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []):torch.testing.assert_close(a,b,rtol=0,atol=0)
        frame=pd.DataFrame([r for r in records if r["run"]==checkpoint.parent.name]);summary={}
        for scope,positions in scopes.items():
            part=frame[frame.scope.eq(scope)];valid=part[part.both_nonzero]
            summary[scope]={"batches":len(part),"batches_with_both_nonzero_gradients":len(valid),
                "undefined_cosine_batches":len(part)-len(valid),
                "batch_mean_cosine":float(valid.cosine.mean()) if len(valid) else None,
                "batch_median_cosine":float(valid.cosine.median()) if len(valid) else None,
                "negative_cosine_batch_fraction":float(valid.cosine.lt(0).mean()) if len(valid) else None,
                "batch_median_name_to_completion_norm":float(valid.name_to_completion_norm.median()) if len(valid) else None,
                "geometry_of_mean_gradients":gradient_geometry([total["completion"][i] for i in positions],[total["name_only"][i] for i in positions]),
                "parameter_names":[items[i][0] for i in positions],"parameter_count":sum(parameters[i].numel() for i in positions)}
        record={"run":checkpoint.parent.name,"checkpoint_sha256":digest(checkpoint),"tasks":count,"scopes":summary,
            "indices_sha256":digest(args.output_dir/f"{checkpoint.parent.name}_task_indices.npy"),
            "parameters_grad_buffers_and_rng_unchanged":True,"elapsed_seconds":time.monotonic()-started}
        models.append(record)
        print({"run":record["run"],"scopes":{key:{field:value for field,value in row.items() if not field.startswith("parameter")} for key,row in summary.items()}},flush=True)
        del wrapper,model,panel,initial,gradients,total,parameters,items
    pd.DataFrame(records).to_csv(args.output_dir/"gradient_batches.csv",index=False)
    write_json(args.output_dir/"summary.json",{"models":models,"seed":args.seed,"data_panel_name_hashes":identity,
        "code_sha256":digest(Path(__file__)),"module_sha256":digest(ROOT/"src/foodcomp/research_gradient_geometry.py"),
        "complete_test_opened":False,"loss_normalization":"Same observed targets, source weights and original187-axis training denominator in both views.",
        "scope":"Local train-only eval-mode MAE gradients at selected checkpoints, no optimizer/update. Euclidean first-order geometry, not AdamW updates, training-history evidence or causal explanation of validation harm."})


if __name__=="__main__":main()

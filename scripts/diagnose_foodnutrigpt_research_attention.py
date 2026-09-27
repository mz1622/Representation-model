"""Train-only masked-key attention audit on completed source-free V9 checkpoints."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_r0 import digest,write_json
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION
from foodcomp.research_attention import attention_category_mass,replay_v9_attention


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,nargs="+",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--tasks",type=int,default=1024)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=20260922)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.tasks<1 or args.batch_size<1:raise ValueError("Positive diagnostic sizes required.")
    if len(set(x.parent.name for x in args.checkpoint))!=len(args.checkpoint):raise ValueError("Unique run names required.")
    args.output_dir.mkdir(parents=True);torch.set_num_threads(2)
    records=[];summaries=[];shared_panel=None
    for checkpoint in args.checkpoint:
        wrapper=NutritionModel(checkpoint);model,data=wrapper.model,wrapper.data
        if wrapper.kind!="v9":raise ValueError("Audit currently supports the unchanged hurdle V9 architecture.")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        panel=FamilyPanel(data,wrapper._cached_text,ROOT/"data/processed"/PANEL_VERSION,wrapper.device)
        panel_hash=digest(panel.root/"manifest.json")
        if panel_hash!=saved["panel_hash"] or (shared_panel is not None and panel_hash!=shared_panel):raise ValueError("Different training panels.")
        shared_panel=panel_hash
        indices=np.random.default_rng(args.seed).choice(len(panel.rows),min(args.tasks,len(panel.rows)),replace=False)
        if not np.isin(panel.rows[indices],data.train).all():raise AssertionError("Non-training audit task.")
        np.save(args.output_dir/f"{checkpoint.parent.name}_task_indices.npy",indices)
        nutrition=torch.as_tensor(data.axes.loss_group.eq("nutrition").to_numpy(),device=wrapper.device)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if wrapper.device.type=="cuda" else []
        model.eval();current={};differences={};seen_calls={i:0 for i in range(len(model.encoder.layers))}

        def collect(layer_index,weights):
            mass=attention_category_mass(weights,current["query_weight"],current["categories"])
            seen_calls[layer_index]+=1
            for head in range(weights.shape[1]):
                records.append({"run":checkpoint.parent.name,"batch":current["batch"],"layer":layer_index,
                    "head":head,"query_weight":current["weight_total"],"supervised_nutrition_queries":current["queries"],
                    "uniform_masked_key_mass":current["uniform_masked"],**{name:float(value[head]) for name,value in mass.items()}})

        try:
            with torch.no_grad():
                batches=0;skipped=0
                for start in range(0,len(indices),args.batch_size):
                    batch=panel.batch(indices[start:start+args.batch_size]);count=len(batch["axis"]);length=batch["axis"].shape[1]+2
                    if not batch["valid"].all():raise ValueError("This audit requires the fixed, unpadded query grid.")
                    target=batch["target"]&nutrition
                    q=batch["cell_weight"]*target/batch["axis_total"].clamp_min(1e-12)
                    if not q.sum()>0:skipped+=1;continue
                    query=torch.cat([torch.zeros((count,2),device=wrapper.device),q],1)
                    categories={name:torch.zeros((count,length),dtype=torch.bool,device=wrapper.device) for name in ["cls","name","visible_axes","masked_axes"]}
                    categories["cls"][:,0]=True;categories["name"][:,1]=True
                    categories["visible_axes"][:,2:]=~batch["masked"];categories["masked_axes"][:,2:]=batch["masked"]
                    uniform=categories["masked_axes"].float().mean(1)
                    current.update(query_weight=query,categories=categories,batch=start//args.batch_size,
                        weight_total=float(query.double().sum()),queries=int(target.sum()),
                        uniform_masked=float((uniform.double()*q.double().sum(1)).sum()/q.double().sum()))
                    original=model(batch)
                    instrumented=replay_v9_attention(model,batch,collect);batches+=1
                    for key,value in original.items():
                        torch.testing.assert_close(value,instrumented[key],rtol=0,atol=0)
                        differences[key]=max(differences.get(key,0.),float((value-instrumented[key]).abs().max()))
        except Exception as error:
            write_json(args.output_dir/"failure.json",{"run":checkpoint.parent.name,"status":"failed",
                "exception_type":type(error).__name__,"message":str(error),"complete_test_opened":False})
            raise
        if any(n!=batches for n in seen_calls.values()):raise AssertionError("Instrumentation skipped an encoder layer.")
        for key,value in initial.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(cpu_rng,torch.get_rng_state(),rtol=0,atol=0)
        for before,after in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []):torch.testing.assert_close(before,after,rtol=0,atol=0)
        frame=pd.DataFrame([r for r in records if r["run"]==checkpoint.parent.name])
        columns=["cls","name","visible_axes","masked_axes","uniform_masked_key_mass"]
        heads=[]
        for (layer,head),part in frame.groupby(["layer","head"]):
            heads.append({"layer":int(layer),"head":int(head),**{column:float(np.average(part[column],weights=part.query_weight)) for column in columns}})
        layers=pd.DataFrame(heads).groupby("layer")[columns].mean().reset_index().to_dict("records")
        summary={"run":checkpoint.parent.name,"checkpoint_sha256":digest(checkpoint),"data_hash":saved["data_hash"],
            "panel_hash":panel_hash,"tasks":len(indices),"batches":batches,"batches_without_nutrition_targets":skipped,
            "head_means":heads,"layer_means":layers,"instrumented_prediction_max_absolute_difference":differences,
            "parameters_and_rng_unchanged":True,"prediction_equality_checked":"all audited batches, bitwise",
            "method":"Off-path attention calls from exact layer inputs; encoder layers execute without hooks using their original forward path."}
        summaries.append(summary);print({"run":summary["run"],"layers":layers,"instrumentation_difference":differences},flush=True)
        del wrapper,model,panel,initial
    pd.DataFrame(records).to_csv(args.output_dir/"attention_batches.csv",index=False)
    write_json(args.output_dir/"summary.json",{"models":summaries,"seed":args.seed,"code_hash":digest(Path(__file__)),
        "module_hash":digest(ROOT/"src/foodcomp/research_attention.py"),"complete_test_opened":False,
        "aggregation":"Train-only sampled family tasks; supervised nutrition query weights use source-equal/train-axis denominators, then normalize over sampled query weight; heads equal. Not the full validation metric.",
        "scope":"Descriptive attention mass, eval mode, no optimizer/update. Masked keys can carry semantic priors and become contextualized in later layers. High attention mass alone does not establish harmful dilution or causal prediction importance. Requires a controlled training intervention before attribution."})


if __name__=="__main__":main()

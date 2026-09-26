"""Fixed exact-name retrieval panel with candidate vectors predicted from names only."""
import argparse
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_r0 import ResearchData,VERSION,write_json,digest
from foodcomp.research_text import prepare_names
from sklearn.linear_model import Ridge

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--method",choices=["name_predictions","ridge_to_text"],default="name_predictions")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started=time.monotonic();torch.set_num_threads(4)
    if args.method=="name_predictions":
        if args.checkpoint is None:raise ValueError("--checkpoint is required for name_predictions.")
        model=NutritionModel(args.checkpoint);data=model.data;device=model.device
    else:
        data=ResearchData(ROOT/"data/processed"/VERSION)
        text,cache=prepare_names(data,ROOT)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names=sorted(set(data.profiles.original_name.astype(str)))
    write_json(args.output_dir/"candidate_names.json",names)
    # Candidate names may include unseen names; vectors use model predictions only.
    axes=np.flatnonzero(data.axes.loss_group.eq("nutrition")&data.axes.loss_eligible)
    if args.method=="name_predictions":
        predicted=model.candidate_profiles(names)
        candidate=torch.as_tensor(np.log1p(predicted[:,axes]/data.scale[axes]),dtype=torch.float32,device=device)
    else:
        # No query name feature: only values and the observedness mask enter this map.
        train=data.train[data.observed[data.train][:,axes].sum(1)>=3]
        pp=data.profiles.iloc[train]
        source_counts=pp.groupby("exact_name_group_id").source_key.transform("nunique").to_numpy()
        profile_counts=pp.groupby(["exact_name_group_id","source_key"]).profile_id.transform("size").to_numpy()
        weights=1./(source_counts*profile_counts)
        xx=np.concatenate([data.values[train][:,axes],data.observed[train][:,axes]],axis=1)
        ridge=Ridge(alpha=1.,solver="lsqr",tol=1e-6).fit(xx,text[train],sample_weight=weights/weights.mean())
        np.savez(args.output_dir/"ridge.npz",coef=ridge.coef_,intercept=ridge.intercept_)
        lookup={str(n):i for i,n in enumerate(data.profiles.original_name)}
        candidate=torch.as_tensor(text[[lookup[n] for n in names]],device=device)
        candidate=torch.nn.functional.normalize(candidate,dim=1)
    name_index={name:i for i,name in enumerate(names)}
    records=[]
    for fraction in [1.,.3]:
        rows=data.validation
        values,visible=data.context(rows,mode="completion",visible_fraction=fraction)
        values=values[:,axes];visible=visible[:,axes]
        kept=visible.sum(1)>=3
        rows,values,visible=rows[kept],values[kept],visible[kept]
        for start in range(0,len(rows),128):
            rr=rows[start:start+128]
            m=torch.as_tensor(visible[start:start+128],dtype=torch.float32,device=device)
            y=torch.as_tensor(values[start:start+128],device=device)*m
            if args.method=="name_predictions":
                distance=(m @ candidate.square().T-2*y @ candidate.T+y.square().sum(1,keepdim=True))/m.sum(1,keepdim=True)
            else:
                features=np.concatenate([values[start:start+128]*visible[start:start+128],visible[start:start+128]],axis=1)
                mapped=torch.as_tensor(ridge.predict(features),dtype=torch.float32,device=device)
                distance=1-torch.nn.functional.normalize(mapped,dim=1)@candidate.T
            if not torch.isfinite(distance).all():raise FloatingPointError("Nonfinite retrieval distance.")
            correct=torch.as_tensor([name_index[str(data.profiles.original_name.iloc[r])] for r in rr],device=device)
            actual=distance[torch.arange(len(rr),device=device),correct]
            index=torch.arange(len(names),device=device)
            rank=1+(distance<actual[:,None]).sum(1)+((distance==actual[:,None])&(index[None,:]<correct[:,None])).sum(1)
            for row,r,n in zip(rr,rank.cpu().numpy(),visible[start:start+128].sum(1)):
                records.append({"profile_index":int(row),"visible_fraction":fraction,"observed_axes":int(n),"rank":int(r),"mrr":1/float(r),
                    "recall_at_1":int(r<=1),"recall_at_5":int(r<=5),"recall_at_10":int(r<=10)})
    result=pd.DataFrame(records).merge(data.profiles[["profile_index","source_key","exact_name_group_id"]],on="profile_index")
    result.to_parquet(args.output_dir/"ranks.parquet",index=False)
    metrics=["mrr","recall_at_1","recall_at_5","recall_at_10"]
    grouped=result.groupby(["visible_fraction","exact_name_group_id","source_key"])[metrics].mean().groupby(["visible_fraction","exact_name_group_id"]).mean()
    scores=grouped.groupby("visible_fraction")[metrics].mean().reset_index().to_dict("records")
    write_json(args.output_dir/"metrics.json",{"metrics":scores,"candidate_count":len(names),"query_profiles":result.groupby("visible_fraction").size().to_dict(),
        "method":args.method,"ridge_alpha":1. if args.method=="ridge_to_text" else None,
        "candidate_sha256":digest(args.output_dir/"candidate_names.json"),"checkpoint_sha256":digest(args.checkpoint) if args.checkpoint else None,
        "code_sha256":digest(Path(__file__)),
        "data_sha256":digest(data.root/"manifest.json"),"elapsed_seconds":time.monotonic()-started,
        "scoring":"source equal within exact-name candidate, candidate macro; at least 3 observed nutrition axes",
        "correct_answers":"exact original name only; no confirmed alias map available; not semantic identity accuracy",
        "unseen_names":"All query groups are validation-only. Candidate names include train+validation text, never their measured nutrition.",
        "complete_test_opened":False,"scientific_claim_allowed":False})
    print(scores)

if __name__=="__main__":main()

"""Compare full training-panel and validation errors of saved dense MLPs without refitting."""
import argparse
import copy
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate
from foodcomp.research_r0 import digest,score_predictions,write_json
from foodcomp.research_r1 import fingerprint_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,nargs="+",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if len({x.parent.name for x in args.checkpoint})!=len(args.checkpoint):raise ValueError("Unique checkpoint run names required.")
    args.output_dir.mkdir(parents=True);torch.set_num_threads(3)
    summaries=[];per_axis=[];data_hash=None
    for checkpoint in args.checkpoint:
        start=time.monotonic();wrapper=NutritionModel(checkpoint)
        if wrapper.kind not in {"mlp","name_mlp"}:raise ValueError("This diagnostic supports fused or name-only MLP runs.")
        actual_hash=digest(wrapper.data.root/"manifest.json")
        if data_hash is not None and actual_hash!=data_hash:raise ValueError("Diagnostic data views differ.")
        data_hash=actual_hash
        # Separate in-memory evaluator view. Frozen validation jobs and all files remain unchanged.
        train=copy.copy(wrapper.data)
        local_rows,local_axes=np.where(train.observed[train.train][:,train.targets])
        rows=train.train[local_rows];axes=train.targets[local_axes]
        if not train.profiles.iloc[rows].partition.eq("train").all():raise AssertionError("Non-training diagnostic target.")
        train.jobs=pd.DataFrame({"profile_index":rows,"axis_index":axes,"target":train.raw[rows,axes],"mask_family":train.families[axes]})
        if train.jobs.duplicated(["profile_index","axis_index"]).any():raise AssertionError("Duplicate training targets.")
        panel_hash=fingerprint_array(np.stack([rows,axes],axis=1))
        pred=evaluate(wrapper.model,train,wrapper._cached_text,wrapper.device)
        training,training_axes,_=score_predictions(train,pred)
        valid_pred=evaluate(wrapper.model,wrapper.data,wrapper._cached_text,wrapper.device)
        validation,validation_axes,_=score_predictions(wrapper.data,valid_pred)
        selected=["axis_index","scaled_log_mae","log_mae","positive_scaled_log_mae","zero_scaled_log_mae","candidate_support"]
        axis=training_axes[selected].merge(validation_axes[selected],on="axis_index",suffixes=("_train","_validation"),validate="one_to_one")
        axis=axis.merge(train.axes[["axis_index","canonical_name","loss_group"]],on="axis_index",validate="one_to_one")
        axis["run"]=checkpoint.parent.name;per_axis.append(axis)
        record={"run":checkpoint.parent.name,"kind":wrapper.kind,"checkpoint_sha256":digest(checkpoint),"training":training,"validation":validation,
            "training_job_count":len(train.jobs),"training_panel_sha256":panel_hash,"data_sha256":data_hash,
            "elapsed_seconds":time.monotonic()-start}
        summaries.append(record)
        print({"run":record["run"],"train_primary":training["nutrition"]["scaled_log_mae"],"validation_primary":validation["nutrition"]["scaled_log_mae"],"seconds":record["elapsed_seconds"]},flush=True)
        del wrapper,train,pred,valid_pred
    pd.concat(per_axis,ignore_index=True).to_csv(args.output_dir/"per_axis_fit.csv",index=False)
    write_json(args.output_dir/"summary.json",{"models":summaries,"code_sha256":digest(Path(__file__)),"complete_test_opened":False,
        "scope":"Full training-family reconstruction diagnostic, no optimization and no transform refitting. Validation panel/scales unchanged. Training and validation have different foods/support/distributions; a gap alone does not causally establish overfitting or rule out a useful capacity change. Checkpoint selection used validation as registered."})


if __name__=="__main__":main()

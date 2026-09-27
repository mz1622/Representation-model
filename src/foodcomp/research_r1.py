"""Shared family task panel and unbiased full-dataset axis-weighted supervision."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .research_r0 import digest,write_json

PANEL_VERSION="foodnutrigpt_v9_r1_tasks_v1"

def panel_arrays(data):
    families=np.unique(data.families[data.targets])
    eligible=np.zeros(len(data.axes),bool);eligible[data.targets]=True
    masks=families[:,None]==data.families[None,:]
    rr=[];ff=[]
    for i,mask in enumerate(masks):
        rows=data.train[data.observed[data.train][:,mask&eligible].any(1)]
        rr.append(rows);ff.append(np.full(len(rows),i,np.int16))
    return np.concatenate(rr),np.concatenate(ff),families,masks

def fingerprint_array(array):
    x=np.ascontiguousarray(array)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()

def build_panel(data,text,cache,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rows,fi,families,masks=panel_arrays(data)
    np.savez_compressed(output/"tasks.npz",rows=rows,families=fi,family_names=families,masks=masks)
    eligible=np.zeros(len(data.axes),bool);eligible[data.targets]=True
    covered=np.zeros_like(data.observed,dtype=np.uint8)
    receipts=[]
    for index,family in enumerate(families):
        rr=rows[fi==index]
        visible=data.observed[rr]&~masks[index]
        target=data.observed[rr]&masks[index]&eligible
        covered[rr]+=target
        tensor_features=np.concatenate([text[rr],np.where(visible,data.values[rr],0),visible.astype(np.float32)],1)
        axis=int(np.flatnonzero(masks[index]&eligible)[0])
        np.testing.assert_array_equal(tensor_features,data.dense_features(rr,text,axis))
        # Verify every axis uses exactly the full row set, labels and source weights of the R0 tree learner.
        for a in np.flatnonzero(masks[index]&eligible):
            train=rr[target[:,a]]
            expected=data.train[data.observed[data.train,a]]
            np.testing.assert_array_equal(train,expected)
            if not np.isfinite(data.values[train,a]).all() or (data.weights[train,a]<=0).any():raise ValueError("Invalid training labels/weights.")
        receipts.append({"family":str(family),"tasks":len(rr),"targets":int(target.sum()),
            "effective_features_sha256":fingerprint_array(tensor_features),
            "raw_labels_sha256":fingerprint_array(np.where(target,data.raw[rr],0)),
            "source_weights_sha256":fingerprint_array(np.where(target,data.weights[rr],0)),
            "target_mask_sha256":fingerprint_array(target),"training_rows_sha256":fingerprint_array(rr)})
    expected=np.zeros_like(covered);expected[data.train]=data.observed[data.train]&eligible
    np.testing.assert_array_equal(covered,expected)
    write_json(output/"manifest.json",{"version":PANEL_VERSION,"data_hash":digest(data.root/"manifest.json"),
        "name_cache_hash":digest(Path(cache)/"manifest.json"),"view":data.view,"tasks":len(rows),
        "observed_target_cells":int(covered.sum()),"families":len(families),"tasks_sha256":digest(output/"tasks.npz"),
        "exhaustive_tree_input_row_target_weight_contract_passed":True,"each_training_target_covered_exactly_once":True,
        "validation_jobs_sha256":data.manifest["artifact_hashes"][f"{data.view}_validation_jobs.parquet"],
        "baseline_reuse":"R0 RF/XGB use exactly these per-axis rows, canonical labels, scales, visible context and source weights. No data/split/scoring change.",
        "neural_weighting":"sum_axis sum_cells(weight*error)/sum_train_axis(weight), macro over 187 axes; unbiased minibatch estimate over uniform family tasks",
        "complete_test_opened":False,"receipts":receipts})

class FamilyPanel:
    def __init__(self,data,text,panel_root,device):
        import json
        self.data=data;self.device=device;self.root=Path(panel_root)
        manifest=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        if digest(self.root/"tasks.npz")!=manifest["tasks_sha256"] or digest(data.root/"manifest.json")!=manifest["data_hash"]:
            raise ValueError("Stale task panel.")
        if manifest["view"]!=data.view:raise ValueError("Task/data view mismatch.")
        with np.load(self.root/"tasks.npz",allow_pickle=False) as f:
            self.rows=f["rows"];self.family_ids=f["families"]
            self.masks=torch.as_tensor(f["masks"],device=device)
        self.manifest=manifest
        self.values=torch.as_tensor(data.values,device=device)
        self.observed=torch.as_tensor(data.observed,device=device)
        self.text=torch.as_tensor(text,device=device)
        self.source=torch.as_tensor(data.profiles.source_index.to_numpy(),dtype=torch.long,device=device)
        self.weights=torch.as_tensor(data.weights,device=device)
        totals=data.weights[data.train].sum(0,dtype=np.float64)
        self.axis_total=torch.as_tensor(totals,dtype=torch.float32,device=device)
        self.eligible=torch.zeros(len(data.axes),device=device,dtype=torch.bool);self.eligible[data.targets]=True
        self.axes=torch.arange(len(data.axes),device=device)
        self.axis_count=len(data.targets)

    def batch(self,indices,*,name_only=False):
        rows=torch.as_tensor(self.rows[indices],dtype=torch.long,device=self.device)
        fi=torch.as_tensor(self.family_ids[indices],dtype=torch.long,device=self.device)
        visible=self.observed[rows]&~self.masks[fi]
        if name_only:visible[:]=False
        target=self.observed[rows]&self.masks[fi]&self.eligible
        return {"axis":self.axes.expand(len(rows),-1),"value":self.values[rows],"text":self.text[rows],
            "masked":~visible,"valid":torch.ones_like(visible),"source":self.source[rows],"target":target,
            "positive":self.values[rows]>0,"cell_weight":self.weights[rows],"axis_total":self.axis_total,
            "objective_multiplier":len(self.rows)/(len(rows)*self.axis_count)}

def panel_loss(outputs,batch,amount_weight=1.,objective="hurdle",*,axis_loss_scale=None):
    amount=F.smooth_l1_loss(outputs["amount_normalized"],batch["value"],reduction="none")
    if objective=="hurdle" and "positive_logit" in outputs:
        error=amount_weight*amount*batch["positive"]+F.binary_cross_entropy_with_logits(outputs["positive_logit"],batch["positive"].float(),reduction="none")
    elif objective=="smooth_l1":error=amount
    elif objective=="mae":error=(outputs["amount_normalized"]-batch["value"]).abs()
    elif objective=="hurdle":error=amount  # direct regression MLP control
    else:raise ValueError(objective)
    weights=batch["cell_weight"]*batch["target"]/batch["axis_total"].clamp_min(1e-12)
    if axis_loss_scale is not None:
        if axis_loss_scale.shape!=batch["axis_total"].shape or not torch.isfinite(axis_loss_scale).all() or not (axis_loss_scale>0).all():
            raise ValueError("Expected positive finite coefficients for the complete axis grid.")
        weights=weights*axis_loss_scale
    value=(weights*error).sum()*batch["objective_multiplier"]
    if not torch.isfinite(value):raise FloatingPointError("Nonfinite panel loss.")
    return value

def model_loss(model,batch,kind,config,objective="hurdle",*,axis_loss_scale=None):
    outputs=model(batch)
    value=panel_loss(outputs,batch,config.amount_loss_weight,objective,axis_loss_scale=axis_loss_scale)
    if kind in {"v9","v9_direct"}:
        calibrated=panel_loss(model.calibrated_outputs(outputs,batch),batch,config.amount_loss_weight,objective,axis_loss_scale=axis_loss_scale)
        weight=config.source_calibrated_loss_weight
        value=(value+weight*calibrated)/(1+weight)+config.source_residual_l2*model.source_residual_penalty()
    if not torch.isfinite(value):raise FloatingPointError("Nonfinite calibrated loss.")
    return value

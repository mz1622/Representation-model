"""R0 controlled neural adapters; fixed query grid is independent of observed labels."""
from pathlib import Path
import copy
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
import train_global_foodnutrigpt_v8_single_stage as legacy_v8
import train_global_foodnutrigpt_v9_source_calibrated as legacy_v9

class DirectSourceCalibratedModel(legacy_v9.SourceCalibratedFoodNutriGPT):
    """Same V9 backbone/amount head; direct value prediction with no presence multiplier.

    Keep construction and forward-call order of the inactive presence head so
    matched-seed initialization and dropout RNG consumption remain comparable.
    Inactive presence parameters are frozen and never enter a loss or prediction.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.presence_head.requires_grad_(False)
        self.source_presence_residual.requires_grad_(False)

    def forward(self,batch):
        outputs=super().forward(batch)
        return {"amount_normalized":outputs["amount_normalized"]}

    def calibrated_outputs(self,base,batch):
        source=batch["source"].unsqueeze(1).expand_as(batch["axis"])
        residual=self._centred_source_offsets(self.source_amount_residual)
        return {"amount_normalized":base["amount_normalized"]+residual[source,batch["axis"]]}

    def source_residual_penalty(self):
        return self.source_amount_residual.weight[self.train_source_indices].square().mean()

class DenseModel(nn.Module):
    def __init__(self, text_dim, axes, kind, width=256, normalization="layer_norm", task_heads="shared", text_conditioning="none", query_residual=False):
        super().__init__()
        if not isinstance(width,int) or width<1:raise ValueError("Positive integer MLP width required.")
        if normalization not in {"layer_norm","none"}:raise ValueError("Unknown MLP normalization.")
        if task_heads not in {"shared","separate"} or (task_heads=="separate" and kind!="mlp"):
            raise ValueError("Separate task heads require the fused MLP.")
        if text_conditioning not in {"none","train_unique_name"} or (text_conditioning!="none" and kind!="mlp"):
            raise ValueError("Name conditioning requires the fused MLP.")
        if not isinstance(query_residual,bool) or (query_residual and (kind!="mlp" or task_heads!="shared")):
            raise ValueError("Axis-query residual requires a fused MLP with a shared head.")
        self.kind = kind
        size = text_dim if kind == "name_mlp" else axes*2 if kind == "numeric_mlp" else text_dim+axes*2
        self.encoder = nn.Sequential(nn.Linear(size,width),nn.GELU(),nn.LayerNorm(width) if normalization=="layer_norm" else nn.Identity(),
                                     nn.Linear(width,width),nn.GELU())
        self.head = nn.Linear(width, axes)
        # Clone exactly; no extra initialization draw changes the matched RNG stream.
        if task_heads=="separate":self.name_head=copy.deepcopy(self.head)
        if text_conditioning=="train_unique_name":
            from .research_conditioning import FrozenNameStandardizer
            self.name_standardizer=FrozenNameStandardizer(text_dim)
        if query_residual:
            from .research_query import AxisQueryResidual
            # Models are constructed on CPU before transfer to the training device.
            # Extra parameters use a local initialization stream without changing
            # the parent's subsequent random draws.
            with torch.random.fork_rng(devices=[]):
                self.query_residual=AxisQueryResidual(width,axes)

    def encode(self, batch):
        numeric = torch.cat([torch.where(batch["masked"],0.,batch["value"]),
                             (~batch["masked"]).float()],1)
        text = self.name_standardizer(batch["text"]) if hasattr(self,"name_standardizer") else batch["text"]
        x = text if self.kind == "name_mlp" else numeric if self.kind == "numeric_mlp" else torch.cat([text,numeric],1)
        return self.encoder(x)

    def forward(self, batch):
        hidden=self.encode(batch)
        amount=self.head(hidden)
        if hasattr(self,"query_residual"):amount=amount+self.query_residual(hidden)
        if hasattr(self,"name_head"):
            name_only=batch["masked"].all(dim=1)
            amount=torch.where(name_only[:,None],self.name_head(hidden),amount)
        return {"amount_normalized": amount}

def make_model(data, text_dim, kind, *, amount_weight=1., source_weight=1., mlp_width=256, mlp_normalization="layer_norm", mlp_task_heads="shared", mlp_text_conditioning="none", mlp_query_residual=False):
    config=legacy_v9.Config(amount_loss_weight=amount_weight, source_calibrated_loss_weight=source_weight)
    if mlp_task_heads!="shared" and kind!="mlp":raise ValueError("Separate task heads require the fused MLP.")
    if mlp_text_conditioning not in {"none","train_unique_name"} or (mlp_text_conditioning!="none" and kind!="mlp"):
        raise ValueError("Name conditioning requires the fused MLP.")
    if not isinstance(mlp_query_residual,bool) or (mlp_query_residual and (kind!="mlp" or mlp_task_heads!="shared")):
        raise ValueError("Axis-query residual requires a fused MLP with a shared head.")
    if kind in {"mlp","name_mlp","numeric_mlp"}:
        return DenseModel(text_dim,len(data.axes),kind,mlp_width,mlp_normalization,mlp_task_heads,mlp_text_conditioning,mlp_query_residual),config
    source_count=int(data.profiles.source_index.max())+1
    if kind in {"v9","v9_direct"}:
        sources=np.unique(data.profiles.iloc[data.train].source_index)
        cls=legacy_v9.SourceCalibratedFoodNutriGPT if kind=="v9" else DirectSourceCalibratedModel
        return cls(text_dim,len(data.axes),source_count,sources,config),config
    if kind=="v8_optimized":
        v8config=legacy_v8.Config(amount_loss_weight=amount_weight,loss_mode="all_axis",source_dropout=1.,text_only_probability=0.)
        return legacy_v8.SourceAwareFoodNutriGPT(text_dim,len(data.axes),source_count,v8config),v8config
    raise ValueError(kind)

def batch_from_arrays(values, visible, text, device, *, source=None, targets=None, weights=None, positive=None):
    n,a=values.shape
    # Masked entries all use one learned mask value, including truly missing cells.
    # Every input uses the same full axis grid. Observed target availability is not an encoder input.
    batch={
        "axis":torch.arange(a,device=device).expand(n,-1),
        "value":torch.as_tensor(values,dtype=torch.float32,device=device),
        "masked":torch.as_tensor(~visible,dtype=torch.bool,device=device),
        "valid":torch.ones((n,a),dtype=torch.bool,device=device),
        "text":torch.as_tensor(text,dtype=torch.float32,device=device),
        "source":torch.as_tensor(np.zeros(n) if source is None else source,dtype=torch.long,device=device),
    }
    if targets is not None:
        batch["target"]=torch.as_tensor(targets,dtype=torch.bool,device=device)
        batch["cell_weight"]=torch.as_tensor(weights,dtype=torch.float32,device=device)
        batch["positive"]=torch.as_tensor(positive,dtype=torch.bool,device=device)
    return batch

def training_batch(data,text,rows,device,rng,kind,mask_ratio=.3,text_only_probability=0.):
    visible=data.observed[rows].copy()
    eligible=np.zeros(len(data.axes),bool);eligible[data.targets]=True
    for i,row in enumerate(rows):
        families=np.unique(data.families[data.observed[row]&eligible])
        if kind=="name_mlp" or rng.random()<text_only_probability:
            visible[i]=False
        elif len(families):
            chosen=rng.choice(families,size=max(1,int(np.ceil(len(families)*mask_ratio))),replace=False)
            visible[i,np.isin(data.families,chosen)]=False
    targets=data.observed[rows]&~visible&eligible
    source=data.profiles.iloc[rows].source_index.to_numpy()
    if kind=="v8_optimized":source=np.zeros_like(source)
    return batch_from_arrays(data.values[rows],visible,text[rows],device,source=source,targets=targets,
                             weights=data.weights[rows],positive=data.raw[rows]>0)

def macro_loss(outputs,batch,amount_weight):
    active=batch["target"]
    weights=batch["cell_weight"]*active
    total=weights.sum(0)
    if not (total>0).any():raise ValueError("Training batch has no supervised targets.")
    y=batch["value"]
    error=F.smooth_l1_loss(outputs["amount_normalized"],y,reduction="none")
    if "positive_logit" in outputs:
        error=amount_weight*error*batch["positive"]+F.binary_cross_entropy_with_logits(
            outputs["positive_logit"],batch["positive"].float(),reduction="none")
    loss=((error*weights).sum(0)/total.clamp_min(1e-12))[total>0].mean()
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss.")
    return loss

def loss(model,batch,kind,config):
    outputs=model(batch)
    free=macro_loss(outputs,batch,config.amount_loss_weight)
    if kind=="v9":
        calibrated=macro_loss(model.calibrated_outputs(outputs,batch),batch,config.amount_loss_weight)
        weight=config.source_calibrated_loss_weight
        result=(free+weight*calibrated)/(1+weight)+config.source_residual_l2*model.source_residual_penalty()
    else:result=free
    if not torch.isfinite(result):raise FloatingPointError("Nonfinite total loss.")
    return result

OUTPUT_QUERY_POLICY = "caller_or_schema_axes_v1"


def validate_query_axes(axes, axis_count):
    indices=np.asarray(axes)
    if indices.ndim!=1 or not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError("Requested axes must be a nonempty one-dimensional integer sequence.")
    if (indices<0).any() or (indices>=axis_count).any() or len(np.unique(indices))!=len(indices):
        raise ValueError("Requested axes must be unique and in range.")
    return np.ascontiguousarray(indices,dtype=np.int64)


def schema_query_axes(data, *, family=None, mode="completion"):
    if mode not in {"completion","name_only"}:raise ValueError("Unknown prediction task.")
    axes=validate_query_axes(data.targets,len(data.axes))
    if mode=="name_only":
        if family is not None:raise ValueError("Name-only queries cannot specify a masked family.")
        return axes
    if family is None:raise ValueError("Completion evaluation requires a caller-specified family.")
    return validate_query_axes(axes[data.families[axes]==family],len(data.axes))


def predictions_from_outputs(outputs,scale,*,target_axes=None):
    amount=outputs["amount_normalized"]
    if not torch.isfinite(amount).all():raise FloatingPointError("Nonfinite pre-inverse prediction.")
    scale_tensor=torch.as_tensor(scale,device=amount.device,dtype=amount.dtype)
    if not torch.isfinite(scale_tensor).all() or not (scale_tensor>0).all():
        raise FloatingPointError("Invalid prediction scale.")
    probability=None
    if "positive_logit" in outputs:
        if not torch.isfinite(outputs["positive_logit"]).all():raise FloatingPointError("Nonfinite presence logit.")
        probability=torch.sigmoid(outputs["positive_logit"])
    if target_axes is not None:
        if amount.ndim!=2 or scale_tensor.ndim!=1 or len(scale_tensor)!=amount.shape[1]:
            raise ValueError("Requested decoding requires batch-by-axis outputs and one scale per axis.")
        if probability is not None and probability.shape!=amount.shape:raise ValueError("Presence shape mismatch.")
        axes=validate_query_axes(target_axes,amount.shape[1])
        index=torch.as_tensor(axes,device=amount.device)
        amount=amount.index_select(1,index);scale_tensor=scale_tensor.index_select(0,index)
        if probability is not None:probability=probability.index_select(1,index)
    u=amount.clamp_min(0)
    raw=torch.expm1(u)*scale_tensor
    if probability is not None:
        raw=raw*probability
    invalid=~torch.isfinite(raw)
    if invalid.any():
        # Preserve every finite historical output bitwise. Only intermediate
        # overflow is recomputed, using the same rounded scale/probability.
        # The result must still fit the original output dtype; never cap it.
        wide=torch.expm1(u.to(torch.float64))*scale_tensor.to(torch.float64)
        if probability is not None:wide=wide*probability.to(torch.float64)
        raw=torch.where(invalid,wide.to(raw.dtype),raw)
    if not torch.isfinite(raw).all():raise FloatingPointError("Nonfinite prediction.")
    return raw,probability

@torch.no_grad()
def predict_arrays(model,data,text,rows,device,*,family=None,mode="completion",batch_size=128,target_axes=None):
    values=data.values[rows]
    visible=data.observed[rows].copy()
    if family is not None:visible[:,data.families==family]=False
    if mode=="name_only":visible[:]=False
    result=[];probabilities=[]
    model.eval()
    for start in range(0,len(rows),batch_size):
        batch=batch_from_arrays(values[start:start+batch_size],visible[start:start+batch_size],text[rows[start:start+batch_size]],device)
        raw,prob=predictions_from_outputs(model(batch),data.scale,target_axes=target_axes)
        result.append(raw.cpu().numpy())
        if prob is not None:probabilities.append(prob.cpu().numpy())
    return np.concatenate(result),np.concatenate(probabilities) if probabilities else None

def evaluate(model,data,text,device,mode="completion"):
    pieces=[]
    if mode=="name_only":
        queries=schema_query_axes(data,mode=mode)
        raw,prob=predict_arrays(model,data,text,data.validation,device,mode=mode,target_axes=queries)
        positions=np.searchsorted(data.validation,data.jobs.profile_index)
        axes=data.jobs.axis_index.to_numpy()
        index=np.full(len(data.axes),-1,dtype=int);index[queries]=np.arange(len(queries));selected=index[axes]
        if (selected<0).any():raise ValueError("Scoring axis is outside caller query set.")
        frame=data.jobs[["profile_index","axis_index"]].copy()
        frame["prediction"]=raw[positions,selected]
        if prob is not None:frame["positive_probability"]=prob[positions,selected]
        return frame
    if mode!="completion":raise ValueError("Unknown prediction task.")
    for family,jobs in data.jobs.groupby("mask_family"):
        rows=np.sort(jobs.profile_index.unique())
        queries=schema_query_axes(data,family=family)
        raw,prob=predict_arrays(model,data,text,rows,device,family=family,target_axes=queries)
        position=np.searchsorted(rows,jobs.profile_index)
        axes=jobs.axis_index.to_numpy()
        index=np.full(len(data.axes),-1,dtype=int);index[queries]=np.arange(len(queries));selected=index[axes]
        if (selected<0).any():raise ValueError("Scoring axis is outside caller query set.")
        frame=jobs[["profile_index","axis_index"]].copy()
        frame["prediction"]=raw[position,selected]
        if prob is not None:frame["positive_probability"]=prob[position,selected]
        pieces.append(frame)
    return pd.concat(pieces,ignore_index=True)

"""R5: nutrition-only mapping into a fixed name-only text space."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .research_r0 import digest,write_json
from .research_r1 import fingerprint_array
from .research_text import prepare_names,NameEncoder

SCORING='source equal within exact-name candidate, candidate macro; at least 3 observed nutrition axes'
CORRECT='exact original name only; no confirmed alias map available; not semantic identity accuracy'


def numeric_features(values,visible):
    values=np.asarray(values);visible=np.asarray(visible)
    if values.ndim!=2 or values.shape!=visible.shape or visible.dtype!=np.bool_:
        raise ValueError('Expected matching two-dimensional values and boolean masks.')
    if not np.isfinite(values[visible]).all() or (values[visible]<0).any():
        raise ValueError('Visible transformed nutrition values must be finite and nonnegative.')
    return np.concatenate([np.where(visible,values,0),visible],axis=1).astype(np.float32)


def source_equal_weights(profiles):
    sources=profiles.groupby('exact_name_group_id').source_key.transform('nunique').to_numpy()
    counts=profiles.groupby(['exact_name_group_id','source_key']).profile_id.transform('size').to_numpy()
    weights=1./(sources*counts)
    if not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid training weights.')
    return weights/weights.mean()


class NameSpace:
    """Vectors are a pure function of names and the frozen training-fitted PCA."""
    def __init__(self,data,repo):
        self.repo=Path(repo);self.text,self.cache=prepare_names(data,self.repo)
        names=data.profiles.original_name.astype(str).to_numpy()
        self.names,first,self.profile_name_ids=np.unique(names,return_index=True,return_inverse=True)
        self.features=self.text[first]
        if not np.array_equal(self.features[self.profile_name_ids],self.text):raise ValueError('Same name has different cached features.')
        if not np.isfinite(self.features).all() or (np.linalg.norm(self.features,axis=1)<=1e-12).any():raise ValueError('Invalid fixed name vectors.')
        self.lookup={name:i for i,name in enumerate(self.names)}
        self.pca=dict(np.load(self.cache/'pca.npz',allow_pickle=False));self.encoder=None

    def encode(self,names):
        result=np.empty((len(names),self.features.shape[1]),np.float32);missing=[]
        for i,name in enumerate(names):
            if not isinstance(name,str) or not name.strip():raise ValueError('Nonempty original names required.')
            if name in self.lookup:result[i]=self.features[self.lookup[name]]
            else:missing.append(i)
        if missing:
            if self.encoder is None:self.encoder=NameEncoder(self.repo/'data/cache/huggingface')
            raw=self.encoder.encode([names[i] for i in missing])
            result[missing]=(raw-self.pca['mean'])@self.pca['components'].T
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite name representation.')
        return result


class AlignmentPanel:
    def __init__(self,data,repo,device):
        self.data=data;self.device=torch.device(device);self.namespace=NameSpace(data,repo)
        self.axes=np.flatnonzero(data.axes.loss_group.eq('nutrition')&data.axes.loss_eligible)
        if len(self.axes)!=142:raise ValueError('Expected142 nutrition axes.')
        self.rows=data.train[data.observed[data.train][:,self.axes].sum(1)>=3]
        profiles=data.profiles.iloc[self.rows]
        if not profiles.partition.eq('train').all():raise ValueError('Nontraining row in training panel.')
        self.features=numeric_features(data.values[self.rows][:,self.axes],data.observed[self.rows][:,self.axes])
        self.weights=source_equal_weights(profiles)
        self.targets=self.namespace.text[self.rows]
        self.name_ids=self.namespace.profile_name_ids[self.rows]
        self.candidates=F.normalize(torch.as_tensor(self.namespace.features,device=self.device),dim=1)
        self.queries={}
        for fraction in [1.,.3]:
            rows=data.validation
            values,visible=data.context(rows,mode='completion',visible_fraction=fraction)
            values=values[:,self.axes];visible=visible[:,self.axes]
            kept=visible.sum(1)>=3;rows=rows[kept];visible=visible[kept]
            self.queries[fraction]={'rows':rows,'features':numeric_features(values[kept],visible),
                'visible_count':visible.sum(1),'correct_name_id':self.namespace.profile_name_ids[rows]}
        self.manifest={'data_sha256':digest(data.root/'manifest.json'),'name_cache_sha256':digest(self.namespace.cache/'manifest.json'),
            'training_rows':len(self.rows),'training_names':int(np.unique(self.name_ids).size),
            'training_rows_sha256':fingerprint_array(self.rows),'training_features_sha256':fingerprint_array(self.features),
            'training_targets_sha256':fingerprint_array(self.targets),'training_weights_sha256':fingerprint_array(self.weights),
            'nutrition_axes':self.axes.tolist(),'candidate_count':len(self.namespace.names),
            'candidate_features_sha256':fingerprint_array(self.namespace.features),
            'queries':{str(f):{'profiles':len(q['rows']),'rows_sha256':fingerprint_array(q['rows']),
                'features_sha256':fingerprint_array(q['features']),'visible_counts_sha256':fingerprint_array(q['visible_count'])} for f,q in self.queries.items()},
            'scope':'Train-only fit; names-only candidate vectors, nutrition-only query features. Names/IDs/source groups are supervision or scoring metadata, not model inputs.',
            'complete_test_opened':False}


class NumericNameMapper(nn.Module):
    def __init__(self,input_dim=284,text_dim=32,width=512):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        self.head=nn.Linear(width,text_dim)

    def forward(self,features):
        if not torch.isfinite(features).all():raise FloatingPointError('Nonfinite mapper input.')
        mapped=self.head(self.encoder(features))
        if not torch.isfinite(mapped).all():raise FloatingPointError('Nonfinite mapped representation.')
        return mapped


def mapping_loss(mapped,target_vectors,name_ids,weights,*,objective,population_size,weight_sum,temperature=.07):
    if mapped.shape!=target_vectors.shape or mapped.ndim!=2 or name_ids.shape!=mapped.shape[:1] or weights.shape!=name_ids.shape:
        raise ValueError('Mismatched mapping batch shapes.')
    if not torch.isfinite(mapped).all() or not torch.isfinite(target_vectors).all() or not torch.isfinite(weights).all() or not (weights>0).all():
        raise FloatingPointError('Invalid mapping loss inputs.')
    if not np.isfinite(weight_sum) or weight_sum<=0 or population_size<1:raise ValueError('Invalid global loss denominator.')
    if objective=='mse':
        per_row=(mapped-target_vectors).square().mean(1)
    elif objective=='contrastive':
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required.')
        unique,inverse=torch.unique(name_ids,sorted=True,return_inverse=True)
        # Exact-name duplicates are one candidate column, never false negatives.
        first=torch.full((len(unique),),len(name_ids),dtype=torch.long,device=name_ids.device)
        first.scatter_reduce_(0,inverse,torch.arange(len(name_ids),device=name_ids.device),reduce='amin',include_self=True)
        candidates=target_vectors[first]
        if not torch.equal(candidates[inverse],target_vectors):raise ValueError('Duplicate exact-name vectors disagree.')
        logits=F.normalize(mapped,dim=1)@F.normalize(candidates,dim=1).T/temperature
        per_row=F.cross_entropy(logits,inverse,reduction='none')
    else:raise ValueError(objective)
    value=(per_row*weights).sum()*(population_size/(len(mapped)*weight_sum))
    if not torch.isfinite(value):raise FloatingPointError('Nonfinite mapping loss.')
    return value


def exact_ranks(distance,correct):
    if distance.ndim!=2 or correct.shape!=(len(distance),) or correct.dtype!=torch.long:raise ValueError('Invalid rank inputs.')
    if not torch.isfinite(distance).all():raise FloatingPointError('Nonfinite retrieval distance.')
    if (correct<0).any() or (correct>=distance.shape[1]).any():raise ValueError('Correct candidate out of bounds.')
    actual=distance[torch.arange(len(distance),device=distance.device),correct]
    index=torch.arange(distance.shape[1],device=distance.device)
    return 1+(distance<actual[:,None]).sum(1)+((distance==actual[:,None])&(index[None,:]<correct[:,None])).sum(1)


@torch.no_grad()
def evaluate_mapping(predict,panel):
    records=[]
    for fraction,q in panel.queries.items():
        for start in range(0,len(q['rows']),128):
            sl=slice(start,start+128);mapped=predict(q['features'][sl])
            mapped=torch.as_tensor(mapped,dtype=torch.float32,device=panel.device)
            if mapped.shape!=(len(q['rows'][sl]),panel.candidates.shape[1]):raise ValueError('Wrong mapped vector shape.')
            distance=1-F.normalize(mapped,dim=1)@panel.candidates.T
            ranks=exact_ranks(distance,torch.as_tensor(q['correct_name_id'][sl],dtype=torch.long,device=panel.device)).cpu().numpy()
            for row,rank,count in zip(q['rows'][sl],ranks,q['visible_count'][sl]):
                records.append({'profile_index':int(row),'visible_fraction':fraction,'observed_axes':int(count),'rank':int(rank),
                    'mrr':1/float(rank),'recall_at_1':int(rank<=1),'recall_at_5':int(rank<=5),'recall_at_10':int(rank<=10)})
    result=pd.DataFrame(records).merge(panel.data.profiles[['profile_index','source_key','exact_name_group_id']],on='profile_index',validate='many_to_one')
    columns=['mrr','recall_at_1','recall_at_5','recall_at_10']
    groups=result.groupby(['visible_fraction','exact_name_group_id','source_key'])[columns].mean().groupby(['visible_fraction','exact_name_group_id']).mean()
    scores=groups.groupby('visible_fraction')[columns].mean().reset_index().to_dict('records')
    return result,scores


def selection_score(scores):
    if {row['visible_fraction'] for row in scores}!={.3,1.} or len(scores)!=2:raise ValueError('Both fixed scenarios required.')
    result=float(np.mean([row['mrr'] for row in scores]))
    if not np.isfinite(result):raise FloatingPointError('Nonfinite retrieval selection.')
    return result


def save_evaluation(out,panel,ranks,scores,*,method,extra=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() or (out/'ranks.parquet').exists():raise FileExistsError('Existing evaluation retained.')
    write_json(out/'candidate_names.json',panel.namespace.names.tolist())
    ranks.to_parquet(out/'ranks.parquet',index=False)
    metrics={'metrics':scores,'selection_mean_mrr':selection_score(scores),'candidate_count':len(panel.namespace.names),
        'query_profiles':ranks.groupby('visible_fraction').size().to_dict(),'method':method,
        'candidate_sha256':digest(out/'candidate_names.json'),'data_sha256':panel.manifest['data_sha256'],
        'name_cache_sha256':panel.manifest['name_cache_sha256'],'scoring':SCORING,'correct_answers':CORRECT,
        'unseen_names':'All query groups are validation-only. Candidate names include train+validation text, never their measured nutrition.',
        'complete_test_opened':False,'scientific_claim_allowed':False}
    if extra:metrics.update(extra)
    write_json(out/'metrics.json',metrics)


def state_fingerprint(model):
    h=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(fingerprint_array(tensor.detach().cpu().numpy()).encode())
    return h.hexdigest()

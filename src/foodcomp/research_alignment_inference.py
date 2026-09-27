"""Independent retrieval specialist: no nutrition prediction head is implied."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .research_r0 import ResearchData,VERSION,digest
from .research_alignment import NameSpace,NumericNameMapper,numeric_features


class AlignmentModel:
    def __init__(self,run_dir,repo=None,device=None):
        self.repo=Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        run_dir=Path(run_dir);manifest=json.loads((run_dir/'run_manifest.json').read_text())
        if manifest['status']!='complete':raise ValueError('Only completed mapping models can be loaded.')
        self.data=ResearchData(self.repo/'data/processed'/VERSION)
        self.namespace=NameSpace(self.data,self.repo)
        if digest(self.data.root/'manifest.json')!=manifest['data_hash'] or digest(self.namespace.cache/'manifest.json')!=manifest['name_cache_hash']:
            raise ValueError('Mapping model data/text fingerprint mismatch.')
        self.axes=np.flatnonzero(self.data.axes.loss_group.eq('nutrition')&self.data.axes.loss_eligible)
        self.axis_position={int(axis):i for i,axis in enumerate(self.axes)}
        self.axis_names={str(self.data.axes.canonical_name.iloc[a]):int(a) for a in self.axes}
        self.kind=manifest['kind'];self.model=None
        if self.kind=='ridge':
            path=run_dir/'ridge.npz'
            if digest(path)!=manifest['checkpoint_hash']:raise ValueError('Ridge checkpoint changed.')
            with np.load(path,allow_pickle=False) as f:self.coef=f['coef'].copy();self.intercept=f['intercept'].copy()
            if self.coef.shape!=(32,284) or self.intercept.shape!=(32,) or not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all():raise ValueError('Invalid Ridge checkpoint.')
        elif self.kind=='mlp':
            path=run_dir/'best_model.pt'
            if digest(path)!=manifest['checkpoint_hash']:raise ValueError('Mapping checkpoint changed.')
            saved=torch.load(path,map_location=self.device,weights_only=True)
            self.model=NumericNameMapper(width=saved['args']['width']).to(self.device)
            self.model.load_state_dict(saved['model_state']);self.model.eval()
        else:raise ValueError('Unknown mapping model.')

    def profile_features(self,observed_profile):
        values=np.zeros((1,len(self.axes)),np.float32);visible=np.zeros_like(values,bool)
        for axis,value in observed_profile.items():
            if isinstance(axis,(int,np.integer)) and not isinstance(axis,bool):index=int(axis)
            elif axis in self.axis_names:index=self.axis_names[axis]
            else:raise ValueError(f'Unknown nutrition axis: {axis}')
            if index not in self.axis_position:raise ValueError('Independent mapper accepts supervised nutrition axes only.')
            if not np.isfinite(value) or value<0:raise ValueError('Observed values must be finite nonnegative g/100g; omit missing axes.')
            position=self.axis_position[index]
            if visible[0,position]:raise ValueError('Duplicate axis under different identifiers.')
            values[0,position]=np.log1p(value/self.data.scale[index]);visible[0,position]=True
        if not visible.any():raise ValueError('At least one observed nutrition value, including explicit zero, is required.')
        return numeric_features(values,visible)

    @torch.no_grad()
    def predict_features(self,features):
        features=np.asarray(features,dtype=np.float32)
        if features.ndim!=2 or features.shape[1]!=284 or not np.isfinite(features).all():raise ValueError('Invalid numeric features.')
        if self.kind=='ridge':
            result=features@self.coef.T+self.intercept
            if not np.isfinite(result).all():raise FloatingPointError('Nonfinite Ridge mapping.')
            return torch.as_tensor(result,device=self.device)
        return self.model(torch.as_tensor(features,device=self.device))

    def encode(self,food_name=None,observed_profile=None,modality='nutrition'):
        if modality=='name':return self.namespace.encode([food_name])[0]
        if modality!='nutrition':raise ValueError('Independent mapping exposes name/nutrition representations only, no fused/prediction head.')
        # food_name deliberately never enters this branch.
        return self.predict_features(self.profile_features(observed_profile or {})).cpu().numpy()[0]

    @torch.no_grad()
    def retrieve_names(self,observed_profile,candidate_names,top_k=10):
        if not isinstance(top_k,int) or top_k<1:raise ValueError('Positive integer top_k required.')
        names=sorted(set(candidate_names))
        if not names:raise ValueError('Nonempty candidate names required.')
        query=self.predict_features(self.profile_features(observed_profile))
        candidates=torch.as_tensor(self.namespace.encode(names),device=self.device)
        similarity=(F.normalize(query,dim=1)@F.normalize(candidates,dim=1).T).cpu().numpy()[0]
        if not np.isfinite(similarity).all():raise FloatingPointError('Nonfinite similarity.')
        order=np.argsort(-similarity,kind='stable')[:top_k]
        return [{'name':names[i],'score':float(similarity[i]),'score_type':'cosine similarity; not probability'} for i in order]

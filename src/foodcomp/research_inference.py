"""Public research interfaces. Numeric retrieval never receives the query name."""
from pathlib import Path
import json
import numpy as np
import torch
from .research_r0 import ResearchData, digest
from .research_text import NameEncoder
from .research_neural import make_model, batch_from_arrays, predictions_from_outputs


class NutritionModel:
    def __init__(self, checkpoint, repo=None, device=None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.data = ResearchData(self.repo/"data/processed"/Path(saved["data_root"].replace("\\", "/")).name, saved["view"])
        self.cache = self.repo/"data/cache/research_name_only"/Path(saved["name_cache"].replace("\\", "/")).name
        if digest(self.data.root/"manifest.json") != saved["data_hash"] or digest(self.cache/"manifest.json") != saved["name_cache_hash"]:
            raise ValueError("Checkpoint data/text fingerprint mismatch.")
        manifest = json.loads((self.cache/"manifest.json").read_text())
        for name, expected in manifest["hashes"].items():
            if digest(self.cache/name) != expected: raise ValueError(f"Stale text cache: {name}")
        self.model, _ = make_model(self.data, saved["text_dim"], saved["kind"],
            amount_weight=saved["config"]["amount_loss_weight"],
            source_weight=saved["config"].get("source_calibrated_loss_weight", 1.))
        self.model.load_state_dict(saved["model_state"])
        self.model.to(self.device).eval()
        self.kind = saved["kind"]
        self.text_dim = saved["text_dim"]
        self._encoder = None
        self._pca = dict(np.load(self.cache/"pca.npz", allow_pickle=False))
        self._cached_text = np.load(self.cache/"features.npy")
        self._name_index = {name: i for i,name in enumerate(self.data.profiles.original_name.astype(str))}
        self._axes = {str(name): int(i) for i,name in zip(self.data.axes.axis_index, self.data.axes.canonical_name)}

    def axis_indices(self, axes):
        result = []
        for axis in axes:
            if isinstance(axis, (int, np.integer)) and 0 <= int(axis) < len(self.data.axes): result.append(int(axis))
            elif str(axis) in self._axes: result.append(self._axes[str(axis)])
            else: raise ValueError(f"Unknown axis: {axis}")
        if len(set(result)) != len(result): raise ValueError("Duplicate axes.")
        return np.array(result, dtype=int)

    def name_features(self, names):
        result = np.empty((len(names), self.text_dim), np.float32)
        missing = []
        for i,name in enumerate(names):
            if not isinstance(name,str) or not name.strip(): raise ValueError("Food names must be nonempty strings.")
            if name in self._name_index: result[i] = self._cached_text[self._name_index[name]]
            else: missing.append(i)
        if missing:
            if self._encoder is None: self._encoder = NameEncoder(self.repo/"data/cache/huggingface")
            raw = self._encoder.encode([names[i] for i in missing])
            result[missing] = (raw-self._pca["mean"]) @ self._pca["components"].T
        return result

    def profile_arrays(self, observed_profile):
        values = np.zeros((1,len(self.data.axes)), np.float32)
        visible = np.zeros_like(values, bool)
        axes = self.axis_indices(list(observed_profile))
        raw = np.array(list(observed_profile.values()), float)
        if not np.isfinite(raw).all() or (raw<0).any(): raise ValueError("Observed values must be finite, nonnegative g/100g; omit missing axes.")
        values[0,axes] = np.log1p(raw/self.data.scale[axes])
        visible[0,axes] = True
        return values,visible

    @torch.no_grad()
    def predict(self, food_name, observed_profile, target_axes):
        axes = self.axis_indices(target_axes)
        if not self.data.axes.loss_eligible.iloc[axes].all():
            raise ValueError("Context-only axes have no validated supervised prediction head.")
        values,visible = self.profile_arrays(observed_profile)
        if visible[0,axes].any(): raise ValueError("Target axes must be withheld from observed_profile.")
        batch = batch_from_arrays(values,visible,self.name_features([food_name]),self.device)
        raw,_ = predictions_from_outputs(self.model(batch),self.data.scale)
        return {self.data.axes.canonical_name.iloc[a]: float(raw[0,a]) for a in axes}

    @torch.no_grad()
    def encode(self, food_name=None, observed_profile=None, modality="fused"):
        if modality not in {"name","nutrition","fused"}: raise ValueError(modality)
        if modality == "fused" and self.kind in {"name_mlp","numeric_mlp"}:
            raise ValueError("Single-modality baselines do not provide a fused representation.")
        values,visible = self.profile_arrays(observed_profile or {})
        if modality == "name": visible[:] = False
        if modality == "nutrition":
            if self.kind == "name_mlp": raise ValueError("A name-only MLP has no nutrition representation.")
            text = np.zeros((1,self.text_dim),np.float32)
        else:
            if self.kind == "numeric_mlp": raise ValueError("A numeric-only MLP has no name/fused representation.")
            text = self.name_features([food_name])
        batch = batch_from_arrays(values,visible,text,self.device)
        if hasattr(self.model,"encode"): hidden = self.model.encode(batch)
        else:
            # Axis-token mean is an explicit probe representation, not a trained contrastive embedding.
            token = self.model.value_encoder(batch["value"].unsqueeze(-1))
            token = torch.where(batch["masked"].unsqueeze(-1),self.model.mask_value.view(1,1,-1),token)
            axis = self.model.axis_embedding(batch["axis"])+token
            prefix = [self.model.cls.expand(1,-1,-1),self.model.text_projection(batch["text"]).unsqueeze(1)]
            if self.kind == "v8_optimized": prefix.append(self.model.source_embedding(batch["source"]).unsqueeze(1))
            hidden = self.model.encoder(torch.cat(prefix+[axis],1))[:,len(prefix):].mean(1)
        if not torch.isfinite(hidden).all(): raise FloatingPointError("Nonfinite representation.")
        return hidden.cpu().numpy()[0]

    @torch.no_grad()
    def candidate_profiles(self, candidate_names, batch_size=256):
        """Only candidate text is used; no candidate measured nutrition is accessed."""
        text = self.name_features(candidate_names)
        result=[]
        for start in range(0,len(text),batch_size):
            features = text[start:start+batch_size]
            values = np.zeros((len(features),len(self.data.axes)),np.float32)
            batch = batch_from_arrays(values,np.zeros_like(values,bool),features,self.device)
            raw,_ = predictions_from_outputs(self.model(batch),self.data.scale)
            result.append(raw.cpu().numpy())
        return np.concatenate(result)

    def retrieve_names(self, observed_profile, candidate_names, top_k=10):
        names = sorted(set(candidate_names))
        if not names or top_k<1: raise ValueError("Nonempty candidates and positive top_k required.")
        values,visible = self.profile_arrays(observed_profile)
        if not visible.any(): raise ValueError("Retrieval needs observed nutrition, including explicit zeros.")
        nutrition = self.data.axes.loss_group.eq("nutrition").to_numpy() & self.data.axes.loss_eligible.to_numpy(bool)
        visible &= nutrition[None,:]
        if not visible.any(): raise ValueError("Retrieval baseline requires an observed supervised nutrition axis.")
        candidate = np.log1p(self.candidate_profiles(names)/self.data.scale)
        distance = ((candidate[:,visible[0]]-values[:,visible[0]])**2).mean(1)
        order = np.argsort(distance,kind="stable")[:top_k]
        return [{"name":names[i],"score":-float(distance[i]),"score_type":"negative_scaled_log_mse; not probability"} for i in order]

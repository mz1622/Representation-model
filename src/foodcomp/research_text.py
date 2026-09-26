"""Name-only text cache: content/revision/split fingerprints guard against stale reuse."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from .research_r0 import digest, write_json

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

class NameEncoder:
    def __init__(self, cache_home: Path):
        os.environ["HF_HOME"] = str(cache_home)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = cache_home/"hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"/REVISION
        if not path.is_dir():
            raise FileNotFoundError(f"Pinned MiniLM cache missing: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def encode(self, names, batch_size=64):
        rows = []
        for start in range(0, len(names), batch_size):
            encoded = self.tokenizer(list(names[start:start+batch_size]), padding=True, truncation=True,
                                     max_length=128, return_tensors="pt").to(self.device)
            hidden = self.encoder(**encoded).last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden*attention).sum(1)/attention.sum(1).clamp_min(1)
            rows.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
            if start % 12800 == 0:
                print(f"name-only encoding {min(start+batch_size,len(names))}/{len(names)}", flush=True)
        return np.concatenate(rows).astype(np.float32)

def prepare_names(data, repo: Path, components=32):
    names = data.profiles.original_name.fillna("").astype(str).tolist()
    identity = hashlib.sha256(json.dumps({
        "ids": data.profiles.profile_id.tolist(), "names": names, "model": MODEL,
        "revision": REVISION, "train_rows": data.train.tolist(), "components": components,
    }, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    cache = repo/"data/cache/research_name_only"/identity
    manifest_path = cache/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["hashes"].items():
            if digest(cache/name) != expected:
                raise ValueError(f"Name cache checksum mismatch: {name}")
        return np.load(cache/"features.npy"), cache
    if cache.exists():
        raise FileExistsError(f"Incomplete cache; retain for inspection and use a new cache version: {cache}")
    cache.mkdir(parents=True)
    encoder = NameEncoder(repo/"data/cache/huggingface")
    unique, inverse = np.unique(names, return_inverse=True)
    raw = encoder.encode(unique.tolist())[inverse]
    np.save(cache/"embeddings.npy", raw)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=20260922)
    pca.fit(raw[data.train])
    features = pca.transform(raw).astype(np.float32)
    np.save(cache/"features.npy", features)
    np.savez(cache/"pca.npz", mean=pca.mean_, components=pca.components_)
    write_json(manifest_path, {"fingerprint": identity, "model": MODEL, "revision": REVISION,
                              "text_field": "original_name", "fit_partition": "train",
                              "rows": len(names), "unique_names": len(unique), "pca_components": components,
                              "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                              "hashes": {x.name: digest(x) for x in cache.iterdir() if x.is_file()}})
    return features, cache

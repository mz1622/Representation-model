"""Audit train-name geometry without loading any nutrient labels or changing caches."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import VERSION,digest,write_json
from foodcomp.research_geometry import nearest_other_rows,neighbor_overlap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name-cache",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--queries",type=int,default=1024)
    p.add_argument("--seed",type=int,default=20260922)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.queries<1:raise ValueError("Positive query count required.")
    args.output_dir.mkdir(parents=True);start=time.monotonic()
    root=ROOT/"data/processed"/VERSION
    manifest=json.loads((root/"manifest.json").read_text())
    if digest(root/"profiles.csv.gz")!=manifest["artifact_hashes"]["profiles.csv.gz"]:raise ValueError("Changed profiles.")
    cache=json.loads((args.name_cache/"manifest.json").read_text())
    if cache["text_field"]!="original_name" or cache["fit_partition"]!="train" or cache["pca_components"]!=32:
        raise ValueError("Expected frozen name-only train-PCA32 cache.")
    for filename,expected in cache["hashes"].items():
        if digest(args.name_cache/filename)!=expected:raise ValueError(f"Changed name cache: {filename}")
    profiles=pd.read_csv(root/"profiles.csv.gz",usecols=["profile_index","original_name","exact_name_group_id","partition"])
    np.testing.assert_array_equal(profiles.profile_index,np.arange(len(profiles)))
    train=profiles[profiles.partition.eq("train")].copy()
    if train.original_name.isna().any() or train.exact_name_group_id.isna().any():raise ValueError("Missing training name/group.")
    representatives=train.sort_values(["exact_name_group_id","original_name","profile_index"]).drop_duplicates("exact_name_group_id")
    rows=representatives.profile_index.to_numpy();train_rows=train.profile_index.to_numpy()
    raw_file=np.load(args.name_cache/"embeddings.npy",mmap_mode="r",allow_pickle=False)
    frozen_file=np.load(args.name_cache/"features.npy",mmap_mode="r",allow_pickle=False)
    if raw_file.shape!=(len(profiles),384) or frozen_file.shape!=(len(profiles),32):raise ValueError("Name cache shape mismatch.")
    raw_train=np.asarray(raw_file[train_rows],dtype=np.float64)
    raw=np.asarray(raw_file[rows],dtype=np.float64)
    if not np.isfinite(raw_train).all():raise FloatingPointError("Nonfinite training text.")
    queries=np.random.default_rng(args.seed).choice(len(rows),min(args.queries,len(rows)),replace=False)
    representatives.to_csv(args.output_dir/"train_candidate_representatives.csv",index=False)
    np.save(args.output_dir/"query_candidate_indices.npy",queries)
    with threadpool_limits(limits=2):
        mean=raw_train.mean(0);centered=raw_train-mean
        covariance=centered.T@centered/(len(centered)-1)
        eigenvalues,basis=np.linalg.eigh(covariance)
        order=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[order];basis=basis[:,order]
        if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -1e-10:raise FloatingPointError("Invalid PCA covariance.")
        np.testing.assert_allclose(basis.T@basis,np.eye(384),rtol=0,atol=1e-12)
        np.savez(args.output_dir/"diagnostic_train_pca.npz",mean=mean,components=basis.T,eigenvalues=eigenvalues)
        features={"frozen_pca32":np.asarray(frozen_file[rows],dtype=np.float64)}
        for dims in [32,64,128]:features[f"exact_pca{dims}"]=(raw-mean)@basis[:,:dims]
        # The full rotation is checked on 64 actual query/candidate vectors.
        check=np.unique(np.concatenate([queries[:32],np.arange(32)]))
        delta=raw[check,None,:]-raw[None,check,:]
        rotated=(raw[check]-mean)@basis
        original_square=np.square(delta).sum(-1)
        rotated_square=np.square(rotated[:,None,:]-rotated[None,:,:]).sum(-1)
        np.testing.assert_allclose(original_square,rotated_square,rtol=1e-11,atol=1e-12)
        reference,reference_square=nearest_other_rows(raw,queries,50)
        np.savez(args.output_dir/"reference_neighbors.npz",indices=reference,squared_distances=reference_square)
        summaries=[];frames=[];all_neighbors={}
        for name,values in features.items():
            neighbors,squared=nearest_other_rows(values,queries,50);all_neighbors[name]=neighbors
            np.savez(args.output_dir/f"{name}_neighbors.npz",indices=neighbors,squared_distances=squared)
            actual_square=np.square(raw[queries]-raw[neighbors[:,0]]).sum(1)
            positive=reference_square[:,0]>1e-12
            inflation=np.full(len(queries),np.nan)
            inflation[positive]=np.sqrt(actual_square[positive]/reference_square[positive,0])
            # Undefined ratios for duplicate embeddings are explicitly counted, never silently averaged.
            if not np.isfinite(inflation[positive]).all() or (inflation[positive]<1-1e-7).any():
                raise FloatingPointError("Invalid nearest-neighbor distance ratio.")
            frame=pd.DataFrame({"feature":name,"candidate_query_index":queries,"reference_nearest_squared_distance":reference_square[:,0],
                "projected_choice_original_squared_distance":actual_square,"distance_ratio_defined":positive,
                "original_distance_inflation":inflation})
            result={"feature":name,"dimensions":values.shape[1],"queries":len(queries),"distance_ratio_defined_queries":int(positive.sum()),
                "zero_distance_reference_queries":int((~positive).sum()),
                "original_distance_inflation_median":float(np.median(inflation[positive])) if positive.any() else None,
                "original_distance_inflation_p90":float(np.quantile(inflation[positive],.9)) if positive.any() else None,
                "training_variance_retained":float(cache["pca_explained_variance"]) if name=="frozen_pca32" else float(eigenvalues[:values.shape[1]].sum()/eigenvalues.sum())}
            for k in [1,5,10,50]:
                overlap=neighbor_overlap(reference,neighbors,k);frame[f"overlap_at{k}"]=overlap;result[f"mean_overlap_at{k}"]=float(overlap.mean())
            frames.append(frame);summaries.append(result);print(result,flush=True)
        solver={f"overlap_at{k}":float(neighbor_overlap(all_neighbors["frozen_pca32"],all_neighbors["exact_pca32"],k).mean()) for k in [1,5,10,50]}
    pd.concat(frames,ignore_index=True).to_csv(args.output_dir/"per_query_geometry.csv",index=False)
    write_json(args.output_dir/"summary.json",{"results":summaries,"frozen_vs_exact32_solver_agreement":solver,
        "train_profiles":len(train),"train_name_candidate_groups":len(rows),"queries":len(queries),"seed":args.seed,
        "data_manifest_sha256":digest(root/"manifest.json"),"cache_manifest_sha256":digest(args.name_cache/"manifest.json"),
        "code_sha256":digest(Path(__file__)),"module_sha256":digest(ROOT/"src/foodcomp/research_geometry.py"),
        "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python":platform.python_version(),"numpy":np.__version__,"cpu_blas_threads":2,
        "elapsed_seconds":time.monotonic()-start,"complete_test_opened":False,"nutrition_labels_loaded":False,
        "validation_vectors_used":False,"full_rotation_distances_verified":True,
        "scope":"Training-name geometry only. Raw MiniLM neighbors are not nutrition truth or confirmed aliases. No model/caches/protocol changed; dimensionality changes require separately registered same-input baselines."})


if __name__=="__main__":main()

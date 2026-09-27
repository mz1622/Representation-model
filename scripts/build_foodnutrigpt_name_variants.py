"""Build R7 train-only names without loading any nutrition labels or replacing V8/R0."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_name_projection import fit_projection,project_names,load_variant


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);start=time.monotonic()
    receipt={'status':'incomplete','nutrition_labels_loaded':False,'complete_test_opened':False,'code_sha256':digest(Path(__file__))}
    try:
        root=ROOT/'data/processed'/VERSION;data=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        parent=ROOT/'data/cache/research_name_only/1f085dcc818fa79dc3ce40257ab8a263404e353b3e9c8158fb0cd98fb838aa05'
        pm=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
        assert digest(parent/'manifest.json')=='705e423f4145ce3a68482bd79f8be3236d8a14cc8836aeb36570751910a919c0'
        for name,expected in pm['hashes'].items():assert digest(parent/name)==expected
        assert digest(root/'profiles.csv.gz')==data['artifact_hashes']['profiles.csv.gz']
        profiles=pd.read_csv(root/'profiles.csv.gz',usecols=['profile_index','profile_id','original_name','partition'])
        np.testing.assert_array_equal(profiles.profile_index,np.arange(len(profiles)))
        assert profiles.partition.isin(['train','validation']).all() and not profiles.original_name.isna().any()
        train=np.flatnonzero(profiles.partition.eq('train'));assert len(train)==64700
        raw=np.load(parent/'embeddings.npy',allow_pickle=False);assert raw.shape==(len(profiles),384)
        names=profiles.original_name.astype(str).tolist()
        unique,first,inverse=np.unique(names,return_index=True,return_inverse=True)
        np.testing.assert_array_equal(raw,raw[first][inverse])
        with threadpool_limits(limits=2):projection=fit_projection(raw,train)
        projection['active_components']=np.array(128,dtype=np.int64)
        features=project_names(raw,projection);probe=np.random.default_rng(20260922).choice(len(raw),512,replace=False)
        np.testing.assert_array_equal(project_names(raw[probe],projection),features[probe])
        np.testing.assert_array_equal(project_names(raw[probe[::-1]],projection)[::-1],features[probe])
        np.testing.assert_array_equal(np.concatenate([project_names(raw[probe[i:i+7]],projection) for i in range(0,len(probe),7)]),features[probe])
        np.testing.assert_array_equal(features,features[first][inverse])
        paths={};manifests={};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        text_identity=hashlib.sha256(json.dumps({'ids':profiles.profile_id.tolist(),'names':names},ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        for active in [32,128]:
            identity={'protocol':'exact_train_pca_padded_v1','data_sha256':digest(root/'manifest.json'),'parent_cache_sha256':digest(parent/'manifest.json'),
                'text_content_sha256':text_identity,'training_rows_sha256':fingerprint_array(train),'active_components':active,'pca_components':128,
                'projection_module_sha256':digest(ROOT/'src/foodcomp/research_name_projection.py')}
            fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            cache=ROOT/'data/cache/research_name_only'/fingerprint
            if cache.exists():raise FileExistsError(cache)
            cache.mkdir();current={**projection,'active_components':np.array(active,dtype=np.int64)}
            transformed=features.copy();transformed[:,active:]=0
            np.testing.assert_array_equal(project_names(raw[probe],current),transformed[probe])
            np.save(cache/'features.npy',transformed);np.savez(cache/'pca.npz',**current)
            m={**identity,'fingerprint':fingerprint,'projection_kind':'exact_train_pca_padded_v1','text_field':'original_name','fit_partition':'train',
                'rows':len(profiles),'unique_names':len(unique),'model':pm['model'],'revision':pm['revision'],'training_profiles':len(train),
                'pca_explained_variance':float(projection['eigenvalues'][:active].sum()/projection['eigenvalues'].sum()),
                'raw_embeddings_reference':str((parent/'embeddings.npy').relative_to(ROOT)),'raw_embeddings_sha256':digest(parent/'embeddings.npy'),
                'code_commit':commit,'builder_sha256':digest(Path(__file__)),'hashes':{f.name:digest(f) for f in cache.iterdir() if f.is_file()},
                'transform':'einsum ij,kj->ik optimize=False float64 centered projection thenfloat32; inactive columnszero',
                'complete_test_opened':False,'nutrition_labels_loaded':False}
            write_json(cache/'manifest.json',m);load_variant(cache,data_hash=identity['data_sha256'],active_components=active)
            paths[str(active)]=str(cache.relative_to(ROOT));manifests[str(active)]=digest(cache/'manifest.json')
        a,_,_=load_variant(ROOT/paths['32']);b,_,_=load_variant(ROOT/paths['128']);np.testing.assert_array_equal(a[:,:32],b[:,:32])
        receipt.update(status='complete',paths=paths,manifest_sha256=manifests,training_rows_sha256=fingerprint_array(train),text_content_sha256=text_identity,
            data_sha256=digest(root/'manifest.json'),parent_cache_sha256=digest(parent/'manifest.json'),training_profiles=len(train),
            input_slots=128,first32_exact=True,all_duplicate_name_features_exact=True,probe512_reorder_chunk7_projection_exact=True,
            variance_retained={str(k):float(projection['eigenvalues'][:k].sum()/projection['eigenvalues'].sum()) for k in [32,128]},
            elapsed_seconds=time.monotonic()-start,environment={'python':platform.python_version(),'numpy':np.__version__,'blas_threads':2})
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)

if __name__=='__main__':main()

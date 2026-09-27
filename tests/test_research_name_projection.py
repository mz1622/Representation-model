import json
import numpy as np
import pytest
from foodcomp.research_name_projection import fit_projection,project_names,load_variant
from foodcomp.research_r0 import digest


def test_training_only_fit_prefix_padding_batch_invariance():
    raw=np.random.default_rng(11).normal(size=(80,7)).astype(np.float32);train=np.arange(60)
    a=fit_projection(raw,train,5);changed=raw.copy();changed[60:]+=1000;b=fit_projection(changed,train,5)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    full={**a,'active_components':5};small={**a,'active_components':2}
    f=project_names(raw,full);s=project_names(raw,small)
    np.testing.assert_array_equal(s[:,:2],f[:,:2]);assert not s[:,2:].any()
    np.testing.assert_array_equal(f,project_names(raw[::-1],full)[::-1])
    np.testing.assert_array_equal(f,np.concatenate([project_names(row[None,:],full) for row in raw]))
    raw[0,0]=np.nan
    with pytest.raises(FloatingPointError):project_names(raw,full)


def test_cache_checksums_and_dimensions_fail_closed(tmp_path):
    raw=np.random.default_rng(7).normal(size=(30,5));projection={**fit_projection(raw,np.arange(20),4),'active_components':2}
    np.save(tmp_path/'features.npy',project_names(raw,projection));np.savez(tmp_path/'pca.npz',**projection)
    manifest={'projection_kind':'exact_train_pca_padded_v1','text_field':'original_name','fit_partition':'train','data_sha256':'data',
        'active_components':2,'rows':30,'pca_components':4,'hashes':{name:digest(tmp_path/name) for name in ['features.npy','pca.npz']}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8');load_variant(tmp_path,data_hash='data',active_components=2)
    with pytest.raises(ValueError,match='fingerprint'):load_variant(tmp_path,data_hash='changed')
    with pytest.raises(ValueError,match='dimensions'):load_variant(tmp_path,active_components=4)
    np.save(tmp_path/'features.npy',np.ones((30,4),np.float32))
    with pytest.raises(ValueError,match='Changed'):load_variant(tmp_path)

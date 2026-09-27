"""R7 matched model initialization, cache boundaries and public inference compatibility."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_name_projection import load_variant,project_names,fit_projection
from foodcomp.research_neural import make_model,training_batch
from foodcomp.research_forward_loss import forward_loss
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_inference import NutritionModel
from audit_foodnutrigpt_v9_r3_control import load_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);registry=json.loads((ROOT/'reports/v9_r7_name_cache_v1/verification.json').read_text(encoding='utf-8'))
        raw=np.load(ROOT/'data/cache/research_name_only/1f085dcc818fa79dc3ce40257ab8a263404e353b3e9c8158fb0cd98fb838aa05/embeddings.npy')
        # Fit only a small training subset twice while mutating heldout name vectors.
        tiny=np.r_[data.train[:400],data.validation[:10]];copy=raw[tiny].copy();changed=copy.copy();changed[400:]+=99
        x=fit_projection(copy,np.arange(400),16);y=fit_projection(changed,np.arange(400),16)
        for key in x:np.testing.assert_array_equal(x[key],y[key])
        models=[];fingerprints=[];texts=[];records=[]
        active=data.train[data.observed[data.train][:,data.targets].any(1)];rows=active[:768]
        for dims in [32,128]:
            cache=ROOT/registry['paths'][str(dims)];text,projection,m=load_variant(cache,data_hash=digest(data.root/'manifest.json'),active_components=dims)
            torch.manual_seed(20260922);model,config=make_model(data,128,'name_mlp');assert sum(p.numel() for p in model.parameters())==164092
            fingerprints.append(state_fingerprint(model));models.append(model);texts.append(text)
            b=training_batch(data,text,rows,'cpu',np.random.default_rng(7),'name_mlp');model.eval()
            predicted=model(b)['amount_normalized'];altered={k:v.clone() for k,v in b.items()}
            altered['value'].fill_(123.);altered['masked'].fill_(False);altered['source']+=99
            assert torch.equal(predicted,model(altered)['amount_normalized'])
            loss=forward_loss(model,b,config,'mae');loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            if dims==32:assert not model.encoder[0].weight.grad[:,32:].any()
            else:assert model.encoder[0].weight.grad[:,32:].abs().sum()>0
            path=args.output_dir/f'initial_fixture_{dims}_not_candidate.pt'
            torch.save({'model_state':model.state_dict(),'kind':'name_mlp','config':asdict(config),'text_dim':128,'data_root':str(data.root),
                'view':data.view,'name_cache':str(cache),'data_hash':digest(data.root/'manifest.json'),'name_cache_hash':digest(cache/'manifest.json'),'args':{'mlp_width':256}},path)
            loaded=NutritionModel(path,device='cpu');assert torch.equal(predicted,loaded.model(b)['amount_normalized'])
            probe_raw=raw[rows[:3]];probe_names=[f'__r7_frozen_projection_probe_{i}__' for i in range(3)]
            assert not any(name in loaded._name_index for name in probe_names)
            class FixtureEncoder:
                def encode(self,names):return probe_raw[[probe_names.index(name) for name in names]]
            loaded._encoder=FixtureEncoder()
            np.testing.assert_array_equal(loaded.name_features(probe_names),project_names(probe_raw,projection))
            np.testing.assert_array_equal(loaded.name_features(probe_names[::-1])[::-1],loaded.name_features(probe_names))
            assert loaded.predict(probe_names[0],{},data.targets[:2])==loaded.predict(probe_names[0],{int(data.targets[2]):0.},data.targets[:2])
            try:loaded.encode(probe_names[0],{},'nutrition')
            except ValueError:pass
            else:raise AssertionError('Name-only baseline claimed nutrition encoder.')
            records.append({'active_dimensions':dims,'name_cache_sha256':digest(cache/'manifest.json'),'loss':float(loss.detach()),
                'fixture_sha256':digest(path),'save_reload_prediction_exact':True,'uncached_name_projection_exact':True})
            del loaded
        assert fingerprints[0]==fingerprints[1]
        np.testing.assert_array_equal(texts[0][:,:32],texts[1][:,:32]);assert not texts[0][:,32:].any()
        # Old checkpoint unknown-name behavior keeps the exact original formula.
        legacy_run=ROOT/'output/v9_r5/name_mlp60_mae';old_module=load_snapshot('foodcomp.r7_old_inference',legacy_run/'code_snapshot/research_inference.py')
        old=old_module.NutritionModel(legacy_run/'best_model.pt',repo=ROOT,device='cpu');new=NutritionModel(legacy_run/'best_model.pt',repo=ROOT,device='cpu')
        old._encoder=FixtureEncoder();new._encoder=FixtureEncoder()
        np.testing.assert_array_equal(old.name_features(probe_names),new.name_features(probe_names))
        real_names=data.profiles.iloc[rows[:3]].original_name.tolist()
        np.testing.assert_array_equal(old.name_features(real_names),new.name_features(real_names))
        assert old.predict(probe_names[0],{},data.targets[:3])==new.predict(probe_names[0],{},data.targets[:3])
        receipt.update(status='complete',parameter_count=164092,initial_state_sha256=fingerprints[0],matched_initialization_exact=True,
            actual_train_rows_checked=768,heldout_name_mutation_does_not_change_fit=True,old_cached_and_uncached_name_inference_exact=True,
            numeric_context_source_mutation_prediction_exact=True,active_input_gradient_checks_passed=True,
            records=records,scope='Train-only768row functional fixture and old/new public interface checks; not a full optimization replay. No nutrition encoder claimed.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt['status'],receipt['parameter_count'])

if __name__=='__main__':main()

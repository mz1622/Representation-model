"""Post-run R7 checks: matched training contracts, name caches and complete replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_name_projection import load_variant
from foodcomp.research_neural import evaluate,training_batch
from foodcomp.research_inference import NutritionModel
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['mlp','knn'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);registry=read(ROOT/'reports/v9_r7_name_cache_v1/verification.json')
        records=[];manifests=[];histories=[];texts=[]
        for dims in [32,128]:
            run=ROOT/f'output/v9_r7/exact_name_{args.kind}{dims}';m=read(run/'run_manifest.json')
            assert m['status']=='complete' and not m['complete_test_opened'] and m['data_hash']==digest(data.root/'manifest.json')
            for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
            cache=ROOT/registry['paths'][str(dims)];text,_,_=load_variant(cache,data_hash=m['data_hash'],active_components=dims)
            assert m['name_cache_hash']==registry['manifest_sha256'][str(dims)]==digest(cache/'manifest.json');texts.append(text)
            record={'run':str(run.relative_to(ROOT)),'manifest_sha256':digest(run/'run_manifest.json'),'name_cache_sha256':m['name_cache_hash']}
            if args.kind=='mlp':
                assert m['epoch_completed']==60 and m['name_input_intervention']['active_components']==dims and m['parameter_count']==164092
                model=NutritionModel(run/'best_model.pt');assert digest(run/'best_model.pt')==m['checkpoint_hash']
                active=data.train[data.observed[data.train][:,data.targets].any(1)]
                for label,x in {'training_rows':active,'training_target_values':data.values[active][:,data.targets],
                    'training_target_mask':data.observed[active][:,data.targets],'training_weights':data.weights[active][:,data.targets]}.items():
                    assert m[label+'_sha256']==fingerprint_array(x)
                h=pd.read_csv(run/'history.csv',float_precision='round_trip');np.testing.assert_array_equal(h.epoch,np.arange(1,61))
                assert np.isfinite(h.select_dtypes('number')).all().all()
                opt=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.001)
                schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=.00001)
                for row in h.itertuples():
                    assert row.training_profiles==len(active) and row.observed_target_cells==m['observed_target_cells']==1828536
                    assert row.training_order_sha256==fingerprint_array(np.random.default_rng(m['seed']+row.epoch).permutation(active))
                    assert row.learning_rate==opt.param_groups[0]['lr'];opt.step();schedule.step()
                assert m['best_epoch']==int(h.loc[h.validation_primary.idxmin(),'epoch']);histories.append(h)
                saved=read(run/'metrics.json');frames=[]
                for mode in ['completion','name_only']:
                    predictions=evaluate(model.model,data,text,model.device,mode)
                    pd.testing.assert_frame_equal(predictions,pd.read_parquet(run/f'{mode}_predictions.parquet'),check_exact=True)
                    assert score_predictions(data,predictions)[0]==saved[mode];frames.append(predictions)
                b=training_batch(data,text,active[:32],model.device,np.random.default_rng(11),'name_mlp')
                with torch.no_grad():
                    first=model.model(b)['amount_normalized'];b['value'].fill_(123.);b['masked'].fill_(False)
                    assert torch.equal(first,model.model(b)['amount_normalized'])
                replay=args.output_dir/f'retrieval_replay_{dims}'
                subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py'),'--checkpoint',str(run/'best_model.pt'),
                    '--output-dir',str(replay.resolve())],cwd=ROOT,check=True,capture_output=True,text=True)
                original=ROOT/f'output/v9_r7/retrieval_name_mlp{dims}'
                pd.testing.assert_frame_equal(pd.read_parquet(replay/'ranks.parquet'),pd.read_parquet(original/'ranks.parquet'),check_exact=True)
                for key in ['metrics','candidate_count','candidate_sha256','query_profiles','checkpoint_sha256','data_sha256','scoring','correct_answers']:
                    assert read(replay/'metrics.json')[key]==read(original/'metrics.json')[key]
                joined=frames[0].merge(frames[1],on=['profile_index','axis_index'],suffixes=('_c','_n'),validate='one_to_one')
                record.update(all60_orders_exposures_lr_verified=True,both323809_predictions_and_metrics_exact=True,all19089_ranks_exact=True,
                    numeric_context_independence=True,completion_name_crossbatch_max_raw_difference=float(np.abs(joined.prediction_c-joined.prediction_n).max()))
            else:
                receipts=read(run/'axis_fitting_manifest.json');assert [r['axis_index'] for r in receipts]==data.targets.tolist()
                for row in receipts:
                    axis=row['axis_index'];train=data.train[data.observed[data.train,axis]]
                    assert row['train_rows_sha256']==fingerprint_array(train)
                    assert row['training_values_sha256']==fingerprint_array(data.values[train,axis])
                    assert row['training_weights_sha256']==fingerprint_array(data.weights[train,axis])
                    assert row['training_features_sha256']==fingerprint_array(text[train,:dims])
                    assert row['reverse_and_subset_validation_exact']
                    if data.axes.loss_group.iloc[axis]=='nutrition':
                        assert row['candidate_validation_neighbor_indices_distances_predictions_exact'] and row['candidate_subbatch256_predictions_exact']
                pred=pd.read_parquet(run/'nutrition_predictions.parquet');assert score_predictions(data,pred)[0]==read(run/'nutrition_metrics.json')
                raw=np.load(run/'candidate_predicted_raw.npy');scaled=np.load(run/'candidate_scaled.npy')
                axes=np.flatnonzero(data.axes.loss_group.eq('nutrition')&data.axes.loss_eligible)
                names=read(run/'candidate_names.json');assert raw.shape==(49913,142) and np.isfinite(raw).all() and (raw>=0).all()
                np.testing.assert_array_equal(scaled,np.log1p(raw/data.scale[axes]).astype(np.float32))
                assert digest(run/'candidate_scaled.npy')==m['candidate_vectors_sha256']
                name_lookup={name:i for i,name in enumerate(names)};axis_lookup={int(axis):i for i,axis in enumerate(axes)}
                rows=pred[pred.axis_index.isin(axes)]
                ids=[name_lookup[str(data.profiles.original_name.iloc[r])] for r in rows.profile_index]
                columns=[axis_lookup[int(a)] for a in rows.axis_index]
                np.testing.assert_array_equal(raw[ids,columns],rows.prediction.to_numpy())
                ranks,scores=evaluate_candidates(data,names,scaled,axes,'cuda' if torch.cuda.is_available() else 'cpu')
                pd.testing.assert_frame_equal(ranks,pd.read_parquet(run/'ranks.parquet'),check_exact=True);assert scores==read(run/'metrics.json')['metrics']
                assert digest(run/'candidate_names.json')==read(run/'metrics.json')['candidate_sha256']
                record.update(all187_training_rows_values_weights_features_verified=True,nutrition_scores_recomputed_exact=True,
                    all317616_nutrition_predictions_match_candidate_names_exact=True,all19089_saved_ranks_metrics_replayed_exact=True,
                    candidate_and_validation_names_use_same_predictor=True)
            records.append(record);manifests.append(m)
        for field in ['code_commit','code_hashes','data_hash']:assert manifests[0][field]==manifests[1][field],field
        np.testing.assert_array_equal(texts[0][:,:32],texts[1][:,:32]);assert not texts[0][:,32:].any()
        if args.kind=='mlp':
            for key in ['parameter_count','initial_state_sha256','training_profiles','observed_target_cells','training_rows_sha256','training_target_values_sha256','training_target_mask_sha256','training_weights_sha256','seed']:
                assert manifests[0][key]==manifests[1][key],key
            for key in ['epoch','training_profiles','observed_target_cells','training_order_sha256','learning_rate']:np.testing.assert_array_equal(histories[0][key],histories[1][key])
        receipt.update(status='complete',kind=args.kind,records=records,only_active_name_dimensions_changed=True,
            scope='Matched R7 active32/128 named input comparison, same labels/weights/evaluation. Not a fair new-input versus old-tree completion superiority claim; only one neural seed.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt['status'],receipt['kind'])

if __name__=='__main__':main()

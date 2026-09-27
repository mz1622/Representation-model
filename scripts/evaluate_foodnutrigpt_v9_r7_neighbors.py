"""R7 fixed KNN: score validation first, then name-only candidate banks on the same fits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_text import prepare_names
from foodcomp.research_name_projection import load_variant
from foodcomp.research_name_neighbors import ObservedAxisNeighbors
from foodcomp.research_alignment import SCORING,CORRECT
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--active-dimensions',type=int,choices=[32,128],required=True)
    p.add_argument('--legacy-control',action='store_true',help='Validation-only187axis oldKDTree replay before new variants.')
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.legacy_control and args.active_dimensions!=32:raise ValueError('Legacy control has32dimensions.')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    files=[Path(__file__),ROOT/'src/foodcomp/research_name_neighbors.py',ROOT/'src/foodcomp/research_name_projection.py',
        ROOT/'src/foodcomp/research_r0.py',ROOT/'src/foodcomp/research_r1.py',ROOT/'src/foodcomp/research_text.py',
        ROOT/'scripts/evaluate_foodnutrigpt_name_neighbors.py',ROOT/'src/foodcomp/research_profile_retrieval.py',ROOT/'src/foodcomp/research_alignment.py']
    snapshot=args.output_dir/'code_snapshot';snapshot.mkdir()
    for file in files:shutil.copyfile(file,snapshot/file.name)
    manifest={'status':'running','version':'V9-R7','kind':'name_knn','args':vars(args),'complete_test_opened':False,'scientific_confirmation':False,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(f.relative_to(ROOT)):digest(f) for f in files},'backend':'kd_tree','neighbors':10,'n_jobs':8,
        'training':'All per-axis observed training rows, canonical scaledlog labels, originalsource/candidate weights; fixedK/no tuning',
        'scope':'Name-onlybaseline; completion ignores numeric context. Candidate vectors use no candidate measured nutrition. Newtext inputs require same-inputtree baselines for futurecompletionclaims.'}
    write_json(args.output_dir/'run_manifest.json',manifest)
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION)
        if args.legacy_control:text,cache=prepare_names(data,ROOT)
        else:
            registry=read(ROOT/'reports/v9_r7_name_cache_v1/verification.json');assert registry['status']=='complete'
            cache=ROOT/registry['paths'][str(args.active_dimensions)]
            assert digest(cache/'manifest.json')==registry['manifest_sha256'][str(args.active_dimensions)]
            text,_,_=load_variant(cache,data_hash=digest(data.root/'manifest.json'),active_components=args.active_dimensions)
            text=text[:,:args.active_dimensions]
        names,first,inverse=np.unique(data.profiles.original_name.astype(str).to_numpy(),return_index=True,return_inverse=True)
        name_features=text[first];np.testing.assert_array_equal(text,name_features[inverse])
        write_json(args.output_dir/'candidate_names.json',names.tolist())
        assert digest(args.output_dir/'candidate_names.json')=='e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
        manifest.update(data_hash=digest(data.root/'manifest.json'),name_cache_hash=digest(cache/'manifest.json'),
            name_cache=str(cache.relative_to(ROOT)),active_dimensions=args.active_dimensions,training_feature_sha256=fingerprint_array(text[data.train]),
            original_name_duplicates_exact=True)
        reference=ROOT/'output/v9_r5/retrieval_name_knn10_tree'
        old=pd.read_parquet(reference/'replayed_nutrition_predictions.parquet')
        axes=np.flatnonzero(data.axes.loss_group.eq('nutrition')&data.axes.loss_eligible)
        axis_column={int(axis):i for i,axis in enumerate(axes)}
        fitted={};receipts=[];predictions=[];validation_neighbors={}
        with threadpool_limits(limits=8):
            for count,axis in enumerate(data.targets,1):
                train=data.train[data.observed[data.train,axis]];rows=data.jobs.loc[data.jobs.axis_index.eq(axis),'profile_index'].to_numpy()
                model=ObservedAxisNeighbors(text[train],data.values[train,axis],data.weights[train,axis],data.scale[axis],n_jobs=8,backend='kd_tree')
                values,indices,distances=model.predict(text[rows],return_neighbors=True)
                np.testing.assert_array_equal(model.predict(text[rows[::-1]])[::-1],values)
                np.testing.assert_array_equal(model.predict(text[rows[:min(32,len(rows))]]),values[:min(32,len(rows))])
                pred=pd.DataFrame({'profile_index':rows,'axis_index':int(axis),'prediction':values});previous=old[old.axis_index.eq(axis)].reset_index(drop=True)
                if args.legacy_control:pd.testing.assert_frame_equal(pred,previous,check_exact=True)
                delta=np.abs(values-previous.prediction.to_numpy())
                record={'axis_index':int(axis),'train_profiles':len(train),'train_rows_sha256':fingerprint_array(train),
                    'training_values_sha256':fingerprint_array(data.values[train,axis]),'training_weights_sha256':fingerprint_array(data.weights[train,axis]),
                    'training_features_sha256':fingerprint_array(text[train]),'reverse_and_subset_validation_exact':True,
                    'legacy_prediction_mismatch_count':int(np.count_nonzero(delta)),'legacy_prediction_max_raw_difference':float(delta.max())}
                if int(axis) in axis_column and not args.legacy_control:
                    fitted[int(axis)]=(model,train,rows);validation_neighbors[int(axis)]=(values,indices,distances)
                predictions.append(pred);receipts.append(record)
                write_json(args.output_dir/'axis_fitting_manifest.json',receipts)
                manifest.update(phase='validation_fitting',axes_completed=count,elapsed_seconds=time.monotonic()-started)
                write_json(args.output_dir/'run_manifest.json',manifest)
                if count%20==0:print(f'PCA{args.active_dimensions} validation axis{count}/187; {time.monotonic()-started:.1f}s',flush=True)
            predictions=pd.concat(predictions,ignore_index=True);scores,by_axis,_=score_predictions(data,predictions)
            predictions.to_parquet(args.output_dir/'nutrition_predictions.parquet',index=False)
            by_axis.to_csv(args.output_dir/'nutrition_axis_metrics.csv',index=False);write_json(args.output_dir/'nutrition_metrics.json',scores)
            manifest.update(phase='validation_complete',nutrition_prediction_sha256=digest(args.output_dir/'nutrition_predictions.parquet'),
                validation_elapsed_seconds=time.monotonic()-started,all187_reverse_subset_prediction_checks_passed=True)
            write_json(args.output_dir/'run_manifest.json',manifest);print('Nutrition',scores['nutrition'],flush=True)
            if args.legacy_control:
                assert scores==read(reference/'nutrition_metrics.json')
                manifest.update(status='complete',all187_legacy_predictions_and_metrics_exact=True,elapsed_seconds=time.monotonic()-started)
                write_json(args.output_dir/'run_manifest.json',manifest);return
            predicted=np.empty((len(names),len(axes)),np.float64);probe=np.random.default_rng(20260922).choice(len(names),256,replace=False)
            for count,axis in enumerate(axes,1):
                model,train,rows=fitted.pop(int(axis));expected,vi,vd=validation_neighbors.pop(int(axis))
                values,indices,distances=model.predict(name_features,return_neighbors=True)
                ids=inverse[rows];np.testing.assert_array_equal(indices[ids],vi);np.testing.assert_array_equal(distances[ids],vd)
                np.testing.assert_array_equal(values[ids],expected);np.testing.assert_array_equal(model.predict(name_features[probe]),values[probe])
                predicted[:,axis_column[int(axis)]]=values
                record=next(x for x in receipts if x['axis_index']==axis)
                record.update(candidate_predictions_sha256=fingerprint_array(values),candidate_neighbor_rows_sha256=fingerprint_array(train[indices]),
                    candidate_validation_neighbor_indices_distances_predictions_exact=True,candidate_subbatch256_predictions_exact=True)
                write_json(args.output_dir/'axis_fitting_manifest.json',receipts)
                manifest.update(phase='candidate_predictions',candidate_axes_completed=count,elapsed_seconds=time.monotonic()-started)
                write_json(args.output_dir/'run_manifest.json',manifest)
                if count%10==0:print(f'PCA{args.active_dimensions} candidate axis{count}/142; {time.monotonic()-started:.1f}s',flush=True)
                del model
        np.save(args.output_dir/'candidate_predicted_raw.npy',predicted)
        scaled=np.log1p(predicted/data.scale[axes]).astype(np.float32);np.save(args.output_dir/'candidate_scaled.npy',scaled)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ranks,metrics=evaluate_candidates(data,names.tolist(),scaled,axes,device);ranks.to_parquet(args.output_dir/'ranks.parquet',index=False)
        replay,rs=evaluate_candidates(data,names.tolist(),np.load(args.output_dir/'candidate_scaled.npy'),axes,device)
        pd.testing.assert_frame_equal(ranks,replay,check_exact=True);assert metrics==rs
        write_json(args.output_dir/'metrics.json',{'metrics':metrics,'candidate_count':len(names),'candidate_sha256':digest(args.output_dir/'candidate_names.json'),
            'query_profiles':ranks.groupby('visible_fraction').size().to_dict(),'data_sha256':manifest['data_hash'],'name_cache_sha256':manifest['name_cache_hash'],
            'scoring':SCORING,'correct_answers':CORRECT,'method':f'exact_pca{args.active_dimensions}_name_knn_kd_tree',
            'complete_test_opened':False,'scientific_claim_allowed':False,'elapsed_seconds':time.monotonic()-started})
        manifest.update(status='complete',phase='complete',elapsed_seconds=time.monotonic()-started,
            candidate_vectors_sha256=digest(args.output_dir/'candidate_scaled.npy'),all142_candidate_validation_and_subbatch_checks_passed=True,
            all19089_ranks_reload_exact=True)
        write_json(args.output_dir/'run_manifest.json',manifest);print('Retrieval',metrics,flush=True)
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);raise

if __name__=='__main__':main()

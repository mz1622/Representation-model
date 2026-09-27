"""Replay the fixed R0 nameKNN and construct candidate vectors exclusively from names."""
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
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import NameSpace,SCORING,CORRECT
from foodcomp.research_name_neighbors import ObservedAxisNeighbors
from foodcomp.research_profile_retrieval import predicted_profile_ranks


def evaluate_candidates(data,names,candidate_scaled,axes,device):
    candidate=torch.as_tensor(candidate_scaled,dtype=torch.float32,device=device)
    lookup={name:i for i,name in enumerate(names)};records=[]
    for fraction in [1.,.3]:
        rows=data.validation;values,visible=data.context(rows,mode='completion',visible_fraction=fraction)
        values=values[:,axes];visible=visible[:,axes];keep=visible.sum(1)>=3
        rows,values,visible=rows[keep],values[keep],visible[keep]
        for start in range(0,len(rows),128):
            rr=rows[start:start+128];sl=slice(start,start+128)
            correct=torch.tensor([lookup[str(data.profiles.original_name.iloc[r])] for r in rr],device=device)
            ranks=predicted_profile_ranks(candidate,torch.as_tensor(values[sl],device=device),torch.as_tensor(visible[sl],device=device),correct).cpu().numpy()
            for row,rank,n in zip(rr,ranks,visible[sl].sum(1)):
                records.append({'profile_index':int(row),'visible_fraction':fraction,'observed_axes':int(n),'rank':int(rank),
                    'mrr':1/float(rank),'recall_at_1':int(rank<=1),'recall_at_5':int(rank<=5),'recall_at_10':int(rank<=10)})
    result=pd.DataFrame(records).merge(data.profiles[['profile_index','source_key','exact_name_group_id']],on='profile_index',validate='many_to_one')
    keys=['mrr','recall_at_1','recall_at_5','recall_at_10']
    groups=result.groupby(['visible_fraction','exact_name_group_id','source_key'])[keys].mean().groupby(['visible_fraction','exact_name_group_id']).mean()
    return result,groups.groupby('visible_fraction')[keys].mean().reset_index().to_dict('records')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4)
    files=[Path(__file__),ROOT/'src/foodcomp/research_name_neighbors.py',ROOT/'src/foodcomp/research_profile_retrieval.py',
        ROOT/'src/foodcomp/research_r0.py',ROOT/'src/foodcomp/research_text.py',ROOT/'src/foodcomp/research_alignment.py']
    snap=args.output_dir/'code_snapshot';snap.mkdir()
    for f in files:shutil.copyfile(f,snap/f.name)
    manifest={'status':'running','kind':'name_knn','version':'V9-R5','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(f.relative_to(ROOT)):digest(f) for f in files},'complete_test_opened':False,'neighbors':10,'n_jobs':8,
        'selection':'none; fixed historical nameKNN','candidate_nutrition':'predicted from name features with training labels only; no measured candidate lookup'}
    write_json(args.output_dir/'run_manifest.json',manifest)
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);text,cache=prepare_names(data,ROOT);namespace=NameSpace(data,ROOT)
        names=namespace.names.tolist();axes=np.flatnonzero(data.axes.loss_group.eq('nutrition')&data.axes.loss_eligible)
        write_json(args.output_dir/'candidate_names.json',names)
        assert digest(args.output_dir/'candidate_names.json')=='e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
        reference=ROOT/'output/v9_r0/name_knn_quarantined';old=pd.read_parquet(reference/'predictions.parquet')
        predicted=np.empty((len(names),len(axes)),dtype=np.float64);axis_column={int(a):i for i,a in enumerate(axes)}
        axis_receipts=[];replayed=[];nameids={name:i for i,name in enumerate(names)}
        with threadpool_limits(limits=8):
            for count,axis in enumerate(data.targets,1):
                train=data.train[data.observed[data.train,axis]];jobs=data.jobs[data.jobs.axis_index.eq(axis)];rows=jobs.profile_index.to_numpy()
                model=ObservedAxisNeighbors(text[train],data.values[train,axis],data.weights[train,axis],data.scale[axis],n_jobs=8)
                value=model.predict(text[rows]);f=pd.DataFrame({'profile_index':rows,'axis_index':int(axis),'prediction':value})
                pd.testing.assert_frame_equal(f.reset_index(drop=True),old[old.axis_index.eq(axis)].reset_index(drop=True),check_exact=True)
                replayed.append(f)
                record={'axis_index':int(axis),'train_profiles':len(train),'train_rows_sha256':fingerprint_array(train),
                    'training_values_sha256':fingerprint_array(data.values[train,axis]),'training_weights_sha256':fingerprint_array(data.weights[train,axis]),'historical_validation_exact':True}
                if int(axis) in axis_column:
                    value_all,indexes,distances=model.predict(namespace.features,return_neighbors=True)
                    predicted[:,axis_column[int(axis)]]=value_all
                    ids=np.array([nameids[str(data.profiles.original_name.iloc[row])] for row in rows])
                    delta=np.abs(value_all[ids]-value)
                    record.update(candidate_predictions_sha256=fingerprint_array(value_all),neighbor_rows_sha256=fingerprint_array(train[indexes]),
                        candidate_vs_validation_mismatch_count=int(np.count_nonzero(delta)),candidate_vs_validation_max_raw_difference=float(delta.max()))
                axis_receipts.append(record);write_json(args.output_dir/'axis_fitting_manifest.json',axis_receipts)
                if count%20==0:print(f'nameKNN axis {count}/187; elapsed {time.monotonic()-started:.1f}s',flush=True)
        replay=pd.concat(replayed,ignore_index=True);scores,axis_scores,_=score_predictions(data,replay)
        assert scores==json.loads((reference/'metrics.json').read_text())
        replay.to_parquet(args.output_dir/'replayed_nutrition_predictions.parquet',index=False);axis_scores.to_csv(args.output_dir/'nutrition_axis_metrics.csv',index=False)
        write_json(args.output_dir/'nutrition_metrics.json',scores)
        np.save(args.output_dir/'candidate_predicted_raw.npy',predicted)
        scaled=np.log1p(predicted/data.scale[axes]).astype(np.float32);np.save(args.output_dir/'candidate_scaled.npy',scaled)
        device='cuda' if torch.cuda.is_available() else 'cpu';ranks,metrics=evaluate_candidates(data,names,scaled,axes,device)
        ranks.to_parquet(args.output_dir/'ranks.parquet',index=False)
        # Recompute complete ranks after loading candidate vectors from disk.
        replay_ranks,replay_metrics=evaluate_candidates(data,names,np.load(args.output_dir/'candidate_scaled.npy'),axes,device)
        pd.testing.assert_frame_equal(ranks,replay_ranks,check_exact=True);assert metrics==replay_metrics
        mismatches=sum(x.get('candidate_vs_validation_mismatch_count',0) for x in axis_receipts)
        maxdiff=max(x.get('candidate_vs_validation_max_raw_difference',0.) for x in axis_receipts)
        write_json(args.output_dir/'metrics.json',{'metrics':metrics,'candidate_count':len(names),'query_profiles':ranks.groupby('visible_fraction').size().to_dict(),
            'method':'name_knn_predicted_profiles','candidate_sha256':digest(args.output_dir/'candidate_names.json'),
            'data_sha256':digest(data.root/'manifest.json'),'name_cache_sha256':digest(cache/'manifest.json'),
            'scoring':SCORING,'correct_answers':CORRECT,'complete_test_opened':False,'elapsed_seconds':time.monotonic()-started,
            'scientific_claim_allowed':False,'candidate_vs_original_validation_mismatches':mismatches,'candidate_vs_original_validation_max_raw_difference':maxdiff})
        manifest.update(status='complete',elapsed_seconds=time.monotonic()-started,data_hash=digest(data.root/'manifest.json'),
            name_cache_hash=digest(cache/'manifest.json'),candidate_vectors_sha256=digest(args.output_dir/'candidate_scaled.npy'),
            all187_axis_validation_predictions_and_scores_exact=True,all19089_ranks_reload_exact=True,
            candidate_vs_original_validation_mismatches=mismatches,candidate_vs_original_validation_max_raw_difference=maxdiff)
        write_json(args.output_dir/'run_manifest.json',manifest);print(metrics);print('Candidate versus historical validation',mismatches,maxdiff)
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'run_manifest.json',manifest);raise


if __name__=='__main__':main()

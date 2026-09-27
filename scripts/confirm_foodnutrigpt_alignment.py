"""Audit the preregistered R5 replication and compare mean seed rates to fixed KNN."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import AlignmentPanel,evaluate_mapping,selection_score
from foodcomp.research_alignment_inference import AlignmentModel
from foodcomp.research_statistics import paired_group_rates

METRICS=['mrr','recall_at_1','recall_at_5','recall_at_10']
KEYS=['visible_fraction','profile_index','source_key','exact_name_group_id','observed_axes']
SEEDS=[20260922,20260923,20260924]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rated(ranks):
    result=ranks.copy()
    result['mrr']=1/result['rank']
    for k in [1,5,10]:result[f'recall_at_{k}']=(result['rank']<=k).astype(float)
    return result


def group_rates(ranks,fraction):
    rows=rated(ranks);rows=rows[rows.visible_fraction.eq(fraction)]
    return rows.groupby(['exact_name_group_id','source_key'])[METRICS].mean().groupby('exact_name_group_id').mean()


def mean_seed_rates(frames):
    """Average rates AFTER source-equal group aggregation; never average ranks."""
    if not frames:raise ValueError('No seed frames.')
    ordered=[x.sort_index() for x in frames]
    for frame in ordered:
        pd.testing.assert_index_equal(frame.index,ordered[0].index)
        if list(frame.columns)!=METRICS or not np.isfinite(frame.to_numpy()).all():raise ValueError('Invalid seed metrics.')
    return pd.DataFrame(np.mean([x.to_numpy() for x in ordered],axis=0),index=ordered[0].index,columns=METRICS)


def replay_gate(run,parent):
    current=read(run/'run_manifest.json');old=read(parent/'run_manifest.json')
    for field in ['initial_state_sha256','data_hash','panel_hash','name_cache_hash','parameter_count','seed','best_epoch','best_selection_mean_mrr','overfit']:
        assert current[field]==old[field],field
    old_args=dict(old['args']);old_args.setdefault('partial_view_probability',0.)
    new_args=dict(current['args']);old_args.pop('output_dir');new_args.pop('output_dir');assert old_args==new_args
    a=pd.read_csv(parent/'history.csv',float_precision='round_trip').drop(columns='elapsed_seconds')
    b=pd.read_csv(run/'history.csv',float_precision='round_trip').drop(columns='elapsed_seconds')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    files=['best_model.pt','latest_training_state.pt']+[f'best_through_epoch_{epoch:03d}.pt' for epoch in [8,20,60]]
    states=[]
    for filename in files:
        left=torch.load(parent/filename,map_location='cpu',weights_only=True)
        right=torch.load(run/filename,map_location='cpu',weights_only=True)
        assert left['best_epoch']==right['best_epoch'] and left['model_state'].keys()==right['model_state'].keys()
        assert all(torch.equal(left['model_state'][key],right['model_state'][key]) for key in left['model_state']),filename
        states.append({'file':filename,'model_tensors_exact':True,'parent_file_sha256':digest(parent/filename),'replication_file_sha256':digest(run/filename)})
    pd.testing.assert_frame_equal(pd.read_parquet(parent/'ranks.parquet'),pd.read_parquet(run/'ranks.parquet'),check_exact=True)
    assert read(parent/'metrics.json')['metrics']==read(run/'metrics.json')['metrics']
    return {'all60_non_timing_history_columns_exact':True,'all19089_ranks_and_metrics_exact':True,'states':states,
        'scope':'Model tensors compared exactly. Serialized checkpoint bytes differ due to registered arguments/output paths; no byte equality claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-only',action='store_true');p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','code_sha256':digest(Path(__file__)),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'complete_test_opened':False,'scientific_confirmation':False,'completion_milestone_met':False,
        'scope':'Conditional validation replication after seed22 model selection; no independent test, alias, label-validity or model-selection uncertainty claim.'}
    try:
        seeds=SEEDS[:1] if args.replay_only else SEEDS
        paths=[ROOT/f'output/v9_r5_confirmation_v1/contrastive_seed{seed}' for seed in seeds]
        receipt['seed22_compatibility']=replay_gate(paths[0],ROOT/'output/v9_r5/mapper_contrastive60')
        data=ResearchData(ROOT/'data/processed'/VERSION);device='cuda' if torch.cuda.is_available() else 'cpu'
        panel=AlignmentPanel(data,ROOT,device)
        baseline=ROOT/'output/v9_r5/retrieval_name_knn10_tree';bm=read(baseline/'metrics.json')
        br=pd.read_parquet(baseline/'ranks.parquet');metadata=[];ranks=[];records=[];histories=[]
        assert digest(baseline/'candidate_names.json')==bm['candidate_sha256'] and not bm['complete_test_opened']
        for seed,run in zip(seeds,paths):
            m=read(run/'run_manifest.json');h=pd.read_csv(run/'history.csv',float_precision='round_trip')
            assert m['status']=='complete' and m['epoch_completed']==60 and m['seed']==seed and not m['test_opened']
            assert m['args']['objective']=='contrastive' and m['args']['partial_view_probability']==0
            for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
            assert read(run/'panel_manifest.json')==panel.manifest and digest(run/'panel_manifest.json')==m['panel_hash']
            np.testing.assert_array_equal(h.epoch,np.arange(1,61));assert np.isfinite(h.select_dtypes('number')).all().all()
            for row in h.itertuples():
                assert row.training_profiles==len(panel.rows)
                assert row.training_order_sha256==fingerprint_array(np.random.default_rng(seed+row.epoch).permutation(len(panel.rows)))
            assert m['best_epoch']==int(h.loc[h.selection_mean_mrr.idxmax(),'epoch'])
            model=AlignmentModel(run,device=device);rr,scores=evaluate_mapping(model.predict_features,panel)
            pd.testing.assert_frame_equal(rr,pd.read_parquet(run/'ranks.parquet'),check_exact=True)
            sm=read(run/'metrics.json');assert scores==sm['metrics'] and selection_score(scores)==m['best_selection_mean_mrr']
            assert digest(run/'candidate_names.json')==sm['candidate_sha256'] and not sm['complete_test_opened']
            for field in ['data_sha256','candidate_sha256','candidate_count','correct_answers','scoring']:assert sm[field]==bm[field],field
            pd.testing.assert_frame_equal(br[KEYS].sort_values(KEYS).reset_index(drop=True),rr[KEYS].sort_values(KEYS).reset_index(drop=True))
            assert not rr.duplicated(['visible_fraction','profile_index']).any()
            assert np.isfinite(rr['rank']).all() and rr['rank'].between(1,sm['candidate_count']).all() and rr['rank'].eq(np.floor(rr['rank'])).all()
            names=panel.namespace.names[:2].tolist();profile={int(panel.axes[0]):0.}
            np.testing.assert_array_equal(model.encode(names[0],profile),model.encode(names[1],profile))
            assert len(model.retrieve_names(profile,names,top_k=2))==2
            for fraction in [1.,.3]:
                expected=next(s for s in scores if s['visible_fraction']==fraction)
                actual=group_rates(rr,fraction).mean()
                np.testing.assert_allclose(actual.to_numpy(),[expected[key] for key in METRICS],rtol=0,atol=1e-15)
            metadata.append(m);ranks.append(rr);histories.append(h)
            records.append({'seed':seed,'run':str(run.relative_to(ROOT)),'best_epoch':m['best_epoch'],'selection_mean_mrr':m['best_selection_mean_mrr'],
                'elapsed_seconds':m['elapsed_seconds'],'checkpoint_sha256':m['checkpoint_hash'],'manifest_sha256':digest(run/'run_manifest.json'),
                'ranks_sha256':digest(run/'ranks.parquet'),'metrics_sha256':digest(run/'metrics.json'),
                'all60_order_exposure_verified':True,'all19089_ranks_metrics_public_loader_replay_exact':True,'nutrition_api_name_independent':True})
        for m,h in zip(metadata[1:],histories[1:]):
            for key in ['code_commit','code_hashes','data_hash','panel_hash','name_cache_hash','parameter_count']:assert m[key]==metadata[0][key],key
            left={k:v for k,v in m['args'].items() if k not in ['seed','output_dir']}
            right={k:v for k,v in metadata[0]['args'].items() if k not in ['seed','output_dir']};assert left==right
            np.testing.assert_array_equal(h.learning_rate,histories[0].learning_rate)
        receipt.update(records=records,baseline={'directory':str(baseline.relative_to(ROOT)),'ranks_sha256':digest(baseline/'ranks.parquet'),
            'metrics_sha256':digest(baseline/'metrics.json')},data_sha256=bm['data_sha256'],candidate_sha256=bm['candidate_sha256'])
        if not args.replay_only:
            comparisons={};all_rows=[];source_rows=[]
            for fraction in [1.,.3]:
                groups=[group_rates(r,fraction) for r in ranks];average=mean_seed_rates(groups)
                comparisons[str(fraction)]=paired_group_rates(group_rates(br,fraction).reset_index(),average.reset_index(),METRICS)
                for key,item in comparisons[str(fraction)].items():
                    values=np.array([g[key].mean() for g in groups]);item['seed_values']=values.tolist();item['seed_sample_sd']=float(values.std(ddof=1))
                    item['scope']='Whole-food-group paired uncertainty conditional on these three fitted models; seed SD reported separately, not a seed bootstrap. Post-selection validation only.'
                for seed,r,g in zip(seeds,ranks,groups):
                    all_rows.append({'seed':seed,'visible_fraction':fraction,**g.mean().to_dict()})
                    for source,rows in rated(r)[lambda x:x.visible_fraction.eq(fraction)].groupby('source_key'):
                        values=rows.groupby('exact_name_group_id')[METRICS].mean().mean()
                        base_rows=rated(br)[lambda x:x.visible_fraction.eq(fraction)&x.source_key.eq(source)]
                        base_values=base_rows.groupby('exact_name_group_id')[METRICS].mean().mean()
                        source_rows.append({'seed':seed,'visible_fraction':fraction,'source_key':source,'profiles':len(rows),
                            'groups':rows.exact_name_group_id.nunique(),**values.to_dict(),**{f'baseline_{k}':base_values[k] for k in METRICS}})
            pd.DataFrame(all_rows).to_csv(args.output_dir/'seed_metrics.csv',index=False)
            pd.DataFrame(source_rows).to_csv(args.output_dir/'source_metrics.csv',index=False)
            selected=np.array([m['best_selection_mean_mrr'] for m in metadata])
            full=comparisons['1.0'];supported=all(all(v>full[key]['baseline'] for v in full[key]['seed_values']) and full[key]['difference_95_interval'][0]>0 for key in ['mrr','recall_at_10'])
            receipt.update(comparisons=comparisons,selection_mean_mrr={'mean':float(selected.mean()),'seed_sample_sd':float(selected.std(ddof=1))},
                full_input_specialist_three_seed_validation_support=bool(supported),general_sparse_superiority=False,
                completion='N/A: independent retrieval specialist',name_only_nutrition='N/A: independent retrieval specialist',
                total_training_elapsed_seconds=sum(m['elapsed_seconds'] for m in metadata))
        receipt['status']='complete'
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt)
    print({'status':receipt['status'],'seeds':seeds,'full_input_support':receipt.get('full_input_specialist_three_seed_validation_support')})


if __name__=='__main__':main()

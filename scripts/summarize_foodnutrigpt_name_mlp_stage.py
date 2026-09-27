"""Record the completed R7 MLP pair without claiming unfinished strong baselines."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    audit=read(ROOT/'reports/v9_r7_mlp_completed_audit_v1/verification.json');assert audit['status']=='complete'
    pair=read(ROOT/'reports/v9_r7_mlp128_vs32_name_only_v1/summary.json')
    retrieval_pair=read(ROOT/'reports/v9_r7_mlp128_vs32_retrieval_v1/summary.json');records=[]
    for dimension in [32,128]:
        run=ROOT/f'output/v9_r7/exact_name_mlp{dimension}';m=read(run/'run_manifest.json');assert m['status']=='complete'
        metrics=read(run/'metrics.json');retrieval=read(ROOT/f'output/v9_r7/retrieval_name_mlp{dimension}/metrics.json')
        records.append({'active_components':dimension,'manifest':m,'metrics':metrics,'retrieval':retrieval,
            'manifest_sha256':digest(run/'run_manifest.json')})
    axis=pd.read_csv(ROOT/'reports/v9_r7_mlp128_vs32_name_only_v1/axis_paired_intervals.csv');axis=axis[axis.loss_group.eq('nutrition')]
    source=pd.read_csv(ROOT/'reports/v9_r7_mlp128_vs32_name_only_v1/source_metrics.csv').pivot(index='source',columns='role',values='scaled_log_mae')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for dimension in [32,128]:
        history=pd.read_csv(ROOT/f'output/v9_r7/exact_name_mlp{dimension}/history.csv')
        axs[0,0].plot(history.epoch,history.validation_primary,label=f'{dimension} active / 128 slots')
        axs[0,1].plot(history.epoch,history.train_loss,label=f'{dimension} active')
    axs[0,0].set(title='Fixed nutrition validation MAE',xlabel='Epoch',ylabel='142-axis scaled-log MAE');axs[0,0].legend(fontsize=9)
    axs[0,1].set(title='Matched name-only training objective',xlabel='Epoch',ylabel='Batch-local 187-axis MAE');axs[0,1].legend(fontsize=9)
    keys=['scaled_log_mae','positive_scaled_log_mae','zero_scaled_log_mae'];x=np.arange(3)
    for i,record in enumerate(records):
        scores=record['metrics']['name_only']['nutrition'];axs[1,0].bar(x+(-.18 if i==0 else .18),[scores[k] for k in keys],.36,label=str(record['active_components'])+' active')
    axs[1,0].set_xticks(x,['Overall','Positive only','Observed zero only']);axs[1,0].set(title='Same selected checkpoints',ylabel='Scaled-log MAE');axs[1,0].legend(fontsize=9)
    for i,fraction in enumerate([1.,.3]):
        points=[next(m['recall_at_10'] for m in r['retrieval']['metrics'] if m['visible_fraction']==fraction) for r in records]
        axs[1,1].bar(np.arange(2)+(-.18 if i==0 else .18),points,.36,label='Full observed' if i==0 else '30% visible')
    axs[1,1].set_xticks([0,1],['32 active','128 active']);axs[1,1].set(title='Nutrition-to-name via name-predicted profiles',ylabel='Recall@10');axs[1,1].legend(fontsize=9)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15)
    fig.suptitle('R7 matched name-compression experiment: single-seed MLP stage\nStrong same-input KNN comparison is a separate required stage',fontsize=12)
    for ext in ['png','svg']:fig.savefig(args.output_dir/f'name_mlp.{ext}',dpi=160)
    result={'status':'mlp_stage_complete_full_r7_not_complete','records':records,'name_only_comparison':pair,'retrieval_comparison':retrieval_pair,
        'audit_sha256':digest(ROOT/'reports/v9_r7_mlp_completed_audit_v1/verification.json'),'code_sha256':digest(Path(__file__)),
        'nutrition_axes_point_better':int((axis.candidate_minus_baseline<0).sum()),'nutrition_axes_interval_better':int((axis.difference_95_high<0).sum()),
        'nutrition_axes_interval_worse':int((axis.difference_95_low>0).sum()),'sources_point_better':int((source.candidate<source.baseline).sum()),
        'total_new_mlp_training_seconds':sum(r['manifest']['elapsed_seconds'] for r in records),'scientific_confirmation':False,'complete_test_opened':False,
        'scope':'Within-MLP active dimension comparison only, one seed. Historical32inputtrees not a valid newinput completion reference. FullR7 requires completed same-inputKNN andall comparisons.'}
    write_json(args.output_dir/'summary.json',result)
    print({key:result[key] for key in ['status','nutrition_axes_point_better','nutrition_axes_interval_better','nutrition_axes_interval_worse','sources_point_better','total_new_mlp_training_seconds']})
    print(pair['paired_intervals']);print(pair['primary_change_decomposition'])

if __name__=='__main__':main()

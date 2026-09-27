"""Close the registered R6 context block only when both arms and all audits exist."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import write_json,digest

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    dest=ROOT/'experiments/foodnutrigpt_v9_research/r6/results_summary.json'
    if dest.exists():raise FileExistsError(dest)
    config=read(ROOT/'experiments/foodnutrigpt_v9_research/r6/config.json');records=[]
    parent=ROOT/'output/v9_r2/mlp60_mae_width512';pm=read(parent/'metrics.json');parent_run=read(parent/'run_manifest.json')
    for tag in ['mix','drop30']:
        run=ROOT/f'output/v9_r6/mlp512_context_{tag}';m=read(run/'run_manifest.json');assert m['status']=='complete' and not m['test_opened']
        audit=ROOT/f'reports/v9_r6_{tag}_completed_audit_v1/verification.json';assert read(audit)['status']=='complete'
        pairs={}
        for task,reference in [('completion','parent'),('completion','xgb22'),('completion','rf22'),('name_only','parent'),('name_only','knn_tree')]:
            name=f'{task}_vs_{reference}';path=ROOT/f'reports/v9_r6_{tag}_{name}_v1/summary.json';pairs[name]=read(path)
        for reference in ['parent','knn_tree']:
            name=f'retrieval_vs_{reference}';pairs[name]=read(ROOT/f'reports/v9_r6_{tag}_{name}_v1/summary.json')
        fit=read(ROOT/f'reports/v9_r6_{tag}_fit_v1/summary.json')
        record={'tag':tag,'run':str(run.relative_to(ROOT)),'manifest':m,'manifest_sha256':digest(run/'run_manifest.json'),
            'metrics':read(run/'metrics.json'),'retrieval':read(ROOT/f'output/v9_r6/retrieval_context_{tag}/metrics.json'),
            'audit':read(audit),'audit_sha256':digest(audit),'fit':fit,'comparisons':pairs}
        counts={}
        for task in ['completion','name_only']:
            folder=ROOT/f'reports/v9_r6_{tag}_{task}_vs_parent_v1'
            axes=pd.read_csv(folder/'axis_paired_intervals.csv');axes=axes[axes.loss_group.eq('nutrition')]
            sources=pd.read_csv(folder/'source_metrics.csv').pivot(index='source',columns='role',values='scaled_log_mae')
            counts[task]={'nutrition_axes_point_better':int((axes.candidate_minus_baseline<0).sum()),
                'nutrition_axes_interval_better':int((axes.difference_95_high<0).sum()),'nutrition_axes_interval_worse':int((axes.difference_95_low>0).sum()),
                'sources_point_better':int((sources.candidate<sources.baseline).sum()),'source_count':len(sources)}
        record['parent_improvement_counts']=counts;records.append(record)
    for key in ['code_commit','code_hashes','data_hash','panel_hash','name_cache_hash','seed','parameter_count']:
        assert records[0]['manifest'][key]==records[1]['manifest'][key],key
    a={k:v for k,v in records[0]['manifest']['args'].items() if k not in ['output_dir','context_dropout']}
    b={k:v for k,v in records[1]['manifest']['args'].items() if k not in ['output_dir','context_dropout']};assert a==b
    output=ROOT/'reports/v9_r6_context_figures_v1';output.mkdir(exist_ok=False)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    runs=[('Parent',parent)]+[(r['tag'],ROOT/r['run']) for r in records]
    for label,path in runs:
        h=pd.read_csv(path/'history.csv');axs[0,0].plot(h.epoch,h.validation_primary,label=label)
    axs[0,0].axhline(.17378389941923483,color='black',ls='--',label='XGB seed22')
    axs[0,0].set(title='Fixed completion validation panel',xlabel='Epoch',ylabel='142-axis scaled-log MAE');axs[0,0].legend(fontsize=8)
    parent_fit=next(row for row in read(ROOT/'reports/v9_r2_mlp60_fit_v1/summary.json')['models'] if row['run']=='mlp60_mae_width512')
    assert parent_fit['checkpoint_sha256']==parent_run['checkpoint_hash']
    train=[parent_fit['training']['nutrition']['scaled_log_mae']]+[r['fit']['models'][0]['training']['nutrition']['scaled_log_mae'] for r in records]
    valid=[pm['completion']['nutrition']['scaled_log_mae']]+[r['metrics']['completion']['nutrition']['scaled_log_mae'] for r in records]
    x=np.arange(3);axs[0,1].bar(x-.18,train,.36,label='Original family training context');axs[0,1].bar(x+.18,valid,.36,label='Validation')
    axs[0,1].set_xticks(x,['Parent','Mixed drop','Fixed 30%']);axs[0,1].set(title='Same-context fitted-model diagnostic',ylabel='142-axis scaled-log MAE');axs[0,1].legend(fontsize=8)
    name=[pm['name_only']['nutrition']['scaled_log_mae']]+[r['metrics']['name_only']['nutrition']['scaled_log_mae'] for r in records]
    axs[1,0].bar(x,name);axs[1,0].axhline(.26175591502686196,color='black',ls='--',label='Name KNN10 / KDTree')
    axs[1,0].set_xticks(x,['Parent','Mixed drop','Fixed 30%']);axs[1,0].set(title='Name-only nutrition prediction',ylabel='142-axis scaled-log MAE');axs[1,0].legend(fontsize=8)
    retrieval=[read(ROOT/'output/v9_r2/retrieval_mlp60_mae_width512/metrics.json')]+[r['retrieval'] for r in records]+[read(ROOT/'output/v9_r5/retrieval_name_knn10_tree/metrics.json')]
    x=np.arange(4)
    for j,fraction in enumerate([1.,.3]):
        values=[next(m['recall_at_10'] for m in item['metrics'] if m['visible_fraction']==fraction) for item in retrieval]
        axs[1,1].bar(x+(-.18 if j==0 else .18),values,.36,label='Full observed input' if j==0 else '30% visible input')
    axs[1,1].set_xticks(x,['Parent','Mixed drop','Fixed 30%','KNN']);axs[1,1].set(title='Same checkpoints: nutrition-to-name retrieval',ylabel='Recall@10');axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15)
    fig.suptitle('R6 context dropout: single-seed controlled exploration\nSupervision unchanged; all fitted models evaluated on the frozen original panels',fontsize=12)
    for ext in ['png','svg']:fig.savefig(output/f'context.{ext}',dpi=160)
    result={'version':'V9-R6','status':'registered_context_block_complete_single_seed','config_sha256':digest(ROOT/'experiments/foodnutrigpt_v9_research/r6/config.json'),
        'parent_manifest_sha256':digest(parent/'run_manifest.json'),'parent_metrics':pm,'records':records,'script_sha256':digest(Path(__file__)),
        'total_new_training_elapsed_seconds':sum(r['manifest']['elapsed_seconds'] for r in records),
        'complete_test_opened':False,'scientific_confirmation':False,'completion_improvement_accepted':False,
        'scope':'Two preregistered training-context configurations plus reused parent; no validation changes or new seed confirmation. Source/category holdout andfewshot remain outstanding.'}
    write_json(dest,result)
    for r in records:
        print(r['tag'],r['manifest']['best_epoch'],r['metrics']['completion']['nutrition'],r['metrics']['name_only']['nutrition'])
        print('train',r['fit']['models'][0]['training']['nutrition'],'counts',r['parent_improvement_counts'])
        for label,pair in r['comparisons'].items():
            if 'paired_intervals' in pair:print(label,pair['paired_intervals'],pair['primary_change_decomposition'])

if __name__=='__main__':main()

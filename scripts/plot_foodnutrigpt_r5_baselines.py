"""Final R5 baseline tradeoffs; separate models and visibility scenarios stay explicit."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained');colors=['#167d9a','#bd7736']
    for name,label,color in zip(['name_mlp60_legacy_loss','name_mlp60_mae'],['Forward SmoothL1','Forward MAE'],colors):
        path=ROOT/'output/v9_r5'/name;h=pd.read_csv(path/'history.csv');m=json.loads((path/'run_manifest.json').read_text());assert m['status']=='complete'
        axes[0,0].plot(h.epoch,h.validation_primary,label=label,color=color)
        sel=h[h.epoch.eq(m['best_epoch'])].iloc[0];axes[0,0].scatter([sel.epoch],[sel.validation_primary],color=color,s=25)
    knn=json.loads((ROOT/'output/v9_r5/retrieval_name_knn10_tree/nutrition_metrics.json').read_text())
    axes[0,0].axhline(knn['nutrition']['scaled_log_mae'],label='Name KNN (tree)',color='#589c66',ls='--')
    axes[0,0].set(title='Name-only nutrition validation',xlabel='Epoch',ylabel='142-axis scaled-log MAE');axes[0,0].legend(frameon=False);axes[0,0].grid(alpha=.2)
    s=json.loads((ROOT/'reports/v9_r5_forward_mae_name_vs_smooth_v1/summary.json').read_text())['primary_change_decomposition']
    values=[s['positive_delta_contribution'],s['zero_delta_contribution']]
    axes[0,1].barh(['Positive labels','Explicit zeros'],values,color=['#bd7736','#167d9a']);axes[0,1].axvline(0,color='gray')
    axes[0,1].set(title='MAE minus SmoothL1: additive error change',xlabel='Contribution to142-axis MAE (negative is better)');axes[0,1].grid(axis='x',alpha=.2)
    labels=['Forward SmoothL1','Forward MAE','Name KNN (tree)','Full-view mapping','Partial-view mapping']
    paths=['retrieval_name_mlp60_legacy_loss','retrieval_name_mlp60_mae','retrieval_name_knn10_tree','mapper_contrastive60','mapper_contrastive60_partial05']
    for fraction,ax in zip([1.,.3],axes[1]):
        vals=[]
        for path in paths:
            m=json.loads((ROOT/'output/v9_r5'/path/'metrics.json').read_text());vals.append(next(x for x in m['metrics'] if x['visible_fraction']==fraction)['recall_at_10']*100)
        bars=ax.barh(labels,vals,color=['#167d9a','#bd7736','#589c66','#a45d70','#846ab0']);ax.invert_yaxis()
        ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set_xlim(0,max(vals)*1.25)
        ax.set(title=f'Retrieval: {"full" if fraction==1. else "30%"} input',xlabel='Recall@10 (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('R5: stronger name baselines change the retrieval conclusion\nSingle-seed neural models; fixed deterministic name KNN; paired intervals reported separately',fontsize=13)
    args.output_dir.mkdir(parents=True)
    for ext in ['png','svg']:fig.savefig(args.output_dir/f'baselines.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()

"""Show the registered training-view intervention without combining model capabilities."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,label,color in [('mapper_contrastive60','Full-view training','#bd7736'),('mapper_contrastive60_partial05','50% partial-view training','#167d9a')]:
        path=ROOT/'output/v9_r5'/name
        m=json.loads((path/'run_manifest.json').read_text());assert m['status']=='complete'
        h=pd.read_csv(path/'history.csv');selected=h[h.epoch.eq(m['best_epoch'])].iloc[0]
        for fraction,ax in zip(['1.0','0.3'],axes[0]):
            ax.plot(h.epoch,h[f'validation_{fraction}_mrr'],label=label,color=color,lw=1.8)
            ax.scatter([selected.epoch],[selected[f'validation_{fraction}_mrr']],s=28,color=color)
            ax.set(xlabel='Epoch',ylabel='Food-group MRR',title=f'Validation: {"full" if fraction=="1.0" else "30%"} nutrition input')
            ax.grid(alpha=.2);ax.legend(frameon=False)
    for fraction,ax in zip(['1.0','0.3'],axes[1]):
        d=json.loads((ROOT/'reports/v9_r5_partial05_vs_parent_v1/summary.json').read_text())['comparisons'][fraction]
        for i,key in enumerate(['mrr','recall_at_1','recall_at_5','recall_at_10']):
            v=d[key];point=v['candidate_minus_baseline']*100;lo,hi=[x*100 for x in v['difference_95_interval']]
            ax.plot([lo,hi],[i,i],color='#167d9a',lw=2.4);ax.scatter([point],[i],color='#167d9a',s=30)
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),['MRR','Recall@1','Recall@5','Recall@10']);ax.invert_yaxis()
        ax.set(xlabel='Partial-view training minus parent (percentage points), 95% interval',title=f'{"Full" if fraction=="1.0" else "30%"} input: paired food groups');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Independent retrieval: changing training visibility only (one seed)',fontsize=14)
    args.output_dir.mkdir(parents=True)
    for ext in ['png','svg']:fig.savefig(args.output_dir/f'partial_views.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()

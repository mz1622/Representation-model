"""Full forward learning curve and single-checkpoint retrieval results."""
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
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for name,color,label in [('v9_r0/name_mlp8_quarantined','#888888','8 epochs / cosine8'),('v9_r5/name_mlp60_legacy_loss','#167d9a','60 epochs / cosine60')]:
        path=ROOT/'output'/name;h=pd.read_csv(path/'history.csv');m=json.loads((path/'run_manifest.json').read_text());assert m['status']=='complete'
        axes[0].plot(h.epoch,h.validation_primary,label=label,color=color);selected=h[h.epoch.eq(m['best_epoch'])].iloc[0]
        axes[0].scatter([selected.epoch],[selected.validation_primary],color=color,s=28)
    axes[0].set(title='Name-only nutrition: validation',xlabel='Epoch',ylabel='142-axis scaled-log MAE (lower is better)');axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    names=['Forward8','Forward60','Full-view mapping','Partial-view mapping']
    paths=['v9_r0/retrieval_name_mlp_v1','v9_r5/retrieval_name_mlp60_legacy_loss','v9_r5/mapper_contrastive60','v9_r5/mapper_contrastive60_partial05']
    for fraction,ax in zip([1.,.3],axes[1:]):
        vals=[]
        for path in paths:
            m=json.loads((ROOT/'output'/path/'metrics.json').read_text());vals.append(next(x for x in m['metrics'] if x['visible_fraction']==fraction)['recall_at_10']*100)
        bars=ax.barh(names,vals,color=['#888888','#167d9a','#bd7736','#589c66']);ax.invert_yaxis()
        ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set_xlim(0,max(vals)*1.3)
        ax.set(title=f'Retrieval: {"full" if fraction==1. else "30%"} input',xlabel='Recall@10 (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Forward budget extension: same loss and width, changed duration and cosine horizon',fontsize=13)
    args.output_dir.mkdir(parents=True)
    for ext in ['png','svg']:fig.savefig(args.output_dir/f'forward_baseline.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()

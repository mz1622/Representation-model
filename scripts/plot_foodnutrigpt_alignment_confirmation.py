"""Plot every seed, both retrieval coverages and conditional food-group intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    receipt=json.loads((ROOT/'reports/v9_r5_confirmation_v1/verification.json').read_text(encoding='utf-8'))
    assert receipt['status']=='complete'
    output=ROOT/'reports/v9_r5_confirmation_figures_v1'
    output.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,fraction in enumerate(['1.0','0.3']):
        ax=axes[0,j]
        for row in receipt['records']:
            history=pd.read_csv(ROOT/row['run']/'history.csv')
            ax.plot(history.epoch,history[f'validation_{fraction}_mrr'],label=str(row['seed']))
        ax.axhline(receipt['comparisons'][fraction]['mrr']['baseline'],color='black',ls='--',label='Name KNN10 / KDTree')
        ax.set(title=f"{'Full observed' if j==0 else '30% visible'} input: all training epochs",xlabel='Epoch',ylabel='MRR')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
        ax=axes[1,j];keys=['mrr','recall_at_1','recall_at_5','recall_at_10']
        for i,key in enumerate(keys):
            item=receipt['comparisons'][fraction][key];point=item['candidate_minus_baseline']
            low,high=item['difference_95_interval']
            ax.errorbar(i,point,yerr=np.array([[point-low],[high-point]]),fmt='o',color='tab:blue',capsize=4)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['MRR','Recall@1','Recall@5','Recall@10'])
        ax.set(title='Mean seed metric minus fixed name KNN',ylabel='Absolute difference; paired group 95% CI');ax.grid(alpha=.15)
    fig.suptitle('R5 three-seed validation replication — separate retrieval specialist\nConditional on selected model family; neither independent test nor completion evidence',fontsize=12)
    for ext in ['png','svg']:fig.savefig(output/f'confirmation.{ext}',dpi=160)
    source=pd.read_csv(ROOT/'reports/v9_r5_confirmation_v1/source_metrics.csv')
    for (seed,fraction),rows in source.groupby(['seed','visible_fraction']):
        print(seed,fraction,'sources with positive MRR:',int((rows.mrr>rows.baseline_mrr).sum()),'/',len(rows))

if __name__=='__main__':main()

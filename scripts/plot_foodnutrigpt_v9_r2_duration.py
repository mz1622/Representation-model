"""Static summary of the predeclared 20/60 budget comparison on one fixed trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    run=root/"output/v9_r2/mlp60_mae_width512"
    load=lambda path:json.loads(path.read_text())
    scores=[load(run/"evaluation_through_020/metrics.json")["completion"]["nutrition"],load(run/"metrics.json")["completion"]["nutrition"]]
    fits=[load(root/"reports/v9_r2_mlp60_through20_fit_v1/summary.json")["models"][0],load(root/"reports/v9_r2_mlp60_fit_v1/summary.json")["models"][0]]
    history=pd.read_csv(run/"history.csv")
    colors=["#457B9D","#E76F51"];labels=["20-epoch budget","60-epoch budget"]
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    ax[0].plot(history.epoch,history.validation_primary,color="#457B9D",lw=2,label="MAE MLP / width512")
    for path,label,color in [("xgb800d10","XGB800 / depth10","#444444"),("rf400_unlimited_leaf1","RF400 / unlimited / leaf1","#2A9D8F")]:
        value=load(root/f"output/v9_r1/{path}/metrics.json")["nutrition"]["scaled_log_mae"]
        ax[0].axhline(value,color=color,ls="--",lw=1.2,label=label)
    for budget,score,color in zip([20,60],scores,colors):
        selected=history[history.epoch.le(budget)].sort_values("validation_primary").iloc[0]
        ax[0].scatter(selected.epoch,score["scaled_log_mae"],color=color,s=35,zorder=3)
    ax[0].set(xlabel="Epoch; same 60-epoch LR horizon",ylabel="Validation nutrition scaled-log MAE",title="Completion improves; tree gap remains")
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    for i,(fit,score,label,color) in enumerate(zip(fits,scores,labels,colors)):
        ax[1].bar(np.arange(2)+(i-.5)*.36,[fit["training"]["nutrition"]["scaled_log_mae"],score["scaled_log_mae"]],width=.36,color=color,label=label)
        ax[2].bar(np.arange(2)+(i-.5)*.36,[score["positive_scaled_log_mae"],score["zero_scaled_log_mae"]],width=.36,color=color,label=label)
    ax[1].set(xticks=[0,1],xticklabels=["Training panel","Validation panel"],ylabel="Nutrition scaled-log MAE",title="Fit and generalization both matter")
    ax[1].legend(fontsize=8)
    ax[2].set(xticks=[0,1],xticklabels=["Positive targets","Explicit zeros"],ylabel="Conditional macro MAE",title="Both types improve within this trajectory")
    fig.suptitle("Single-seed fixed-schedule duration study; model selection uses validation only",fontsize=12)
    fig.savefig(args.output_dir/"duration_comparison.png",dpi=180);fig.savefig(args.output_dir/"duration_comparison.svg");plt.close(fig)


if __name__=="__main__":main()

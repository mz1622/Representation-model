"""Static figures for the registered R2 SmoothL1-to-MAE comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    paths=[root/"output/v9_r1/mlp20_panel",root/"output/v9_r2/mlp20_mae"]
    labels=["SmoothL1 (R1)","MAE (R2)"];colors=["#457B9D","#E76F51"]
    scores=[json.loads((path/"metrics.json").read_text())["completion"]["nutrition"] for path in paths]
    tree=json.loads((root/"output/v9_r1/xgb500d8/metrics.json").read_text())["nutrition"]["scaled_log_mae"]
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    for path,label,color in zip(paths,labels,colors):
        history=pd.read_csv(path/"history.csv")
        ax[0].plot(history.epoch,history.validation_primary,label=label,color=color,lw=2)
    ax[0].axhline(tree,color="#444444",ls="--",label="XGB500 depth 8")
    ax[0].set(xlabel="Epoch (fixed 20-epoch LR horizon)",ylabel="Validation nutrition scaled-log MAE",title="Same task panel and selection metric")
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    positions=np.arange(3)
    for i,(score,label,color) in enumerate(zip(scores,labels,colors)):
        ax[1].bar(positions+(i-.5)*.36,[score[k] for k in ["scaled_log_mae","positive_scaled_log_mae","zero_scaled_log_mae"]],width=.36,color=color,label=label)
    ax[1].set(xticks=positions,xticklabels=["All", "Positive only", "Explicit zero"],ylabel="Macro MAE (conditional subsets)",title="Positive/zero trade-off")
    report=json.loads((root/"reports/v9_r2_mae_vs_smoothl1_v2/summary.json").read_text())
    delta=report["primary_change_decomposition"]
    ax[2].bar(["Positive cells","Zero cells","Net"],[delta["positive_delta_contribution"],delta["zero_delta_contribution"],sum(delta.values())],color=["#457B9D","#E76F51","#444444"])
    ax[2].axhline(0,color="black",lw=.7)
    ax[2].set(ylabel="Change in primary MAE (R2 minus R1)",title="Additive contributions; same denominator")
    fig.suptitle("Single-seed exploration: loss alignment improves zeros, but does not beat XGBoost",fontsize=12)
    fig.savefig(args.output_dir/"loss_comparison.png",dpi=180)
    fig.savefig(args.output_dir/"loss_comparison.svg")
    plt.close(fig)


if __name__=="__main__":main()

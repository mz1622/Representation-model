"""Plot completed R3 task-proportion controls and food-group uncertainty."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--percent",type=int,choices=[10,20],required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    run=ROOT/f"output/v9_r3/mlp60_name_mix{args.percent}"
    baseline=ROOT/"output/v9_r2/mlp60_mae_width512"
    for folder in [run,baseline]:
        manifest=json.loads((folder/"run_manifest.json").read_text())
        if manifest["status"]!="complete" or manifest["args"]["epochs"]!=60:raise ValueError("Require completed 60-epoch runs.")
    summaries={task:json.loads((ROOT/f"reports/v9_r3_mix{args.percent}_{suffix}_vs0_v1/summary.json").read_text())
        for task,suffix in [("Completion","completion"),("Name only","name")]}
    tree=json.loads((ROOT/"output/v9_r1/xgb800d10/metrics.json").read_text())["nutrition"]["scaled_log_mae"]
    args.output_dir.mkdir(parents=True)
    fig,(curve,interval)=plt.subplots(1,2,figsize=(11.8,4.8),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
    for folder,label,color in [(baseline,"0% control","#167d9a"),(run,f"{args.percent}% name-only tasks","#bd7736")]:
        history=pd.read_csv(folder/"history.csv")
        if not np.array_equal(history.epoch,np.arange(1,61)):raise ValueError("Incomplete epoch history.")
        curve.plot(history.epoch,history.validation_primary,label=label,color=color,lw=2)
        best=history.loc[history.validation_primary.idxmin()]
        curve.scatter([best.epoch],[best.validation_primary],color=color,s=38,zorder=4)
    curve.axhline(tree,color="#a9473e",ls="--",lw=1.2,label=f"XGB800d10: {tree:.4f}")
    curve.set(xlabel="Epoch (fixed 60-epoch cosine schedule)",ylabel="142-axis scaled-log MAE (lower is better)",title="Same model and target supervision")
    curve.legend(frameon=False,fontsize=8.5);curve.grid(alpha=.17)
    labels=[];row=0
    for task,color in [("Completion","#167d9a"),("Name only","#bd7736")]:
        for metric,label in [("scaled_log_mae","Primary"),("log_mae","Legacy log-MAE")]:
            r=summaries[task]["paired_intervals"][metric]
            point=r["relative_improvement"]*100;low,high=np.array(r["relative_improvement_95_interval"])*100
            interval.plot([low,high],[row,row],color=color,lw=2)
            interval.scatter([point],[row],color=color,s=40)
            labels.append(f"{task}\n{label}");row+=1
    interval.axvline(0,color="#666666",ls="--",lw=1)
    interval.set_yticks(range(row),labels);interval.invert_yaxis();interval.grid(axis="x",alpha=.17)
    interval.set(xlabel="Relative improvement (%) with 95% food-group interval",title="Task changes versus the 0% control")
    fig.suptitle(f"Name-only training fraction: 0% to {args.percent}%",fontsize=14)
    fig.supxlabel("Single seed; validation only. Intervals exclude seed and selection uncertainty. Both tasks use one completion-selected checkpoint.",fontsize=8.5)
    for extension in ["png","svg"]:fig.savefig(args.output_dir/f"mix{args.percent}_vs0.{extension}",dpi=160)
    plt.close(fig);print(args.output_dir)


if __name__=="__main__":main()

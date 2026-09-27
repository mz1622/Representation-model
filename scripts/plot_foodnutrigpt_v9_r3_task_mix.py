"""Plot completed R3 task-proportion or prediction-head controls."""
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
    comparison=p.add_mutually_exclusive_group(required=True)
    comparison.add_argument("--percent",type=int,choices=[10,20])
    comparison.add_argument("--separate-heads",action="store_true")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.separate_heads:
        run=ROOT/"output/v9_r3/mlp60_mix10_separate_heads"
        baseline=ROOT/"output/v9_r3/mlp60_name_mix10"
        report_paths={"Completion":"v9_r3_heads_completion_vs_shared_v1",
                      "Name only":"v9_r3_heads_name_only_vs_shared_v1"}
        control_label="Shared head, 10% name-only"
        candidate_label="Separate heads, 10% name-only"
        title="Shared versus separate prediction heads"
        curve_title="Fixed 10% name-only task fraction"
        interval_title="Task changes versus the shared head"
        filename="heads_vs_shared"
        note="Separate heads add 129,276 parameters; this does not isolate gradient conflict."
    else:
        run=ROOT/f"output/v9_r3/mlp60_name_mix{args.percent}"
        baseline=ROOT/"output/v9_r2/mlp60_mae_width512"
        report_paths={task:f"v9_r3_mix{args.percent}_{suffix}_vs0_v1"
                      for task,suffix in [("Completion","completion"),("Name only","name")]}
        control_label="0% control"
        candidate_label=f"{args.percent}% name-only tasks"
        title=f"Name-only training fraction: 0% to {args.percent}%"
        curve_title="Same model and target supervision"
        interval_title="Task changes versus the 0% control"
        filename=f"mix{args.percent}_vs0"
        note=""
    for folder in [run,baseline]:
        manifest=json.loads((folder/"run_manifest.json").read_text())
        if manifest["status"]!="complete" or manifest["args"]["epochs"]!=60:raise ValueError("Require completed 60-epoch runs.")
    summaries={task:json.loads((ROOT/"reports"/path/"summary.json").read_text())
        for task,path in report_paths.items()}
    tree=json.loads((ROOT/"output/v9_r1/xgb800d10/metrics.json").read_text())["nutrition"]["scaled_log_mae"]
    args.output_dir.mkdir(parents=True)
    fig,(curve,interval)=plt.subplots(1,2,figsize=(11.8,4.8),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
    for folder,label,color in [(baseline,control_label,"#167d9a"),(run,candidate_label,"#bd7736")]:
        history=pd.read_csv(folder/"history.csv")
        if not np.array_equal(history.epoch,np.arange(1,61)):raise ValueError("Incomplete epoch history.")
        if not np.isfinite(history.validation_primary).all():raise ValueError("Non-finite epoch metric.")
        curve.plot(history.epoch,history.validation_primary,label=label,color=color,lw=2)
        best=history.loc[history.validation_primary.idxmin()]
        curve.scatter([best.epoch],[best.validation_primary],color=color,s=38,zorder=4)
    curve.axhline(tree,color="#a9473e",ls="--",lw=1.2,label=f"XGB800d10: {tree:.4f}")
    curve.set(xlabel="Epoch (fixed 60-epoch cosine schedule)",ylabel="142-axis scaled-log MAE (lower is better)",title=curve_title)
    curve.legend(frameon=False,fontsize=8.5);curve.grid(alpha=.17)
    labels=[];row=0
    for task,color in [("Completion","#167d9a"),("Name only","#bd7736")]:
        for metric,label in [("scaled_log_mae","Primary"),("log_mae","Legacy log-MAE")]:
            r=summaries[task]["paired_intervals"][metric]
            point=r["relative_improvement"]*100;low,high=np.array(r["relative_improvement_95_interval"])*100
            if not np.isfinite([point,low,high]).all() or low>high:raise ValueError("Invalid paired interval.")
            interval.plot([low,high],[row,row],color=color,lw=2)
            interval.scatter([point],[row],color=color,s=40)
            labels.append(f"{task}\n{label}");row+=1
    interval.axvline(0,color="#666666",ls="--",lw=1)
    interval.set_yticks(range(row),labels);interval.invert_yaxis();interval.grid(axis="x",alpha=.17)
    interval.set(xlabel="Relative improvement (%) with 95% food-group interval",title=interval_title)
    fig.suptitle(title,fontsize=14)
    footer="Single seed; validation only. Intervals exclude seed and selection uncertainty. Both tasks use one completion-selected checkpoint."
    if note:footer+="\n"+note
    fig.supxlabel(footer,fontsize=8.5)
    for extension in ["png","svg"]:fig.savefig(args.output_dir/f"{filename}.{extension}",dpi=160)
    plt.close(fig);print(args.output_dir)


if __name__=="__main__":main()

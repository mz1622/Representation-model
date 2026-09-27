"""Plot the completed V9 fixed-schedule duration comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"reports/v9_r1_v9_duration_figures_v1"
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
history=pd.read_csv(ROOT/"output/v9_r1/v9_20_a1_s1/history.csv")
summaries={task:json.loads((ROOT/f"reports/v9_r1_v9_duration{suffix}_v1/summary.json").read_text())
           for task,suffix in [("Completion",""),("Name only","_nameonly")]}
tree=json.loads((ROOT/"output/v9_r1/xgb800d10/metrics.json").read_text())["nutrition"]["scaled_log_mae"]
fig,(ax,left)=plt.subplots(1,2,figsize=(11.8,4.8),gridspec_kw={"width_ratios":[1.25,1]},layout="constrained")
ax.plot(history.epoch,history.validation_primary,color="#167d9a",lw=2,label="V9 validation primary")
for budget,color in [(8,"#bd7736"),(20,"#2450a4")]:
    part=history[history.epoch<=budget];best=part.loc[part.validation_primary.idxmin()]
    ax.scatter([best.epoch],[best.validation_primary],s=55,color=color,zorder=4)
    ax.annotate(f"Through {budget}: {best.validation_primary:.4f}",(best.epoch,best.validation_primary),
                xytext=(-12,18 if budget==8 else -26),textcoords="offset points",ha="right",fontsize=9,color=color)
ax.axhline(tree,color="#bd473b",ls="--",lw=1.3,label=f"XGB800d10: {tree:.4f}")
ax.set(xlabel="Epoch (fixed 20-epoch cosine schedule)",ylabel="142-axis scaled-log MAE (lower is better)",title="More training improves completion")
ax.legend(loc="upper right",frameon=False,fontsize=8.5);ax.grid(alpha=.17)
labels=[];points=[];bounds=[];colors=[]
for task in ["Completion","Name only"]:
    for metric,label in [("scaled_log_mae","Primary"),("log_mae","Legacy log-MAE")]:
        x=summaries[task]["paired_intervals"][metric]
        labels.append(f"{task}\n{label}");points.append(x["relative_improvement"]*100)
        bounds.append(np.array(x["relative_improvement_95_interval"])*100)
        colors.append("#167d9a" if task=="Completion" else "#bd7736")
for i,(point,bound,color) in enumerate(zip(points,bounds,colors)):
    left.errorbar(point,i,xerr=[[point-bound[0]],[bound[1]-point]],fmt="o",color=color,capsize=4,lw=1.8)
left.axvline(0,color="#666666",ls="--",lw=1)
left.set_yticks(range(4),labels);left.invert_yaxis();left.grid(axis="x",alpha=.17)
left.set(xlabel="Relative improvement (%) with 95% food-group interval",title="Name-only metrics move differently")
fig.suptitle("V9: budget 8 to 20 epochs, same training trajectory",fontsize=14)
fig.supxlabel("Single seed; validation only. Intervals exclude seed and model-selection uncertainty.",fontsize=9)
for extension in ["png","svg"]:fig.savefig(out/f"v9_duration.{extension}",dpi=160)
plt.close(fig)
print(out)

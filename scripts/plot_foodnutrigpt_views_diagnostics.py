"""Plot completed R4 two-view fit, consistency, clipping and spread diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    histories=[];receipts=[]
    fields=[("validation_primary","Validation nutrition (142 axes)","Scaled-log MAE"),
        ("train_supervised_mae","Supervised training objective (187 axes)","MAE"),
        ("train_consistency_unweighted_coefficient","Representation view distance","Axis/source-weighted C, before coefficient"),
        ("train_loss","Total training objective","MAE + coefficient x C"),
        ("mean_preclip_gradient_norm","Mean gradient norm before clipping","Norm, batch mean"),
        ("gradient_clip_fraction","Batches exceeding clip norm1","Fraction"),
        ("mean_unit_batch_std","Within-batch unit representation spread","Mean feature population std"),
        ("mean_representation_norm","Representation amplitude","Mean L2 norm")]
    for path,weight in [(args.control,0.),(args.candidate,.1)]:
        m=json.loads((path/"run_manifest.json").read_text())
        if m["status"]!="complete" or m["test_opened"] or m["args"]["consistency_weight"]!=weight:
            raise ValueError("Completed registered test-closed arms required.")
        h=pd.read_csv(path/"history.csv")
        np.testing.assert_array_equal(h.epoch,np.arange(1,61))
        if not np.isfinite(h[[name for name,_,_ in fields]].to_numpy()).all():raise ValueError("Nonfinite history.")
        histories.append(h);receipts.append({"run":str(path),"manifest_sha256":sha(path/"run_manifest.json"),"history_sha256":sha(path/"history.csv")})
    for field in ["training_tasks","visible_cells_A","removed_visible_cells_B","view_mask_sha256","learning_rate"]:
        np.testing.assert_array_equal(histories[0][field],histories[1][field])
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout="constrained")
    for axis,(name,title,label) in zip(axes.ravel(),fields):
        for history,text,color in zip(histories,["Coefficient 0","Coefficient 0.1"],["#167d9a","#bd7736"]):
            axis.plot(history.epoch,history[name],color=color,label=text,lw=1.8)
        axis.set(title=title,xlabel="Epoch",ylabel=label)
        axis.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Same views and supervision: adding representation consistency",fontsize=15)
    fig.supxlabel("Single seed, train/validation only. Batch spread and clipping are descriptive diagnostics, not transfer or causal isolation of gradient scale.",fontsize=9)
    args.output_dir.mkdir(parents=True)
    for extension in ["png","svg"]:fig.savefig(args.output_dir/f"diagnostics.{extension}",dpi=160)
    plt.close(fig)
    (args.output_dir/"figure_manifest.json").write_text(json.dumps({"inputs":receipts,"script_sha256":sha(Path(__file__)),"complete_test_opened":False},indent=2)+"\n")
    print(args.output_dir)


if __name__=="__main__":main()

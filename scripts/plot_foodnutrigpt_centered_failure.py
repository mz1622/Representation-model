"""Show only the retained pre-failure trajectory; never present it as a completed model."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed-run",type=Path,default=Path("output/v9_r4/mlp60_views_centered_weight01"))
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    root=Path(__file__).resolve().parents[1]
    failed=root/args.failed_run
    manifest=json.loads((failed/"run_manifest.json").read_text())
    if manifest["status"]!="failed" or manifest["epoch_completed"] not in {16,17,29}:raise ValueError("Expected retained centred-view failure.")
    completed=manifest["epoch_completed"];failed_epoch=completed+1
    paths=[root/"output/v9_r4/mlp60_views_weight0",root/"output/v9_r4/mlp60_views_weight01",failed]
    labels=["No consistency","Original origin, weight0.1","Joint batch centre, weight0.1"]
    histories=[pd.read_csv(path/"history.csv").iloc[:completed] for path in paths]
    fields=[("validation_primary","Validation nutrition MAE"),("train_supervised_mae","Supervised training MAE (187 axes)"),
            ("train_consistency_unweighted_coefficient","Training C (different origin definitions)"),
            ("mean_representation_norm","Mean representation L2 norm"),
            ("gradient_clip_fraction","Fraction of batches clipped at norm1"),
            ("mean_unit_batch_std","Unit representation feature std")]
    for history in histories:
        np.testing.assert_array_equal(history.epoch,np.arange(1,completed+1))
        if not np.isfinite(history[[f for f,_ in fields]].to_numpy()).all():raise ValueError("Nonfinite saved history.")
    for field in ["training_tasks","visible_cells_A","removed_visible_cells_B","view_mask_sha256","learning_rate"]:
        for history in histories[1:]:np.testing.assert_array_equal(histories[0][field],history[field])
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained")
    for axis,(field,title) in zip(axes.ravel(),fields):
        for history,label,color in zip(histories,labels,["#167d9a","#bd7736","#ba3d50"]):
            axis.plot(history.epoch,history[field],label=label,color=color,lw=1.8)
        axis.set(title=title,xlabel=f"Completed epoch (1-{completed} only)");axis.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[1,0].plot(histories[-1].epoch,histories[-1].mean_centred_representation_norm,
                   color="#ba3d50",ls="--",label="Centred residual norm (candidate only)")
    axes[1,0].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Centred-view run failed during epoch{failed_epoch} validation",fontsize=16)
    reason="final values exceed float32 range" if "inverse_numerics" in manifest else "float32 inverse-transform intermediate overflow"
    if manifest["args"].get("output_query_policy")=="caller_or_schema_axes_v1":reason="requested output exceeds float32 range"
    fig.supxlabel(f"Retained partial history only; no completed three-task score or 60-epoch comparison. Failure: {reason}.",fontsize=9)
    args.output_dir.mkdir(parents=True)
    for extension in ["png","svg"]:fig.savefig(args.output_dir/f"failure.{extension}",dpi=160)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output_dir/"figure_manifest.json").write_text(json.dumps({"failed_manifest_sha256":sha(failed/"run_manifest.json"),
        "history_sha256":{str(path.relative_to(root)):sha(path/"history.csv") for path in paths},
        "completed_epochs_shown":completed,"failed_evaluation_epoch":failed_epoch,"complete_test_opened":False,
        "script_sha256":sha(Path(__file__))},indent=2)+"\n")
    print(args.output_dir)


if __name__=="__main__":main()

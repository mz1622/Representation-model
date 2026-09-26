"""Summarize completed R0 runs, source diagnostics and paired uncertainty."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
from importlib.metadata import distributions
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,write_json,digest
from foodcomp.research_statistics import paired_interval

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=ROOT/"output/v9_r0")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    axes=data.axes.loc[data.axes.loss_group.eq("nutrition")&data.axes.loss_eligible,"axis_index"].to_numpy()
    results=[];errors={};sources=[];runs=[]
    for folder in sorted(args.runs.iterdir()):
        manifest=folder/"run_manifest.json"
        if not manifest.exists():continue
        run=json.loads(manifest.read_text(encoding="utf-8"));runs.append({"directory":str(folder.relative_to(ROOT)),**run})
        if run["status"]!="complete":continue
        metrics=json.loads((folder/"metrics.json").read_text())
        modes=metrics if "completion" in metrics else {run["configuration"]["mode"]:metrics}
        for mode,score in modes.items():
            results.append({"run":folder.name,"mode":mode,"view":run.get("args",run.get("configuration",{}))["view"],
                "elapsed_seconds":run["elapsed_seconds"],**score["nutrition"],"metabolome_scaled_log_mae":score["food_metabolome"]["scaled_log_mae"],
                "all_187_log_mae":score["all"]["log_mae"]})
            prefix=mode+"_" if "completion" in metrics else ""
            errors[(folder.name,mode)]=pd.read_parquet(folder/f"{prefix}candidate_errors.parquet")
            pred=pd.read_parquet(folder/f"{prefix}predictions.parquet")
            view=run.get("args",run.get("configuration",{}))["view"]
            jobs=pd.read_parquet(data.root/f"{view}_validation_jobs.parquet")
            scored=jobs.merge(pred,on=["profile_index","axis_index"],validate="one_to_one").merge(data.profiles[["profile_index","source_key","exact_name_group_id"]],on="profile_index")
            cells=scored.groupby(["source_key","exact_name_group_id","axis_index"])[["target","prediction"]].median().reset_index()
            cells["error"]=np.abs(np.log1p(cells.prediction/data.scale[cells.axis_index])-np.log1p(cells.target/data.scale[cells.axis_index]))
            for source,g in cells[cells.axis_index.isin(axes)].groupby("source_key"):
                sources.append({"run":folder.name,"mode":mode,"source":source,"supported_axes":g.axis_index.nunique(),
                    "candidate_groups":g.exact_name_group_id.nunique(),"scaled_log_mae":g.groupby("axis_index").error.mean().mean(),
                    "warning":"Source-specific axis support differs; not a cross-source quality ranking."})
    frame=pd.DataFrame(results);frame.to_csv(args.output_dir/"results.csv",index=False)
    pd.DataFrame(sources).to_csv(args.output_dir/"source_metrics.csv",index=False)
    write_json(args.output_dir/"run_inventory.json",runs)
    baseline=("xgb300_quarantined_completion","completion")
    intervals={}
    for key,value in errors.items():
        row=frame[(frame.run==key[0])&frame["mode"].eq(key[1])].iloc[0]
        if key[1]!="completion" or row["view"]!="quarantined" or key==baseline:continue
        intervals[key[0]]={m:paired_interval(errors[baseline],value,axes,metric=m) for m in ["scaled_log_mae","log_mae"]}
    write_json(args.output_dir/"paired_intervals.json",intervals)
    # Inclusive/exclusive median share fitted medians for most axes; compare matched cells explicitly.
    inc=errors[("median_inclusive","completion")];exc=errors[("median_quarantined","completion")]
    common=inc.merge(exc,on=["exact_name_group_id","axis_index"],suffixes=("_inclusive","_quarantined"),validate="one_to_one")
    matched=common[common.axis_index.isin(axes)].groupby("axis_index")[["scaled_log_mae_inclusive","scaled_log_mae_quarantined"]].mean().mean().to_dict()
    write_json(args.output_dir/"sensitivity.json",{"matched_candidate_axis_mean_errors":matched,
        "interpretation":"Same scales and intersection of scored cells. Exclusion can change training medians and source support; not a causal estimate of corrected true labels."})
    write_json(args.output_dir/"reproducibility.json",{"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "code_hashes":{str(x.relative_to(ROOT)):digest(x) for pat in ["src/foodcomp/research*.py","scripts/*v9_r0*.py","tests/test_research*.py"] for x in ROOT.glob(pat)},
        "environment":sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name']}),
        "python":sys.version,"complete_test_opened":False})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Optional matplotlib missing; numeric curves are retained in each run's history.csv.")
    else:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for folder in sorted(args.runs.iterdir()):
            if (folder/"history.csv").exists():
                h=pd.read_csv(folder/"history.csv");ax.plot(h.epoch,h.validation_primary,marker=".",label=folder.name)
        ax.set(xlabel="Epoch",ylabel="Validation nutrition scaled-log macro MAE")
        ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.output_dir/"validation_curves.png",dpi=160);plt.close(fig)
    print(frame[["run","mode","scaled_log_mae","log_mae"]].to_string(index=False))

if __name__=="__main__":main()

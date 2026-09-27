"""Recompute fair three-seed completion evidence from local saved predictions."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions
from foodcomp.research_confirmation import SEEDS,confirm_completion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for role in ["candidate","rf","xgb"]:parser.add_argument("--"+role+"-dirs",type=Path,nargs=3,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    data_hash=digest(data.root/"manifest.json")
    panel_hash=digest(ROOT/"data/processed/foodnutrigpt_v9_r1_tasks_v1/manifest.json")
    fingerprint=None;runs={};receipts={}
    try:
        for role in ["candidate","rf","xgb"]:
            runs[role]={};receipts[role]=[];configuration=None;code_hashes=None
            for folder in getattr(args,role+"_dirs"):
                manifest_path=folder/"run_manifest.json"
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["status"]!="complete":raise ValueError(f"Incomplete run: {folder}")
                if manifest.get("test_opened",manifest.get("complete_test_opened",True)):
                    raise ValueError("Confirmation inputs must document a closed test.")
                if manifest.get("data_hash",manifest.get("data_manifest_sha256"))!=data_hash:
                    raise ValueError("Data view mismatch.")
                current=manifest.get("name_cache_hash",manifest.get("name_cache_manifest_sha256"))
                if current is None:raise ValueError("Text cache fingerprint missing.")
                if fingerprint is None:fingerprint=current
                elif current!=fingerprint:raise ValueError("Different name inputs across confirmation runs.")
                settings=dict(manifest.get("args",manifest.get("configuration",{})))
                seed=int(settings.pop("seed"));settings.pop("output_dir",None)
                if seed not in SEEDS or seed in runs[role]:raise ValueError("Missing, unregistered or duplicate seed.")
                if configuration is None:configuration=settings
                elif settings!=configuration:raise ValueError("Hyperparameters changed across confirmation seeds.")
                current_code=manifest["code_hashes"]
                if code_hashes is None:code_hashes=current_code
                elif current_code!=code_hashes:raise ValueError("Implementation changed across seeds; rerun fixed-code confirmation.")
                if role=="candidate":
                    if manifest.get("panel_hash")!=panel_hash:raise ValueError("Candidate training panel mismatch.")
                    prediction=folder/"completion_predictions.parquet"
                else:
                    if settings.get("method")!=role or settings.get("mode")!="completion":raise ValueError("Incorrect tree method/task.")
                    if manifest.get("training_row_cap","missing") is not None:raise ValueError("Tree baseline uses a row cap or lacks its declaration.")
                    if manifest.get("jobs_sha256")!=data.manifest["artifact_hashes"][f"{data.view}_validation_jobs.parquet"]:
                        raise ValueError("Tree validation task mismatch.")
                    prediction=folder/"predictions.parquet"
                scores,axes,groups=score_predictions(data,pd.read_parquet(prediction))
                runs[role][seed]=groups
                axes.to_csv(args.output_dir/f"{role}_{seed}_axis_metrics.csv",index=False)
                receipts[role].append({"seed":seed,"directory":str(folder),"manifest_sha256":digest(manifest_path),
                    "prediction_sha256":digest(prediction),"code_commit":manifest["code_commit"],"scores":scores})
        axes=data.axes[data.axes.loss_group.eq("nutrition")&data.axes.loss_eligible].axis_index.to_numpy()
        if len(axes)!=142:raise ValueError("Primary axis protocol changed.")
        result,axis_intervals=confirm_completion(runs,axes)
        result.update(status="complete",receipts=receipts,data_hash=data_hash,panel_hash=panel_hash,name_cache_hash=fingerprint)
        axis_intervals.merge(data.axes[["axis_index","canonical_name"]],on="axis_index").to_csv(args.output_dir/"axis_paired_intervals.csv",index=False)
        write_json(args.output_dir/"confirmation.json",result)
        lines=["# Three-seed completion confirmation","",f"Stronger tree: {result['stronger_tree']}. Conditional milestone passed: {result['conditional_completion_milestone_passed']}.","",
            "| Method | Primary mean ± seed SD | Legacy log-MAE mean ± seed SD |","|---|---:|---:|"]
        for role,summary in result["seed_summaries"].items():
            m=summary["metrics"];p=m["scaled_log_mae"];l=m["log_mae"]
            lines.append(f"| {role} | {p['mean']:.8f} ± {p['sample_std']:.8f} | {l['mean']:.8f} ± {l['sample_std']:.8f} |")
        lines.extend(["",result["aggregation"],"",result["uncertainty"],"",result["claim_scope"],"","See confirmation.json for individual seeds, paired intervals, all gates, fingerprints and full nutrition/metabolome metrics."])
        (args.output_dir/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        print({"stronger_tree":result["stronger_tree"],"gates":result["gates"]})
    except Exception as error:
        write_json(args.output_dir/"confirmation.json",{"status":"incomplete_or_invalid","error_type":type(error).__name__,"error":str(error),
            "conditional_completion_milestone_passed":False,"complete_test_opened":False})
        raise


if __name__=="__main__":main()

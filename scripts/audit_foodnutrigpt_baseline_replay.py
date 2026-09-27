"""Check a fixed-seed tree replay against its saved exploratory run."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json,score_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    result={"status":"incomplete","complete_test_opened":False,"code_sha256":digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/"data/processed"/VERSION)
        settings=None;identity=None;frames={};counts={};scores={};receipts={}
        for role,path in [("original",args.original),("replay",args.replay)]:
            m=json.loads((path/"run_manifest.json").read_text())
            if m["status"]!="complete" or m.get("complete_test_opened",True):raise ValueError("Require completed, test-closed runs.")
            if m["data_manifest_sha256"]!=digest(data.root/"manifest.json"):raise ValueError("Changed data version.")
            if m.get("training_row_cap","missing") is not None:raise ValueError("Full training rows required.")
            config=dict(m["configuration"]);config.pop("output_dir")
            if config["method"] not in {"rf","xgb"} or config["mode"]!="completion":raise ValueError("Only completion tree replay is supported.")
            # Verified historical script hardcoded squared error; no other default is inferred.
            if "xgb_objective" not in config:
                if m["code_hashes"]["scripts/run_foodnutrigpt_v9_r0_baseline.py"]!="5799580cd54e98c661c44e51c3fea80db14041de8e1dadf554b7f4a07efb9896":
                    raise ValueError("Unknown legacy objective implementation.")
                config["xgb_objective"]="reg:squarederror"
            if settings is None:settings=config
            elif settings!=config:raise ValueError("Fixed-seed replay configuration changed.")
            current={key:m[key] for key in ["data_manifest_sha256","protocol_sha256","jobs_sha256","name_cache_manifest_sha256"]}
            current.update({key:m["code_hashes"][key] for key in ["src/foodcomp/research_r0.py","src/foodcomp/research_text.py"]})
            if identity is None:identity=current
            elif identity!=current:raise ValueError("Replay labels, task, text, or shared implementation changed.")
            frame=pd.read_parquet(path/"predictions.parquet").sort_values(["profile_index","axis_index"]).reset_index(drop=True)
            score,_,_=score_predictions(data,frame);scores[role]=score;frames[role]=frame
            cost=pd.read_csv(path/"fitting_cost.csv").sort_values("axis_index").reset_index(drop=True)
            counts[role]=cost[["axis_index","train_profiles","validation_jobs"]]
            if not np.array_equal(cost.axis_index,data.targets):raise ValueError("Training-axis coverage changed.")
            receipts[role]={"directory":str(path),"manifest_sha256":digest(path/"run_manifest.json"),
                "prediction_sha256":digest(path/"predictions.parquet"),"code_commit":m["code_commit"],
                "baseline_script_sha256":m["code_hashes"]["scripts/run_foodnutrigpt_v9_r0_baseline.py"]}
        keys=["profile_index","axis_index"]
        pd.testing.assert_frame_equal(frames["original"][keys],frames["replay"][keys])
        pd.testing.assert_frame_equal(counts["original"],counts["replay"])
        a=frames["original"].prediction.to_numpy();b=frames["replay"].prediction.to_numpy()
        delta=np.abs(a-b)
        result.update(configuration=settings,identical_shared_fingerprints=identity,receipts=receipts,scores=scores,
            prediction_count=len(a),bitwise_equal_predictions=bool(np.array_equal(a,b)),
            changed_prediction_count=int(np.count_nonzero(a!=b)),max_absolute_prediction_difference=float(delta.max()),
            training_counts_identical=True)
        if not result["bitwise_equal_predictions"]:raise ValueError("Replay predictions differ; investigate before attributing to seed variation.")
        if scores["original"]!=scores["replay"]:raise AssertionError("Equal predictions received different scores.")
        result.update(status="complete",interpretation="Fixed-seed reproducibility only; not three-seed stability, model superiority, or provenance confirmation.")
    except Exception as error:
        result.update(status="failed",exception_type=type(error).__name__,message=str(error))
        write_json(args.output_dir/"audit.json",result)
        raise
    write_json(args.output_dir/"audit.json",result)
    print({k:result[k] for k in ["status","prediction_count","changed_prediction_count","max_absolute_prediction_difference"]})


if __name__=="__main__":main()

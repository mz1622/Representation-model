"""Rescore all fixed strong tree seeds and replay fixed name/retrieval references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,digest,score_predictions,write_json
from foodcomp.research_neural import OUTPUT_QUERY_POLICY
from audit_foodnutrigpt_stable_inverse import assert_saved_fields_equal


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);started=time.monotonic()
    data=ResearchData(ROOT/"data/processed"/VERSION)
    result={"status":"incomplete","output_query_policy":OUTPUT_QUERY_POLICY,"complete_test_opened":False,
        "data_sha256":digest(data.root/"manifest.json"),"score_replays":[],"retrieval_replays":[],
        "code_sha256":{str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/"src/foodcomp/research_neural.py",ROOT/"src/foodcomp/research_inference.py",ROOT/"src/foodcomp/research_selection.py",ROOT/"scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py",ROOT/"scripts/train_foodnutrigpt_v9_r4_views.py"]}}
    try:
        paths=[ROOT/"output/v9_baseline_confirmation_v1"/f"{method}_seed{seed}" for method in ["rf","xgb"] for seed in [20260922,20260923,20260924]]
        paths.append(ROOT/"output/v9_r0/name_knn_quarantined")
        for run in paths:
            manifest=json.loads((run/"run_manifest.json").read_text())
            if manifest["status"]!="complete" or manifest["complete_test_opened"] or manifest["data_manifest_sha256"]!=result["data_sha256"]:
                raise ValueError("Invalid fixed reference manifest.")
            prediction=pd.read_parquet(run/"predictions.parquet")
            score,_,_=score_predictions(data,prediction)
            assert_saved_fields_equal(score,json.loads((run/"metrics.json").read_text()))
            result["score_replays"].append({"run":str(run.relative_to(ROOT)),"rows":len(prediction),
                "all_saved_metrics_exact":True,"predictions_sha256":digest(run/"predictions.parquet"),
                "manifest_sha256":digest(run/"run_manifest.json"),"model_refitted":False})
        retrieval=[("w0","output/v9_r4/mlp60_views_weight0","output/v9_r4/retrieval_mlp60_views_weight0"),
                   ("raw01","output/v9_r4/mlp60_views_weight01","output/v9_r4/retrieval_mlp60_views_weight01"),
                   ("name_mlp","output/v9_r0/name_mlp8_quarantined","output/v9_r0/retrieval_name_mlp_v1")]
        for tag,model_run,old_path in retrieval:
            checkpoint=ROOT/model_run/"best_model.pt";reference=ROOT/old_path
            old=json.loads((reference/"metrics.json").read_text())
            if digest(checkpoint)!=old["checkpoint_sha256"]:raise ValueError("Fixed retrieval reference checkpoint mismatch.")
            destination=args.output_dir/f"retrieval_{tag}"
            subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py"),"--checkpoint",str(checkpoint),"--output-dir",str(destination)],cwd=ROOT,check=True)
            actual=pd.read_parquet(destination/"ranks.parquet");before=pd.read_parquet(reference/"ranks.parquet")
            pd.testing.assert_frame_equal(actual,before,check_exact=True)
            current=json.loads((destination/"metrics.json").read_text())
            for key in ["metrics","candidate_count","query_profiles","candidate_sha256","checkpoint_sha256","data_sha256"]:
                if current[key]!=old[key]:raise AssertionError(f"Fixed retrieval result changed: {tag}/{key}")
            if current["output_query_policy"]!=OUTPUT_QUERY_POLICY:raise ValueError("Missing query policy receipt.")
            result["retrieval_replays"].append({"tag":tag,"checkpoint_sha256":digest(checkpoint),
                "rank_rows":len(actual),"all_rank_columns_exact":True,"metrics_exact":True,
                "reference_ranks_sha256":digest(reference/"ranks.parquet"),
                "current_ranks_sha256":digest(destination/"ranks.parquet"),
                "candidate_sha256":current["candidate_sha256"],"requested_axes":len(current["requested_candidate_nutrition_axes"])})
        result.update(status="complete",elapsed_seconds=time.monotonic()-started,
            scope="All six saved strong-tree seeds plus name-kNN rescored, no refitting or new tree predictions on unlabelled rows. Three neural retrieval references regenerated on fixed candidates/queries with142 caller-specified axes and identical ranks. Scalar tree requests do not compute unrelated output heads; scored cells, inputs, labels and weights unchanged. Additional unlabelled family guard is retained for neural evaluation.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/"verification.json",result);raise
    write_json(args.output_dir/"verification.json",result)
    print({"status":result["status"],"scored_references":len(result["score_replays"]),"retrieval_references":len(result["retrieval_replays"])})


if __name__=="__main__":main()

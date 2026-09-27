"""Create a current aggregate ledger from registered runs; never exports individual nutrition values."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version",choices=["r1","r2","r3","baseline_confirmation_v1"],required=True)
    args=p.parse_args()
    folder=ROOT/"experiments/foodnutrigpt_v9_research"/args.version
    config=json.loads((folder/"config.json").read_text())
    registered=config.get("registered_candidates",config.get("neural_runs",[])+config.get("tree_runs",[]))
    runs=[]
    for item in registered:
        path=ROOT/item.get("output_dir",f"output/v9_{args.version}/{item['name']}")
        record={"registered_configuration":item,"local_output":str(path.relative_to(ROOT))}
        manifest=path/"run_manifest.json"
        if not manifest.exists():
            record["manifest_status"]="not started; check active process queues separately"
        else:
            m=json.loads(manifest.read_text())
            record.update(manifest_status=m["status"],manifest_sha256=digest(manifest),code_commit=m["code_commit"],
                          elapsed_seconds=m.get("elapsed_seconds"),epoch_completed=m.get("epoch_completed"),
                          best_epoch=m.get("best_epoch"),checkpoint_sha256=m.get("checkpoint_hash"))
            if m["status"]=="complete":
                record["metrics"]=json.loads((path/"metrics.json").read_text())
            elif m["status"]=="failed":
                record["failure"]={k:m.get(k) for k in ["error_type","error"]}
        budgets=[]
        for budget in item.get("duration_checkpoints",[]):
            evaluation=path/f"evaluation_through_{budget:03d}"
            if (evaluation/"evaluation_manifest.json").exists():
                receipt=json.loads((evaluation/"evaluation_manifest.json").read_text())
                if receipt["selection_budget_epochs"]!=budget:raise ValueError("Mismatched duration-budget evaluation.")
                budgets.append({"budget_epochs":budget,"evaluation_manifest":receipt,
                    "metrics":json.loads((evaluation/"metrics.json").read_text())})
        if budgets:record["completed_budget_evaluations"]=budgets
        alternate=path/"evaluation_hurdle"
        if (alternate/"evaluation_manifest.json").exists():
            record["hurdle_selected_evaluation"]={"manifest":json.loads((alternate/"evaluation_manifest.json").read_text()),
                "metrics":json.loads((alternate/"metrics.json").read_text())}
        runs.append(record)
    retrieval=[]
    for path in sorted((ROOT/f"output/v9_{args.version}").glob("retrieval_*/metrics.json")):
        retrieval.append({"local_output":str(path.parent.relative_to(ROOT)),"metrics":json.loads(path.read_text()),"metrics_sha256":digest(path)})
    write_json(folder/"results_summary.json",{
        "version":config["version"],"generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"exploration in progress; no accepted improvement over stronger tree baseline",
        "configuration_sha256":digest(folder/"config.json"),"runs":runs,"completed_retrieval_evaluations":retrieval,
        "process_liveness":"Manifest status only. Check actual process/session before scheduling or declaring a live job.",
        "confirmation_seeds_completed":False,"complete_test_opened":False,
        "provenance_unresolved":True,"individual_predictions_included":False,
        "interpretation":"Aggregate ledger, not a success certificate. Historical baseline and retrieval comparisons are in README and ignored local result files."
    })
    print([(r["registered_configuration"]["name"],r["manifest_status"]) for r in runs])


if __name__=="__main__":main()

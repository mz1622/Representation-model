"""Evaluate a selected checkpoint without refitting, on the original shared panel."""
import argparse
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate
from foodcomp.research_r0 import score_predictions,write_json,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--selection-budget-epochs",type=int,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(4)
    model=NutritionModel(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if saved["best_epoch"]>args.selection_budget_epochs:raise ValueError("Checkpoint exceeds claimed selection budget.")
    results={}
    for mode in ["completion","name_only"]:
        pred=evaluate(model.model,model.data,model._cached_text,model.device,mode)
        score,axes,candidates=score_predictions(model.data,pred);results[mode]=score
        pred.to_parquet(args.output_dir/f"{mode}_predictions.parquet",index=False)
        axes.to_csv(args.output_dir/f"{mode}_axis_metrics.csv",index=False)
        candidates.to_parquet(args.output_dir/f"{mode}_candidate_errors.parquet",index=False)
    write_json(args.output_dir/"metrics.json",results)
    write_json(args.output_dir/"evaluation_manifest.json",{"checkpoint":str(args.checkpoint),"checkpoint_hash":digest(args.checkpoint),
        "selection_budget_epochs":args.selection_budget_epochs,"best_epoch":saved["best_epoch"],
        "data_hash":saved["data_hash"],"name_cache_hash":saved["name_cache_hash"],"seed":saved["seed"],
        "elapsed_seconds":time.monotonic()-start,"complete_test_opened":False,
        "output_query_policy":model.output_query_policy})
    print({mode:score["nutrition"]["scaled_log_mae"] for mode,score in results.items()})

if __name__=="__main__":main()

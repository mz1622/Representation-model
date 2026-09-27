"""Check the dual selector against original inference on a real saved checkpoint."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate
from foodcomp.research_selection import evaluate_dual_selection
from foodcomp.research_r0 import digest,write_json,score_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(4)
    wrapped=NutritionModel(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if wrapped.device.type=="cuda" else []
    pred,loss=evaluate_dual_selection(wrapped.model,wrapped.data,wrapped._cached_text,wrapped.device,
                                    amount_weight=saved["config"]["amount_loss_weight"])
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for before,after in zip(gpu_rng,torch.cuda.get_rng_state_all() if gpu_rng else []):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    original=evaluate(wrapped.model,wrapped.data,wrapped._cached_text,wrapped.device)
    keys=["profile_index","axis_index"]
    a=pred.sort_values(keys).reset_index(drop=True);b=original.sort_values(keys).reset_index(drop=True)
    np.testing.assert_array_equal(a[keys],b[keys])
    differences={}
    for column in ["prediction","positive_probability"]:
        np.testing.assert_array_equal(a[column],b[column])
        differences[column]=float(np.abs(a[column]-b[column]).max())
    metric,_,_=score_predictions(wrapped.data,pred)
    receipt={"checkpoint":str(args.checkpoint),"checkpoint_hash":digest(args.checkpoint),
        "data_hash":saved["data_hash"],"name_cache_hash":saved["name_cache_hash"],
        "validation_jobs":len(pred),"max_absolute_prediction_difference":differences,
        "cpu_and_cuda_rng_unchanged":True,"selection_loss":loss,"primary":metric["nutrition"]["scaled_log_mae"],
        "elapsed_seconds":time.monotonic()-start,"complete_test_opened":False}
    write_json(args.output_dir/"verification.json",receipt);print(receipt)


if __name__=="__main__":main()

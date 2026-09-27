"""Compare saved zero-consistency artifacts with the completed original MLP control."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,default=ROOT/"output/v9_r2/mlp60_mae_width512")
    p.add_argument("--control",type=Path,default=ROOT/"output/v9_r4/mlp60_views_weight0")
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    result={"status":"incomplete","complete_test_opened":False,"script_sha256":digest(Path(__file__)),
        "parent":str(args.parent),"control":str(args.control),"artifacts":[]}
    try:
        manifests=[json.loads((folder/"run_manifest.json").read_text()) for folder in [args.parent,args.control]]
        if any(m["status"]!="complete" or m["test_opened"] for m in manifests):raise ValueError("Completed test-closed runs required.")
        if manifests[1]["args"]["consistency_weight"]!=0:raise ValueError("Zero-weight control required.")
        for field in ["data_hash","panel_hash","name_cache_hash","seed","best_epoch"]:
            if manifests[0][field]!=manifests[1][field]:raise AssertionError(f"Different {field}.")
        for field in ["kind","objective","mlp_width","epochs","schedule_epochs","batch_size","learning_rate"]:
            if manifests[0]["args"][field]!=manifests[1]["args"][field]:raise AssertionError(f"Different configuration: {field}.")
        histories=[pd.read_csv(folder/"history.csv") for folder in [args.parent,args.control]]
        fields=["epoch","train_loss","validation_primary","validation_legacy_log_mae","learning_rate","training_tasks"]
        for field in fields:np.testing.assert_array_equal(histories[0][field].to_numpy(),histories[1][field].to_numpy())
        result["history_epochs_exact"]=len(histories[0]);result["history_fields"]=fields
        for name in ["best_model.pt","best_through_epoch_008.pt","best_through_epoch_020.pt","best_through_epoch_060.pt","latest_training_state.pt"]:
            paths=[folder/name for folder in [args.parent,args.control]]
            states=[torch.load(path,map_location="cpu",weights_only=True)["model_state"] for path in paths]
            if states[0].keys()!=states[1].keys():raise AssertionError("Checkpoint parameter keys differ.")
            for key in states[0]:
                a,b=states[0][key],states[1][key]
                if a.dtype!=b.dtype or a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise AssertionError(f"Invalid checkpoint tensor: {key}")
                if a.contiguous().numpy().tobytes()!=b.contiguous().numpy().tobytes():raise AssertionError(f"Checkpoint tensor bytes differ: {name}/{key}")
            result["artifacts"].append({"artifact":name,"model_tensor_bytes_equal":True,
                "parent_file_sha256":digest(paths[0]),"control_file_sha256":digest(paths[1]),
                "note":"Whole checkpoint file hashes may differ because args/protocol metadata differ."})
        for mode in ["completion","name_only"]:
            paths=[folder/f"{mode}_predictions.parquet" for folder in [args.parent,args.control]]
            a,b=[pd.read_parquet(path) for path in paths]
            if not np.isfinite(a.prediction).all() or not np.isfinite(b.prediction).all():raise FloatingPointError("Nonfinite saved prediction.")
            pd.testing.assert_frame_equal(a,b,check_exact=True)
            if a.prediction.to_numpy().tobytes()!=b.prediction.to_numpy().tobytes():raise AssertionError("Prediction bytes differ.")
            result["artifacts"].append({"artifact":f"{mode}_predictions.parquet","rows":len(a),
                "prediction_bytes_and_all_columns_equal":True,"parent_file_sha256":digest(paths[0]),"control_file_sha256":digest(paths[1])})
        result.update(status="complete",data_sha256=manifests[0]["data_hash"],
            scope="Exact saved60-epoch loss/validation/LR/task histories, five saved model states and both full prediction tables. Does not prove equality of unsaved intermediate parameter states or other seeds/devices; metadata and elapsed times intentionally differ.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(args.output_dir/"verification.json",result);raise
    write_json(args.output_dir/"verification.json",result)
    print({k:result[k] for k in ["status","history_epochs_exact"]})


if __name__=="__main__":main()

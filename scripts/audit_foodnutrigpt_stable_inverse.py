"""Actual prediction/backward-compatibility and overflow-repair audit; no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate, predictions_from_outputs
from foodcomp.research_r0 import digest, score_predictions, write_json


def assert_saved_fields_equal(actual, saved):
    for key, value in saved.items():
        if isinstance(value,dict):assert_saved_fields_equal(actual[key],value)
        elif actual[key]!=value:raise AssertionError(f"Saved metric changed: {key}: {actual[key]} vs {value}")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    result={"status":"incomplete","complete_test_opened":False,"neural_controls":[],"tree_rescoring":[],
        "code_sha256":{file:digest(ROOT/file) for file in ["src/foodcomp/research_neural.py","src/foodcomp/research_views.py","src/foodcomp/research_inference.py","src/foodcomp/research_r1.py","src/foodcomp/research_r0.py","scripts/train_foodnutrigpt_v9_r4_views.py",str(Path(__file__).relative_to(ROOT))]}}
    try:
        for name in ["mlp60_views_weight0","mlp60_views_weight01"]:
            run=ROOT/"output/v9_r4"/name;wrapper=NutritionModel(run/"best_model.pt")
            saved_metrics=json.loads((run/"metrics.json").read_text())
            record={"run":name,"checkpoint_sha256":digest(run/"best_model.pt"),"tasks":{}}
            for mode in ["completion","name_only"]:
                new=evaluate(wrapper.model,wrapper.data,wrapper._cached_text,wrapper.device,mode)
                old=pd.read_parquet(run/f"{mode}_predictions.parquet")
                pd.testing.assert_frame_equal(new,old,check_exact=True)
                if new.prediction.to_numpy().tobytes()!=old.prediction.to_numpy().tobytes():raise AssertionError("Prediction bytes changed.")
                score,_,_=score_predictions(wrapper.data,new)
                assert_saved_fields_equal(score,saved_metrics[mode])
                record["tasks"][mode]={"rows":len(new),"all_columns_exact":True,"prediction_bytes_exact":True,
                    "saved_metrics_exact":True,"reference_prediction_sha256":digest(run/f"{mode}_predictions.parquet")}
            result["neural_controls"].append(record)
        for name in ["xgb800d10","rf400_unlimited_leaf1"]:
            run=ROOT/"output/v9_r1"/name
            score,_,_=score_predictions(wrapper.data,pd.read_parquet(run/"predictions.parquet"))
            assert_saved_fields_equal(score,json.loads((run/"metrics.json").read_text()))
            result["tree_rescoring"].append({"run":name,"saved_metrics_exact":True,"prediction_sha256":digest(run/"predictions.parquet"),"uses_changed_neural_decoder":False})
        failure=ROOT/"data/local/research_diagnostics/r4_centered_failure_v1"
        diagnostic=json.loads((failure/"summary.json").read_text())
        if diagnostic["status"]!="complete_failure_reproduced":raise ValueError("Expected completed failure diagnosis.")
        raw_model=torch.load(failure/"replayed_epoch17_diagnostic_not_candidate.pt",map_location="cpu",weights_only=True)
        wrapper.model.load_state_dict(raw_model["model_state"])
        prediction=evaluate(wrapper.model,wrapper.data,wrapper._cached_text,wrapper.device)
        if not np.isfinite(prediction.prediction).all() or len(prediction)!=323809:raise AssertionError("Repaired full evaluation did not produce finite expected query panel.")
        private=pd.read_parquet(failure/"overflow_cells_private.parquet")
        values=torch.tensor(private.amount_normalized.to_numpy(),device=wrapper.device,dtype=torch.float32)
        scales=torch.tensor(private.scale.to_numpy(),device=wrapper.device,dtype=torch.float32)
        if torch.isfinite(torch.expm1(values)*scales).any():raise AssertionError("Stored overflow did not replay.")
        repaired,_=predictions_from_outputs({"amount_normalized":values},scales)
        expected=(torch.expm1(values.double())*scales.double()).float()
        if not torch.equal(repaired,expected) or not torch.isfinite(repaired).all():raise AssertionError("Recoverable overflow values differ from float64 intermediate reference.")
        result.update(status="complete",repaired_overflow_cells=len(private),
            replayed_failed_epoch_full_query_panel_finite=True,diagnostic_epoch_promoted_to_candidate=False,
            no_values_capped_or_skipped=True,invalid_or_truly_out_of_range_outputs_still_fail="Covered by numerical unit tests",
            failure_diagnostic_sha256=digest(failure/"summary.json"),
            scope="Two selected neural controls regenerated and rescored, strongest same-seed trees rescored from fixed predictions. Original failed epoch remains diagnostic only. No test access, training, metric or selection change.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(args.output_dir/"verification.json",result);raise
    write_json(args.output_dir/"verification.json",result)
    print({k:result[k] for k in ["status","repaired_overflow_cells","replayed_failed_epoch_full_query_panel_finite"]})


if __name__=="__main__":main()

"""R0 fixed-prediction diagnostics; no model selection or test access."""
import argparse
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,score_predictions,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    raw=pd.read_parquet(data.root/"canonical_cells.parquet")
    nutrition=data.axes.loc[data.axes.loss_group.eq("nutrition")&data.axes.loss_eligible,"axis_index"]
    support=data.jobs.merge(data.profiles[["profile_index","exact_name_group_id","source_key"]],on="profile_index")
    support["positive"]=support.target>0;support["explicit_zero"]=support.target==0
    support.groupby("axis_index").agg(profile_targets=("target","size"),candidate_support=("exact_name_group_id","nunique"),
        sources=("source_key","nunique"),positive_profiles=("positive","sum"),explicit_zero_profiles=("explicit_zero","sum")).to_csv(args.output_dir/"support.csv")
    # Hold predictions and panel fixed, change only the within-profile duplicate label rule.
    counterfactual=copy.copy(data)
    counterfactual.jobs=data.jobs.drop(columns="target").merge(raw[["profile_index","axis_index","old_rf_log_median"]],on=["profile_index","axis_index"],validate="one_to_one")
    counterfactual.jobs["target"]=np.expm1(counterfactual.jobs.pop("old_rf_log_median"))
    scoring={}
    for run,prefix in [("xgb300_quarantined_completion",""),("v9_8_quarantined","completion_")]:
        pred=pd.read_parquet(ROOT/f"output/v9_r0/{run}/{prefix}predictions.parquet")
        score,_,_=score_predictions(data,pred);old,_,_=score_predictions(counterfactual,pred)
        scoring[run]={"raw_median_truth":score["nutrition"],"inverse_log_median_truth":old["nutrition"]}
        if "positive_probability" in pred:
            if pred.positive_probability.le(1e-12).any():raise ValueError("Cannot recover conditional amount from saturated zero probabilities.")
            no_hurdle=pred.copy();no_hurdle.prediction/=no_hurdle.positive_probability
            changed,_,_=score_predictions(data,no_hurdle)
            scoring[run]["conditional_amount_without_probability_multiplication"]=changed["nutrition"]
            joined=data.jobs.merge(pred,on=["profile_index","axis_index"],validate="one_to_one")
            joined["underprediction_positive"]=np.where(joined.target>0,joined.prediction<joined.target,np.nan)
            joined.groupby("axis_index").agg(positive_underprediction_fraction=("underprediction_positive","mean"),
                mean_positive_probability=("positive_probability","mean")).to_csv(args.output_dir/"hurdle_diagnostics.csv")
    train=raw[raw.partition.eq("train")]
    write_json(args.output_dir/"diagnostics.json",{"scoring_rule":scoring,
        "nutrition_train_changed_aggregation_cells":int((train.axis_index.isin(nutrition)&train.aggregation_gap.gt(1e-10)).sum()),
        "nutrition_validation_changed_aggregation_cells":int((raw.partition.eq("validation")&raw.axis_index.isin(nutrition)&raw.aggregation_gap.gt(1e-10)).sum()),
        "interpretation":"Scoring sensitivity holds model predictions fixed, not retraining labels. Removing presence multiplication is a post-hoc diagnostic, not an accepted model or a comparison of training objectives.",
        "complete_test_opened":False})
    # Original release preserves units/basis/status but omits the numeric source field needed to assign causality.
    cols=["profile_id","axis_index","raw_unit","raw_basis","source_value_origin","value_status","normalized_value_g_per_100g"]
    profiles=data.profiles[["profile_id","partition","source_key"]]
    pieces=[]
    frozen=ROOT/"data/processed/global_foodnutrigpt_v8_single_stage_v2_complete_test/source_native_axis_tokens.csv.gz"
    for chunk in pd.read_csv(frozen,usecols=cols,chunksize=250000,low_memory=False):
        chunk=chunk.merge(profiles,on="profile_id",how="inner",validate="many_to_one")
        chunk["zero"]=chunk.normalized_value_g_per_100g.eq(0)
        chunk["above_100"]=chunk.normalized_value_g_per_100g.gt(100)
        pieces.append(chunk.groupby(["partition","source_key","raw_unit","raw_basis","source_value_origin","value_status"],dropna=False).agg(
            observations=("axis_index","size"),zeros=("zero","sum"),above_100=("above_100","sum"),maximum=("normalized_value_g_per_100g","max")).reset_index())
    units=pd.concat(pieces)
    keys=["partition","source_key","raw_unit","raw_basis","source_value_origin","value_status"]
    units.groupby(keys,dropna=False).agg(observations=("observations","sum"),zeros=("zeros","sum"),above_100=("above_100","sum"),maximum=("maximum","max")).to_csv(args.output_dir/"unit_basis_audit.csv")

if __name__=="__main__":main()

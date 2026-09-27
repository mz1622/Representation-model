"""Decompose a neural/tree validation gap without changing the scored panel or fitting models."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,score_predictions,write_json,digest
from foodcomp.research_statistics import paired_interval,paired_axis_intervals

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neural-dir",type=Path,required=True);p.add_argument("--tree-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    neural=pd.read_parquet(args.neural_dir/"completion_predictions.parquet")
    tree=pd.read_parquet(args.tree_dir/"predictions.parquet")
    nn_score,nn_axes,nn_groups=score_predictions(data,neural)
    tree_score,tree_axes,tree_groups=score_predictions(data,tree)
    axes=data.axes.copy()
    axes["training_candidates"]=[data.profiles.iloc[data.train[data.observed[data.train,a]]].exact_name_group_id.nunique() for a in axes.axis_index]
    axes["support_bin"]=pd.cut(axes.training_candidates,bins=[-1,99,999,np.inf],labels=["below_100","100_to_999","at_least_1000"])
    axes["scale_bin"]=np.where(axes.research_scale<.001,"below_1mg_typical","at_least_1mg_typical")
    a=nn_axes[["axis_index","scaled_log_mae","log_mae","positive_scaled_log_mae","zero_scaled_log_mae"]].merge(
        tree_axes[["axis_index","scaled_log_mae","log_mae","positive_scaled_log_mae","zero_scaled_log_mae"]],on="axis_index",suffixes=("_neural","_tree"),validate="one_to_one")
    a=a.merge(axes,on="axis_index",validate="one_to_one")
    a["primary_gap"]=a.scaled_log_mae_neural-a.scaled_log_mae_tree
    a.to_csv(args.output_dir/"axis_comparison.csv",index=False)
    nutrition=a[a.loss_group.eq("nutrition")]
    if len(nutrition)!=142:raise ValueError("Primary-axis coverage changed.")
    decomposition=[]
    for by in ["mask_family","support_bin","scale_bin"]:
        for label,g in nutrition.groupby(by,observed=True):
            decomposition.append({"stratification":by,"stratum":str(label),"axes":len(g),
                "neural_primary":g.scaled_log_mae_neural.mean(),"tree_primary":g.scaled_log_mae_tree.mean(),
                "macro_gap_contribution":g.primary_gap.sum()/142,
                "neural_better_axes":int(g.primary_gap.lt(0).sum())})
    pd.DataFrame(decomposition).to_csv(args.output_dir/"gap_decomposition.csv",index=False)
    intervals=paired_axis_intervals(tree_groups,nn_groups).merge(axes[["axis_index","canonical_name","loss_group"]],on="axis_index")
    intervals.to_csv(args.output_dir/"axis_paired_intervals.csv",index=False)
    selected=nutrition.axis_index.to_numpy()
    pair={m:paired_interval(tree_groups,nn_groups,selected,metric=m) for m in ["scaled_log_mae","log_mae"]}
    # Exact same source-cell aggregation as the main evaluator; diagnostic bins do not relabel observations.
    key=["profile_index","axis_index"]
    joined=data.jobs.merge(neural[key+["prediction"]].rename(columns={"prediction":"neural"}),on=key,validate="one_to_one")
    joined=joined.merge(tree[key+["prediction"]].rename(columns={"prediction":"tree"}),on=key,validate="one_to_one")
    pinfo=data.profiles[["profile_index","profile_id","source_key","exact_name_group_id","original_name"]]
    joined=joined.merge(pinfo,on="profile_index",validate="many_to_one")
    visible=np.zeros(len(joined),int)
    for family,index in joined.groupby("mask_family").groups.items():
        rows=joined.loc[index,"profile_index"].to_numpy()
        visible[index]=data.observed[rows][:,data.families!=family].sum(1)
    joined["visible_axes"]=visible
    near=pd.read_csv(ROOT/"reports/v9_r0_sources_v1/near_name_candidates.csv")
    joined["near_name_flag"]=joined.original_name.isin(set(near.validation_name))
    cells=joined.groupby(["source_key","exact_name_group_id","axis_index"],as_index=False).agg(
        target=("target","median"),neural=("neural","median"),tree=("tree","median"),
        visible_axes=("visible_axes","median"),near_name_flag=("near_name_flag","max"))
    cells=cells[cells.axis_index.isin(selected)].copy()
    scale=data.scale[cells.axis_index.to_numpy()]
    for name in ["neural","tree"]:cells[name+"_error"]=np.abs(np.log1p(cells[name]/scale)-np.log1p(cells.target/scale))
    cells["context_bin"]=pd.cut(cells.visible_axes,bins=[-1,0,4,14,39,np.inf],labels=["none","1_to_4","5_to_14","15_to_39","at_least_40"])
    grouped=[]
    for by in ["source_key","context_bin","near_name_flag"]:
        for label,g in cells.groupby(by,observed=True):
            candidates=g.groupby(["exact_name_group_id","axis_index"])[["neural_error","tree_error"]].mean()
            per_axis=candidates.groupby("axis_index").mean()
            grouped.append({"stratification":by,"stratum":str(label),"supported_axes":len(per_axis),
                "candidate_groups":g.exact_name_group_id.nunique(),"neural_primary":per_axis.neural_error.mean(),
                "tree_primary":per_axis.tree_error.mean(),"warning":"Conditional observational strata; coverage differs, not a causal masking/source intervention."})
    pd.DataFrame(grouped).to_csv(args.output_dir/"conditional_groups.csv",index=False)
    ss=data.scale[joined.axis_index.to_numpy()]
    joined["neural_error"]=np.abs(np.log1p(joined.neural/ss)-np.log1p(joined.target/ss))
    joined["tree_error"]=np.abs(np.log1p(joined.tree/ss)-np.log1p(joined.target/ss))
    joined["gap"]=joined.neural_error-joined.tree_error
    cases=pd.concat([joined[joined.axis_index.isin(selected)].nsmallest(20,"gap"),joined[joined.axis_index.isin(selected)].nlargest(20,"gap")])
    cases.merge(axes[["axis_index","canonical_name"]],on="axis_index").to_csv(args.output_dir/"local_success_failure_cases.csv",index=False)
    summary={"neural":nn_score,"tree":tree_score,"neural_better_nutrition_axes":int(nutrition.primary_gap.lt(0).sum()),
        "primary_gap":float(nutrition.primary_gap.mean()),"paired_intervals":pair,
        "largest_gap_families":sorted([r for r in decomposition if r["stratification"]=="mask_family"],key=lambda r:r["macro_gap_contribution"],reverse=True),
        "data_hash":digest(data.root/"manifest.json"),"neural_prediction_hash":digest(args.neural_dir/"completion_predictions.parquet"),
        "tree_prediction_hash":digest(args.tree_dir/"predictions.parquet"),"code_hash":digest(Path(__file__)),"complete_test_opened":False,
        "interpretation":"Exploratory diagnosis only; bins do not change primary metric, training or model selection. Original-value cases stay in ignored local reports."}
    write_json(args.output_dir/"summary.json",summary)
    print({k:summary[k] for k in ["neural_better_nutrition_axes","primary_gap","largest_gap_families"]})

if __name__=="__main__":main()

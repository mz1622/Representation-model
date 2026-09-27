"""Compare two saved predictions on the unchanged validation panel; never refits models."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,score_predictions,digest,write_json
from foodcomp.research_statistics import paired_interval,paired_axis_intervals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--task",choices=["completion","name_only"],required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    scores={};groups={};axis_metrics={};source_metrics=[];source_cells={}
    for role,path in [("baseline",args.baseline),("candidate",args.candidate)]:
        pred=pd.read_parquet(path)
        scores[role],axis_metrics[role],groups[role]=score_predictions(data,pred)
        axis_metrics[role].to_csv(args.output_dir/f"{role}_axis_metrics.csv",index=False)
        jobs=data.jobs.merge(pred,on=["profile_index","axis_index"],validate="one_to_one")
        jobs=jobs.merge(data.profiles[["profile_index","exact_name_group_id","source_key"]],on="profile_index",validate="many_to_one")
        cells=jobs.groupby(["exact_name_group_id","axis_index","source_key"],as_index=False).agg(target=("target","median"),prediction=("prediction","median"))
        source_cells[role]=cells
        for source,g in cells.groupby("source_key"):
            g=g[g.axis_index.isin(data.axes[data.axes.loss_group.eq("nutrition")].axis_index)].copy()
            if g.empty:continue
            scale=data.scale[g.axis_index.to_numpy()]
            g["scaled_log_mae"]=np.abs(np.log1p(g.prediction/scale)-np.log1p(g.target/scale))
            g["log_mae"]=np.abs(np.log1p(g.prediction)-np.log1p(g.target))
            by_axis=g.groupby("axis_index")[["scaled_log_mae","log_mae"]].mean()
            source_metrics.append({"role":role,"source":source,"supported_nutrition_axes":len(by_axis),"candidate_groups":g.exact_name_group_id.nunique(),**by_axis.mean().to_dict()})
    pd.DataFrame(source_metrics).to_csv(args.output_dir/"source_metrics.csv",index=False)
    axes=data.axes[data.axes.loss_group.eq("nutrition")&data.axes.loss_eligible].axis_index.to_numpy()
    intervals={m:paired_interval(groups["baseline"],groups["candidate"],axes,metric=m) for m in ["scaled_log_mae","log_mae"]}
    keys=["exact_name_group_id","axis_index","source_key"]
    cells=source_cells["baseline"].merge(source_cells["candidate"],on=keys,suffixes=("_baseline","_candidate"),validate="one_to_one",how="outer",indicator=True)
    if not cells._merge.eq("both").all():raise ValueError("Different source-cell coverage.")
    np.testing.assert_array_equal(cells.target_baseline,cells.target_candidate)
    cells=cells[cells.axis_index.isin(axes)].copy();scale=data.scale[cells.axis_index.to_numpy()]
    error={role:np.abs(np.log1p(cells["prediction_"+role]/scale)-np.log1p(cells.target_baseline/scale)) for role in scores}
    delta=error["candidate"]-error["baseline"]
    # Keep all cells in both denominators, assigning zero contribution to the other stratum.
    # Conditional positive/zero MAEs alone do not add to the total macro improvement.
    cells["positive_delta_contribution"]=delta*(cells.target_baseline>0)
    cells["zero_delta_contribution"]=delta*(cells.target_baseline==0)
    contributions=cells.groupby(["exact_name_group_id","axis_index"])[["positive_delta_contribution","zero_delta_contribution"]].mean().groupby("axis_index").mean().mean().to_dict()
    np.testing.assert_allclose(sum(contributions.values()),scores["candidate"]["nutrition"]["scaled_log_mae"]-scores["baseline"]["nutrition"]["scaled_log_mae"],rtol=1e-10,atol=1e-12)
    per_axis=paired_axis_intervals(groups["baseline"],groups["candidate"])
    per_axis.merge(data.axes[["axis_index","canonical_name","loss_group"]],on="axis_index").to_csv(args.output_dir/"axis_paired_intervals.csv",index=False)
    summary={"task":args.task,"scores":scores,"paired_intervals":intervals,"primary_change_decomposition":contributions,
        "baseline_path":str(args.baseline),"candidate_path":str(args.candidate),
        "baseline_sha256":digest(args.baseline),"candidate_sha256":digest(args.candidate),
        "data_sha256":digest(data.root/"manifest.json"),"code_sha256":digest(Path(__file__)),
        "complete_test_opened":False,"confirmation":False,
        "scope":"Same frozen validation labels and weights. Intervals condition on selected models; seed variation and selection uncertainty are not included. Source-specific averages have different axis coverage."}
    write_json(args.output_dir/"summary.json",summary)
    print(intervals)


if __name__=="__main__":main()

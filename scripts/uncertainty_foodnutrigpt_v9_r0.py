"""Per-axis paired group intervals, explicitly retaining sparse-axis support diagnostics."""
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import write_json,digest
from foodcomp.research_statistics import paired_axis_intervals

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    baseline=pd.read_parquet(ROOT/"output/v9_r0/xgb300_quarantined_completion/candidate_errors.parquet")
    axes=pd.read_csv(ROOT/"data/processed/foodnutrigpt_v9_r0_v1/axes.csv")[["axis_index","canonical_name","loss_group"]]
    for run in ["v9_8_quarantined","v8_optimized20_quarantined","mlp8_quarantined"]:
        candidate=pd.read_parquet(ROOT/f"output/v9_r0/{run}/completion_candidate_errors.parquet")
        frame=paired_axis_intervals(baseline,candidate).merge(axes,on="axis_index",validate="one_to_one")
        frame.to_csv(args.output_dir/f"{run}_axis_intervals.csv",index=False)
    write_json(args.output_dir/"manifest.json",{"seed":20260922,"repeats":1000,"axes":187,"comparison_baseline":"xgb300_quarantined_completion",
        "sampling_unit":"whole exact-name candidate group, jointly across axes",
        "interpretation":"Conditional on fitted models; not seed variation. Sparse axes have coarse/unstable intervals. Empty-support resamples explicitly counted per axis, never treated as zero error. No multiple-comparison significance claim.",
        "code_hashes":{str(x.relative_to(ROOT)):digest(x) for x in [Path(__file__),ROOT/"src/foodcomp/research_statistics.py"]},
        "complete_test_opened":False})

if __name__=="__main__":main()

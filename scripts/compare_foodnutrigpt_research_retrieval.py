"""Compare fixed-candidate retrieval using paired food-group bootstrap."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from foodcomp.research_r0 import digest,write_json
from foodcomp.research_statistics import paired_group_rates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir",type=Path,required=True)
    p.add_argument("--candidate-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    ranks={};metadata={};hashes={}
    metrics=["mrr","recall_at_1","recall_at_5","recall_at_10"]
    for role,path in [("baseline",args.baseline_dir),("candidate",args.candidate_dir)]:
        meta=json.loads((path/"metrics.json").read_text(encoding="utf-8"));metadata[role]=meta
        if digest(path/"candidate_names.json")!=meta["candidate_sha256"]:raise ValueError("Candidate list changed.")
        if meta["complete_test_opened"]:raise ValueError("Retrieval comparison requires closed test.")
        ranks[role]=pd.read_parquet(path/"ranks.parquet")
        r=ranks[role]
        if r.duplicated(["visible_fraction","profile_index"]).any():raise ValueError("Duplicate query profile.")
        if not np.isfinite(r["rank"]).all() or not r["rank"].between(1,meta["candidate_count"]).all() or not r["rank"].eq(np.floor(r["rank"])).all():
            raise ValueError("Invalid rank.")
        r["mrr"]=1/r["rank"]
        for k in [1,5,10]:r[f"recall_at_{k}"]=(r["rank"]<=k).astype(float)
        hashes[role]={"ranks":digest(path/"ranks.parquet"),"metrics":digest(path/"metrics.json")}
    for field in ["data_sha256","candidate_sha256","candidate_count","correct_answers","scoring"]:
        if metadata["baseline"][field]!=metadata["candidate"][field]:raise ValueError(f"Different retrieval protocol: {field}")
    keys=["visible_fraction","profile_index","source_key","exact_name_group_id","observed_axes"]
    a=ranks["baseline"][keys].sort_values(keys).reset_index(drop=True)
    b=ranks["candidate"][keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    result={}
    for fraction in sorted(a.visible_fraction.unique()):
        groups={}
        for role,r in ranks.items():
            groups[role]=r[r.visible_fraction.eq(fraction)].groupby(["exact_name_group_id","source_key"])[metrics].mean().groupby("exact_name_group_id").mean().reset_index()
        result[str(fraction)]=paired_group_rates(groups["baseline"],groups["candidate"],metrics)
    write_json(args.output_dir/"summary.json",{"comparisons":result,"hashes":hashes,
        "baseline":str(args.baseline_dir),"candidate":str(args.candidate_dir),
        "data_sha256":metadata["baseline"]["data_sha256"],"candidate_sha256":metadata["baseline"]["candidate_sha256"],
        "query_profiles":metadata["baseline"]["query_profiles"],"code_sha256":digest(Path(__file__)),
        "complete_test_opened":False,"confirmation":False,
        "scope":"Compare models only within the same visibility fraction. Fractions have different eligible query populations and are not a controlled visibility effect. Exact-name relevance only; no confirmed alias map."})
    print({fraction:items["recall_at_10"] for fraction,items in result.items()})


if __name__=="__main__":main()

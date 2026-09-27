"""Read-only prototype audit of caller/family query scope; not model selection."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import batch_from_arrays,predictions_from_outputs
from foodcomp.research_r0 import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4);started=time.monotonic()
    result={"status":"incomplete","post_result_diagnostic":True,"candidate_resumed":False,
        "complete_test_opened":False,"script_sha256":digest(Path(__file__)),"cases":[]}
    try:
        wrapper=NutritionModel(ROOT/"output/v9_r4/mlp60_views_weight0/best_model.pt")
        data=wrapper.data
        queries={str(f):data.targets[data.families[data.targets]==f].astype(int).tolist() for f in np.unique(data.families[data.targets])}
        result["schema_queries_by_family"]=queries
        for dirname,epoch in [("r4_centered_failure_v1",17),("r4_stable_centered_failure_v1",18)]:
            folder=ROOT/"data/local/research_diagnostics"/dirname
            diagnostic=json.loads((folder/"summary.json").read_text())
            if diagnostic["status"]!="complete_failure_reproduced":raise ValueError("Expected retained failure diagnostic.")
            saved=torch.load(folder/f"replayed_epoch{epoch}_diagnostic_not_candidate.pt",map_location="cpu",weights_only=True)
            wrapper.model.load_state_dict(saved["model_state"]);wrapper.model.eval()
            overflow=pd.read_parquet(folder/"overflow_cells_private.parquet")
            if any(int(r.axis_index) in queries[str(r.mask_family)] for r in overflow.itertuples()):
                raise AssertionError("An overflow is actually inside a schema-defined family request.")
            checked=0;unlabelled=0;original_failed_batches=0;families=[]
            with torch.no_grad():
                for family,jobs in data.jobs.groupby("mask_family"):
                    rows=np.sort(jobs.profile_index.unique());axes=np.array(queries[str(family)],dtype=np.int64)
                    visible=data.observed[rows].copy();visible[:,data.families==family]=False
                    count=len(rows)*len(axes)
                    missing=int((~data.observed[np.ix_(rows,axes)]).sum())
                    for start in range(0,len(rows),128):
                        rr=rows[start:start+128]
                        batch=batch_from_arrays(data.values[rr],visible[start:start+128],wrapper._cached_text[rr],wrapper.device)
                        outputs=wrapper.model(batch)
                        if not all(torch.isfinite(v).all() for v in outputs.values()):
                            raise FloatingPointError("Nonfinite pre-inverse output, including outside the request.")
                        try:predictions_from_outputs(outputs,data.scale)
                        except FloatingPointError as error:
                            if str(error)!="Nonfinite prediction.":raise
                            original_failed_batches+=1
                        index=torch.as_tensor(axes,device=wrapper.device)
                        selected={k:v.index_select(1,index) for k,v in outputs.items()}
                        raw,_=predictions_from_outputs(selected,data.scale[axes])
                        if not torch.isfinite(raw).all():raise AssertionError("Invalid requested output.")
                    checked+=count;unlabelled+=missing
                    families.append({"family":str(family),"profiles":len(rows),"query_axes":len(axes),"requested_cells":count,"requested_cells_without_label":missing})
            # The current decoder already repairs the epoch17 intermediate
            # overflow. Epoch18 still exceeds its return dtype outside the query.
            example=overflow.iloc[0];row=int(example.profile_index);family=str(example.mask_family)
            axes=np.array(queries[family],dtype=np.int64)
            context_axes=np.flatnonzero(data.observed[row] & (data.families!=family))
            context={int(a):float(data.raw[row,a]) for a in context_axes}
            name=str(data.profiles.iloc[row].original_name)
            api_failed=False
            try:wrapper.predict(name,context,axes)
            except FloatingPointError as error:
                if str(error)!="Nonfinite prediction.":raise
                api_failed=True
            if api_failed!=(epoch==18):raise AssertionError("Current API behavior differs from repaired17/unrepaired18 expectation.")
            result["cases"].append({"diagnostic":dirname,"replayed_epoch":epoch,
                "model_sha256":digest(folder/f"replayed_epoch{epoch}_diagnostic_not_candidate.pt"),
                "all_logged_overflows_outside_schema_family_request":True,
                "family_requested_cells_checked":checked,"requested_cells_without_scoring_label":unlabelled,
                "all_requested_outputs_finite":True,"original_all_axis_failed_batches":original_failed_batches,
                "current_public_api_fails_for_finite_requested_family":api_failed,"families":families})
        result.update(status="complete",elapsed_seconds=time.monotonic()-started,
            data_sha256=digest(data.root/"manifest.json"),name_cache_sha256=digest(wrapper.cache/"manifest.json"),
            scope="Full schema-defined supervised family queries, including unlabelled requested axes; no labels choose the query set, no scoring or checkpoint selection, no training. Full252-axis encoder inputs/forward outputs unchanged. All pre-inverse outputs must be finite; only requested coordinates are inversely transformed in the prototype. This is not acceptance of either failed candidate or a claim of good off-task outputs.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(args.output_dir/"verification.json",result);raise
    write_json(args.output_dir/"verification.json",result)
    print([{k:c[k] for k in ["replayed_epoch","family_requested_cells_checked","requested_cells_without_scoring_label","original_all_axis_failed_batches","all_requested_outputs_finite"]} for c in result["cases"]])


if __name__=="__main__":main()

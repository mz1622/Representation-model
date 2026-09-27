"""Registered fixed train-only probe of two-view distance and representation spread."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import digest,write_json
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,fingerprint_array
from foodcomp.research_views import extra_view_masks,subset_view
from foodcomp.research_view_geometry import geometry_summary
from foodcomp.research_inference import NutritionModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,nargs=2,required=True)
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(3)
    result={"status":"incomplete","complete_test_opened":False,
        "sampling":"8192 uniform train family tasks without replacement; default_rng20260925",
        "view_masks":"registered extra_view_masks: full frozen task IDs by252axes, p0.3 seed20260922 epoch1001",
        "models":[],"code_sha256":{str(x.relative_to(ROOT)):digest(x) for x in
            [Path(__file__),ROOT/"src/foodcomp/research_view_geometry.py",ROOT/"src/foodcomp/research_views.py"]}}
    expected=None;task_hash=None;mask_hash=None
    try:
        for checkpoint in args.checkpoint:
            started=time.monotonic()
            manifest=json.loads((checkpoint.parent/"run_manifest.json").read_text())
            if manifest["status"]!="complete" or manifest["test_opened"]:raise ValueError("Completed test-closed model required.")
            wrapper=NutritionModel(checkpoint);model=wrapper.model
            if wrapper.kind!="mlp" or hasattr(model,"query_residual"):raise ValueError("Original linear-readout MLP required.")
            current=(digest(wrapper.data.root/"manifest.json"),digest(wrapper.cache/"manifest.json"))
            if expected is not None and current!=expected:raise ValueError("Data or name cache changed between models.")
            expected=current
            panel=FamilyPanel(wrapper.data,wrapper._cached_text,ROOT/"data/processed"/PANEL_VERSION,wrapper.device)
            ids=np.random.default_rng(20260925).choice(len(panel.rows),8192,replace=False)
            if not np.isin(panel.rows[ids],wrapper.data.train).all():raise ValueError("Nontraining probe input.")
            masks=extra_view_masks(len(panel.rows),len(wrapper.data.axes),.3,20260922,1001)
            current_task_hash=fingerprint_array(ids);current_mask_hash=fingerprint_array(masks)
            if task_hash is not None and (task_hash,mask_hash)!=(current_task_hash,current_mask_hash):raise ValueError("Probe views changed.")
            task_hash=current_task_hash;mask_hash=current_mask_hash
            first=[];second=[];weights=[]
            model.eval()
            with torch.no_grad():
                for start in range(0,len(ids),256):
                    index=ids[start:start+256];batch=panel.batch(index)
                    first.append(model.encode(batch).cpu().numpy())
                    second.append(model.encode(subset_view(batch,masks[index])).cpu().numpy())
                    q=(batch["cell_weight"]*batch["target"]/batch["axis_total"].clamp_min(1e-12)).sum(1)
                    weights.append(q.cpu().numpy())
            record=geometry_summary(np.concatenate(first),np.concatenate(second),np.concatenate(weights),len(panel.rows),panel.axis_count)
            record.update(run=checkpoint.parent.name,checkpoint_sha256=digest(checkpoint),elapsed_seconds=time.monotonic()-started)
            result["models"].append(record)
            del wrapper,model,panel,masks,first,second,weights
        result.update(status="complete",data_sha256=expected[0],name_cache_sha256=expected[1],
            task_ids_sha256=task_hash,full_extra_mask_sha256=mask_hash,
            scope="Descriptive fixed train-input geometry only; no fitting, validation/test selection or new transfer task. Uniform family-task spread is not source-balanced food identity variance. The weighted distance uses original source/axis weights; the finite sample weight mass need not be exactly1. Nonzero spread alone does not prove absence of semantic collapse or useful transfer.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(args.output_dir/"summary.json",result);raise
    write_json(args.output_dir/"summary.json",result)
    print([{k:x[k] for k in ["run","uniform_task_view_distance","axis_weighted_mean_view_distance"]} for x in result["models"]])


if __name__=="__main__":main()

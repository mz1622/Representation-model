"""Validate retained failed runs and exact pre-failure replay, without resuming them."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    runs=[ROOT/"output/v9_r4"/n for n in ["mlp60_views_centered_weight01","mlp60_views_centered_weight01_stable_inverse"]]
    manifests=[json.loads((r/"run_manifest.json").read_text()) for r in runs]
    if any(m["status"]!="failed" or m["test_opened"] for m in manifests):raise ValueError("Both retained test-closed failures required.")
    for run,m in zip(runs,manifests):
        for relative,sha in m["code_hashes"].items():
            if digest(run/"code_snapshot"/Path(relative).name)!=sha:raise ValueError("Original source snapshot changed.")
    first,second=[pd.read_csv(r/"history.csv") for r in runs]
    if len(first)!=16 or len(second)!=17:raise ValueError("Unexpected partial history.")
    fields=[c for c in first if c!="elapsed_seconds"]
    pd.testing.assert_frame_equal(first[fields],second[fields].iloc[:16],check_exact=True)
    exposure=["epoch","training_tasks","visible_cells_A","removed_visible_cells_B","view_mask_sha256","learning_rate"]
    for name in ["mlp60_views_weight0","mlp60_views_weight01"]:
        reference=pd.read_csv(ROOT/"output/v9_r4"/name/"history.csv")
        for history in [first,second]:
            pd.testing.assert_frame_equal(history[exposure],reference[exposure].iloc[:len(history)],check_exact=True)
    states=[torch.load(r/"best_model.pt",map_location="cpu",weights_only=True) for r in runs]
    if any(s["best_epoch"]!=16 for s in states):raise ValueError("Unexpected retained partial best epoch.")
    if states[0]["model_state"].keys()!=states[1]["model_state"].keys():raise ValueError("Model state keys changed.")
    for key,first_tensor in states[0]["model_state"].items():
        second_tensor=states[1]["model_state"][key]
        if first_tensor.numpy().tobytes()!=second_tensor.numpy().tobytes():raise AssertionError("Partial best model tensor bytes changed.")
    private=ROOT/"data/local/research_diagnostics/r4_stable_centered_failure_v1/overflow_cells_private.parquet"
    values=pd.read_parquet(private).raw_float64_diagnostic.to_numpy(dtype=np.float64)
    if len(values)!=2137 or not np.isfinite(values).all() or not (values>float(np.finfo(np.float32).max)).all():
        raise AssertionError("Final-range classification not reproduced in explicit float64.")
    result={"status":"complete","complete_test_opened":False,"script_sha256":digest(Path(__file__)),
        "original_source_snapshots_intact":True,"all_first16_history_fields_except_walltime_exact":True,
        "partial_best_epoch16_model_tensor_bytes_exact":True,"partial_exposure_matches_both_original_controls":True,
        "second_failure_final_values_above_float32_max":len(values),
        "failed_runs":[{"run":str(r.relative_to(ROOT)),"status":m["status"],"manifest_sha256":digest(r/"run_manifest.json"),
            "completed_epochs":m["epoch_completed"],"elapsed_seconds":m["elapsed_seconds"],"code_commit":m["code_commit"]} for r,m in zip(runs,manifests)],
        "scope":"Both candidates remain failed with no completed three-task score. The original lost failing states were not saved; diagnostic replays are not asserted bitwise equal to those lost states. No numerical cap, ignored output, resumed candidate or test use."}
    write_json(args.output_dir/"verification.json",result);print(result["status"])


if __name__=="__main__":main()

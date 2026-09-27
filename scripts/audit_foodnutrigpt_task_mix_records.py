"""Audit actual completed 10/20% histories and training states, without rerunning training."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import PANEL_VERSION, fingerprint_array
from foodcomp.research_task_mix import name_only_tasks


def main():
    out = ROOT / "reports/v9_r3_mixture_record_audit_v1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    result = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        data = ResearchData(ROOT / "data/processed" / VERSION)
        panel = ROOT / "data/processed" / PANEL_VERSION
        panel_manifest = json.loads((panel / "manifest.json").read_text())
        if digest(panel / "tasks.npz") != panel_manifest["tasks_sha256"]:
            raise ValueError("Changed training tasks.")
        with np.load(panel / "tasks.npz", allow_pickle=False) as saved:
            rows, families, masks = saved["rows"], saved["families"], saved["masks"]
        if not np.isin(rows, data.train).all():
            raise ValueError("Audit task is outside training.")
        counts = np.zeros(len(rows), dtype=np.int64)
        eligible = np.zeros(len(data.axes), dtype=bool); eligible[data.targets] = True
        for family, mask in enumerate(masks):
            subset = families == family
            counts[subset] = data.observed[rows[subset]][:, mask & eligible].sum(axis=1)
        if counts.sum() != panel_manifest["observed_target_cells"] or np.any(counts <= 0):
            raise ValueError("Target supervision coverage changed.")
        runs = {}; configuration = implementation = None
        for percent in [10, 20]:
            folder = ROOT / f"output/v9_r3/mlp60_name_mix{percent}"
            manifest = json.loads((folder / "run_manifest.json").read_text())
            if manifest["status"] != "complete" or manifest["test_opened"]:
                raise ValueError("Both completed, test-closed runs are required.")
            settings = dict(manifest["args"])
            if settings.pop("name_only_probability") != percent / 100:
                raise ValueError("Incorrect task probability.")
            settings.pop("output_dir")
            if configuration is None:
                configuration, implementation = settings, manifest["code_hashes"]
            elif settings != configuration or implementation != manifest["code_hashes"]:
                raise ValueError("More than task probability changed across runs.")
            if settings["epochs"] != 60 or settings["schedule_epochs"] != 60 or settings["seed"] != 20260922:
                raise ValueError("Registered budget/seed changed.")
            if manifest["data_hash"] != digest(data.root / "manifest.json") or manifest["panel_hash"] != digest(panel / "manifest.json"):
                raise ValueError("Data or training panel mismatch.")
            if manifest["name_cache_hash"] != panel_manifest["name_cache_hash"]:
                raise ValueError("Text input changed.")
            for source, hashed in manifest["code_hashes"].items():
                if digest(folder / "code_snapshot" / Path(source).name) != hashed:
                    raise ValueError("Frozen training source snapshot changed.")
            history = pd.read_csv(folder / "history.csv")
            np.testing.assert_array_equal(history.epoch, np.arange(1, 61))
            np.testing.assert_array_equal(history.training_tasks, np.full(60, len(rows)))
            state = torch.load(folder / "latest_training_state.pt", map_location="cpu", weights_only=True)
            if state["best_epoch"] != 60:
                raise ValueError("Last training state does not represent epoch60.")
            runs[percent] = {"folder": folder, "history": history, "state": state, "manifest": manifest}
        records = []
        for epoch in range(1, 61):
            selected = {}
            for percent in [10, 20]:
                selected[percent] = name_only_tasks(len(rows), percent / 100, 20260922, epoch)
                record = runs[percent]["history"].iloc[epoch - 1]
                task_count = int(selected[percent].sum())
                target_count = int(counts[selected[percent]].sum())
                mask_hash = fingerprint_array(selected[percent])
                if task_count != record.name_only_tasks or target_count != record.name_only_target_cells or mask_hash != record.name_only_task_mask_sha256:
                    raise ValueError(f"Actual epoch record cannot be reproduced: {percent}% epoch{epoch}")
                records.append({"percent": percent, "epoch": epoch, "name_only_tasks": task_count,
                                "name_only_target_cells": target_count, "name_only_task_mask_sha256": mask_hash})
            if np.any(selected[10] & ~selected[20]):
                raise AssertionError("10% assignment is not nested in 20%.")
        first, second = runs[10]["state"], runs[20]["state"]
        torch.testing.assert_close(first["cpu_rng"], second["cpu_rng"], rtol=0, atol=0)
        if len(first["cuda_rng"]) != len(second["cuda_rng"]):
            raise ValueError("CUDA device count differs.")
        for a, b in zip(first["cuda_rng"], second["cuda_rng"]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if first["scheduler"] != second["scheduler"] or first["optimizer"]["param_groups"] != second["optimizer"]["param_groups"]:
            raise ValueError("Learning-rate or optimizer schedule differs.")
        steps = []
        for key, state in first["optimizer"]["state"].items():
            torch.testing.assert_close(state["step"], second["optimizer"]["state"][key]["step"], rtol=0, atol=0)
            steps.append(int(state["step"]))
        expected_steps = 60 * int(np.ceil(len(rows) / configuration["batch_size"]))
        if not steps or set(steps) != {expected_steps}:
            raise ValueError("Parameters did not receive the registered optimizer-step count.")
        pd.DataFrame(records).to_csv(out / "epoch_assignments.csv", index=False)
        result.update(status="complete", configuration_except_probability=configuration,
            all_120_assignment_hashes_counts_and_target_counts_match=True, nested_assignments_all_60_epochs=True,
            final_cpu_and_cuda_rng_identical=True, optimizer_steps_per_parameter=expected_steps,
            final_learning_rate_schedule_identical=True, train_tasks=len(rows), observed_targets_per_epoch=int(counts.sum()),
            actual_selected_task_fraction={str(p): float(runs[p]["history"].name_only_tasks.sum() / (60 * len(rows))) for p in runs},
            data_sha256=digest(data.root / "manifest.json"), panel_sha256=digest(panel / "manifest.json"),
            receipts={str(p): {"manifest_sha256": digest(runs[p]["folder"] / "run_manifest.json"),
                "history_sha256": digest(runs[p]["folder"] / "history.csv"),
                "latest_training_state_sha256": digest(runs[p]["folder"] / "latest_training_state.pt")} for p in runs},
            elapsed_seconds=time.monotonic() - started,
            scope="Audit saved histories, source snapshots, assignment reproducibility and final random/optimizer states. Does not replay every forward/gradient or prove superiority or absence of label errors. Selected-task probability is not the fraction of changed contexts because some family tasks already lack visible numeric inputs.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out / "verification.json", result)
        raise
    write_json(out / "verification.json", result)
    print({k: result[k] for k in ["status", "actual_selected_task_fraction", "optimizer_steps_per_parameter", "final_cpu_and_cuda_rng_identical"]})


if __name__ == "__main__":
    main()

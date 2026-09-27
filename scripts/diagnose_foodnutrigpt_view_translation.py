"""Train-only post-result algebraic counterfactual on fixed R4 view checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, fingerprint_array
from foodcomp.research_views import extra_view_masks, subset_view
from foodcomp.research_translation import translation_summary
from foodcomp.research_inference import NutritionModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry-report", type=Path, default=ROOT/"reports/v9_r4_views_geometry_v1/summary.json")
    p.add_argument("--checkpoint", type=Path, nargs=2, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    reference = json.loads(args.geometry_report.read_text())
    if reference["status"] != "complete" or reference["complete_test_opened"]:
        raise ValueError("Completed test-closed reference geometry required.")
    torch.set_num_threads(3)
    result = {
        "status": "incomplete", "complete_test_opened": False, "post_result_diagnostic": True,
        "new_training_candidates": 0, "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reference_geometry_sha256": digest(args.geometry_report),
        "code_sha256": {str(x.relative_to(ROOT)): digest(x) for x in [
            Path(__file__), ROOT/"src/foodcomp/research_translation.py",
            ROOT/"src/foodcomp/research_view_geometry.py", ROOT/"src/foodcomp/research_views.py"]},
        "scope": "Train-input coordinate counterfactual; shared mean of probe views only. "
                 "No optimization or validation/test evaluation. Same float32 encoder features "
                 "cast to float64 for both sides of the linear-head equality. No claim of "
                 "bitwise float32 replay, causal optimization path, semantic collapse or transfer.",
        "models": [],
    }
    try:
        for checkpoint, old in zip(args.checkpoint, reference["models"]):
            started = time.monotonic()
            if digest(checkpoint) != old["checkpoint_sha256"]:
                raise ValueError("Checkpoint order/identity changed from fixed probe.")
            wrapper = NutritionModel(checkpoint); model = wrapper.model
            if wrapper.kind != "mlp" or hasattr(model, "query_residual") or hasattr(model, "name_head"):
                raise ValueError("Single original linear readout required.")
            if digest(wrapper.data.root/"manifest.json") != reference["data_sha256"] or digest(wrapper.cache/"manifest.json") != reference["name_cache_sha256"]:
                raise ValueError("Data/name cache identity changed.")
            panel = FamilyPanel(wrapper.data, wrapper._cached_text, ROOT/"data/processed"/PANEL_VERSION, wrapper.device)
            ids = np.random.default_rng(20260925).choice(len(panel.rows), 8192, replace=False)
            if not np.isin(panel.rows[ids], wrapper.data.train).all():
                raise ValueError("Nontraining probe.")
            masks = extra_view_masks(len(panel.rows), len(wrapper.data.axes), .3, 20260922, 1001)
            if fingerprint_array(ids) != reference["task_ids_sha256"] or fingerprint_array(masks) != reference["full_extra_mask_sha256"]:
                raise ValueError("Probe tasks or view masks changed.")
            first = []; second = []; weights = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(ids), 256):
                    index = ids[start:start+256]; batch = panel.batch(index)
                    first.append(model.encode(batch).cpu().numpy())
                    second.append(model.encode(subset_view(batch, masks[index])).cpu().numpy())
                    weights.append((batch["cell_weight"] * batch["target"] /
                                    batch["axis_total"].clamp_min(1e-12)).sum(1).cpu().numpy())
            record = translation_summary(np.concatenate(first), np.concatenate(second),
                np.concatenate(weights), len(panel.rows), panel.axis_count,
                model.head.weight.detach().cpu().numpy(), model.head.bias.detach().cpu().numpy())
            geometry = record["coordinate_cases"][0]["geometry"]
            if any(geometry[k] != old[k] for k in geometry):
                raise AssertionError("Original geometry did not exactly reproduce the fixed probe.")
            record.update(run=checkpoint.parent.name, checkpoint_sha256=digest(checkpoint),
                          elapsed_seconds=time.monotonic()-started, original_geometry_exactly_replayed=True)
            result["models"].append(record)
            del wrapper, model, panel, masks, first, second, weights
        result.update(status="complete", data_sha256=reference["data_sha256"],
                      name_cache_sha256=reference["name_cache_sha256"],
                      task_ids_sha256=reference["task_ids_sha256"],
                      full_extra_mask_sha256=reference["full_extra_mask_sha256"])
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir/"summary.json", result)
        raise
    write_json(args.output_dir/"summary.json", result)
    print(json.dumps([{ "run": x["run"], "weighted_distances": {
        c["case"]: c["geometry"]["axis_weighted_mean_view_distance"] for c in x["coordinate_cases"]},
        "translation_invariant_variance_ratio": x["weighted_pair_distance_over_centred_second_moment"]
    } for x in result["models"]], indent=2))


if __name__ == "__main__":
    main()

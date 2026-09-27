"""Measure training name scales and initial first-layer contributions; no optimization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION
from foodcomp.research_neural import make_model
from foodcomp.research_conditioning import fit_training_name_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    torch.set_num_threads(1)
    result = {"status": "incomplete", "complete_test_opened": False}
    try:
        data = ResearchData(ROOT / "data/processed" / VERSION)
        text, cache = prepare_names(data, ROOT)
        mean, std, metadata = fit_training_name_statistics(data, text)
        panel = FamilyPanel(data, text, ROOT / "data/processed" / PANEL_VERSION, "cpu")
        tasks = np.random.default_rng(20260922).choice(len(panel.rows), 8192, replace=False)
        if not np.isin(panel.rows[tasks], data.train).all():
            raise ValueError("Non-training context task.")
        torch.manual_seed(20260922)
        model, _ = make_model(data, text.shape[1], "mlp", mlp_width=512)
        weight = model.encoder[0].weight.detach()
        sums = dict(name=0., standardized_name=0., visible_values=0., visibility_mask=0., numeric_combined=0.)
        count = 0
        with torch.no_grad():
            for start_row in range(0, len(tasks), 256):
                batch = panel.batch(tasks[start_row:start_row+256])
                name = batch["text"]
                numeric = torch.where(batch["masked"], 0., batch["value"])
                visible = (~batch["masked"]).float()
                a, d = numeric.shape[1], name.shape[1]
                components = {
                    "name": F.linear(name, weight[:, :d]),
                    "standardized_name": F.linear((name-torch.from_numpy(mean))/torch.from_numpy(std), weight[:, :d]),
                    "visible_values": F.linear(numeric, weight[:, d:d+a]),
                    "visibility_mask": F.linear(visible, weight[:, d+a:]),
                    "numeric_combined": F.linear(torch.cat([numeric, visible], 1), weight[:, d:]),
                }
                for key, value in components.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("Nonfinite diagnostic contribution.")
                    sums[key] += float(value.double().square().sum())
                count += name.shape[0] * weight.shape[0]
        rms = {key: float(np.sqrt(value/count)) for key, value in sums.items()}
        np.savez(args.output_dir / "train_name_statistics.npz", mean=mean, std=std)
        result.update(status="complete", fit=metadata, coordinate_std_min=float(std.min()),
                      coordinate_std_median=float(np.median(std)), coordinate_std_max=float(std.max()),
                      max_absolute_coordinate_mean=float(np.abs(mean).max()), initial_contribution_rms=rms,
                      name_to_numeric_rms_ratio=rms["name"]/rms["numeric_combined"],
                      standardized_name_to_numeric_rms_ratio=rms["standardized_name"]/rms["numeric_combined"],
                      sampled_train_family_tasks=len(tasks), seed=20260922,
                      data_sha256=digest(data.root/"manifest.json"), name_cache_sha256=digest(cache/"manifest.json"),
                      code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      script_sha256=digest(Path(__file__)), module_sha256=digest(ROOT/"src/foodcomp/research_conditioning.py"),
                      statistics_artifact_sha256=digest(args.output_dir/"train_name_statistics.npz"),
                      elapsed_seconds=time.monotonic()-start,
                      scope="Descriptive training-only input/initial-linear-contribution diagnostic. No loss/optimizer, held-out score, cache or model change. Small contribution is not evidence that text is ignored or that standardization will improve prediction.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir/"summary.json", result)
        raise
    write_json(args.output_dir/"summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

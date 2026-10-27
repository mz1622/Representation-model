#!/usr/bin/env python3
"""Zero-shot Taiwan TFDA evaluation using canonical-axis dependency masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.dependency_mask_v1 import DependencyMaskPolicy, VERSION as MASK_VERSION  # noqa: E402
from foodcomp.research_dependency_mask_v1 import (  # noqa: E402
    FoodPromptCceTransformer, predict_transformer, score_profile_axis_macro,
)
from foodcomp.research_no_foodname_v1 import Config, numeric_only_training_view  # noqa: E402
from foodcomp.research_r0 import ResearchData, digest, write_json  # noqa: E402


def attach_external_panel(data, audit_dir: Path):
    """Map TFDA values only through frozen canonical axis indices."""
    foods = pd.read_parquet(audit_dir / "foods.parquet")
    mapped = pd.read_parquet(audit_dir / "mapped_values.parquet")
    coverage = pd.read_csv(audit_dir / "axis_coverage.csv")
    axes = np.sort(coverage.loc[
        coverage.strict_sample_backed_foods.ge(20), "axis_index"
    ].to_numpy(dtype=np.int64))
    if len(axes) != 80 or not set(axes).issubset(set(map(int, data.targets))):
        raise ValueError("unexpected TFDA canonical-axis panel")
    eligible = mapped[mapped.numeric_eligible.astype(bool)].copy()
    strict = mapped[
        mapped.strict_sample_backed_target.astype(bool)
        & mapped.axis_index.isin(axes)
    ].copy()
    panel = foods[
        foods.food_id.isin(strict.food_id) & foods.mapped_eligible_axes.ge(10)
    ].sort_values("food_id").reset_index(drop=True)
    if len(panel) != 2_212:
        raise ValueError("unexpected TFDA profile count")
    eligible = eligible[eligible.food_id.isin(panel.food_id)]
    strict = strict[strict.food_id.isin(panel.food_id)]
    if len(strict) != 128_470 or strict.axis_index.nunique() != 80:
        raise ValueError("unexpected TFDA target jobs")
    if eligible.duplicated(["food_id", "axis_index"]).any():
        raise ValueError("duplicate mapped TFDA cell")
    offset = len(data.profiles)
    panel["profile_index"] = offset + np.arange(len(panel), dtype=np.int64)
    row_lookup = panel.set_index("food_id").profile_index
    raw = np.full((len(panel), len(data.axes)), np.nan, dtype=np.float64)
    row = eligible.food_id.map(row_lookup).to_numpy(dtype=np.int64) - offset
    axis = eligible.axis_index.to_numpy(dtype=np.int64)
    value = eligible.value_g_per_100g.to_numpy(dtype=np.float64)
    raw[row, axis] = value
    observed = np.isfinite(raw)
    transformed = np.where(observed, np.log1p(raw / data.scale), 0).astype(np.float32)
    data.raw = np.concatenate((data.raw, raw))
    data.observed = np.concatenate((data.observed, observed))
    data.values = np.concatenate((data.values, transformed))
    data.weights = np.concatenate((data.weights, np.zeros_like(transformed)))
    jobs = strict[["food_id", "axis_index", "value_g_per_100g"]].copy()
    jobs["profile_index"] = jobs.food_id.map(row_lookup).to_numpy(dtype=np.int64)
    jobs = jobs.rename(columns={"value_g_per_100g": "target"})[
        ["profile_index", "axis_index", "target"]
    ].sort_values(["profile_index", "axis_index"]).reset_index(drop=True)
    novel_rows = panel.loc[
        ~panel.english_name_seen_in_pretraining.astype(bool), "profile_index"
    ].to_numpy(dtype=np.int64)
    if len(novel_rows) != 1_789:
        raise ValueError("unexpected name-novel TFDA profile count")
    return jobs, novel_rows, {
        "profiles": int(len(panel)), "jobs": int(len(jobs)), "axes": int(len(axes)),
        "name_novel_profiles": int(len(novel_rows)),
        "name_novel_jobs": int(jobs.profile_index.isin(novel_rows).sum()),
        "legacy_family_column_used": False,
        "axis_mapping": "frozen TFDA analyte -> canonical internal axis_index",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "cce":
        raise ValueError("checkpoint is not V5+CCE")
    if checkpoint.get("mask_policy_version") != MASK_VERSION:
        raise ValueError("checkpoint dependency-mask version mismatch")
    if checkpoint.get("mask_policy_sha256") != digest(args.policy_manifest):
        raise ValueError("checkpoint dependency-mask hash mismatch")
    data = numeric_only_training_view(ResearchData(args.data_dir, "quarantined"))
    jobs, novel_rows, panel = attach_external_panel(data, args.audit_dir)
    policy = DependencyMaskPolicy(data.axes)
    config = Config(**checkpoint["config"])
    model = FoodPromptCceTransformer(len(data.axes), config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model.to(device).eval()
    prediction = predict_transformer(
        model, data, policy, jobs, device, args.batch_size, two_pass=True
    )
    manifest = {
        "status": "complete", "version": "taiwan_dependency_mask_v1_zero_shot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": digest(args.data_dir / "manifest.json"),
        "audit_manifest_sha256": digest(args.audit_dir / "audit_summary.json"),
        "axis_mapping_sha256": digest(args.audit_dir / "axis_mapping.csv"),
        "code_sha256": {
            "scripts/evaluate_taiwan_dependency_mask_v1.py": digest(Path(__file__)),
            "src/foodcomp/dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/dependency_mask_v1.py"
            ),
            "src/foodcomp/research_dependency_mask_v1.py": digest(
                ROOT / "src/foodcomp/research_dependency_mask_v1.py"
            ),
        },
        "mask_policy_version": MASK_VERSION,
        "mask_policy_manifest_sha256": digest(args.policy_manifest),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_epoch": checkpoint["completed_epoch"],
        "taiwan_training_rows": 0, "taiwan_model_selection": False,
        "taiwan_normalizer_fitting": False, "food_name_model_input": False,
        "source_model_input": False, "family_label_model_input": False,
        "panel": panel, "metrics": {},
    }
    for name, part in (
        ("strict_all", jobs),
        ("name_novel", jobs[jobs.profile_index.isin(novel_rows)]),
    ):
        part_prediction = prediction[prediction.profile_index.isin(part.profile_index.unique())]
        metrics, per_axis, _ = score_profile_axis_macro(data, part, part_prediction)
        per_axis.to_csv(args.output_dir / f"{name}_axis_metrics.csv", index=False)
        manifest["metrics"][name] = metrics
    prediction.to_parquet(args.output_dir / "external_predictions.parquet", index=False)
    write_json(args.output_dir / "manifest.json", manifest)
    print(manifest["metrics"], flush=True)


if __name__ == "__main__":
    main()

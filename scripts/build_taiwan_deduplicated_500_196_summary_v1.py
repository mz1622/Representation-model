#!/usr/bin/env python3
"""Consolidate the frozen 500-test/196-adaptation Taiwan experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_complete(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete input: {path}")
    expected = {
        "test_profiles": 500,
        "adaptation_pool_profiles": 196,
        "test_axes": 80,
    }
    aliases = {"test_profiles": ("test_profiles", "fixed_test_profiles")}
    for key, expected_value in expected.items():
        candidates = aliases.get(key, (key,))
        value = next((manifest[name] for name in candidates if name in manifest), None)
        if value != expected_value:
            raise ValueError(f"{path} has {key}={value}, expected {expected_value}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-shot-dir", type=Path, required=True)
    parser.add_argument("--taiwan-only-dir", type=Path, required=True)
    parser.add_argument("--neural-dir", type=Path, required=True)
    parser.add_argument("--pretrained-trees-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True)

    inputs = {
        "zero_shot": args.zero_shot_dir,
        "taiwan_only": args.taiwan_only_dir,
        "v5_finetune": args.neural_dir,
        "pretrained_trees": args.pretrained_trees_dir,
    }
    manifests = {name: load_complete(path) for name, path in inputs.items()}
    split_reference = pd.read_csv(args.zero_shot_dir / "profile_split.csv")[[
        "profile_index", "fewshot_partition"
    ]].sort_values("profile_index").reset_index(drop=True)
    for path in (args.taiwan_only_dir,):
        current = pd.read_csv(path / "profile_split.csv")[[
            "profile_index", "fewshot_partition"
        ]].sort_values("profile_index").reset_index(drop=True)
        pd.testing.assert_frame_equal(split_reference, current, check_dtype=False)

    zero = pd.read_csv(args.zero_shot_dir / "metrics.csv")
    zero = zero[zero.panel.eq("fixed_test")].copy()
    zero["method"] = zero.method.replace({"V5_CCE": "V5+CCE frozen"})
    zero.insert(0, "analysis", "zero_shot")
    zero.insert(1, "shots", 0)

    taiwan = pd.read_csv(args.taiwan_only_dir / "fewshot_summary.csv")
    neural = pd.read_csv(args.neural_dir / "finetune_summary.csv")
    neural.insert(0, "method", "V5+CCE finetuned")
    continued = pd.read_csv(args.pretrained_trees_dir / "continued_summary.csv")
    frozen = pd.DataFrame([
        {
            "method": "V5+CCE frozen",
            "shots": shots,
            "macro_scaled_log_mae_mean": float(
                zero.loc[zero.method.eq("V5+CCE frozen"), "macro_scaled_log_mae"].iloc[0]
            ),
            "macro_scaled_log_mae_sd": 0.0,
            "macro_scaled_log_rmse_mean": float(
                zero.loc[zero.method.eq("V5+CCE frozen"), "macro_scaled_log_rmse"].iloc[0]
            ),
            "macro_scaled_log_rmse_sd": 0.0,
            "macro_raw_mae_mean": float(
                zero.loc[zero.method.eq("V5+CCE frozen"), "macro_raw_mae"].iloc[0]
            ),
            "macro_raw_mae_sd": 0.0,
            "min_axis_labels": 0,
            "max_axis_labels": 0,
        }
        for shots in (8, 32, 128)
    ])
    fewshot = pd.concat((
        frozen,
        neural,
        taiwan[~taiwan.method.eq("V5+CCE frozen")],
        continued,
    ), ignore_index=True, sort=False).sort_values(["shots", "method"])
    fewshot.to_csv(args.output_dir / "fewshot_all_methods.csv", index=False)
    zero.to_csv(args.output_dir / "zero_shot_500_test.csv", index=False)

    headline_rows = []
    for row in zero.itertuples(index=False):
        headline_rows.append({
            "analysis": "zero_shot",
            "panel_or_shots": "fixed_500_test",
            "method": row.method,
            "macro_scaled_log_mae_mean": row.macro_scaled_log_mae,
            "macro_scaled_log_mae_sd": 0.0,
        })
    for row in fewshot.itertuples(index=False):
        headline_rows.append({
            "analysis": "few_shot",
            "panel_or_shots": row.shots,
            "method": row.method,
            "macro_scaled_log_mae_mean": row.macro_scaled_log_mae_mean,
            "macro_scaled_log_mae_sd": row.macro_scaled_log_mae_sd,
        })
    pd.DataFrame(headline_rows).to_csv(
        args.output_dir / "headline_results.csv", index=False
    )

    styles = {
        "V5+CCE frozen": ("#7F7F7F", "--", "o"),
        "V5+CCE finetuned": ("#17365D", "-", "o"),
        "RF source continued": ("#59A14F", "-", "s"),
        "XGBoost source continued": ("#F28E2B", "-", "s"),
        "RF Taiwan-only": ("#8CD17D", ":", "^"),
        "XGBoost Taiwan-only": ("#FFBE7D", ":", "^"),
    }
    fig, axis = plt.subplots(figsize=(8.8, 5.0), facecolor="white")
    for method, (color, linestyle, marker) in styles.items():
        current = fewshot[fewshot.method.eq(method)].sort_values("shots")
        axis.errorbar(
            current.shots,
            current.macro_scaled_log_mae_mean,
            yerr=current.macro_scaled_log_mae_sd.fillna(0),
            label=method,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            capsize=3,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks([8, 32, 128], ["8", "32", "128"])
    axis.set_xlabel("Taiwan adaptation profiles (from the fixed 196-profile pool)")
    axis.set_ylabel("Macro scaled-log MAE")
    axis.set_title("Taiwan transfer: fixed 500-profile test set")
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "fewshot_500_test_196_adaptation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    copies = {
        args.zero_shot_dir / "fixed_test_paired_bootstrap.csv": "zero_shot_paired_bootstrap.csv",
        args.neural_dir / "finetune_runs.csv": "v5_finetune_runs.csv",
        args.taiwan_only_dir / "fewshot_runs.csv": "taiwan_only_tree_runs.csv",
        args.pretrained_trees_dir / "continued_runs.csv": "pretrained_tree_continued_runs.csv",
    }
    for source, name in copies.items():
        shutil.copy2(source, args.output_dir / name)

    manifest = {
        "status": "complete",
        "version": "taiwan_deduplicated_500_test_196_adaptation_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clean_profiles": 696,
        "fixed_test_profiles": 500,
        "adaptation_pool_profiles": 196,
        "fixed_test_jobs": int(manifests["zero_shot"]["fixed_test_jobs"]),
        "test_axes": 80,
        "shots": [8, 32, 128],
        "seeds": [20261027, 20261028, 20261029],
        "identity_group_aware_split": True,
        "split_sha256": digest(args.zero_shot_dir / "profile_split.csv"),
        "inputs": {
            name: {
                "version": manifests[name]["version"],
                "manifest_sha256": digest(path / "manifest.json"),
            }
            for name, path in inputs.items()
        },
        "raw_data_committed": False,
        "predictions_committed": False,
        "model_parameters_committed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

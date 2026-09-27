"""Plot both registered checkpoint criteria on a completed single trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_file = args.run_dir/"run_manifest.json"
    alternate_file = args.run_dir/"evaluation_hurdle/evaluation_manifest.json"
    history_file = args.run_dir/"history.csv"
    manifest = json.loads(manifest_file.read_text())
    alternate = json.loads(alternate_file.read_text())
    if manifest["status"] != "complete" or manifest["test_opened"] or alternate["complete_test_opened"] or not manifest["args"]["dual_selection"]:
        raise ValueError("Completed test-closed dual selection and alternate evaluation required.")
    if alternate["checkpoint_hash"] != manifest["hurdle_checkpoint_hash"]:
        raise ValueError("Alternate evaluation/checkpoint mismatch.")
    history = pd.read_csv(history_file)
    np.testing.assert_array_equal(history.epoch, np.arange(1, manifest["args"]["epochs"]+1))
    columns = ["validation_primary", "validation_source_free_hurdle", "validation_presence_bce", "validation_positive_amount_smooth_l1", "validation_amount_weight"]
    if not np.isfinite(history[columns].to_numpy()).all():
        raise FloatingPointError("Nonfinite validation curve.")
    np.testing.assert_allclose(history.validation_source_free_hurdle,
        history.validation_presence_bce + history.validation_amount_weight*history.validation_positive_amount_smooth_l1,
        rtol=0, atol=1e-12)
    primary_epoch = int(history.loc[history.validation_primary.idxmin(), "epoch"])
    hurdle_epoch = int(history.loc[history.validation_source_free_hurdle.idxmin(), "epoch"])
    if primary_epoch != manifest["best_epoch"] or hurdle_epoch != manifest["best_hurdle_epoch"] or hurdle_epoch != alternate["best_epoch"]:
        raise ValueError("Saved selections differ from earliest strict curve minima.")
    selections = [(primary_epoch, "Primary selection", "#167d9a", "o"),
                  (hurdle_epoch, "Hurdle selection", "#bd7736", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    for panel, metric, title in zip(axes[:2], columns[:2], ["Nutrition primary (142 axes)", "Source-free hurdle (187 axes)"]):
        panel.plot(history.epoch, history[metric], color="#374d5e", lw=2)
        for epoch, label, color, marker in selections:
            value = float(history.loc[history.epoch.eq(epoch), metric].iloc[0])
            panel.scatter([epoch], [value], marker=marker, s=60, facecolors="none", edgecolors=color, linewidths=2, label=f"{label}: epoch {epoch}")
            panel.axvline(epoch, color=color, ls=":", alpha=.5)
        panel.set(title=title, xlabel="Epoch", ylabel="Validation error / objective")
        panel.legend(frameon=False, fontsize=8)
    axes[2].plot(history.epoch, history.validation_presence_bce, label="Presence BCE", color="#8b62ae", lw=2)
    axes[2].plot(history.epoch, history.validation_amount_weight*history.validation_positive_amount_smooth_l1,
                 label="Weighted positive SmoothL1", color="#bd7736", lw=2)
    for epoch, _, color, _ in selections:
        axes[2].axvline(epoch, color=color, ls=":", alpha=.5)
    axes[2].set(title="Hurdle components (187 axes)", xlabel="Epoch", ylabel="Contribution to objective")
    axes[2].legend(frameon=False, fontsize=8)
    for panel in axes:
        panel.grid(alpha=.2)
    fig.suptitle("Checkpoint selection on the same training trajectory", fontsize=14)
    fig.supxlabel("Single seed, fixed validation panel; different criteria and axis scopes. Historical test remains closed.", fontsize=9)
    args.output_dir.mkdir(parents=True)
    for extension in ["png", "svg"]:
        fig.savefig(args.output_dir/f"selection.{extension}", dpi=160)
    plt.close(fig)
    (args.output_dir/"figure_manifest.json").write_text(json.dumps({
        "run_manifest_sha256":sha256(manifest_file), "history_sha256":sha256(history_file),
        "alternate_evaluation_sha256":sha256(alternate_file), "script_sha256":sha256(Path(__file__)),
        "primary_epoch":primary_epoch, "hurdle_epoch":hurdle_epoch, "complete_test_opened":False,
        "scope":"Saved completed curves and registered selections only; no new model, scoring or selection rule."
    }, indent=2), encoding="utf-8")
    print({"primary_epoch":primary_epoch, "hurdle_epoch":hurdle_epoch, "output_dir":str(args.output_dir)})


if __name__ == "__main__":
    main()

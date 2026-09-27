"""Plot completed matched-budget neural controls and saved paired intervals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline-dir", "candidate-dir", "completion-report", "name-report", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ["title", "baseline-label", "candidate-label"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    histories = []
    epochs = None
    for folder in [args.baseline_dir, args.candidate_dir]:
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["test_opened"]:
            raise ValueError("Completed, test-closed runs required.")
        budget = manifest["args"]["epochs"]
        if manifest["args"]["schedule_epochs"] != budget or (epochs is not None and budget != epochs):
            raise ValueError("This plot requires the same full schedule/budget.")
        epochs = budget
        history = pd.read_csv(folder / "history.csv")
        np.testing.assert_array_equal(history.epoch, np.arange(1, budget + 1))
        if not np.isfinite(history.validation_primary).all():
            raise ValueError("Non-finite validation history.")
        histories.append(history)
    summaries = [json.loads((folder / "summary.json").read_text())
                 for folder in [args.completion_report, args.name_report]]
    fig, (curve, intervals) = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    for history, label, color in zip(histories, [args.baseline_label, args.candidate_label], ["#167d9a", "#bd7736"]):
        curve.plot(history.epoch, history.validation_primary, label=label, color=color, lw=2)
        best = history.loc[history.validation_primary.idxmin()]
        curve.scatter([best.epoch], [best.validation_primary], color=color, s=40)
    curve.set(xlabel=f"Epoch (fixed {epochs}-epoch cosine schedule)", ylabel="142-axis scaled-log MAE (lower is better)", title="Completion validation history")
    curve.grid(alpha=.18)
    curve.legend(frameon=False)
    labels = []
    for summary, task, color in zip(summaries, ["Completion", "Name only"], ["#167d9a", "#bd7736"]):
        for metric, label in [("scaled_log_mae", "Primary"), ("log_mae", "Legacy log-MAE")]:
            result = summary["paired_intervals"][metric]
            point = result["relative_improvement"] * 100
            low, high = np.array(result["relative_improvement_95_interval"]) * 100
            if not np.isfinite([point, low, high]).all() or low > high:
                raise ValueError("Invalid paired interval.")
            row = len(labels)
            intervals.plot([low, high], [row, row], color=color, lw=2)
            intervals.scatter([point], [row], color=color, s=40)
            labels.append(f"{task}\n{label}")
    intervals.axvline(0, color="#666666", ls="--", lw=1)
    intervals.set_yticks(range(len(labels)), labels)
    intervals.invert_yaxis()
    intervals.grid(axis="x", alpha=.18)
    intervals.set(xlabel="Relative improvement (%) with 95% food-group interval", title="Candidate versus registered parent")
    fig.suptitle(args.title, fontsize=14)
    fig.supxlabel("Single seed, validation only; intervals exclude seed and selection uncertainty. Each model uses one completion-selected checkpoint.", fontsize=8.5)
    args.output_dir.mkdir(parents=True)
    for extension in ["png", "svg"]:
        fig.savefig(args.output_dir / f"comparison.{extension}", dpi=160)
    plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()

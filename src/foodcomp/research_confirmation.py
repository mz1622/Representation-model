"""Three-seed confirmation without treating seeds/cells as independent foods."""
import numpy as np
from .research_statistics import paired_interval, paired_axis_intervals

SEEDS = (20260922, 20260923, 20260924)
METRICS = ("scaled_log_mae", "log_mae")
KEYS = ["exact_name_group_id", "axis_index"]


def seed_average_errors(runs, axes):
    """Average already-scored errors, never raw predictions (no ensemble)."""
    if set(runs) != set(SEEDS):
        raise ValueError("Exactly the three registered distinct seeds are required.")
    arrays = []; reference = None; per_seed = []
    for seed in SEEDS:
        frame = runs[seed]
        frame = frame[frame.axis_index.isin(axes)][KEYS + list(METRICS)].sort_values(KEYS).reset_index(drop=True)
        if frame.duplicated(KEYS).any() or set(frame.axis_index) != set(axes):
            raise ValueError("Duplicate cells or missing registered axis support.")
        if reference is None:reference = frame[KEYS].copy()
        elif not reference.equals(frame[KEYS]):raise ValueError("Seeds must share identical candidate/axis coverage.")
        values = frame[list(METRICS)].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid confirmation errors.")
        arrays.append(values)
        per_seed.append({"seed": seed, **frame.groupby("axis_index")[list(METRICS)].mean().mean().to_dict()})
    averaged = reference.copy()
    averaged[list(METRICS)] = np.mean(arrays, axis=0)
    summary = {metric: {"mean": float(np.mean([r[metric] for r in per_seed])),
                        "sample_std": float(np.std([r[metric] for r in per_seed], ddof=1)),
                        "min": float(min(r[metric] for r in per_seed)),
                        "max": float(max(r[metric] for r in per_seed))} for metric in METRICS}
    return averaged, {"per_seed": per_seed, "metrics": summary}


def confirm_completion(runs, axes, *, repeats=1000):
    if set(runs) != {"candidate", "rf", "xgb"}:
        raise ValueError("Candidate, RF and XGBoost must all be confirmed.")
    frames = {}; summaries = {}
    for role, seeded in runs.items():
        frames[role], summaries[role] = seed_average_errors(seeded, axes)
        if role in {"rf", "xgb"} and any(row[metric] <= 0 for row in summaries[role]["per_seed"] for metric in METRICS):
            raise ValueError("A zero-error baseline has undefined relative improvement; report absolute errors separately.")
    stronger = min(("rf", "xgb"), key=lambda role: summaries[role]["metrics"]["scaled_log_mae"]["mean"])
    intervals = {metric: paired_interval(frames[stronger], frames["candidate"], axes,
                    metric=metric, repeats=repeats, seed=SEEDS[0]) for metric in METRICS}
    # Check both trees' coverage even when only the stronger is the primary reference.
    comparisons = {}
    for role in ("rf", "xgb"):
        if not frames[role][KEYS].equals(frames["candidate"][KEYS]):
            raise ValueError("Tree and neural candidate/axis panels differ.")
        pairs = []
        for base, candidate in zip(summaries[role]["per_seed"], summaries["candidate"]["per_seed"]):
            pairs.append({"seed": base["seed"], **{metric: 1-candidate[metric]/base[metric] for metric in METRICS}})
        comparisons[role] = {"per_seed_relative_improvement": pairs,
            "relative_improvement_sample_std": {metric: float(np.std([p[metric] for p in pairs], ddof=1)) for metric in METRICS}}
    primary = intervals["scaled_log_mae"]; legacy = intervals["log_mae"]
    gates = {"three_registered_seeds_for_all_methods": True,
             "primary_mean_gain_at_least_5_percent": primary["relative_improvement"] >= .05,
             "food_group_interval_supports_improvement": primary["relative_improvement_95_interval"][0] > 0,
             "legacy_mean_regression_at_most_2_percent": legacy["relative_improvement"] >= -.02}
    result = {"seeds": list(SEEDS), "stronger_tree": stronger, "seed_summaries": summaries,
        "paired_seed_comparisons": comparisons, "food_group_intervals": intervals, "gates": gates,
        "conditional_completion_milestone_passed": all(gates.values()),
        "aggregation": "Score each model independently. Average errors across the same three seeds, then resample whole food groups once for all seeds and axes. No prediction ensemble and no best-seed selection.",
        "uncertainty": "Report seed sample SD separately. Food-group intervals condition on these three trained models and do not include model selection, label validity or population seed uncertainty.",
        "claim_scope": "Conditional completion milestone only. Does not certify label provenance, unseen-source generalization, name-only/retrieval or foundation-model status.",
        "complete_test_opened": False}
    per_axis = paired_axis_intervals(frames[stronger], frames["candidate"], repeats=repeats, seed=SEEDS[0])
    return result, per_axis

"""Three neural seeds versus genuinely fixed references, without an ensemble."""
import numpy as np

from .research_confirmation import SEEDS
from .research_final_statistics import CELL_KEYS, ERRORS, mean_seed_frames, number_summary
from .research_statistics import paired_interval, paired_axis_intervals


def validate_score_panel(frame, axes):
    selected = frame[frame.axis_index.isin(axes)][CELL_KEYS + ERRORS].sort_values(CELL_KEYS).reset_index(drop=True)
    if (not len(selected) or selected[CELL_KEYS].isna().any().any()
            or selected.duplicated(CELL_KEYS).any() or set(selected.axis_index) != set(axes)):
        raise ValueError('Invalid, duplicate or incomplete score panel.')
    values = selected[ERRORS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative model errors.')
    return selected


def acceptance_gates(primary, legacy):
    """Registered R9 threshold is positive improvement over RF, not 5% or XGB."""
    numbers = [primary['relative_improvement'], *primary['relative_improvement_95_interval'],
               legacy['relative_improvement']]
    if not np.isfinite(numbers).all():
        raise ValueError('Acceptance evidence must be finite.')
    return {'three_registered_neural_seeds': True,
            'primary_mean_improvement_over_fixed_rf': primary['relative_improvement'] > 0,
            'food_group_interval_supports_improvement': primary['relative_improvement_95_interval'][0] > 0,
            'legacy_mean_regression_at_most_2_percent': legacy['relative_improvement'] >= -.02}


def confirm_fixed_references(candidate_runs, references, axes, *, repeats=1000):
    if set(candidate_runs) != set(SEEDS):
        raise ValueError('Exactly the three registered neural seeds are required.')
    if set(references) != {'rf', 'xgb'}:
        raise ValueError('Exactly the frozen RF and XGBoost references are required.')
    if not len(axes) or len(set(axes)) != len(axes):
        raise ValueError('Distinct registered axes are required.')
    candidates = {seed: validate_score_panel(frame, axes) for seed, frame in candidate_runs.items()}
    average = mean_seed_frames(candidates, CELL_KEYS, ERRORS)
    per_seed = [{'seed': seed, **candidates[seed].groupby('axis_index')[ERRORS].mean().mean().to_dict()}
                for seed in SEEDS]
    seed_summary = {metric: number_summary([row[metric] for row in per_seed]) for metric in ERRORS}
    comparisons, axis_intervals = {}, {}
    for role, frame in references.items():
        base = validate_score_panel(frame, axes)
        if not base[CELL_KEYS].equals(average[CELL_KEYS]):
            raise ValueError('Reference and neural panels differ.')
        points = base.groupby('axis_index')[ERRORS].mean().mean().to_dict()
        if any(points[metric] <= 0 for metric in ERRORS[:2]):
            raise ValueError('Relative improvement requires a positive reference error.')
        intervals = {metric: paired_interval(base, average, axes, metric=metric,
            repeats=repeats, seed=SEEDS[0]) for metric in ERRORS[:2]}
        for interval in intervals.values():
            if not np.isfinite([interval['relative_improvement'], *interval['relative_improvement_95_interval']]).all():
                raise ValueError('Nonfinite fixed-reference uncertainty.')
        comparisons[role] = {'reference_seed_count': 1, 'reference_metrics': points,
            'paired_intervals': intervals,
            'per_seed_relative_improvement': [{'seed': row['seed'], **{
                metric: 1-row[metric]/points[metric] for metric in ERRORS[:2]}} for row in per_seed]}
        axis_intervals[role] = paired_axis_intervals(base, average, repeats=repeats, seed=SEEDS[0])
    rf = comparisons['rf']['paired_intervals']
    gates = acceptance_gates(rf['scaled_log_mae'], rf['log_mae'])
    return {'seeds': list(SEEDS), 'per_seed': per_seed, 'seed_summary': seed_summary,
        'comparisons': comparisons, 'gates': gates,
        'conditional_fixed_rf_milestone_passed': all(gates.values()),
        'aggregation': 'Score each model first; average errors over all three seeds. Resample whole food groups shared across seeds and axes. No prediction ensemble or best-seed selection.',
        'scope': 'Internal validation conditional on these three trained neural models and the fixed RF/XGBoost artifacts. Food-group intervals exclude training-seed population uncertainty, tree randomness, model-selection bias and label validity. Seed sample SD is reported separately. This gate does not establish auxiliary-task or external-transfer performance.'}, axis_intervals

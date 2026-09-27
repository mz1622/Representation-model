"""Post-result coordinate counterfactual; never modifies or fits a model."""
import numpy as np
from foodcomp.research_view_geometry import geometry_summary


def translation_summary(first, second, task_weights, task_count, axis_count, weight, bias):
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    q = np.asarray(task_weights, dtype=np.float64)
    original_geometry = geometry_summary(a, b, q, task_count, axis_count)
    w = np.asarray(weight, dtype=np.float64)
    intercept = np.asarray(bias, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != a.shape[1] or intercept.shape != (len(w),):
        raise ValueError("Linear head shapes do not match representation.")
    if not np.isfinite(w).all() or not np.isfinite(intercept).all():
        raise FloatingPointError("Nonfinite linear head.")
    centre = (a.mean(0) + b.mean(0)) / 2
    variance = float((np.mean(np.sum((a-centre)**2, axis=1)) +
                      np.mean(np.sum((b-centre)**2, axis=1))) / 2)
    pair_distance = .5*np.sum((a-b)**2, axis=1)
    base_predictions = [x @ w.T + intercept for x in (a, b)]
    result = {
        "centre_norm": float(np.linalg.norm(centre)),
        "pooled_centred_second_moment": variance,
        "uniform_half_squared_pair_distance": float(pair_distance.mean()),
        "weighted_half_squared_pair_distance": float(q @ pair_distance / q.sum()),
        "weighted_pair_distance_over_centred_second_moment":
            float(q @ pair_distance / q.sum() / variance) if variance > 0 else None,
        "variance_ratio_status": "defined" if variance > 0 else "undefined_zero_variance",
        "coordinate_cases": [],
    }
    for label, multiplier in (("original", 0.), ("subtract_shared_train_mean", -1.),
                              ("add_ten_shared_train_means", 10.)):
        shift = multiplier * centre
        shifted = [x + shift for x in (a, b)]
        compensated_bias = intercept - w @ shift
        differences = [np.max(np.abs(x @ w.T + compensated_bias - reference))
                       for x, reference in zip(shifted, base_predictions)]
        pair_error = float(np.max(np.abs((shifted[0]-shifted[1]) - (a-b))))
        if not np.isfinite(differences).all() or max(differences) > 1e-10 or pair_error > 1e-10:
            raise AssertionError("Prediction/pair difference was not preserved within float64 tolerance.")
        geometry = (original_geometry if multiplier == 0 else
                    geometry_summary(*shifted, q, task_count, axis_count))
        result["coordinate_cases"].append({
            "case": label, "mean_multiplier": multiplier, "geometry": geometry,
            "max_abs_prediction_error_A": float(differences[0]),
            "max_abs_prediction_error_B": float(differences[1]),
            "max_abs_pair_vector_error": pair_error,
        })
    return result

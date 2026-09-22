"""Training-only descriptive distributions; never selects a new normalizer."""

import numpy as np
import pandas as pd


QUANTILES = {"min": 0, "p01": .01, "p05": .05, "q1": .25, "median": .5,
             "q3": .75, "p95": .95, "p99": .99, "max": 1}
SCALE_EPS = 1e-8


def distribution_statistics(values):
    """SD uses n-1; population SD is additionally reported for scale comparison."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Statistics require a finite one-dimensional observed-value array")
    keys = [*QUANTILES, "mean", "sd", "sd_population", "iqr", "mad", "skewness",
            "bowley_skewness", "tukey_outside_pct", "top_one_percent_variance_share_pct"]
    if not len(values):
        return {key: np.nan for key in keys}
    result = dict(zip(QUANTILES, np.quantile(values, list(QUANTILES.values()), method="linear")))
    constant = result['min'] == result['max']
    # Avoid treating rounding noise in the mean of a constant float vector as spread.
    result["mean"] = float(values[0] if constant else values.mean())
    result["sd"] = (0.0 if constant else float(values.std(ddof=1))) if len(values) > 1 else np.nan
    result["sd_population"] = 0.0 if constant else float(values.std(ddof=0))
    result["iqr"] = result["q3"] - result["q1"]
    result["mad"] = float(np.median(np.abs(values - result["median"])))
    sd = result["sd_population"]
    n = len(values)
    result["skewness"] = (float(np.sqrt(n * (n - 1)) / (n - 2) * np.mean(((values - result["mean"]) / sd) ** 3))
                          if n >= 3 and sd > 0 else np.nan)
    result["bowley_skewness"] = ((result["q1"] + result["q3"] - 2 * result["median"]) / result["iqr"]
                                  if result["iqr"] > 0 else np.nan)
    outside = (values < result["q1"] - 1.5 * result["iqr"]) | (values > result["q3"] + 1.5 * result["iqr"])
    result["tukey_outside_pct"] = float(100 * outside.mean())
    residual_squares = (values - result["mean"]) ** 2
    result["top_one_percent_variance_share_pct"] = (
        float(100 * residual_squares[np.argsort(values)[-max(1, int(np.ceil(.01 * n))):]].sum() / residual_squares.sum())
        if residual_squares.sum() > 0 else np.nan)
    return result


def summarize_compositions(values, foods, components):
    """One record per retained axis, with all-observed and positive-only summaries."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(foods), len(components)):
        raise ValueError("Food/axis metadata do not align with the value matrix")
    if not components.component_concept_id.is_unique:
        raise ValueError("Composition IDs must be unique")
    rows = foods.partition.eq("train").to_numpy()
    if not rows.any():
        raise ValueError("No training foods available")
    train = values[rows]
    observed = np.isfinite(train)
    if np.isinf(train).any() or (train[observed] < 0).any() or (train[observed] > 100).any():
        raise ValueError("Training values must be missing or finite masses in [0, 100]")
    result = []
    for j, axis in enumerate(components.to_dict("records")):
        x = train[observed[:, j], j]
        positive = x[x > 0]
        record = {key: axis[key] for key in ("component_concept_id", "canonical_name", "display_id",
                  "classification_category", "classification_category_en", "training_role", "prediction_stage") if key in axis}
        record.update(partition="train", n_foods=len(train), n_observed=len(x), n_missing=len(train) - len(x),
                      observed_pct=100 * len(x) / len(train), n_zero=int((x == 0).sum()), n_positive=len(positive),
                      zero_pct_observed=100 * (x == 0).mean() if len(x) else np.nan, n_unique=len(np.unique(x)))
        for prefix, data in (("raw", x), ("log", np.log1p(x)), ("positive_raw", positive), ("positive_log", np.log1p(positive))):
            record.update({f"{prefix}_{key}": value for key, value in distribution_statistics(data).items()})
        robust = max(record["log_iqr"] / 1.349, 1.4826 * record["log_mad"])
        fallback = not np.isfinite(robust) or robust <= SCALE_EPS
        record["robust_scale_unfloored"] = robust
        record["current_scale_fallback"] = fallback
        record["current_effective_scale"] = 1.0 if fallback else robust
        sd = record["log_sd_population"]
        record["log_abs_mean_median_gap_sd"] = abs(record["log_mean"] - record["log_median"]) / sd if sd > 0 else np.nan
        record["log_sd_over_robust"] = sd / robust if not fallback else np.nan
        record["robust_vs_sd_squared_loss_weight"] = (sd / robust) ** 2 if not fallback else np.nan
        record["current_log_z_min"] = (record["log_min"] - record["log_median"]) / record["current_effective_scale"]
        record["current_log_z_max"] = (record["log_max"] - record["log_median"]) / record["current_effective_scale"]
        flags = []
        if fallback:
            flags.append("robust_scale_fallback")
        if record["zero_pct_observed"] >= 50:
            flags.append("zero_majority")
        if record["log_skewness"] > 1:
            flags.append("log_right_skew_gt_1")
        if record["log_skewness"] < -1:
            flags.append("log_left_skew_lt_minus_1")
        if record["log_sd_over_robust"] >= 5:
            flags.append("sd_over_robust_ge_5")
        record["review_flags"] = ";".join(flags)
        result.append(record)
    return pd.DataFrame(result)

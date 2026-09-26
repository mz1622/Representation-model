"""Versioned, source-free research data and evaluation contracts.

Frozen V8 files are read only. Test records are never emitted into this view.
All learners consume the same raw-space cell labels and family-hidden context.
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

VERSION = "foodnutrigpt_v9_r0_v1"
FROZEN = "global_foodnutrigpt_v8_single_stage_v2_complete_test"
SEEDS = [20260922, 20260923, 20260924]
POLICY = {
    "version": VERSION, "primary_axes": "nutrition and loss_eligible (142 expected)",
    "primary_metric": "macro_axis_train_typical_concentration_scaled_log_mae",
    "transform": "log1p(g_per_100g / train_source_equal_positive_weighted_median)",
    "cell_target": "median in raw g/100g within profile/axis; never median in log space",
    "scoring": "median profiles within candidate/axis/source, mean source errors within candidate/axis, mean candidates within axis, macro axes",
    "text": "original_name only; no enriched food_text, source, group or other fields",
    "context": "all observed axes outside target chemical family; no token truncation",
    "quarantine": "whole profile/axis cell if positive max/min is 1000 or 1000000 within log10 tolerance 1e-6",
    "quarantine_interpretation": "suspected scale inconsistency, not a proven unit error; neither value is corrected",
    "missing": "NaN/absent; explicit zero remains observed",
    "test_opened": False, "seeds": SEEDS,
    "max_candidates_per_round": 12, "confirmation_seeds": 3,
    "round_wall_hours": [48, 72],
    "required_relative_primary_gain": 0.05, "allowed_legacy_nutrition_log_mae_regression": 0.02,
}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+"\n", encoding="utf-8")

def weighted_median(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if v.shape != w.shape or not len(v) or not np.isfinite(v).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Weighted median requires finite values and strictly positive aligned weights.")
    order = np.argsort(v, kind="stable")
    return float(v[order][np.searchsorted(np.cumsum(w[order]), w.sum()/2, side="left")])

def scale_conflict(low, high):
    low, high = np.asarray(low, float), np.asarray(high, float)
    ratio_log = np.log10(np.divide(high, low, out=np.ones_like(high), where=low > 0))
    return (low > 0) & (np.isclose(ratio_log, 3, atol=1e-6, rtol=0) | np.isclose(ratio_log, 6, atol=1e-6, rtol=0))

def cell_weights(cells: pd.DataFrame) -> np.ndarray:
    """One total weight per candidate/axis, equal sources then profiles."""
    keys = ["exact_name_group_id", "axis_index"]
    ns = cells.groupby(keys)["source_key"].transform("nunique").to_numpy(float)
    np_ = cells.groupby(keys+["source_key"])["profile_id"].transform("nunique").to_numpy(float)
    return 1.0 / (ns*np_)

def canonical_cells(tokens: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    t = tokens.copy()
    y = t["normalized_value_g_per_100g"].to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Nonfinite or negative source label.")
    t["_log"] = np.log1p(y)
    t["_positive"] = np.where(y > 0, y, np.nan)
    cells = t.groupby(["profile_id", "axis_index"], sort=True).agg(
        value=("normalized_value_g_per_100g", "median"),
        old_rf_log_median=("_log", "median"),
        observations=("measurement_id", "size"),
        minimum=("normalized_value_g_per_100g", "min"),
        maximum=("normalized_value_g_per_100g", "max"),
        positive_minimum=("_positive", "min"),
    ).reset_index()
    cells["aggregation_gap"] = np.abs(cells.value-np.expm1(cells.old_rf_log_median))
    cells["quarantined"] = scale_conflict(cells.positive_minimum, cells.maximum)
    cells["zero_positive_conflict"] = cells.minimum.eq(0) & cells.maximum.gt(0)
    cells = cells.merge(profiles[["profile_id", "partition", "source_key", "exact_name_group_id", "profile_index"]],
                        on="profile_id", validate="many_to_one")
    return cells

def typical_scales(cells, axis_count):
    positive = cells[cells.partition.eq("train") & cells.value.gt(0)].copy()
    positive["weight"] = cell_weights(positive)
    scale = np.ones(axis_count, dtype=np.float64)
    support = np.zeros(axis_count, dtype=np.int64)
    for axis, group in positive.groupby("axis_index"):
        scale[axis] = weighted_median(group.value, group.weight)
        support[axis] = group.exact_name_group_id.nunique()
    return scale, support

def build_view(repo: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output/"protocol.json", POLICY)  # freeze before looking at model results
    data = repo/"data/processed"/FROZEN
    split = repo/"data/splits"/FROZEN/"splits.json"
    release_path = repo/"data/release_manifests"/f"{FROZEN}.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    frozen_hashes = {}
    for entry in release["files"]:
        folder, name = entry["path"].split("/", 1)
        path = data/name if folder == "data" else split
        actual = digest(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Frozen checksum mismatch: {path}")
        frozen_hashes[entry["path"]] = actual
    p = pd.read_csv(data/"food_profiles.csv.gz", low_memory=False)
    p = p[p.partition.isin(["train", "validation"])].sort_values("profile_id").reset_index(drop=True)
    p["profile_index"] = np.arange(len(p))
    s = json.loads(split.read_text(encoding="utf-8"))
    for part in ["train", "validation"]:
        if set(p.loc[p.partition.eq(part), "profile_id"]) != set(s[part]):
            raise ValueError("Profile partitions do not exactly match the frozen split.")
    if p.groupby("exact_name_group_id").partition.nunique().max() != 1:
        raise ValueError("Train/validation exact-name group overlap.")
    axes = pd.read_csv(data/"axis_registry.csv").sort_values("axis_index").reset_index(drop=True)
    if not np.array_equal(axes.axis_index, np.arange(len(axes))):
        raise ValueError("Axis indices must be contiguous.")
    ids = set(p.profile_id)
    columns = ["profile_id", "axis_index", "measurement_id", "component_observation_id",
               "raw_unit", "raw_basis", "source_value_origin", "value_status",
               "normalized_value_g_per_100g", "mask_family"]
    parts = []
    for chunk in pd.read_csv(data/"source_native_axis_tokens.csv.gz", usecols=columns, chunksize=250000, low_memory=False):
        parts.append(chunk[chunk.profile_id.isin(ids)])  # test filtered before analysis
    t = pd.concat(parts, ignore_index=True)
    if not t.value_status.isin(["observed", "explicit_zero"]).all():
        raise ValueError("Censored or unknown values entered the research view.")
    if not t.groupby("axis_index").mask_family.nunique().eq(1).all():
        raise ValueError("Axis has ambiguous masking-family membership.")
    cells = canonical_cells(t, p)
    primary = cells[~cells.quarantined].copy()
    scales, support = typical_scales(primary, len(axes))
    if (support[axes.loss_eligible.to_numpy(bool)] == 0).any():
        raise ValueError("A target axis lacks a positive training scale after quarantine.")
    axes["research_scale"] = scales
    axes["positive_train_candidate_support"] = support
    axes.to_csv(output/"axes.csv", index=False)
    p.to_csv(output/"profiles.csv.gz", index=False)
    cells.to_parquet(output/"canonical_cells.parquet", index=False)
    flagged = cells[cells.quarantined]
    evidence = t.merge(flagged[["profile_id", "axis_index"]], on=["profile_id", "axis_index"], how="inner")
    evidence = evidence.merge(p[["profile_id", "original_name", "source_key", "partition"]], on="profile_id")
    evidence.to_csv(output/"quarantine_evidence.csv.gz", index=False)
    cells[cells.aggregation_gap.gt(1e-10)].to_csv(output/"aggregation_disagreements.csv.gz", index=False)
    shape = (len(p), len(axes))
    matrices = {}
    for name, frame in [("inclusive", cells), ("quarantined", primary)]:
        raw = np.full(shape, np.nan, dtype=np.float64)
        raw[frame.profile_index, frame.axis_index] = frame.value
        weights = np.zeros(shape, dtype=np.float32)
        train = frame[frame.partition.eq("train")].copy()
        weights[train.profile_index, train.axis_index] = cell_weights(train)
        matrices[name] = raw
        np.savez_compressed(output/f"{name}.npz", values=raw, train_weights=weights)
    source_summary = cells.groupby(["partition", "source_key"]).agg(
        cells=("value", "size"), quarantined_cells=("quarantined", "sum"),
        zero_positive_cells=("zero_positive_conflict", "sum"),
        changed_aggregation_cells=("aggregation_gap", lambda v: int((v > 1e-10).sum())),
        raw_max=("maximum", "max"),
    ).reset_index()
    source_summary.to_csv(output/"source_audit.csv", index=False)
    for view, raw in matrices.items():
        validation = np.flatnonzero(p.partition.eq("validation"))
        r, a = np.where(np.isfinite(raw[validation]) & axes.loss_eligible.to_numpy(bool)[None, :])
        jobs = pd.DataFrame({"profile_index": validation[r], "axis_index": a})
        jobs["target"] = raw[jobs.profile_index, jobs.axis_index]
        jobs["mask_family"] = axes.mask_family.to_numpy()[a]
        jobs.to_parquet(output/f"{view}_validation_jobs.parquet", index=False)
    train_cells = cells[cells.partition.eq("train")]
    axis_stats = train_cells.merge(axes[["axis_index", "canonical_name", "loss_group", "loss_eligible"]], on="axis_index")
    axis_stats.groupby(["axis_index", "canonical_name", "loss_group", "loss_eligible"]).agg(
        profile_cells=("value", "size"), zero_fraction=("value", lambda x: float(x.eq(0).mean())),
        duplicate_cells=("observations", lambda x: int((x>1).sum())),
        quarantine_cells=("quarantined", "sum"),
        aggregation_gap_cells=("aggregation_gap", lambda x: int((x>1e-10).sum())),
        min_value=("value", "min"), median_value=("value", "median"), max_value=("value", "max"),
    ).to_csv(output/"train_axis_audit.csv")
    counts = t.groupby("profile_id").size()
    unit_check = __import__("foodcomp.schema", fromlist=["normalize_unit"]).normalize_unit
    manifest = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_input_hashes": frozen_hashes, "protocol_sha256": digest(output/"protocol.json"),
        "profiles": len(p), "train_profiles": int(p.partition.eq("train").sum()),
        "validation_profiles": int(p.partition.eq("validation").sum()),
        "source_tokens": len(t), "canonical_cells": len(cells),
        "quarantine_cells": len(flagged), "quarantine_observations": len(evidence),
        "training_quarantine_cells": int(train_cells.quarantined.sum()),
        "aggregation_disagreement_train_cells": int(train_cells.aggregation_gap.gt(1e-10).sum()),
        "train_profiles_above_legacy_256_cap": int((counts.reindex(p.loc[p.partition.eq("train"), "profile_id"]) > 256).sum()),
        "mg_space_factor": unit_check("mg/100 g", "mg/100 g")["conversion_factor"],
        "mg_compact_factor": unit_check("mg/100g", "mg/100g")["conversion_factor"],
        "raw_evidence_status": "original FooDB Content.csv/staging not available; public access returned HTTP 403",
        "root_cause_status": "unresolved; current unit parser treats both spellings identically",
        "confirmation_allowed": False,
        "complete_test_opened": False,
    }
    manifest["artifact_hashes"] = {x.name: digest(x) for x in output.iterdir() if x.is_file()}
    write_json(output/"manifest.json", manifest)
    return manifest

@dataclass
class ResearchData:
    root: Path
    view: str = "quarantined"

    def __post_init__(self):
        self.root = Path(self.root)
        self.manifest = json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        for name in ["axes.csv", "profiles.csv.gz", f"{self.view}.npz", f"{self.view}_validation_jobs.parquet", "protocol.json"]:
            if digest(self.root/name) != self.manifest["artifact_hashes"][name]:
                raise ValueError(f"Stale research artifact: {name}")
        self.profiles = pd.read_csv(self.root/"profiles.csv.gz", low_memory=False)
        self.axes = pd.read_csv(self.root/"axes.csv")
        with np.load(self.root/f"{self.view}.npz", allow_pickle=False) as archive:
            self.raw = archive["values"]
            self.weights = archive["train_weights"]
        self.observed = np.isfinite(self.raw)
        self.scale = self.axes.research_scale.to_numpy(float)
        self.values = np.where(self.observed, np.log1p(self.raw/self.scale), 0).astype(np.float32)
        self.families = self.axes.mask_family.astype(str).to_numpy()
        self.targets = np.flatnonzero(self.axes.loss_eligible.to_numpy(bool))
        self.train = np.flatnonzero(self.profiles.partition.eq("train"))
        self.validation = np.flatnonzero(self.profiles.partition.eq("validation"))
        self.jobs = pd.read_parquet(self.root/f"{self.view}_validation_jobs.parquet")

    def context(self, rows, axis=None, *, mode="completion", visible_fraction=1.0):
        rows = np.asarray(rows, dtype=np.int64)
        visible = self.observed[rows].copy()
        if axis is not None:
            visible[:, self.families == self.families[axis]] = False
        if mode == "name_only":
            visible[:] = False
        elif mode != "completion":
            raise ValueError(mode)
        if visible_fraction < 1:
            if not 0 <= visible_fraction <= 1:
                raise ValueError("visible_fraction must be in [0,1].")
            # Label-independent, deterministic per profile/axis position, shared by all learners.
            rr = rows[:, None].astype(np.uint64)
            aa = np.arange(len(self.axes))[None, :].astype(np.uint64)
            code = (rr*2654435761 + aa*2246822519 + 20260922) % 4294967291
            visible &= code/4294967291 < visible_fraction
        return np.where(visible, self.values[rows], 0), visible

    def dense_features(self, rows, text, axis=None, *, mode="completion", visible_fraction=1.0):
        values, mask = self.context(rows, axis, mode=mode, visible_fraction=visible_fraction)
        return np.concatenate([text[rows], values, mask.astype(np.float32)], axis=1)

    def targets_for_jobs(self):
        return self.jobs[["profile_index", "axis_index", "target"]].copy()

def score_predictions(data: ResearchData, predictions: pd.DataFrame):
    keys = ["profile_index", "axis_index"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate prediction jobs.")
    truth = data.targets_for_jobs()
    scored = truth.merge(predictions, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("Predictions must exactly cover the frozen validation panel.")
    if not np.isfinite(scored.prediction).all() or scored.prediction.lt(0).any():
        raise ValueError("Nonfinite or negative predictions are invalid, never silently dropped.")
    if "target_y" in scored or "target_x" in scored:
        raise ValueError("Learners may not provide or replace evaluator-owned labels.")
    scored = scored.merge(data.profiles[["profile_index", "exact_name_group_id", "source_key"]],
                          on="profile_index", validate="many_to_one")
    aggregations = {"target": ("target", "median"), "prediction": ("prediction", "median")}
    if "positive_probability" in scored:
        if not scored.positive_probability.between(0, 1).all():
            raise ValueError("Presence probabilities must be finite and in [0,1].")
        aggregations["positive_probability"] = ("positive_probability", "median")
    source_cells = scored.groupby(["exact_name_group_id", "axis_index", "source_key"], as_index=False).agg(**aggregations)
    scale = data.scale[source_cells.axis_index.to_numpy()]
    y, pred = source_cells.target.to_numpy(), source_cells.prediction.to_numpy()
    source_cells["scaled_log_mae"] = np.abs(np.log1p(pred/scale)-np.log1p(y/scale))
    source_cells["log_mae"] = np.abs(np.log1p(pred)-np.log1p(y))
    source_cells["log_mse"] = (np.log1p(pred)-np.log1p(y))**2
    source_cells["raw_mae"] = np.abs(pred-y)
    source_cells["positive_scaled_log_mae"] = np.where(y > 0, source_cells.scaled_log_mae, np.nan)
    source_cells["zero_scaled_log_mae"] = np.where(y == 0, source_cells.scaled_log_mae, np.nan)
    metrics = ["scaled_log_mae", "log_mae", "log_mse", "raw_mae", "positive_scaled_log_mae", "zero_scaled_log_mae"]
    if "positive_probability" in source_cells:
        source_cells["presence_brier"] = (source_cells.positive_probability-(y>0))**2
        metrics.append("presence_brier")
    candidates = source_cells.groupby(["exact_name_group_id", "axis_index"])[metrics].mean().reset_index()
    axes = candidates.groupby("axis_index")[metrics].mean()
    axes["candidate_support"] = candidates.groupby("axis_index").size()
    axes = axes.reset_index().merge(data.axes[["axis_index", "canonical_name", "loss_group"]], on="axis_index")
    if set(axes.axis_index) != set(data.targets):
        raise ValueError("A supervised axis has no validation support.")
    result = {"complete_test_opened": False, "profile_axis_jobs": len(truth), "candidate_axis_cells": len(candidates)}
    for group in ["nutrition", "food_metabolome", "all"]:
        frame = axes if group == "all" else axes[axes.loss_group.eq(group)]
        result[group] = {"axes": len(frame)}
        for metric in metrics:
            v = frame[metric].mean()
            result[group][metric] = float(v) if np.isfinite(v) else None
        result[group]["log_rmse_root_mean_axis_mse"] = float(np.sqrt(frame.log_mse.mean()))
    return result, axes, candidates

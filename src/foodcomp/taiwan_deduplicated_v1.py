"""Shared protocol for the deduplicated Taiwan external panel."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from foodcomp.dependency_mask_analysis_v1 import attach_taiwan_panel


SHOTS = (8, 32, 128)
SEEDS = (20261027, 20261028, 20261029)
SPLIT_SEED = 20261026
TEST_PROFILES = 500


def row_hash(rows) -> str:
    values = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def attach_clean_panel(data, audit_dir: Path, overlap_ledger: Path):
    jobs, _, panel, axes = attach_taiwan_panel(data, audit_dir)
    ledger = pd.read_csv(overlap_ledger, dtype={"food_id": str})
    required = {
        "food_id", "in_strict_panel", "external_clean_eligible",
        "identity_group_key",
    }
    if not required <= set(ledger.columns):
        raise ValueError("Taiwan overlap ledger is missing required columns")
    if ledger.food_id.duplicated().any():
        raise ValueError("Taiwan overlap ledger has duplicate food IDs")
    merged = panel.merge(
        ledger[list(required)], on="food_id", how="left", validate="one_to_one"
    )
    if merged.external_clean_eligible.isna().any():
        raise ValueError("Taiwan overlap ledger does not cover the strict panel")
    clean = merged[merged.external_clean_eligible.astype(bool)].copy()
    clean_rows = np.sort(clean.profile_index.to_numpy(dtype=np.int64))
    clean_jobs = jobs[jobs.profile_index.isin(clean_rows)].reset_index(drop=True)
    if len(clean) < TEST_PROFILES + max(SHOTS):
        raise ValueError("deduplicated Taiwan panel is too small for the planned split")
    if clean_jobs.axis_index.nunique() != len(axes):
        raise ValueError("deduplicated Taiwan panel lost target-axis coverage")
    return jobs, clean_jobs, clean, panel, axes


def group_aware_split(clean: pd.DataFrame):
    """Select exactly 500 test profiles without splitting lexical identity groups."""
    groups = (
        clean.groupby("identity_group_key", sort=True).profile_index
        .apply(lambda values: np.sort(values.to_numpy(dtype=np.int64)))
    )
    keys = groups.index.to_numpy(dtype=object)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(keys))
    selected = []
    count = 0
    for position in order:
        rows = groups.iloc[int(position)]
        if count + len(rows) <= TEST_PROFILES:
            selected.extend(rows.tolist())
            count += len(rows)
        if count == TEST_PROFILES:
            break
    if count != TEST_PROFILES:
        raise ValueError(f"could not form an intact-group {TEST_PROFILES}-profile test")
    test = np.sort(np.asarray(selected, dtype=np.int64))
    adaptation = np.sort(clean.loc[
        ~clean.profile_index.isin(test), "profile_index"
    ].to_numpy(dtype=np.int64))
    key_by_row = clean.set_index("profile_index").identity_group_key
    if set(key_by_row.loc[test]) & set(key_by_row.loc[adaptation]):
        raise ValueError("Taiwan lexical identity group leaked across adaptation/test")
    return test, adaptation


def coverage_nested_order(pool, axes, observed, seed):
    """Missingness-only coverage first, then a seeded random nested order."""
    rng = np.random.default_rng(seed)
    remaining = np.asarray(pool, dtype=np.int64).copy()
    uncovered = np.ones(len(axes), dtype=bool)
    selected = []
    while uncovered.any():
        score = observed[np.ix_(remaining, axes)][:, uncovered].sum(1)
        best = score.max(initial=0)
        if best <= 0:
            raise ValueError("adaptation pool cannot cover all Taiwan axes")
        tied = np.flatnonzero(score == best)
        position = int(rng.choice(tied))
        row = int(remaining[position])
        selected.append(row)
        uncovered &= ~observed[row, axes]
        remaining = np.delete(remaining, position)
    if len(selected) > min(SHOTS):
        raise ValueError("axis coverage needs more than eight Taiwan profiles")
    return np.concatenate((np.asarray(selected), rng.permutation(remaining))), len(selected)


def split_frame(panel, clean, test_rows, adaptation_rows):
    columns = [
        "food_id", "profile_index", "external_clean_eligible",
        "identity_group_key",
    ]
    frame = panel.merge(
        clean[columns], on=["food_id", "profile_index"], how="left",
        validate="one_to_one", suffixes=("", "_clean"),
    )
    frame["external_clean_eligible"] = frame.external_clean_eligible.eq(True)
    frame["fewshot_partition"] = np.where(
        frame.profile_index.isin(test_rows), "fixed_test",
        np.where(frame.profile_index.isin(adaptation_rows), "adaptation_pool", "excluded"),
    )
    return frame


__all__ = [
    "SHOTS", "SEEDS", "SPLIT_SEED", "TEST_PROFILES", "row_hash",
    "attach_clean_panel", "group_aware_split", "coverage_nested_order",
    "split_frame",
]

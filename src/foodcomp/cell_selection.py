"""User-specified single-record precedence, not an analytical quality ranking."""

import numpy as np
import pandas as pd


SOURCE_PRIORITY = ("foodb", "usda_foundation", "usda_sr_legacy", "cnf", "frida", "ciqual")
CELL_KEYS = ["food_concept_id", "component_concept_id"]


def selection_policy():
    return {
        "policy_id": "single_source_record_v1",
        "source_priority": list(SOURCE_PRIORITY),
        "authorization": "User-confirmed FooDB > USDA Foundation > SR Legacy; remaining sources fill gaps",
        "fallback_priority": "CNF > Frida > CIQUAL; retained deterministic supplementary order",
        "within_source_tiebreak": ["quality_tier_A_B_C_D_unknown", "larger_reported_sample_count", "lexical_measurement_id"],
        "validation": "Apply existing reference eligibility, frozen observation and Foundation holdout gates before ranking",
        "numerical_output": "One existing normalized record per food-composition cell; no averaging or pooled source output",
        "interpretation": "Precedence is a dataset construction decision, not proof that a selected value is analytically superior",
    }


def select_cell_records(eligible, source_priority=SOURCE_PRIORITY):
    """Rank already-eligible records without consulting targets or model errors."""
    if not source_priority or len(source_priority) != len(set(source_priority)):
        raise ValueError("Source precedence must be nonempty and unique")
    unknown = set(eligible.source_key) - set(source_priority)
    if unknown:
        raise ValueError(f"No source precedence configured for: {sorted(unknown)}")
    if eligible.measurement_id.duplicated().any():
        raise ValueError("Duplicate measurement IDs in cell selection")
    values = pd.to_numeric(eligible.normalized_value_g_per_100g, errors="raise")
    if not (np.isfinite(values) & values.between(0, 100)).all():
        raise ValueError("Single-record selection requires finite mass values in [0, 100]")
    ranked = eligible.assign(
        _source_rank=eligible.source_key.map({s: i for i, s in enumerate(source_priority)}),
        _quality_rank=eligible.quality_tier.map({"A": 0, "B": 1, "C": 2, "D": 3}).fillna(99),
        _sample_rank=pd.to_numeric(eligible.sample_count, errors="coerce").fillna(0).clip(lower=0),
    ).sort_values(["_source_rank", "_quality_rank", "_sample_rank", "measurement_id"],
                  ascending=[True, True, False, True], kind="stable")
    winners = ranked.drop_duplicates(CELL_KEYS).drop(columns=["_source_rank", "_quality_rank", "_sample_rank"])
    audit = eligible.groupby(CELL_KEYS).agg(
        candidate_record_count=("measurement_id", "size"),
        candidate_source_count=("source_key", "nunique"),
        candidate_min_g_per_100g=("normalized_value_g_per_100g", "min"),
        candidate_max_g_per_100g=("normalized_value_g_per_100g", "max"),
    ).reset_index()
    audit["candidate_value_disagreement"] = audit.candidate_min_g_per_100g.ne(audit.candidate_max_g_per_100g)
    audit = audit.merge(winners[CELL_KEYS + ["measurement_id", "source_key"]].rename(columns={
        "measurement_id": "selected_measurement_id", "source_key": "selected_source_key"}),
        on=CELL_KEYS, validate="one_to_one")
    audit["selection_policy"] = "single_source_record_v1"
    return winners, audit

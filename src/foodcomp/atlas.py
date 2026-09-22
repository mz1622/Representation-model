"""Build a static, provenance-first Food Composition Atlas.

The Atlas is an audit interface over immutable source-level data. It does not
pool measurements, choose a preferred source value, merge food identities, or
turn missing values into zeros.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .util import normalize_text, require_columns, sha256_file, write_csv, write_json


ATLAS_VERSION = "food_composition_atlas_v7"
DEFAULT_TOP_K = 10
ACQUIRED_SOURCE_REGISTRY = Path("data/raw/source_acquisition_2026_09_16/acquired_source_registry.csv")

# Staging adapters retain their concise local keys. The source registry uses
# release-specific keys. This map changes display/provenance joins only; it
# never identifies, merges, or pools food records or measurements.
STAGING_TO_REGISTRY_KEY = {
    "afcd": "afcd_release_3",
    "ciqual": "ciqual_2025",
    "cofid": "cofid_2021",
    "fndds": "usda_fndds_2021_2023",
    "foodb": "foodb_2020",
    "frida": "frida_6_1",
    "norway": "norwegian_fcdb",
}


def _as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _load_source_registry(project_root: Path, registry_path: Path) -> pd.DataFrame:
    """Combine the immutable global registry with versioned acquired releases."""
    registry = pd.read_csv(registry_path, keep_default_na=False)
    supplement_path = project_root / ACQUIRED_SOURCE_REGISTRY
    if supplement_path.exists():
        supplement = pd.read_csv(supplement_path, keep_default_na=False)
        registry = pd.concat([registry, supplement], ignore_index=True, sort=False)
    if registry.source_key.duplicated().any():
        duplicate = registry.loc[registry.source_key.duplicated(keep=False), "source_key"].tolist()
        raise ValueError(f"Source registry keys must be unique; duplicates={duplicate}")
    return registry


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _records(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    return [{key: _json_value(value) for key, value in row.items()} for row in selected.to_dict("records")]


def _shard_key(identifier: str) -> str:
    """Use the stable identifier digest prefix for deterministic evidence shards."""
    digest = identifier.rsplit(":", 1)[-1]
    if len(digest) < 2 or not re.fullmatch(r"[0-9a-fA-F]+", digest):
        raise ValueError(f"Expected stable hexadecimal identifier, received {identifier!r}")
    return digest[:2].lower()


def _join_values(values: Iterable[Any]) -> str:
    cleaned = sorted({_as_text(value) for value in values if _as_text(value)})
    return "; ".join(cleaned)


def _safe_quantile(series: pd.Series, q: float) -> float | None:
    if series.empty:
        return None
    return float(series.quantile(q))


def _use_registry_source_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Use release-stable registry keys in source-level Atlas summaries."""
    result = frame.copy()
    result["source_key"] = result.source_key.replace(STAGING_TO_REGISTRY_KEY)
    return result


def _prepare_axis_registry(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_axis_id",
        "canonical_name",
        "original_axis_names",
        "axis_family",
        "chemical_identity",
        "food_composition_role",
        "nutritional_role",
        "measurement_modality_required",
        "selection_tier",
        "recommended_training_stage",
        "aliases_for_review",
        "direct_prediction_target",
        "numeric_pooling_permitted",
        "notes",
    }
    require_columns(panel, required, "composition-axis registry")
    if panel.target_axis_id.duplicated().any() or panel.canonical_name.duplicated().any():
        raise ValueError("The Atlas requires unique composition-axis IDs and canonical names.")
    result = panel.copy()
    result["search_terms"] = result.apply(
        lambda row: _join_values(
            [row.canonical_name, row.original_axis_names, row.aliases_for_review, row.target_axis_id]
        ),
        axis=1,
    )
    # Training-stage and target-policy fields belong to the modeling plan, not
    # the read-only scientific Atlas. Keep them in the immutable input registry
    # while omitting them from the public composition-axis data product.
    result = result.drop(
        columns=[
            "recommended_training_stage",
            "direct_prediction_target",
            "numeric_pooling_permitted",
        ]
    )
    return result


def _build_source_table(registry: pd.DataFrame, ingestion: pd.DataFrame) -> pd.DataFrame:
    require_columns(registry, {"source_key", "name", "official_url", "license_or_terms"}, "source registry")
    require_columns(
        ingestion,
        {
            "source_key",
            "source_name",
            "region",
            "countries_or_coverage",
            "access_status",
            "ingestion_status",
            "raw_food_observations_loaded",
            "mapped_numeric_measurements_retained",
        },
        "source ingestion ledger",
    )
    right = ingestion.rename(columns={"source_name": "ingestion_source_name"})
    result = registry.merge(
        right,
        how="outer",
        on="source_key",
        validate="one_to_one",
        suffixes=("_registry", "_ingestion"),
    )
    result["display_name"] = result["name"].fillna(result["ingestion_source_name"])
    for field in ("region", "countries_or_coverage", "access_status", "source_version"):
        registry_field = f"{field}_registry"
        ingestion_field = f"{field}_ingestion"
        if registry_field in result and ingestion_field in result:
            result[field] = result[registry_field].replace("", np.nan).fillna(result[ingestion_field])
        elif registry_field in result:
            result[field] = result[registry_field]
        elif ingestion_field in result:
            result[field] = result[ingestion_field]
    result["source_status"] = np.where(
        result["ingestion_status"].fillna("").str.startswith("integrated"),
        "numerically_integrated",
        "registered_or_pending",
    )
    return result.sort_values(["region", "display_name"], na_position="last").reset_index(drop=True)


def _axis_measurement_summary(direct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and source-specific direct-mass summaries without pooling values."""
    required = {
        "target_axis_id",
        "source_key",
        "exact_name_group_id",
        "normalized_value_g_per_100g",
        "value_status",
    }
    require_columns(direct, required, "direct mass measurements")
    usable = direct.loc[direct.normalized_value_g_per_100g.notna()].copy()
    usable["is_positive"] = usable.normalized_value_g_per_100g.gt(0)
    usable["is_explicit_zero"] = usable.value_status.eq("explicit_zero")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        positive = group.loc[group.is_positive, "normalized_value_g_per_100g"]
        values = group["normalized_value_g_per_100g"]
        return pd.Series(
            {
                "direct_measurement_count": int(len(group)),
                "food_group_count": int(group.exact_name_group_id.nunique()),
                "source_count": int(group.source_key.nunique()),
                "positive_value_count": int(len(positive)),
                "explicit_zero_count": int(group.is_explicit_zero.sum()),
                "minimum_g_per_100g": None if values.empty else float(values.min()),
                "maximum_g_per_100g": None if values.empty else float(values.max()),
                "mean_g_per_100g": None if values.empty else float(values.mean()),
                "median_g_per_100g": _safe_quantile(values, 0.5),
                "q1_g_per_100g": _safe_quantile(values, 0.25),
                "q3_g_per_100g": _safe_quantile(values, 0.75),
                "sd_g_per_100g": None if len(values) < 2 else float(values.std(ddof=1)),
                "positive_minimum_g_per_100g": None if positive.empty else float(positive.min()),
                "positive_maximum_g_per_100g": None if positive.empty else float(positive.max()),
                "positive_median_g_per_100g": _safe_quantile(positive, 0.5),
                "positive_mean_g_per_100g": None if positive.empty else float(positive.mean()),
            }
        )

    overall = usable.groupby("target_axis_id", sort=False, group_keys=False).apply(aggregate).reset_index()
    by_source = (
        usable.groupby(["target_axis_id", "source_key"], sort=False, group_keys=False)
        .apply(aggregate)
        .reset_index()
    )
    return overall, by_source


def _food_summary(mapped: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """Summarize reported mappings separately from model-eligible labels."""
    required = {"exact_name_group_id", "target_axis_id", "source_key", "measurement_id"}
    require_columns(mapped, required, "mapped numeric measurements")
    require_columns(direct, required, "direct mass measurements")
    mapped_summary = (
        mapped.groupby("exact_name_group_id", as_index=False)
        .agg(
            mapped_measurement_count=("measurement_id", "size"),
            mapped_axis_count=("target_axis_id", "nunique"),
            mapped_source_count=("source_key", "nunique"),
        )
        .reset_index(drop=True)
    )
    direct_summary = (
        direct.groupby("exact_name_group_id", as_index=False)
        .agg(
            direct_measurement_count=("measurement_id", "size"),
            direct_axis_count=("target_axis_id", "nunique"),
            direct_source_count=("source_key", "nunique"),
        )
        .reset_index(drop=True)
    )
    return mapped_summary.merge(direct_summary, how="left", on="exact_name_group_id", validate="one_to_one")


def _build_axis_source_coverage(
    axis_registry: pd.DataFrame,
    source_axis_map: pd.DataFrame,
    mapped_by_source: pd.DataFrame,
    direct_by_source: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        source_axis_map,
        {"source_key", "target_axis_id", "mapping_status", "source_original_name", "source_axis_id"},
        "global source-axis mapping",
    )
    mapped = source_axis_map.copy()
    mapped["source_axis_presence"] = True
    map_summary = (
        mapped.groupby(["target_axis_id", "source_key"], as_index=False)
        .agg(
            source_axis_count=("source_axis_id", "nunique"),
            source_axis_names=("source_original_name", _join_values),
            mapping_statuses=("mapping_status", _join_values),
        )
    )
    result = map_summary.merge(mapped_by_source, how="outer", on=["target_axis_id", "source_key"])
    result = result.merge(direct_by_source, how="outer", on=["target_axis_id", "source_key"])
    axis_names = axis_registry.loc[:, ["target_axis_id", "canonical_name"]]
    result = result.merge(axis_names, how="left", on="target_axis_id")
    result["catalogue_status"] = np.where(
        result.source_axis_count.fillna(0).gt(0), "source_axis_present", "not_catalogued_in_mapped_snapshot"
    )
    result["numeric_status"] = np.where(
        result.mapped_numeric_measurement_count.fillna(0).gt(0),
        "mapped_numeric_values_available",
        "no_mapped_numeric_values",
    )
    result["model_label_status"] = np.where(
        result.direct_measurement_count.fillna(0).gt(0),
        "model_eligible_direct_mass_values_available",
        "no_model_eligible_direct_mass_values",
    )
    return result.sort_values(["canonical_name", "source_key"], na_position="last").reset_index(drop=True)


def _mapped_axis_source_summary(mapped: pd.DataFrame) -> pd.DataFrame:
    """Count source-reported mapped values before applying the model-label gate."""
    require_columns(
        mapped,
        {"target_axis_id", "source_key", "exact_name_group_id", "measurement_id"},
        "mapped numeric measurements",
    )
    return (
        mapped.groupby(["target_axis_id", "source_key"], as_index=False)
        .agg(
            mapped_numeric_measurement_count=("measurement_id", "size"),
            mapped_numeric_food_group_count=("exact_name_group_id", "nunique"),
        )
    )


def _mapped_axis_summary(mapped: pd.DataFrame) -> pd.DataFrame:
    """Count source-reported numeric evidence before the model-label quality gate."""
    require_columns(
        mapped,
        {"target_axis_id", "source_key", "exact_name_group_id", "measurement_id"},
        "mapped numeric measurements",
    )
    return (
        mapped.groupby("target_axis_id", as_index=False)
        .agg(
            mapped_numeric_measurement_count=("measurement_id", "size"),
            mapped_numeric_food_group_count=("exact_name_group_id", "nunique"),
            mapped_numeric_source_count=("source_key", "nunique"),
        )
    )


def _annotate_training_eligibility(mapped: pd.DataFrame) -> pd.DataFrame:
    """Attach a row-level, non-destructive explanation of training eligibility.

    Source values remain visible regardless of this classification.  The role
    describes the current model-label policy; it is neither a scientific
    judgment on the source nor an automatic promotion decision.
    """
    required = {
        "direct_mass_label_candidate",
        "value_status",
        "normalized_value_g_per_100g",
        "conversion_status",
        "source_value_origin",
        "source_note",
    }
    require_columns(mapped, required, "mapped numeric measurements")
    result = mapped.copy()
    direct = result.direct_mass_label_candidate.astype(str).str.lower().eq("true")
    value = pd.to_numeric(result.normalized_value_g_per_100g, errors="coerce")
    status = result.value_status.fillna("").astype(str)
    origin = result.source_value_origin.fillna("").astype(str)
    source_note = result.source_note.fillna("").astype(str)

    result["training_review_role"] = "needs_record_review"
    result["training_review_code"] = "unclassified_non_strict_record"
    result["training_review_note"] = (
        "Retained as source-specific evidence. The current rule set could not assign a model-label role; "
        "review the original expression, source provenance and analytical documentation."
    )

    def assign(mask: pd.Series, role: str, code: str, note: str) -> None:
        result.loc[mask, "training_review_role"] = role
        result.loc[mask, "training_review_code"] = code
        result.loc[mask, "training_review_note"] = note

    assign(
        direct,
        "strict_model_label",
        "direct_mass_quality_gate_passed",
        "Eligible for the current strict direct-mass model-label set: an observed source value with an exact "
        "fresh-weight mass conversion passed the present provenance and quality gate.",
    )

    non_strict = ~direct
    status_notes = {
        "reported_zero_unresolved": (
            "evidence_only_value_semantics",
            "reported_zero_unresolved",
            "The source reports zero, but its documentation does not distinguish an analytical zero from a below-limit, "
            "imputed, rounded or otherwise encoded value. Retain as evidence; do not use as a regression label.",
        ),
        "assumed_zero": (
            "evidence_only_value_semantics",
            "assumed_zero",
            "This zero was inferred or assumed rather than explicitly measured. Retain as evidence; do not use as a regression label.",
        ),
        "below_limit": (
            "evidence_only_value_semantics",
            "below_detection_or_quantification_limit",
            "This is a censored below-limit expression, not a numeric zero. Retain the source statement; do not use it as an ordinary regression label.",
        ),
        "range_only": (
            "evidence_only_value_semantics",
            "range_without_point_estimate",
            "The source provides a range without a single reported value. Retain it as evidence; do not convert it into a point regression label.",
        ),
        "trace_or_censored": (
            "evidence_only_value_semantics",
            "trace_or_censored",
            "The source reports trace or censored content. It is not equivalent to zero and is not an ordinary regression label.",
        ),
        "invalid": (
            "evidence_only_value_semantics",
            "invalid_source_numeric_expression",
            "The parsed source expression is marked invalid and needs source-level review before any modelling use.",
        ),
    }
    for source_status, (role, code, note) in status_notes.items():
        assign(non_strict & status.eq(source_status), role, code, note)

    bracketed = non_strict & status.eq("bracketed_or_parenthesized_estimate")
    assign(
        bracketed,
        "review_required_estimated_value",
        "bracketed_or_parenthesized_estimate",
        "The source presents an estimated or qualified numeric expression. Retain it visibly; require an explicit estimation policy before model-label use.",
    )

    non_mass = non_strict & ~result.conversion_status.eq("converted_exact_mass")
    assign(
        non_mass,
        "evidence_only_measurement_modality",
        "not_exact_fresh_weight_mass",
        "The expression cannot currently be verified as an exact fresh-weight mass value in g/100 g. Retain source evidence; exclude from the mass-regression target set.",
    )
    invalid_mass_range = non_strict & result.conversion_status.eq("converted_exact_mass") & ~value.between(0, 100, inclusive="both")
    assign(
        invalid_mass_range,
        "review_required_measurement_range",
        "mass_fraction_outside_physical_range",
        "The normalized mass value lies outside the current 0 to 100 g/100 g physical check. Retain source evidence; review the unit, denominator and source expression.",
    )

    provenance_candidates = non_strict & status.isin(["observed", "explicit_zero"]) & result.conversion_status.eq("converted_exact_mass") & value.between(0, 100, inclusive="both")
    afcd_analysed = provenance_candidates & origin.eq("food_level_derivation_not_individual_assay_certification") & source_note.eq(
        "profile_derivation=Analysed; food_details_derivation=Analysed"
    )
    assign(
        afcd_analysed,
        "tier_b_curated_analytical_candidate",
        "afcd_food_and_profile_marked_analysed",
        "AFCD marks both the food profile and food detail as Analysed. The value is mass-convertible and is a Tier B training candidate, but the imported release lacks a per-value assay certificate; do not use it for strict validation without source-level approval.",
    )
    source_linked_analytical = provenance_candidates & origin.eq("source_linked_analytical_project")
    assign(
        source_linked_analytical,
        "tier_b_source_linked_analytical_candidate",
        "source_linked_analytical_project",
        "The source links this food-level value to an analytical project, but the current snapshot lacks complete per-value analytical metadata. It is a Tier B training candidate pending source-level review and is not a strict-validation label.",
    )
    compiled = provenance_candidates & origin.isin(["compiled_food_level_analytical_reference", "official_harmonized_compilation"])
    assign(
        compiled,
        "review_required_compilation",
        "compiled_or_harmonized_reference",
        "This is an official compiled or harmonized food-level reference, not a confirmed independent assay in the current snapshot. Retain it visibly; verify original analytical provenance before training use.",
    )
    unresolved = provenance_candidates & origin.eq("bibliography_contradicts_independent_assay_or_remains_unresolved")
    assign(
        unresolved,
        "evidence_only_unresolved_provenance",
        "unresolved_or_conflicting_bibliography",
        "The source bibliography conflicts with independent-assay evidence or remains unresolved. Retain the value and provenance; exclude it from model labels until reviewed.",
    )
    derived = provenance_candidates & origin.isin([
        "survey_or_calculated_dish",
        "source_derived_or_calculated",
        "unreviewed_calculated_or_borrowed",
        "recipe_or_estimation",
        "recipe_calculated",
    ])
    assign(
        derived,
        "evidence_only_derived_or_recipe",
        "derived_recipe_or_borrowed_expression",
        "The source identifies this value as calculated, recipe-based, estimated, borrowed or otherwise derived. Retain it as source evidence; exclude it from the direct composition-regression target set.",
    )
    other_nonreported = provenance_candidates & origin.ne("source_reported") & result.training_review_role.eq("needs_record_review")
    assign(
        other_nonreported,
        "review_required_provenance",
        "non_source_reported_value_origin",
        "The value is mass-convertible, but its recorded provenance is not a direct source-reported measurement. Retain it visibly and review the source method before training use.",
    )
    return result


def _write_jsonl_shards(mapped: pd.DataFrame, destination: Path) -> dict[str, int]:
    """Write every mapped numeric source record into deterministic food shards.

    The display layer needs to show a source-reported value even when it is not
    eligible as a model label. Training eligibility remains a separate field.
    """
    destination.mkdir(parents=True, exist_ok=True)
    fields = [
        "measurement_id",
        "source_key",
        "food_observation_id",
        "component_observation_id",
        "target_axis_id",
        "raw_value",
        "numeric_value",
        "raw_unit",
        "raw_basis",
        "value_status",
        "normalized_value_g_per_100g",
        "conversion_factor_to_g_per_100g",
        "conversion_status",
        "measurement_modality",
        "source_measurement_locator",
        "source_value_origin",
        "direct_mass_label_candidate",
        "training_review_role",
        "training_review_code",
        "training_review_note",
        "source_note",
        "mapping_evidence",
        "mapping_status",
        "exact_name_group_id",
        "exact_food_name_key",
    ]
    require_columns(mapped, fields, "mapped numeric measurements")
    counts: dict[str, int] = defaultdict(int)
    for key, chunk in mapped.groupby(mapped.exact_name_group_id.map(_shard_key), sort=False):
        path = destination / f"{key}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in _records(chunk, fields):
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                counts[key] += 1
    return dict(sorted(counts.items()))


def _write_food_observation_shards(observations: pd.DataFrame, destination: Path) -> dict[str, int]:
    """Write source-native food provenance rows into the same stable food shards."""
    fields = [
        "food_observation_id",
        "source_key",
        "source_food_id",
        "original_name",
        "original_name_local",
        "scientific_name",
        "food_group",
        "food_subgroup",
        "food_type",
        "foodon_id",
        "foodex2_code",
        "langual_code",
        "taxonomy_id",
        "part",
        "maturity",
        "processing",
        "cooking",
        "preservation",
        "physical_state",
        "packing_medium",
        "geography",
        "cultivar",
        "recipe_status",
        "brand_status",
        "edible_status",
        "source_record_url",
        "source_lineage",
        "source_table",
        "source_row",
        "exact_food_name_key",
        "exact_name_group_id",
        "exact_name_group_status",
        "mapped_axis_count_source_record",
        "model_eligible_axis_count_source_record",
        "unmapped_axis_reason",
    ]
    require_columns(observations, fields, "food observations")
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for key, chunk in observations.groupby(observations.exact_name_group_id.map(_shard_key), sort=False):
        path = destination / f"{key}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in _records(chunk, fields):
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                counts[key] += 1
    return dict(sorted(counts.items()))


def _add_source_record_coverage(
    observations: pd.DataFrame,
    mapped: pd.DataFrame,
    direct: pd.DataFrame,
    foodb_unmapped_reasons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach source-record coverage without changing source values or identity."""
    mapped_counts = (
        mapped.groupby("food_observation_id", as_index=False)
        .agg(mapped_axis_count_source_record=("target_axis_id", "nunique"))
    )
    direct_counts = (
        direct.groupby("food_observation_id", as_index=False)
        .agg(model_eligible_axis_count_source_record=("target_axis_id", "nunique"))
    )
    result = observations.merge(mapped_counts, how="left", on="food_observation_id", validate="one_to_one")
    result = result.merge(direct_counts, how="left", on="food_observation_id", validate="one_to_one")
    for column in ("mapped_axis_count_source_record", "model_eligible_axis_count_source_record"):
        result[column] = result[column].fillna(0).astype(int)
    result["unmapped_axis_reason"] = ""
    if foodb_unmapped_reasons is not None and not foodb_unmapped_reasons.empty:
        require_columns(foodb_unmapped_reasons, {"food_observation_id", "reason"}, "FooDB unmapped-axis audit")
        reason_map = foodb_unmapped_reasons.set_index("food_observation_id")["reason"]
        result["unmapped_axis_reason"] = result.food_observation_id.map(reason_map).fillna("")
    return result


def _build_food_search_index(groups: pd.DataFrame, observations: pd.DataFrame, food_stats: pd.DataFrame) -> pd.DataFrame:
    required_groups = {
        "exact_name_group_id",
        "display_name",
        "source_count",
        "source_keys",
        "source_food_ids",
        "original_names",
        "food_group_values",
        "facet_disagreement_fields",
        "has_facet_disagreement",
    }
    require_columns(groups, required_groups, "exact name food groups")
    require_columns(observations, {"exact_name_group_id", "source_food_id", "source_key", "original_name"}, "food observations")
    result = groups.merge(food_stats, how="left", on="exact_name_group_id", validate="one_to_one")
    obs_terms = (
        observations.groupby("exact_name_group_id", as_index=False)
        .agg(
            observation_original_names=("original_name", _join_values),
            observation_source_ids=("source_food_id", _join_values),
            observation_source_keys=("source_key", _join_values),
        )
    )
    result = result.merge(obs_terms, how="left", on="exact_name_group_id", validate="one_to_one")
    result["search_terms"] = result.apply(
        lambda row: _join_values(
            [
                row.display_name,
                row.original_names,
                row.source_food_ids,
                row.observation_original_names,
                row.observation_source_ids,
                row.exact_name_group_id,
            ]
        ),
        axis=1,
    )
    result["normalized_search_terms"] = result.search_terms.map(normalize_text)
    result["result_kind"] = "food"
    return result.sort_values("display_name", kind="stable").reset_index(drop=True)


def _write_gzip_json_shards(
    frame: pd.DataFrame,
    destination: Path,
    shard_size: int = 1_500,
) -> dict[str, Any]:
    """Write a large JSON array as small gzip-compressed static shards.

    The food search index is intentionally source-rich and can exceed hosting
    providers' single-object limits. Sharding preserves the same records while
    keeping each static artifact independently transferable.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    records = _records(frame)
    for start in range(0, len(records), shard_size):
        index = start // shard_size
        name = f"{index:03d}.json.gz"
        path = destination / name
        chunk = records[start : start + shard_size]
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(chunk, handle, ensure_ascii=True, separators=(",", ":"))
        shards.append({"path": f"food_search_index/{name}", "records": len(chunk), "sha256": sha256_file(path)})
    return {"format": "gzip_json_array", "records": len(records), "shards": shards}


def _build_review_queues(
    output_dir: Path,
    source_axis_map: pd.DataFrame,
    groups: pd.DataFrame,
    conflicts: pd.DataFrame,
    mapped_measurements: pd.DataFrame,
    sources: pd.DataFrame,
    axes: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    review_dir = output_dir / "review_queues"
    review_dir.mkdir(parents=True, exist_ok=True)
    pending_mapping = source_axis_map.loc[
        ~source_axis_map.mapping_status.fillna("").eq("mapped_exact_or_registered")
    ].copy()
    pending_mapping["review_type"] = "axis_mapping"
    pending_mapping["recommended_decision"] = "needs evidence"

    identity = groups.loc[
        groups.source_count.gt(1) | groups.has_facet_disagreement.fillna(False)
    ].copy()
    identity["review_type"] = "food_identity"
    identity["recommended_decision"] = "keep separate"

    conflict_queue = conflicts.merge(
        axes.loc[:, ["target_axis_id", "canonical_name", "axis_family"]], how="left", on="target_axis_id"
    )
    conflict_queue["review_type"] = "cross_source_conflict"
    conflict_queue["recommended_decision"] = "retain all source values"

    exception_statuses = {"censored", "range_only", "trace", "unparsed_requires_review", "invalid_negative"}
    exceptions = mapped_measurements.loc[
        mapped_measurements.value_status.fillna("").isin(exception_statuses)
        | ~mapped_measurements.conversion_status.fillna("").eq("converted_exact_mass")
    ].copy()
    exceptions["review_type"] = "measurement_exception"
    exceptions["recommended_decision"] = "needs evidence"

    queue_frames = {
        "axis_mapping_review": pending_mapping,
        "food_identity_review": identity,
        "cross_source_conflict_review": conflict_queue,
        "measurement_exception_review": exceptions,
        "source_registry_review": sources.copy(),
    }
    manifest: dict[str, dict[str, Any]] = {}
    for name, frame in queue_frames.items():
        path = review_dir / f"{name}.csv"
        write_csv(frame, path)
        archive_path = review_dir / f"{name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(path, arcname=path.name)
        manifest[name] = {
            "csv": f"review_queues/{path.name}",
            "archive": f"review_queues/{archive_path.name}",
            "rows": int(len(frame)),
            "sha256": sha256_file(path),
            "archive_sha256": sha256_file(archive_path),
        }
    return manifest


def _snapshot_inputs(paths: dict[str, Path]) -> list[dict[str, str]]:
    result = []
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Atlas input {label!r} is missing: {path}")
        result.append({"label": label, "path": str(path), "sha256": sha256_file(path)})
    return result


def build_atlas(
    project_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    audit_dir: Path | None = None,
) -> Path:
    """Build a static read-only Atlas and CSV review queues from audited inputs."""
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Atlas output: {output_dir}")
        shutil.rmtree(output_dir)

    default_audit_v7 = project_root / "data/processed/global_frozen_prediction_panel_food_audit_v7"
    default_audit_v6 = project_root / "data/processed/global_frozen_prediction_panel_food_audit_v6"
    default_audit_v5 = project_root / "data/processed/global_frozen_prediction_panel_food_audit_v5"
    default_audit_v4 = project_root / "data/processed/global_frozen_prediction_panel_food_audit_v4"
    default_audit_v3 = project_root / "data/processed/global_frozen_prediction_panel_food_audit_v3"
    audit = (
        audit_dir
        or (
            default_audit_v7
            if default_audit_v7.exists()
            else default_audit_v6
            if default_audit_v6.exists()
            else default_audit_v5
            if default_audit_v5.exists()
            else default_audit_v4
            if default_audit_v4.exists()
            else default_audit_v3
            if default_audit_v3.exists()
            else project_root / "data/processed/global_frozen_prediction_panel_food_audit_v2"
        )
    ).resolve()
    global_atlas = project_root / "data/processed/global_food_metabolome_axis_atlas_v1"
    audited_panel = project_root / "data/processed/audited_prediction_axis_panel_v1"
    proposed = (
        audited_panel
        if (audited_panel / "proposed_prediction_axis_registry.csv").exists()
        else project_root / "data/processed/proposed_prediction_axis_panel_v2"
    )
    input_paths = {
        "proposed_axis_registry": proposed / "proposed_prediction_axis_registry.csv",
        "source_registry": global_atlas / "source_registry.csv",
        "source_ingestion_ledger": audit / "global_source_ingestion_ledger.csv",
        "food_groups": audit / "exact_name_food_groups.csv",
        "food_observations": audit / "food_observation_to_exact_name_group.csv.gz",
        "direct_mass_measurements": audit / "direct_mass_label_candidate_measurements.csv.gz",
        "mapped_measurements": audit / "mapped_numeric_target_measurements.csv.gz",
        "food_axis_summary": audit / "exact_name_food_axis_cell_summary_direct_mass.csv.gz",
        "source_axis_mapping": audit / "global_source_axis_to_frozen_prediction_axis.csv.gz",
        "cross_source_conflicts": audit / "cross_source_axis_conflicts_preserved.csv.gz",
        "audit_manifest": audit / "audit_manifest.json",
    }
    acquired_source_registry = project_root / ACQUIRED_SOURCE_REGISTRY
    if acquired_source_registry.exists():
        input_paths["acquired_source_registry"] = acquired_source_registry
    hierarchy_path = (
        project_root
        / "reports"
        / audit.name
        / "food_hierarchy_evidence"
        / "authority_backed_foodon_relation_candidates.csv"
    )
    if hierarchy_path.exists():
        input_paths["authority_backed_food_hierarchy"] = hierarchy_path
    foodb_reason_path = (
        project_root
        / "reports"
        / audit.name
        / "food_axis_coverage"
        / "foodb_foods_without_mapped_prediction_axes_explained.csv.gz"
    )
    if foodb_reason_path.exists():
        input_paths["foodb_unmapped_axis_reasons"] = foodb_reason_path
    input_manifest = _snapshot_inputs(input_paths)

    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / "site" / "assets" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    panel = _prepare_axis_registry(pd.read_csv(input_paths["proposed_axis_registry"], keep_default_na=False))
    retained_axis_ids = set(panel.target_axis_id.astype(str))
    registry = _load_source_registry(project_root, input_paths["source_registry"])
    ingestion = pd.read_csv(input_paths["source_ingestion_ledger"], keep_default_na=False)
    sources = _build_source_table(registry, ingestion)
    # The public Atlas is a numerical data product, not a source-acquisition
    # tracker. Sources without parsed numerical observations remain only in
    # the versioned acquisition audit and are not emitted to the site.
    sources = sources.loc[sources.source_status.eq("numerically_integrated")].copy()
    integrated_source_keys = set(sources.source_key.astype(str))
    groups = pd.read_csv(input_paths["food_groups"], keep_default_na=False)
    observations = pd.read_csv(input_paths["food_observations"], keep_default_na=False, low_memory=False)
    # Mixed source-native fields are expected. Reading in one pass prevents a
    # chunk-dependent dtype inference from changing the exported audit records.
    direct = pd.read_csv(input_paths["direct_mass_measurements"], keep_default_na=False, low_memory=False)
    mapped = pd.read_csv(input_paths["mapped_measurements"], keep_default_na=False, low_memory=False)
    mapped = mapped.loc[mapped.target_axis_id.astype(str).isin(retained_axis_ids)].copy()
    direct = direct.loc[direct.target_axis_id.astype(str).isin(retained_axis_ids)].copy()
    mapped = _annotate_training_eligibility(mapped)
    source_axis_map = _use_registry_source_keys(
        pd.read_csv(input_paths["source_axis_mapping"], keep_default_na=False)
    )
    source_axis_map = source_axis_map.loc[
        source_axis_map.source_key.astype(str).isin(integrated_source_keys)
        & source_axis_map.target_axis_id.astype(str).isin(retained_axis_ids)
    ].copy()
    conflicts = pd.read_csv(input_paths["cross_source_conflicts"], keep_default_na=False)
    hierarchy_edges = (
        pd.read_csv(hierarchy_path, keep_default_na=False)
        if hierarchy_path.exists()
        else pd.DataFrame()
    )
    foodb_unmapped_reasons = (
        pd.read_csv(foodb_reason_path, keep_default_na=False, low_memory=False)
        if foodb_reason_path.exists()
        else pd.DataFrame()
    )
    for field in ("foodon_id", "foodex2_code", "langual_code", "taxonomy_id"):
        if field not in observations:
            observations[field] = ""

    axis_overall, axis_by_source = _axis_measurement_summary(direct)
    mapped_axis_overall = _mapped_axis_summary(mapped)
    mapped_axis_by_source = _use_registry_source_keys(_mapped_axis_source_summary(mapped))
    axis_by_source = _use_registry_source_keys(axis_by_source)
    axes = panel.merge(axis_overall, how="left", on="target_axis_id", validate="one_to_one")
    axes = axes.merge(mapped_axis_overall, how="left", on="target_axis_id", validate="one_to_one")
    for column in (
        "direct_measurement_count",
        "food_group_count",
        "source_count",
        "mapped_numeric_measurement_count",
        "mapped_numeric_food_group_count",
        "mapped_numeric_source_count",
    ):
        axes[column] = axes[column].fillna(0).astype(int)
    axes["value_mapping_status"] = np.select(
        [
            axes.direct_measurement_count.gt(0),
            axes.mapped_numeric_measurement_count.gt(0),
        ],
        [
            "direct_mass_values_available",
            "source_numeric_values_not_model_eligible",
        ],
        default="proposed_axis_without_numeric_source_values",
    )
    food_stats = _food_summary(mapped, direct)
    for column in (
        "mapped_measurement_count",
        "mapped_axis_count",
        "mapped_source_count",
        "direct_measurement_count",
        "direct_axis_count",
        "direct_source_count",
    ):
        food_stats[column] = food_stats[column].fillna(0).astype(int)
    observations = _add_source_record_coverage(observations, mapped, direct, foodb_unmapped_reasons)
    food_search = _build_food_search_index(groups, observations, food_stats)
    axis_source_coverage = _build_axis_source_coverage(
        axes, source_axis_map, mapped_axis_by_source, axis_by_source
    )

    conflicts = conflicts.merge(
        axes.loc[:, ["target_axis_id", "canonical_name", "axis_family"]], how="left", on="target_axis_id"
    ).merge(
        groups.loc[:, ["exact_name_group_id", "display_name"]], how="left", on="exact_name_group_id"
    )

    evidence_counts = _write_jsonl_shards(mapped, data_dir / "food_evidence")
    observation_counts = _write_food_observation_shards(observations, data_dir / "food_observations")
    review_manifest = _build_review_queues(
        output_dir, source_axis_map, groups, conflicts, mapped, sources, axes
    )
    static_review_dir = data_dir / "review_queues"
    static_review_dir.mkdir(parents=True, exist_ok=True)
    for item in review_manifest.values():
        archive_source = output_dir / item["archive"]
        archive_destination = static_review_dir / Path(item["archive"]).name
        shutil.copy2(archive_source, archive_destination)
        item["static_download"] = f"assets/data/{item['archive']}"

    with input_paths["audit_manifest"].open(encoding="utf-8") as handle:
        audit_manifest = json.load(handle)
    summary = {
        "atlas_version": ATLAS_VERSION,
        "title": "Food Composition Atlas",
        "data_sources": int(len(sources)),
        "source_food_observations": int(len(observations)),
        "exact_name_candidate_groups": int(len(groups)),
        "mapped_numeric_measurements": int(len(mapped)),
        "direct_mass_measurements": int(len(direct)),
        "data_review_roles": {
            str(role): int(count)
            for role, count in mapped.training_review_role.value_counts().sort_index().items()
        },
        "cross_source_conflicts": int(len(conflicts)),
        "composition_axes": int(len(axes)),
        "authority_backed_hierarchy_edges": int(len(hierarchy_edges)),
        "audit_snapshot": {
            "audit_version": audit_manifest["audit_version"],
            "food_observations": audit_manifest["food_observations"],
            "mapped_numeric_measurements": audit_manifest["mapped_numeric_measurements"],
            "direct_mass_label_candidates": audit_manifest["direct_mass_label_candidates"],
        },
        "definitions": {
            "exact_name_candidate_group": "A conservative name-normalized candidate group. It is not a confirmed food merge.",
            "mapped_numeric_source_value": "A source-reported numeric value mapped to a registered composition axis. It remains source-specific and is not pooled.",
            "direct_mass_measurement": "A mapped source value with a compatible direct-mass conversion. It remains source-specific and is not pooled.",
            "data_review_role": "A row-level explanation of conversion and comparability. It preserves source evidence and does not declare a value scientifically incorrect.",
            "missing": "Not reported or unknown. Missing is never displayed as zero.",
            "explicit_zero": "The source explicitly reported a numeric zero.",
            "conflict": "More than one source reported non-equal direct-mass values for the same exact-name candidate group and axis. Values remain separate.",
        },
    }
    manifest = {
        "atlas_version": ATLAS_VERSION,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_files": input_manifest,
        "entity_counts": summary,
        "evidence_shards": {
            "strategy": "first two hexadecimal characters of exact_name_group_id digest",
            "mapped_numeric_evidence_rows_by_shard": evidence_counts,
            "food_observation_rows_by_shard": observation_counts,
        },
        "review_queues": review_manifest,
        "scientific_safeguards": [
            "No missing-to-zero conversion.",
            "No automatic fuzzy food or component merge.",
            "Source-specific measurements remain separate.",
            "A mapping does not authorize numerical pooling.",
            "Food hierarchy edges are absent unless an authority-backed crosswalk is available.",
        ],
    }

    write_json(summary, data_dir / "summary.json")
    write_json(_records(sources), data_dir / "sources.json")
    write_json(_records(axes), data_dir / "axes.json")
    write_json(_records(axis_source_coverage), data_dir / "axis_source_coverage.json")
    food_search_manifest = _write_gzip_json_shards(food_search, data_dir / "food_search_index")
    write_json(food_search_manifest, data_dir / "food_search_index_manifest.json")
    write_json(_records(conflicts), data_dir / "conflicts.json")
    write_json(
        {
            "policy": (
                "Only source-supplied FoodOn identifiers connected through FoodOn owl:subClassOf are shown. "
                "Names, FoodEx2, LanguaL, taxonomy and semantic similarity do not create hierarchy edges. "
                "No numeric value is inherited along a hierarchy edge."
            ),
            "edges": _records(hierarchy_edges),
        },
        data_dir / "food_hierarchy_relations.json",
    )
    write_json([], data_dir / "axis_relations.json")
    write_json(review_manifest, data_dir / "review_queue_manifest.json")
    write_json(manifest, output_dir / "atlas_manifest.json")

    static_source = project_root / "web" / "food_composition_atlas"
    if not static_source.exists():
        raise FileNotFoundError(f"Static Atlas source is missing: {static_source}")
    shutil.copytree(static_source, output_dir / "site", dirs_exist_ok=True)
    (output_dir / "README.md").write_text(
        "# Food Composition Atlas\n\n"
        "Run `python -m http.server 8000 --directory site` from this directory, then open "
        "http://localhost:8000. The site is read-only and must not be interpreted as a value-pooling release. "
        "Review queue CSV files are in `review_queues/`.\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the static Food Composition Atlas.")
    parser.add_argument("--output", type=Path, default=None, help="Atlas output directory.")
    parser.add_argument(
        "--audit-dir", type=Path, default=None,
        help="Global source audit directory; defaults to the newest available audited snapshot.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated Atlas output.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output or root / "outputs" / "food_composition_atlas"
    built = build_atlas(root, output, overwrite=args.overwrite, audit_dir=args.audit_dir)
    print(f"Built Food Composition Atlas at {built}")


if __name__ == "__main__":
    main()

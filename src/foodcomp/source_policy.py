"""Versioned source trust without claiming individual analytical certification."""

from __future__ import annotations

import pandas as pd

from .util import require_columns


TRUSTED_REFERENCE_POLICY = "trusted_sr_cnf_foodb_v1"
TRUSTED_REFERENCE_SOURCES = frozenset({"usda_sr_legacy", "cnf", "foodb"})
SOURCE_ONLY_EXCLUSIONS = frozenset({
    "", "non_independent_or_derived_value", "insufficient_value_level_quality",
    "non_primary_data_layer", "specialist_layer_not_primary_benchmark",
    "copied_value_retained_for_provenance_only",
})


def _true(values: pd.Series) -> pd.Series:
    return values.astype("string").str.casefold().eq("true").fillna(False)


def validation_reference_mask(frame: pd.DataFrame) -> pd.Series:
    """Old releases retain their old gate; new reference labels are explicit."""
    column = "validation_reference_eligible" if "validation_reference_eligible" in frame else "strict_validation_eligible"
    return _true(frame[column]) & _true(frame.main_value_eligible)


def apply_source_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Accept curated reference values, preserving all earlier evidence fields.

    Trust removes missing-documentation and database-layer barriers. It does
    not resolve copied food IDs, convert non-mass expressions, or certify an
    estimate as a measured value. Known copies remain in the origin ledger.
    """
    if policy != TRUSTED_REFERENCE_POLICY:
        raise ValueError(f"Unknown source-admission policy: {policy}")
    require_columns(frame, [
        "measurement_id", "source_key", "food_observation_id", "component_observation_id",
        "main_value_eligible", "normalized_value_g_per_100g",
        "value_status", "conversion_status", "measurement_modality", "is_censored",
        "is_range_only", "source_reference", "method_expression", "analytical_method",
        "lineage_source_key", "independent_evidence", "exclusion_reason",
    ], "source-policy input")
    if "source_admission_policy" in frame:
        raise ValueError("Apply the policy to immutable staging, not an already reviewed table")
    result = frame.copy()
    missing_legacy_flag = "strict_validation_eligible" not in result
    if missing_legacy_flag:
        if _true(result.main_value_eligible).any() or not result.source_key.eq("foodb").all():
            raise ValueError("Missing legacy validation flag is only supported for wholly archival FooDB partitions")
        result["strict_validation_eligible"] = False
    result["source_gate_previous_validation_status"] = (
        "legacy_foodb_archive_without_validation_field" if missing_legacy_flag else "legacy_validation_field_present"
    )
    trusted = result.source_key.isin(TRUSTED_REFERENCE_SOURCES)
    old_main = _true(result.main_value_eligible)
    old_validation = _true(result.strict_validation_eligible) & old_main
    old_reason = result.exclusion_reason.fillna("")
    result["source_admission_policy"] = policy
    result["source_trusted"] = trusted
    result["source_gate_previous_eligible"] = old_main
    result["source_gate_previous_validation_eligible"] = old_validation
    result["source_gate_previous_exclusion_reason"] = old_reason
    result["source_policy_decision"] = "unchanged_other_source"
    result["source_policy_reason"] = "existing_source_policy_preserved"
    result["source_independence_status"] = "not_established"
    result.loc[_true(result.independent_evidence), "source_independence_status"] = "legacy_source_rule_reports_independence"

    reason = pd.Series("", index=result.index, dtype="object")

    def hold(mask: pd.Series, explanation: str) -> None:
        reason.loc[trusted & mask & reason.eq("")] = explanation

    numeric = pd.to_numeric(result.normalized_value_g_per_100g, errors="coerce")
    numeric_ok = (
        numeric.between(0, 100) & result.value_status.isin(["observed", "explicit_zero"])
        & result.conversion_status.eq("converted_exact_mass")
        & result.measurement_modality.eq("mass_fraction_fresh_weight")
        & ~_true(result.is_censored) & ~_true(result.is_range_only)
    )
    hold(~numeric_ok, "invalid_missing_censored_assumed_zero_or_non_mass_value")
    hold(~old_reason.isin(SOURCE_ONLY_EXCLUSIONS), "existing_value_or_definition_hold")

    reference = result.source_reference.astype("string").fillna("").str.strip()
    method = result.method_expression.astype("string").fillna("").str.strip()
    # A paper title mentioning phenolics or prediction is not an origin code.
    origin = (reference + "; " + method).where(~result.source_key.eq("foodb"),
                                              method + "; " + result.analytical_method.astype("string").fillna(""))
    hold(origin.str.contains(r"assumed zero|assumed to be zero", case=False, regex=True), "assumed_zero_not_measured_zero")
    hold(origin.str.contains(r"label|claim|declaration|regulations|product standard", case=False, regex=True),
         "label_or_regulatory_value_not_reference_target")
    hold(origin.str.contains(r"recipe|based on physical composition|surveys only|nutrition canada survey", case=False, regex=True),
         "recipe_or_survey_calculation_auxiliary_only")
    hold(method.str.contains(r"imput|similar food|another form|different food|regression|estimated formulation", case=False, regex=True),
         "explicit_imputation_or_prediction_not_reference_target")
    hold(origin.str.contains(r"less than|greater than|below.*(?:limit|detection|quantification)", case=False, regex=True),
         "censoring_in_source_metadata_requires_review")

    # Routine sums/differences are reference expressions, not automatically
    # imputations; component-definition gates still reject activity equivalents.
    routine = method.str.fullmatch(r"Calculated|Summed|Calculated field", case=False)
    routine |= method.str.startswith("Nutrient that is based on other nutrient/s", na=False)
    routine |= reference.str.fullmatch(r"Calculated from analytical Canadian data|Calculated field", case=False)
    mixed_imputation = reference.str.contains(r"imput", case=False, regex=True)
    hold(mixed_imputation & ~routine, "imputed_or_unresolved_calculated_imputed_origin")

    foodb_copy = result.source_key.eq("foodb") & reference.str.match(
        r"^(?:USDA|DTU|FRIDA|PHENOL[- ]EXPLORER)(?:$|[\s:/,(-])", case=False,
    )
    cnf_copy = result.source_key.eq("cnf") & result.lineage_source_key.fillna("cnf").ne("cnf")
    sr_copy = result.source_key.eq("usda_sr_legacy") & method.str.contains(
        r"another source|other tables of food composition", case=False, regex=True,
    )
    copied = foodb_copy | cnf_copy | sr_copy
    result.loc[copied, "source_independence_status"] = "copied_or_derived_from_external_reference"
    hold(copied, "external_reference_requires_origin_resolution_and_deduplication")

    admitted = trusted & reason.eq("")
    result.loc[trusted, "main_value_eligible"] = admitted[trusted]
    result.loc[trusted, "source_policy_decision"] = "held_record_level_check"
    result.loc[trusted, "source_policy_reason"] = reason[trusted]
    result.loc[admitted, "source_policy_decision"] = "admitted_trusted_reference"
    result.loc[admitted, "source_policy_reason"] = "trusted_database_no_individual_method_or_sample_count_requirement"
    result.loc[trusted & ~admitted, "exclusion_reason"] = reason[trusted & ~admitted]
    result.loc[admitted, "exclusion_reason"] = ""
    result["main_value_eligible"] = _true(result.main_value_eligible).astype(bool)
    result["strict_validation_eligible"] = (old_validation & result.main_value_eligible).astype(bool)
    result["validation_reference_eligible"] = (
        result.main_value_eligible & (result.strict_validation_eligible | admitted)
    ).astype(bool)
    result["validation_evidence_basis"] = "not_eligible"
    result.loc[result.strict_validation_eligible, "validation_evidence_basis"] = "legacy_source_evidence_rules"
    result.loc[admitted, "validation_evidence_basis"] = "trusted_database_reference_not_individual_assay_certification"
    result["source_value_expression"] = "source_reported_reference"
    result.loc[trusted & routine, "source_value_expression"] = "definition_derived_reference"
    return result


def policy_metadata() -> dict:
    return {
        "policy_id": TRUSTED_REFERENCE_POLICY,
        "trusted_sources": sorted(TRUSTED_REFERENCE_SOURCES),
        "authorization": "User-approved database trust policy, 2026-09-08",
        "benchmark_estimand": "Held-out curated food-composition reference values, not guaranteed individual analytical measurements",
        "quality_tiers_and_independent_evidence": "Preserved; source trust never upgrades these fields",
        "missing_methods_sample_counts_and_full_text": "Not source-admission barriers for the three trusted databases",
        "copied_values": "Retained in provenance; unresolved origins do not supply additional labels or independent sample counts",
        "other_sources": "Existing policies unchanged",
    }

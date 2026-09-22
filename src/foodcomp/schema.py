"""Canonical table schemas and value parsing for food composition records."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import CENSORED_TOKENS, MAIN_BASIS, MAIN_MODALITY, MASS_TO_GRAMS, MISSING_TOKENS
from .util import normalize_text, parse_number


FOOD_COLUMNS = [
    "food_observation_id", "source_key", "source_food_id", "original_name",
    "original_name_local", "description", "scientific_name", "food_group",
    "food_subgroup", "food_type", "foodon_id", "foodex2_code", "langual_code",
    "taxonomy_id", "part", "maturity", "processing", "cooking", "preservation",
    "physical_state", "packing_medium", "geography", "cultivar", "recipe_status",
    "brand_status", "edible_status", "source_lineage_id", "source_record_url",
    "potential_reference_lineages",
]

COMPONENT_COLUMNS = [
    "component_observation_id", "source_key", "source_component_id", "original_name",
    "original_name_local", "description", "original_unit", "infoods_tag",
    "eurofir_component_id", "cdno_id", "chebi_id", "lipidmaps_id", "pubchem_id",
    "inchikey", "cas_number", "formula", "source_component_group",
    "source_chemical_class", "source_definition", "authority_status",
]

MEASUREMENT_COLUMNS = [
    "measurement_id", "source_key", "source_measurement_id", "food_observation_id",
    "component_observation_id", "raw_value", "raw_min", "raw_max", "raw_unit",
    "raw_basis", "value_status", "numeric_value", "is_explicit_zero", "is_censored",
    "is_range_only", "censor_limit", "normalized_value_g_per_100g",
    "normalized_min_g_per_100g", "normalized_max_g_per_100g", "conversion_factor",
    "conversion_status", "measurement_modality", "analytical_method", "method_expression",
    "sample_count", "standard_error", "quality_tier", "quality_score_available",
    "source_reference", "lineage_source_key", "independent_evidence",
    "data_layer", "main_value_eligible", "exclusion_reason", "range_estimate_flag",
    "reported_zero", "zero_semantics", "value_origin", "strict_validation_eligible",
    "quality_evidence", "source_component_definition",
]


def empty_food_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FOOD_COLUMNS)


def empty_component_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPONENT_COLUMNS)


def empty_measurement_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MEASUREMENT_COLUMNS)


def coerce_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df.loc[:, columns]


def parse_value(raw_value: Any, raw_min: Any = None, raw_max: Any = None) -> dict[str, Any]:
    """Parse value state without conflating missing, zero, censoring or ranges."""
    minimum = parse_number(raw_min)
    maximum = parse_number(raw_max)
    if raw_value is None or pd.isna(raw_value):
        if minimum is not None or maximum is not None:
            return _state("range_only", None, minimum, maximum, False, False, True, None)
        return _state("missing", None, minimum, maximum, False, False, False, None)

    if isinstance(raw_value, (int, float, np.number)):
        number = parse_number(raw_value)
        if number is None:
            return _state("invalid", None, minimum, maximum, False, False, False, None)
        return _state("explicit_zero" if number == 0 else "observed", number, minimum, maximum, number == 0, False, False, None)

    text = str(raw_value).strip()
    normalized = normalize_text(text)
    if normalized in MISSING_TOKENS:
        return _state("missing", None, minimum, maximum, False, False, False, None)
    if normalized in CENSORED_TOKENS:
        return _state("trace", None, minimum, maximum, False, True, False, None)
    censored_match = re.match(r"^\s*<\s*([0-9]+(?:[.,][0-9]+)?)\s*$", text)
    if censored_match:
        limit = float(censored_match.group(1).replace(",", "."))
        return _state("below_limit", None, minimum, maximum, False, True, False, limit)
    range_match = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*[-–]\s*([0-9]+(?:[.,][0-9]+)?)\s*$", text)
    if range_match:
        lo = float(range_match.group(1).replace(",", "."))
        hi = float(range_match.group(2).replace(",", "."))
        return _state("range_only", None, lo, hi, False, False, True, None)
    number = parse_number(text)
    if number is not None:
        return _state("explicit_zero" if number == 0 else "observed", number, minimum, maximum, number == 0, False, False, None)
    return _state("invalid", None, minimum, maximum, False, False, False, None)


def _state(status: str, value: float | None, minimum: float | None, maximum: float | None,
           zero: bool, censored: bool, range_only: bool, limit: float | None) -> dict[str, Any]:
    return {
        "value_status": status, "numeric_value": value, "parsed_min": minimum,
        "parsed_max": maximum, "is_explicit_zero": zero, "is_censored": censored,
        "is_range_only": range_only, "censor_limit": limit,
    }


def normalize_unit(raw_unit: Any, raw_basis: Any = None, implicit_per_100g: bool = False) -> dict[str, Any]:
    """Return exact mass conversion metadata for the primary fresh-weight modality."""
    unit_text = "" if raw_unit is None or pd.isna(raw_unit) else str(raw_unit).strip()
    basis_text = "" if raw_basis is None or pd.isna(raw_basis) else str(raw_basis).strip()
    compact = unicase(unit_text)
    basis = unicase(basis_text)
    joined = f"{compact} {basis}".strip()

    if any(token in joined for token in ("kcal", "kilocal", "kiloj", "kj", "energy")):
        return _unit_result(None, "excluded_energy", "energy")
    if re.search(r"\b(iu|rae|dfe|re|ne|ate)\b", joined) or any(token in joined for token in ("alpha-te", "α-te", "retinol equivalent")):
        return _unit_result(None, "excluded_biological_equivalent", "biological_activity_equivalent")
    if any(token in joined for token in ("um", "µm", "micromol", "mmol", "mol/")):
        return _unit_result(None, "excluded_molar", "molar_concentration")
    relative_basis = any(token in joined for token in (
        "per 100g fa", "per 100 g fa", "100gfa", "100g fatty acid",
        "100 g fatty acid", "per 100 fatty acid",
    ))
    if "%" in unit_text or relative_basis:
        return _unit_result(None, "excluded_relative_fraction", "relative_fraction")
    if re.search(r"(?:/|per\s+)(?:\d+\s*)?g\s*(?:n\b|nitrogen|protein)", joined):
        return _unit_result(None, "requires_nitrogen_or_protein_conversion", "mass_per_nitrogen_or_protein")
    if any(token in joined for token in ("dry weight", "dry matter", "dry basis", " dw")):
        return _unit_result(None, "requires_moisture_conversion", "mass_fraction_dry_weight")
    if any(token in joined for token in ("100ml", "100 ml", "/ml", "per ml", "litre", "liter")):
        return _unit_result(None, "requires_density_conversion", "mass_per_volume")
    serving_basis = "serving" in joined or "household" in joined or ("portion" in joined and "edible portion" not in joined)
    if serving_basis:
        return _unit_result(None, "requires_portion_conversion", "mass_per_serving")

    # A table-wide basis cannot override a different denominator in a column.
    if "/" in compact and not re.fullmatch(
        r"(?:g|mg|ug|µg|mcg)\s*/\s*100\s*g(?:\s+(?:food|edible(?: portion)?|fresh(?: weight)?))*", compact
    ):
        return _unit_result(None, "unsupported_explicit_denominator", "unknown_denominator")

    numerator = None
    for token in ("micrograms", "microgram", "mcg", "ug", "µg", "milligrams", "milligram", "mg", "grams", "gram", "g"):
        token_pattern = re.escape(unicase(token))
        if re.search(rf"(?:^|[^a-z]){token_pattern}(?:$|[^a-z])", compact):
            numerator = "ug" if token in {"micrograms", "microgram", "mcg", "ug", "µg"} else "mg" if token.startswith("milli") or token == "mg" else "g"
            break
    if numerator is None:
        return _unit_result(None, "unsupported_or_missing_unit", "unknown")

    explicit_100g = any(token in joined for token in ("/100g", "/100 g", "per100g", "per 100g", "per 100 g", "100 grams"))
    if not explicit_100g and not implicit_per_100g:
        return _unit_result(None, "missing_denominator", "mass_unspecified_basis")
    return _unit_result(MASS_TO_GRAMS[numerator], "converted_exact_mass", MAIN_MODALITY)


def unicase(value: str) -> str:
    return value.casefold().replace("μ", "µ").replace("\u00a0", " ").strip()


def _unit_result(factor: float | None, status: str, modality: str) -> dict[str, Any]:
    return {
        "conversion_factor": factor,
        "conversion_status": status,
        "measurement_modality": modality,
        "canonical_basis": MAIN_BASIS if factor is not None else None,
    }


def apply_value_and_unit_normalization(
    frame: pd.DataFrame,
    *,
    implicit_per_100g: bool = False,
) -> pd.DataFrame:
    parsed = pd.DataFrame([
        parse_value(value, minimum, maximum)
        for value, minimum, maximum in zip(frame["raw_value"], frame["raw_min"], frame["raw_max"], strict=True)
    ])
    units = pd.DataFrame([
        normalize_unit(unit, basis, implicit_per_100g=implicit_per_100g)
        for unit, basis in zip(frame["raw_unit"], frame["raw_basis"], strict=True)
    ])
    for column in parsed.columns:
        frame[column] = parsed[column].values
    for column in units.columns:
        frame[column] = units[column].values
    frame["reported_zero"] = frame["numeric_value"].eq(0)
    frame["zero_semantics"] = np.where(frame["reported_zero"], "source_reported_numeric_zero", "not_zero")
    frame["normalized_value_g_per_100g"] = frame["numeric_value"] * frame["conversion_factor"]
    frame["normalized_min_g_per_100g"] = frame["parsed_min"] * frame["conversion_factor"]
    frame["normalized_max_g_per_100g"] = frame["parsed_max"] * frame["conversion_factor"]
    frame["range_estimate_flag"] = frame["is_range_only"].fillna(False)
    float_tolerance = frame["normalized_value_g_per_100g"].gt(100.0) & frame["normalized_value_g_per_100g"].le(100.0 + 1e-6)
    frame.loc[float_tolerance, "normalized_value_g_per_100g"] = 100.0
    valid = frame["normalized_value_g_per_100g"].notna()
    nonnegative = frame["normalized_value_g_per_100g"].ge(0)
    physically_bounded = frame["normalized_value_g_per_100g"].le(100.0)
    frame["main_value_eligible"] = valid & nonnegative & physically_bounded
    frame.loc[valid & ~nonnegative, "value_status"] = "invalid_negative"
    frame.loc[valid & ~nonnegative, "main_value_eligible"] = False
    frame.loc[valid & nonnegative & ~physically_bounded, "value_status"] = "invalid_mass_fraction_above_100"
    return frame.drop(columns=["parsed_min", "parsed_max", "canonical_basis"], errors="ignore")


def finite_or_none(value: Any) -> float | None:
    parsed = parse_number(value)
    return parsed if parsed is not None and math.isfinite(parsed) else None

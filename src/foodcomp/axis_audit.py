"""Conservative deduplication and scientific audit of atlas measurement axes.

This is deliberately downstream of the source-native atlas and upstream of
any food-value transformation. It deduplicates only documented identities and
exact expression signatures. It never joins foods, pools measurements,
converts units, creates splits, or uses observation frequency to decide that a
nutrient is scientifically required.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json

import pandas as pd

from .util import normalize_text, stable_id


NUMERIC_STATES = {"food_value_table_present", "food_numeric_measurement_present"}
RANGE_ONLY_STATE = "food_range_only_measurement_present"


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def _join_unique(values: pd.Series, limit: int | None = None) -> str:
    seen: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in seen:
            seen.append(text)
    if limit is not None and len(seen) > limit:
        return " | ".join(seen[:limit]) + f" | … ({len(seen) - limit} more)"
    return " | ".join(seen)


def _identity_basis(concept_id: str) -> str:
    if concept_id.startswith("INFOODS:"):
        return "INFOODS_standardized_component_expression"
    if concept_id.startswith("INCHI:"):
        return "exact_structural_identity_validated_InChI"
    if concept_id.startswith("INCHIKEY:"):
        return "exact_structural_identity_validated_InChIKey"
    return "source_native_axis_only_no_cross_source_merge"


def _availability(states: set[str]) -> str:
    if states & NUMERIC_STATES:
        return "numeric_measurement_available"
    if RANGE_ONLY_STATE in states:
        return "range_only_measurement_available"
    if "food_association_without_quantified_value" in states:
        return "food_association_without_numeric_measurement"
    return "catalogue_definition_only"


def _expression_signature(row: pd.Series) -> str:
    """Retain units, denominators and stated expressions at deduplication."""
    return stable_id(
        "measurement_axis",
        row.axis_concept_id,
        normalize_text(row.raw_unit),
        normalize_text(row.raw_denominator),
        normalize_text(row.analytical_method_or_expression),
        normalize_text(row.source_definition),
    )


def build_deduplicated_measurement_axes(mapped: pd.DataFrame) -> pd.DataFrame:
    """Create measurement-expression axes without name-only or unit-blind merges."""
    work = mapped.copy()
    work["audit_axis_id"] = work.apply(_expression_signature, axis=1)
    records = []
    for audit_axis_id, group in work.groupby("audit_axis_id", sort=True):
        first = group.iloc[0]
        states = set(group.value_availability)
        requirements = sorted({value for value in group.requirement_id if _clean(value)})
        tiers = sorted({value for value in group.scientific_inclusion_tier if _clean(value)})
        roles = sorted({value for value in group.food_composition_role if _clean(value)})
        modalities = sorted({value for value in group.measurement_modality if _clean(value)})
        if len(tiers) > 1 or len(roles) > 1:
            semantic_status = "classification_conflict_requires_review"
        elif first.authority_namespace == "SOURCE":
            semantic_status = "source_defined_expression_requires_semantic_review"
        else:
            semantic_status = "identifier_or_standard_supported"
        records.append({
            "audit_axis_id": audit_axis_id,
            "axis_concept_id": first.axis_concept_id,
            "identity_deduplication_basis": _identity_basis(first.axis_concept_id),
            "canonical_name": first.original_name,
            "canonical_name_basis": first.canonical_name_basis,
            "authority_namespace": first.authority_namespace,
            "authority_id": first.authority_id,
            "requirement_ids": ";".join(requirements),
            "scientific_inclusion_tier": ";".join(tiers),
            "food_composition_role": ";".join(roles),
            "nutritional_role": _join_unique(group.nutritional_role),
            "chemical_identity": _join_unique(group.chemical_identity),
            "chemical_class": _join_unique(group.source_chemical_class),
            "measurement_modality": ";".join(modalities),
            "raw_unit": _join_unique(group.raw_unit),
            "raw_denominator": _join_unique(group.raw_denominator),
            "analytical_method_or_expression": _join_unique(group.analytical_method_or_expression),
            "source_definition": _join_unique(group.source_definition, limit=4),
            "source_axis_count": len(group),
            "source_count": group.source_key.nunique(),
            "source_keys": ";".join(sorted(group.source_key.unique())),
            "source_axis_ids": ";".join(group.source_axis_id),
            "original_name_variants": _join_unique(group.original_name, limit=12),
            "value_availability_states": ";".join(sorted(states)),
            "measurement_availability": _availability(states),
            "numeric_evidence_source_count": group.loc[group.value_availability.isin(NUMERIC_STATES), "source_key"].nunique(),
            "semantic_audit_status": semantic_status,
            "source_field_integrity_notes": _join_unique(group.source_field_integrity_note),
        })
    return pd.DataFrame(records)


def audit_axis_decisions(deduped: pd.DataFrame) -> pd.DataFrame:
    """Assign a transparent scientific role without making a model selection."""
    result = deduped.copy()
    collections: list[str] = []
    decisions: list[str] = []
    rationales: list[str] = []
    review: list[bool] = []
    for row in result.itertuples(index=False):
        tiers = set(_clean(row.scientific_inclusion_tier).split(";"))
        availability = _clean(row.measurement_availability)
        source_defined = _clean(row.semantic_audit_status) != "identifier_or_standard_supported"
        if "must_have_nutrition_core" in tiers:
            collections.append("final_required_nutrition_core")
            if availability == "numeric_measurement_available":
                decisions.append("retain_as_core_measurement_expression")
                rationales.append("Instantiates a normative nutrition-core requirement; preserve expression and method definition.")
            else:
                decisions.append("retain_as_core_data_gap")
                rationales.append("Scientifically required nutrition-core expression but no numeric measurement is currently catalogued.")
            review.append(source_defined)
        elif "include_capable_food_metabolome" in tiers:
            collections.append("food_metabolome_extension")
            if availability == "numeric_measurement_available":
                decisions.append("retain_as_quantified_metabolome_candidate")
                rationales.append("A defined food-related chemical with numeric source evidence; not equivalent to a core nutrient.")
            else:
                decisions.append("retain_as_catalogue_only_metabolome_candidate")
                rationales.append("A chemically defined food-metabolome candidate without catalogued numeric evidence.")
            review.append(False)
        elif "separate_exposure_safety_layer" in tiers:
            collections.append("exposure_and_safety_layer")
            decisions.append("retain_outside_nutrition_target_space")
            rationales.append("Exposure, residue, contaminant, toxin, or processing-safety analyte; retain separately from nutrition prediction.")
            review.append(True)
        elif "retain_nonchemical_expression" in tiers:
            collections.append("derived_and_nonchemical_expression_layer")
            decisions.append("retain_for_provenance_not_as_chemical_axis")
            rationales.append("Calculated expression, activity equivalent, energy, coefficient, or metadata; not a standalone chemical entity.")
            review.append(False)
        else:
            collections.append("unresolved_axis_layer")
            decisions.append("manual_review_before_any_inclusion")
            rationales.append("Identity, food-composition role, or measurement definition is not yet adequate for a final scientific collection.")
            review.append(True)
    result["final_collection"] = collections
    result["audit_decision"] = decisions
    result["audit_rationale"] = rationales
    result["manual_review_required"] = review
    return result


def final_required_axes(requirements: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Return the 50 semantic requirements plus their audited expression evidence."""
    rows = []
    for requirement in requirements.itertuples(index=False):
        identifier = requirement.requirement_id
        relevant = audit[audit.requirement_ids.str.split(";").apply(lambda ids: identifier in ids)]
        numeric = relevant[relevant.measurement_availability.eq("numeric_measurement_available")]
        standardized = relevant[relevant.identity_deduplication_basis.ne("source_native_axis_only_no_cross_source_merge")]
        rows.append({
            "requirement_id": identifier,
            "canonical_name": requirement.canonical_name,
            "scientific_group": requirement.scientific_group,
            "component_kind": requirement.component_kind,
            "normative_status": "must_have_nutrition_core",
            "scientific_rationale": requirement.rationale,
            "primary_authority": requirement.primary_authority,
            "authority_url": requirement.authority_url,
            "preferred_infoods_tags": requirement.preferred_infoods_tags,
            "related_forms_note": requirement.related_forms_note,
            "audited_measurement_expression_count": len(relevant),
            "numeric_measurement_expression_count": len(numeric),
            "identifier_or_standard_supported_expression_count": len(standardized),
            "source_defined_expression_count": int(relevant.semantic_audit_status.ne("identifier_or_standard_supported").sum()),
            "catalogue_status": (
                "numeric_expression_available" if len(numeric) else "core_data_gap_no_numeric_expression_catalogued"
            ),
            "selection_note": "Required by scientific scope independent of source count, measurement count, or model performance.",
        })
    return pd.DataFrame(rows)


def write_final_axis_report(path: Path, *, audit: pd.DataFrame, required: pd.DataFrame,
                            source_axes: int, identity_concepts: int) -> None:
    identity_counts = audit.groupby("identity_deduplication_basis").size().sort_values(ascending=False)
    collection_counts = audit.groupby("final_collection").size().sort_values(ascending=False)
    decision_counts = audit.groupby("audit_decision").size().sort_values(ascending=False)
    available = int(required.catalogue_status.eq("numeric_expression_available").sum())
    lines = [
        "# Final Scientific Axis Audit",
        "",
        "## Decision",
        "",
        "The final **required nutrition axis set** contains 50 normative requirements. It is scientifically defined from food-composition and dietary-reference scope, not from how often an axis appears in a dataset. These 50 requirements are not treated as 50 interchangeable numerical columns: their observed source expressions retain chemical form, analytical definition, unit, denominator and method distinctions.",
        "",
        f"The audit began with {source_axes:,} source-native axes and {identity_concepts:,} evidence-supported identity concepts. Conservative measurement-expression deduplication produced {len(audit):,} distinct axes. No fuzzy name match or unit-blind merge was applied.",
        "",
        "## Deduplication policy",
        "",
        "1. An INFOODS tagname can define a shared standardized component expression.",
        "2. A syntactically valid InChI/InChIKey can define a shared molecular identity.",
        "3. A source-native entry without either evidence type stays source-axis-specific. Identical or similar names do not merge it with any other source.",
        "4. Units, denominators, source definitions and stated analytical/calculation methods are part of the measurement-expression key. Therefore kcal and kJ, RAE and retinol mass, crude and total fibre, and calculated versus directly measured carbohydrate remain separate axes.",
        "",
        "### Evidence-supported identity bases",
        "",
        "| Identity basis | Measurement-expression axes |",
        "|---|---:|",
        *[f"| {key} | {value:,} |" for key, value in identity_counts.items()],
        "",
        "## Audited collections",
        "",
        "| Collection | Measurement-expression axes | Meaning |",
        "|---|---:|---|",
        *[f"| {key} | {value:,} | See `axis_audit_decisions.csv.gz`; this is not a numerical pooling decision. |" for key, value in collection_counts.items()],
        "",
        "| Audit decision | Axes |",
        "|---|---:|",
        *[f"| {key} | {value:,} |" for key, value in decision_counts.items()],
        "",
        "## Required nutrition core",
        "",
        f"All 50 requirements remain required. {available} currently have at least one numeric measurement-expression candidate in the audited catalogue. This is a data-availability statement only; it does not validate unit compatibility, source quality, or future target eligibility.",
        "",
        "| Requirement | Scientific group | Kind | Audited expressions | Numeric expressions | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in required.itertuples(index=False):
        lines.append(
            f"| {row.canonical_name} | {row.scientific_group} | {row.component_kind} | "
            f"{row.audited_measurement_expression_count} | {row.numeric_measurement_expression_count} | {row.catalogue_status} |"
        )
    lines.extend([
        "",
        "## What is not a nutrition prediction target",
        "",
        "- Exposure/safety chemicals stay in a separate layer even when structurally defined and quantitatively measured.",
        "- Energy, activity equivalents, calculated-by-difference values, coefficients and metadata remain provenance/derived-expression records, not molecular identities.",
        "- Structurally defined food metabolites are retained as an extension collection. Their future inclusion requires an explicit research question and measurement-definition audit; they are not silently promoted to essential nutrients.",
        "- Unresolved axes are retained in the review queue and excluded from any claimed final scientific target set until their identity and measurement expression are reviewed.",
        "",
        "## Files",
        "",
        "- `deduplicated_measurement_axis_registry.csv.gz`: all deduplicated axes and audit fields.",
        "- `axis_audit_decisions.csv.gz`: final collection and explicit retain/review decision for every axis.",
        "- `final_required_nutrition_axes.csv`: the 50 normative requirements and their current expression-level evidence.",
        "",
        "This audit does not create a numerical matrix, transform a value, deduplicate a food, make a data-quality ranking, generate a split, or train a model.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_axis_audit(atlas_dir: Path, output_dir: Path, report_dir: Path) -> dict[str, int]:
    mapped_path = atlas_dir / "source_axis_to_concept.csv.gz"
    requirements_path = atlas_dir / "must_have_requirement_registry.csv"
    if not mapped_path.is_file() or not requirements_path.is_file():
        raise FileNotFoundError("Build the global_food_metabolome_axis_atlas before the deduplication audit.")
    mapped = pd.read_csv(mapped_path, compression="gzip", keep_default_na=False, low_memory=False)
    requirements = pd.read_csv(requirements_path, keep_default_na=False)
    required_columns = {"source_axis_id", "axis_concept_id", "value_availability", "raw_unit", "raw_denominator"}
    missing = sorted(required_columns - set(mapped.columns))
    if missing:
        raise ValueError(f"Atlas mapping is missing audit columns: {missing}")
    if mapped.source_axis_id.duplicated().any():
        raise ValueError("Atlas source-axis map must have one row per source_axis_id.")

    deduped = build_deduplicated_measurement_axes(mapped)
    audit = audit_axis_decisions(deduped)
    required = final_required_axes(requirements, audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(deduped, output_dir / "deduplicated_measurement_axis_registry.csv.gz")
    _write(audit, output_dir / "axis_audit_decisions.csv.gz")
    _write(required, output_dir / "final_required_nutrition_axes.csv")
    (output_dir / "manifest.json").write_text(json.dumps({
        "dataset": "global_food_metabolome_axis_audit_v1",
        "input_source_axis_count": int(len(mapped)),
        "input_identity_concept_count": int(mapped.axis_concept_id.nunique()),
        "deduplicated_measurement_axis_count": int(len(deduped)),
        "final_required_nutrition_requirement_count": int(len(required)),
        "invariants": [
            "No fuzzy name matching was used for deduplication.",
            "Units, denominators, source definitions and stated methods remain part of a measurement-expression axis.",
            "No food records or numeric values were merged or transformed.",
            "Scientific core membership is independent of coverage and model performance.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    write_final_axis_report(
        report_dir / "FINAL_REQUIRED_AXIS_AUDIT.md", audit=audit, required=required,
        source_axes=len(mapped), identity_concepts=mapped.axis_concept_id.nunique(),
    )
    return {
        "source_axes": len(mapped),
        "identity_concepts": mapped.axis_concept_id.nunique(),
        "measurement_axes": len(deduped),
        "required_nutrition_axes": len(required),
    }

"""Audit global source-axis coverage against the scientific target panel.

This module does semantic retrieval only.  It does not assert equivalence or
merge numerical values.  In particular, a high embedding cosine score is a
review lead, never a chemical identity decision.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from .util import normalize_text, stable_id, write_csv, write_json


SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
TARGET_TO_SOURCE_TOP_K = 20
UNRESOLVED_NEIGHBOURS_PER_TARGET = 5
HIGH_SEMANTIC_REVIEW_THRESHOLD = 0.72
MIN_LEXICAL_SUPPORT = 0.72
MIN_TOKEN_SUPPORT = 0.50
NUMERIC_AVAILABILITY = {"food_value_table_present", "food_numeric_measurement_present"}


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_name_key(value: str) -> str:
    return normalize_text(value)


def _strip_parenthetical(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", value).strip()


def _name_variants(value: str) -> set[str]:
    """Generate conservative, display-level name variants.

    These variants address harmless label conventions such as ``Selenium, Se``,
    ``Tocotrienol, alpha``, and parenthetical vitamin abbreviations. They do
    not remove chemical-form words or alter a source definition.
    """
    raw = _clean(value)
    if not raw:
        return set()
    variants = {raw, _strip_parenthetical(raw)}
    comma_parts = [part.strip() for part in raw.split(",") if part.strip()]
    if comma_parts:
        variants.add(comma_parts[0])
    if len(comma_parts) == 2:
        variants.add(f"{comma_parts[1]} {comma_parts[0]}")
    return {
        key for item in variants
        if len(key := _canonical_name_key(item)) >= 3
    }


def _target_alias_index(targets: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in targets.itertuples(index=False):
        aliases = {
            _clean(row.canonical_name),
            _strip_parenthetical(_clean(row.canonical_name)),
        }
        aliases.update(
            _clean(item)
            for item in _clean(row.aliases_for_review).split("|")
            if _clean(item)
        )
        aliases.update(
            _clean(item)
            for item in _clean(row.original_axis_names).split("|")
            if _clean(item)
        )
        for alias in aliases:
            for key in _name_variants(alias):
                index[key].add(row.target_axis_id)
    return index


def _exact_match_basis(source: pd.Series, target: pd.Series) -> str:
    """Describe why a label-level match exists, without asserting identity."""
    source_variants = _source_aliases(source)
    display_variants: set[str] = set()
    for value in [
        _clean(target.canonical_name),
        *[_clean(item) for item in _clean(target.aliases_for_review).split("|") if _clean(item)],
    ]:
        display_variants.update(_name_variants(value))
    if source_variants & display_variants:
        return "canonical_name_or_review_alias"
    source_native_variants: set[str] = set()
    for value in [_clean(item) for item in _clean(target.original_axis_names).split("|") if _clean(item)]:
        source_native_variants.update(_name_variants(value))
    if source_variants & source_native_variants:
        return "registered_source_native_alias"
    return ""


def _source_aliases(row: pd.Series) -> set[str]:
    return _name_variants(_clean(row.original_name)) | _name_variants(_clean(row.original_name_local))


def _source_text(row: pd.Series) -> str:
    fields = [
        ("name", row.original_name),
        ("local name", row.original_name_local),
        ("definition", row.source_definition),
        ("component group", row.source_component_group),
        ("chemical class", row.source_chemical_class),
        ("INFOODS tag", row.infoods_tag),
    ]
    rendered = "; ".join(f"{label}: {_clean(value)}" for label, value in fields if _clean(value))
    return f"Food composition analyte. {rendered}" if rendered else "Food composition analyte."


def _target_text(row: pd.Series) -> str:
    aliases = "; ".join(
        item for item in _clean(row.aliases_for_review).split("|")[:8] if item
    )
    return (
        f"Food composition target. Name: {_clean(row.canonical_name)}. "
        f"Chemical identity: {_clean(row.chemical_identity)}. "
        f"Food composition role: {_clean(row.food_composition_role)}. "
        f"Chemical family: {_clean(row.axis_family)}. Aliases: {aliases}."
    )


def _token_overlap(source_name: str, target_name: str) -> float:
    source = set(_canonical_name_key(source_name).split())
    target = set(_canonical_name_key(target_name).split())
    ignored = {"acid", "vitamin", "total", "food", "content", "compound", "fatty"}
    source -= ignored
    target -= ignored
    union = source | target
    return len(source & target) / len(union) if union else 0.0


def _lexical_similarity(source_name: str, target_name: str) -> float:
    return SequenceMatcher(
        a=_canonical_name_key(source_name), b=_canonical_name_key(target_name)
    ).ratio()


def _chemical_number_signatures(value: str) -> set[str]:
    """Return only explicit structural/nutrient number signatures.

    A source label such as ``20:5 n-3`` is stronger identity evidence than an
    embedding score. Generic row numbers and database identifiers are not
    parsed, so they cannot create a false chemistry match.
    """
    text = _clean(value).casefold().replace("ω", "omega")
    signatures: set[str] = set()
    for chain, bonds, omega in re.findall(
        r"(?:\bc\s*)?(\d{1,2})\s*:\s*(\d{1,2})(?:\s*(?:n|omega)\s*-?\s*(\d+))?",
        text,
    ):
        signatures.add(f"fatty_acid:{chain}:{bonds}:{omega or ''}")
    for family, number in re.findall(r"\b(?:vitamin\s*)?([bdek])\s*-?\s*(\d+)\b", text):
        signatures.add(f"vitamer:{family}:{number}")
    for number in re.findall(r"\bmenaquinone\s*-?\s*(\d+)\b", text):
        signatures.add(f"menaquinone:{number}")
    return signatures


def _form_markers(value: str) -> set[str]:
    text = _canonical_name_key(value)
    text = re.sub(r"\bcis\s+trans\s+(?:unspecified|not specified)\b", "", text)
    return {
        marker for marker in {"alpha", "beta", "gamma", "delta", "cis", "trans", "hydroxy"}
        if re.search(rf"\b{marker}\b", text)
    }


def _fatty_acid_isomer_state(value: str) -> set[str]:
    text = _canonical_name_key(value)
    if not re.search(r"\b\d{1,2}\s+\d{1,2}(?:[ct])?\b", text):
        return set()
    if re.search(r"\bcis\s+trans\s+(?:unspecified|not specified)\b", text):
        return {"unspecified"}
    states: set[str] = set()
    if re.search(r"\b(cis|c)\b", text) or re.search(r"\b\d+\s+\d+c\b", text):
        states.add("cis")
    if re.search(r"\b(trans|t)\b", text) or re.search(r"\b\d+\s+\d+t\b", text):
        states.add("trans")
    if re.search(r"\b\d+z\b", text):
        states.add("cis")
    if re.search(r"\b\d+e\b", text):
        states.add("trans")
    if "conjugated" in text or " cla " in f" {text} ":
        states.add("conjugated")
    if any(marker in text for marker in {"undifferentiated", "unspecified", "isomeric"}):
        states.add("unspecified")
    return states


def _positional_locator_signatures(value: str) -> set[str]:
    """Capture position-bearing labels such as 3,4-dicaffeoyl- or 11-eicosenoic.

    These locants change molecular identity. They cannot be resolved by a
    generic embedding similarity score.
    """
    text = _clean(value).casefold()
    return {
        re.sub(r"\s+", "", match)
        for match in re.findall(r"\b\d+(?:\s*,\s*\d+)+\s*-[a-z]", text)
    } | {
        re.sub(r"\s+", "", match)
        for match in re.findall(r"\b\d+\s*-(?=[a-z])", text)
    } | {
        re.sub(r"\s+", "", match)
        for match in re.findall(r"\b\d+[ez](?=-)", text)
    }


def _glycoside_markers(value: str) -> set[str]:
    text = _canonical_name_key(value)
    markers = {
        "glucoside", "galactoside", "rhamnoside", "rutinoside", "xyloside",
        "xylosyl", "glucosyl", "glucuronide", "acetyl", "malonyl", "caffeoyl",
        "feruloyl", "succinyl", "dioxalyl", "gallate",
    }
    return {marker for marker in markers if marker in text}


def _expression_markers(value: str) -> set[str]:
    text = _canonical_name_key(value)
    markers = {"available", "total", "by difference", "calculated", "labelling"}
    return {marker for marker in markers if marker in text}


def _chemical_scaffold_markers(value: str) -> set[str]:
    """Detect visibly incompatible parent scaffolds without structure inference."""
    text = _canonical_name_key(value)
    markers = {
        "benzoic", "benzaldehyde", "phenylacetic", "quinic", "cinnamic", "caffeoylquinic",
        "feruloylquinic", "hydroxycinnamic",
    }
    return {marker for marker in markers if marker in text}


def _glycoside_aglycone(value: str) -> str:
    """Return the named aglycone for a simple numbered glycoside label."""
    text = _canonical_name_key(value)
    if not _glycoside_markers(text):
        return ""
    match = re.search(r"\b([a-z]+)\s+\d+(?:\s+o)?\s+[a-z]", text)
    return match.group(1) if match else ""


def _positioned_small_molecule_signature(value: str) -> set[str]:
    """Capture stereochemical/positional labels not represented by C:D notation."""
    text = _clean(value).casefold()
    signatures = {
        f"bracket-position:{number}:{name}"
        for number, name in re.findall(r"\[(\d+)\]\s*-\s*([a-z]+)", text)
    }
    signatures |= {
        f"coumaric-position:{position}"
        for position in re.findall(r"\b([omp])\s*-\s*coumaric\b", text)
    }
    return signatures


def _vitamer_form_markers(value: str) -> set[str]:
    text = _canonical_name_key(value).replace("pyridoxin", "pyridoxine")
    forms = {"pyridoxine", "pyridoxal", "pyridoxamine", "nicotinic acid", "nicotinamide"}
    return {form for form in forms if form in text}


def _best_name_similarity(source: pd.Series, target: pd.Series) -> tuple[float, float, bool]:
    source_variants = _source_aliases(source)
    target_variants: set[str] = set()
    for value in [
        _clean(target.canonical_name),
        *[_clean(item) for item in _clean(target.aliases_for_review).split("|") if _clean(item)],
        *[_clean(item) for item in _clean(target.original_axis_names).split("|") if _clean(item)],
    ]:
        target_variants.update(_name_variants(value))
    if source_variants & target_variants:
        return 1.0, 1.0, True
    comparisons = [
        (_lexical_similarity(source_name, target_name), _token_overlap(source_name, target_name))
        for source_name in source_variants
        for target_name in target_variants
    ]
    if not comparisons:
        return 0.0, 0.0, False
    lexical, overlap = max(comparisons, key=lambda item: (item[0] + item[1], item[1]))
    return lexical, overlap, False


def _identity_evidence(source: pd.Series, target: pd.Series, *, exact_name: bool,
                       semantic_score: float, lexical_score: float,
                       token_overlap: float) -> tuple[bool, str]:
    """Return whether a candidate is a plausible *same-analyte* lead.

    It purposefully rejects close semantic neighbours whose labels disclose a
    different chemical form. A fuzzy match is not allowed to create a pooled
    numerical axis; it only creates a manual identity-review lead.
    """
    # Source definitions can be long narrative text (especially FooDB), with
    # incidental chemical names, measurements and literature citations. They
    # are useful for semantic discovery but are unsafe for deterministic
    # identity/form parsing. Use only source-native label fields here.
    source_identity = " ".join([
        _clean(source.original_name), _clean(source.original_name_local),
        _clean(source.source_component_group), _clean(source.infoods_tag),
    ])
    # An alias may contain a longer systematic synonym than a source's common
    # analyte label. Form compatibility therefore uses the canonical target
    # label; aliases are used only for display-name matching.
    target_identity = _clean(target.canonical_name)
    source_numbers = _chemical_number_signatures(source_identity)
    target_numbers = _chemical_number_signatures(target_identity)
    if source_numbers and target_numbers and source_numbers.isdisjoint(target_numbers):
        return False, "different_explicit_structural_or_vitamer_signature"
    if source_numbers and target_numbers and source_numbers != target_numbers:
        return False, "aggregate_or_different_explicit_structural_definition"

    source_expression = _expression_markers(source_identity)
    target_expression = _expression_markers(target_identity)
    if source_expression != target_expression and (source_expression or target_expression):
        return False, "different_calculated_or_nutritional_expression"

    source_scaffolds = _chemical_scaffold_markers(source_identity)
    target_scaffolds = _chemical_scaffold_markers(target_identity)
    if source_scaffolds and target_scaffolds and source_scaffolds.isdisjoint(target_scaffolds):
        return False, "different_explicit_chemical_scaffold"

    source_vitamer_forms = _vitamer_form_markers(source_identity)
    target_vitamer_forms = _vitamer_form_markers(target_identity)
    if source_vitamer_forms != target_vitamer_forms and (source_vitamer_forms or target_vitamer_forms):
        return False, "different_explicit_vitamer_chemical_form"

    source_forms = _form_markers(source_identity)
    target_forms = _form_markers(target_identity)
    if source_forms and target_forms and source_forms.isdisjoint(target_forms):
        return False, "different_explicit_chemical_form"
    if source_forms != target_forms and (source_forms or target_forms):
        return False, "under_or_over_specified_chemical_form"

    source_isomer_state = _fatty_acid_isomer_state(source_identity)
    target_isomer_state = _fatty_acid_isomer_state(target_identity)
    source_specific_isomer = source_isomer_state - {"unspecified"}
    target_specific_isomer = target_isomer_state - {"unspecified"}
    if source_specific_isomer and target_specific_isomer and source_specific_isomer.isdisjoint(target_specific_isomer):
        return False, "different_explicit_fatty_acid_isomer_state"
    if source_specific_isomer and not target_specific_isomer:
        return False, "source_is_more_specific_fatty_acid_isomer"
    if target_specific_isomer and not source_specific_isomer:
        return False, "target_is_more_specific_fatty_acid_isomer"

    source_glycosides = _glycoside_markers(source_identity)
    target_glycosides = _glycoside_markers(target_identity)
    if source_glycosides != target_glycosides and (source_glycosides or target_glycosides):
        return False, "different_glycoside_or_acylated_form"
    source_aglycone = _glycoside_aglycone(source_identity)
    target_aglycone = _glycoside_aglycone(target_identity)
    if source_aglycone and target_aglycone and source_aglycone != target_aglycone:
        return False, "different_glycoside_aglycone"

    source_locators = _positional_locator_signatures(source_identity)
    target_locators = _positional_locator_signatures(target_identity)
    if source_locators != target_locators and (source_locators or target_locators):
        return False, "different_explicit_positional_isomer"

    source_small_molecule_position = _positioned_small_molecule_signature(source_identity)
    target_small_molecule_position = _positioned_small_molecule_signature(target_identity)
    if source_small_molecule_position != target_small_molecule_position and (
        source_small_molecule_position or target_small_molecule_position
    ):
        return False, "different_explicit_small_molecule_position"

    if exact_name:
        return True, "exact_name_or_registered_alias"
    if source_numbers and target_numbers and source_numbers == target_numbers:
        return True, "matching_explicit_structural_signature"
    if semantic_score < HIGH_SEMANTIC_REVIEW_THRESHOLD:
        return False, "semantic_score_below_review_threshold"
    if lexical_score >= MIN_LEXICAL_SUPPORT and token_overlap >= MIN_TOKEN_SUPPORT:
        return True, "strong_name_similarity_with_shared_identity_tokens"
    return False, "semantic_similarity_without_identity_bearing_name_evidence"


def _relation_hint(source: pd.Series, target: pd.Series, *, exact_name: bool,
                   semantic_score: float, lexical_score: float) -> tuple[str, str, str]:
    """Classify a candidate's review type without claiming biochemical identity."""
    source_text = " ".join([
        _clean(source.original_name), _clean(source.source_definition),
        _clean(source.analytical_method_or_expression), _clean(source.raw_unit),
    ]).casefold()
    target_text = " ".join([
        _clean(target.canonical_name), _clean(target.chemical_identity),
        _clean(target.axis_family),
    ]).casefold()
    derived = bool(re.search(r"\b(by difference|calculated|equivalent|rae|dfe|iu|energy)\b", source_text))
    aggregate = bool(re.search(r"\b(total|sum|aggregate|ratio|percentage|fraction)\b", source_text))
    form = bool(re.search(r"\b(cis|trans|isomer|vitamer|glycoside|ester|salt|d[23]|alpha|beta|gamma|delta)\b", source_text))
    if derived:
        return (
            "derived_or_activity_expression_not_mergeable",
            "exclude_from_direct_value_merge",
            "high",
        )
    if exact_name:
        return (
            "exact_name_candidate_requires_definition_check",
            "manual_identity_and_expression_review",
            "high" if aggregate or form else "medium",
        )
    if aggregate:
        return (
            "possible_aggregate_or_member_relation",
            "manual_relation_review_no_pooling",
            "high",
        )
    if form:
        return (
            "possible_chemical_form_or_method_variant",
            "manual_relation_review_no_pooling",
            "high",
        )
    if semantic_score >= HIGH_SEMANTIC_REVIEW_THRESHOLD and lexical_score >= 0.35:
        return (
            "high_semantic_same_analyte_candidate",
            "manual_identity_and_expression_review",
            "medium",
        )
    return (
        "semantic_nearest_neighbour_only",
        "do_not_merge_without_external_identity_evidence",
        "low",
    )


def _encode(texts: list[str], *, cache_dir: Path, cache_name: str, model_name: str) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / cache_name
    if cache.exists():
        values = np.load(cache)
        if len(values) == len(texts):
            return values.astype(np.float32)
        cache.unlink()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for semantic axis retrieval. "
            "Install requirements-colab.txt before running this audit."
        ) from error
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir / "model"))
    values = model.encode(
        texts, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache, values)
    return values


def _top_semantic_candidates(source_embeddings: np.ndarray, target_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(source_embeddings), 2048):
        block = source_embeddings[start:start + 2048] @ target_embeddings.T
        indices = np.argpartition(block, -TOP_K, axis=1)[:, -TOP_K:]
        scores = np.take_along_axis(block, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        all_indices.append(np.take_along_axis(indices, order, axis=1))
        all_scores.append(np.take_along_axis(scores, order, axis=1))
    return np.vstack(all_indices), np.vstack(all_scores)


def _top_source_candidates(target_embeddings: np.ndarray, source_embeddings: np.ndarray) -> np.ndarray:
    """Return the closest source-axis positions for every target axis.

    Retrieval must be bidirectional. A source axis can be closer to three
    other targets and therefore absent from a source-to-target top-three list,
    while still being the best available spelling variant for a given target.
    """
    all_indices: list[np.ndarray] = []
    for start in range(0, len(target_embeddings), 128):
        block = target_embeddings[start:start + 128] @ source_embeddings.T
        indices = np.argpartition(block, -TARGET_TO_SOURCE_TOP_K, axis=1)[:, -TARGET_TO_SOURCE_TOP_K:]
        scores = np.take_along_axis(block, indices, axis=1)
        order = np.argsort(-scores, axis=1)
        all_indices.append(np.take_along_axis(indices, order, axis=1))
    return np.vstack(all_indices)


def _candidate_frame(sources: pd.DataFrame, targets: pd.DataFrame,
                     source_embeddings: np.ndarray, target_embeddings: np.ndarray) -> pd.DataFrame:
    alias_index = _target_alias_index(targets)
    target_indices, _scores = _top_semantic_candidates(source_embeddings, target_embeddings)
    target_source_indices = _top_source_candidates(target_embeddings, source_embeddings)
    target_positions_by_source: dict[int, set[int]] = {
        source_position: set(target_indices[source_position])
        for source_position in range(len(sources))
    }
    for target_position, source_positions in enumerate(target_source_indices):
        for source_position in source_positions:
            target_positions_by_source[int(source_position)].add(target_position)
    records: list[dict[str, Any]] = []
    for source_position, source in sources.iterrows():
        source_aliases = _source_aliases(source)
        exact_target_ids = set().union(*(alias_index.get(alias, set()) for alias in source_aliases))
        candidate_positions = list(target_positions_by_source[source_position])
        for target_id in sorted(exact_target_ids):
            target_pos = int(targets.index[targets.target_axis_id.eq(target_id)][0])
            candidate_positions.append(target_pos)
        seen: set[str] = set()
        ranked = sorted(
            {int(position) for position in candidate_positions},
            key=lambda position: -float(source_embeddings[source_position] @ target_embeddings[position]),
        )
        for rank, target_pos in enumerate(ranked, start=1):
            target = targets.iloc[int(target_pos)]
            if target.target_axis_id in seen:
                continue
            seen.add(target.target_axis_id)
            source_name = _clean(source.original_name) or _clean(source.original_name_local)
            lexical, overlap, variant_exact_name = _best_name_similarity(source, target)
            match_basis = _exact_match_basis(source, target)
            exact_name = target.target_axis_id in exact_target_ids or variant_exact_name
            same_analyte_lead, identity_evidence = _identity_evidence(
                source, target, exact_name=exact_name,
                semantic_score=float(source_embeddings[source_position] @ target_embeddings[target_pos]), lexical_score=lexical,
                token_overlap=overlap,
            )
            if not same_analyte_lead:
                continue
            if identity_evidence == "exact_name_or_registered_alias":
                identity_evidence = match_basis or "normalized_name_variant"
            relation, decision, priority = _relation_hint(
                source, target, exact_name=exact_name,
                semantic_score=float(source_embeddings[source_position] @ target_embeddings[target_pos]), lexical_score=lexical,
            )
            records.append({
                "candidate_id": stable_id("semantic_axis_candidate", source.source_axis_id, target.target_axis_id),
                "source_axis_id": source.source_axis_id,
                "source_key": source.source_key,
                "source_name": source.source_name,
                "source_component_id": source.source_component_id,
                "source_original_name": source_name,
                "source_original_name_local": _clean(source.original_name_local),
                "source_definition": _clean(source.source_definition),
                "source_component_group": _clean(source.source_component_group),
                "source_chemical_class": _clean(source.source_chemical_class),
                "source_infoods_tag": _clean(source.infoods_tag),
                "source_inchi": _clean(source.inchi),
                "source_raw_unit": _clean(source.raw_unit),
                "source_raw_denominator": _clean(source.raw_denominator),
                "source_expression": _clean(source.analytical_method_or_expression),
                "source_value_availability": _clean(source.value_availability),
                "source_has_numeric_values": _clean(source.value_availability) in NUMERIC_AVAILABILITY,
                "target_axis_id": target.target_axis_id,
                "target_canonical_name": target.canonical_name,
                "target_axis_family": target.axis_family,
                "target_chemical_identity": target.chemical_identity,
                "target_training_stage": target.recommended_training_stage,
                "candidate_rank": rank,
                "exact_name_or_registered_alias": exact_name,
                "exact_match_basis": match_basis,
                "identity_evidence": identity_evidence,
                "candidate_class": "same_analyte_manual_resolution_lead",
                "semantic_cosine_similarity": round(float(source_embeddings[source_position] @ target_embeddings[target_pos]), 6),
                "lexical_similarity": round(lexical, 6),
                "informative_token_overlap": round(overlap, 6),
                "proposed_relation_for_review": relation,
                "automated_merge_decision": decision,
                "review_priority": priority,
                "numerical_value_merge_permitted": False,
            })
    return pd.DataFrame(records)


def _target_coverage(targets: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        subset = candidates[candidates.target_axis_id.eq(target.target_axis_id)]
        exact = subset[subset.exact_name_or_registered_alias]
        direct_exact = exact[~exact.exact_match_basis.eq("registered_source_native_alias")]
        source_code_exact = exact[exact.exact_match_basis.eq("registered_source_native_alias")]
        high_semantic = subset[
            (~subset.exact_name_or_registered_alias)
            & (subset.semantic_cosine_similarity >= HIGH_SEMANTIC_REVIEW_THRESHOLD)
        ]
        numeric_direct_exact = direct_exact[direct_exact.source_has_numeric_values]
        numeric_source_code_exact = source_code_exact[source_code_exact.source_has_numeric_values]
        numeric_high = high_semantic[high_semantic.source_has_numeric_values]
        if len(numeric_direct_exact):
            status = "numeric_exact_name_candidate_present_requires_identity_check"
        elif len(numeric_source_code_exact):
            status = "numeric_registered_source_code_candidate_requires_mapping_check"
        elif len(numeric_high):
            status = "numeric_high_semantic_candidate_requires_identity_check"
        elif len(direct_exact):
            status = "catalogue_exact_name_candidate_but_numeric_value_not_confirmed"
        elif len(source_code_exact):
            status = "catalogue_registered_source_code_candidate_requires_mapping_check"
        elif len(high_semantic):
            status = "catalogue_high_semantic_candidate_requires_manual_review"
        else:
            status = "no_candidate_in_available_source_axis_catalogues"
        rows.append({
            "target_axis_id": target.target_axis_id,
            "canonical_name": target.canonical_name,
            "axis_family": target.axis_family,
            "selection_tier": target.selection_tier,
            "recommended_training_stage": target.recommended_training_stage,
            "exact_name_candidate_axis_count": int(len(exact)),
            "exact_name_candidate_source_count": int(exact.source_key.nunique()),
            "numeric_exact_name_candidate_axis_count": int(len(numeric_direct_exact)),
            "numeric_exact_name_candidate_source_count": int(numeric_direct_exact.source_key.nunique()),
            "registered_source_code_candidate_axis_count": int(len(source_code_exact)),
            "registered_source_code_candidate_source_count": int(source_code_exact.source_key.nunique()),
            "numeric_registered_source_code_candidate_axis_count": int(len(numeric_source_code_exact)),
            "numeric_registered_source_code_candidate_source_count": int(numeric_source_code_exact.source_key.nunique()),
            "high_semantic_candidate_axis_count": int(len(high_semantic)),
            "high_semantic_candidate_source_count": int(high_semantic.source_key.nunique()),
            "numeric_high_semantic_candidate_axis_count": int(len(numeric_high)),
            "numeric_high_semantic_candidate_source_count": int(numeric_high.source_key.nunique()),
            "best_semantic_similarity": float(subset.semantic_cosine_similarity.max()) if len(subset) else np.nan,
            "coverage_status": status,
            "numeric_value_merge_permitted": False,
        })
    return pd.DataFrame(rows)


def _source_coverage(source_registry: pd.DataFrame, sources: pd.DataFrame,
                     candidates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for source in source_registry.itertuples(index=False):
        source_axes = sources[sources.source_key.eq(source.source_key)]
        source_candidates = candidates[candidates.source_key.eq(source.source_key)]
        records.append({
            "source_key": source.source_key,
            "source_name": source.name,
            "region": source.region,
            "access_status": source.access_status,
            "catalogued_source_axis_count": int(len(source_axes)),
            "numeric_source_axis_count": int(source_axes.value_availability.isin(NUMERIC_AVAILABILITY).sum()),
            "exact_name_candidate_axis_count": int(source_candidates.exact_name_or_registered_alias.sum()),
            "high_semantic_candidate_axis_count": int((
                (~source_candidates.exact_name_or_registered_alias)
                & (source_candidates.semantic_cosine_similarity >= HIGH_SEMANTIC_REVIEW_THRESHOLD)
            ).sum()),
            "candidate_target_count": int(source_candidates.target_axis_id.nunique()),
            "coverage_interpretation": (
                "No local axis catalogue was available; this is an acquisition gap, not evidence of absent composition data."
                if len(source_axes) == 0 else
                "Candidate coverage only; every candidate requires identity and expression review before any value use."
            ),
        })
    return pd.DataFrame(records)


def _unresolved_target_neighbours(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    coverage: pd.DataFrame,
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Provide discovery-only nearby source labels for unresolved target axes.

    These rows are deliberately separate from ``semantic_axis_candidates``.
    They answer whether an apparent gap may be a naming/catalogue issue, but
    they have no identity or numerical-mapping status.
    """
    unresolved_ids = set(
        coverage.loc[
            coverage.coverage_status.eq("no_candidate_in_available_source_axis_catalogues"),
            "target_axis_id",
        ]
    )
    numeric_positions = np.flatnonzero(sources.value_availability.isin(NUMERIC_AVAILABILITY).to_numpy())
    records: list[dict[str, Any]] = []
    for target_position, target in targets.iterrows():
        if target.target_axis_id not in unresolved_ids:
            continue
        scores = source_embeddings[numeric_positions] @ target_embeddings[target_position]
        take = min(UNRESOLVED_NEIGHBOURS_PER_TARGET, len(scores))
        indices = np.argpartition(scores, -take)[-take:]
        indices = indices[np.argsort(-scores[indices])]
        for rank, relative_position in enumerate(indices, start=1):
            source = sources.iloc[int(numeric_positions[relative_position])]
            source_name = _clean(source.original_name) or _clean(source.original_name_local)
            lexical, overlap, exact = _best_name_similarity(source, target)
            same_analyte, rejection_or_evidence = _identity_evidence(
                source,
                target,
                exact_name=exact,
                semantic_score=float(scores[relative_position]),
                lexical_score=lexical,
                token_overlap=overlap,
            )
            records.append({
                "target_axis_id": target.target_axis_id,
                "target_canonical_name": target.canonical_name,
                "target_axis_family": target.axis_family,
                "target_training_stage": target.recommended_training_stage,
                "neighbour_rank": rank,
                "source_axis_id": source.source_axis_id,
                "source_key": source.source_key,
                "source_component_id": source.source_component_id,
                "source_original_name": source_name,
                "source_definition": _clean(source.source_definition),
                "source_component_group": _clean(source.source_component_group),
                "source_raw_unit": _clean(source.raw_unit),
                "source_raw_denominator": _clean(source.raw_denominator),
                "semantic_cosine_similarity": round(float(scores[relative_position]), 6),
                "lexical_similarity": round(lexical, 6),
                "informative_token_overlap": round(overlap, 6),
                "same_analyte_eligibility_under_current_rules": same_analyte,
                "identity_rejection_or_evidence": rejection_or_evidence,
                "manual_resolution_action": (
                    "Inspect source definition and an external chemical authority; do not merge numerical values."
                ),
                "numerical_value_merge_permitted": False,
            })
    return pd.DataFrame(records)


def _write_report(targets: pd.DataFrame, sources: pd.DataFrame, source_registry: pd.DataFrame,
                  candidates: pd.DataFrame, coverage: pd.DataFrame, source_coverage: pd.DataFrame,
                  path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = coverage.coverage_status.value_counts().sort_index()
    source_rows = "\n".join(
        f"| {row.source_name} | {row.region} | {int(row.catalogued_source_axis_count):,} | "
        f"{int(row.numeric_source_axis_count):,} | {int(row.candidate_target_count):,} |"
        for row in source_coverage.sort_values(["catalogued_source_axis_count", "source_key"], ascending=[False, True]).itertuples(index=False)
    )
    status_rows = "\n".join(f"| {key} | {int(value)} |" for key, value in status_counts.items())
    stage_rows: list[str] = []
    for stage, subset in coverage.groupby("recommended_training_stage", sort=True):
        counts = subset.coverage_status.value_counts()
        stage_rows.append(
            f"| {stage} | {len(subset)} | "
            f"{int(counts.get('numeric_exact_name_candidate_present_requires_identity_check', 0))} | "
            f"{int(counts.get('numeric_registered_source_code_candidate_requires_mapping_check', 0))} | "
            f"{int(counts.get('numeric_high_semantic_candidate_requires_identity_check', 0))} | "
            f"{int(counts.get('catalogue_exact_name_candidate_but_numeric_value_not_confirmed', 0) + counts.get('catalogue_high_semantic_candidate_requires_manual_review', 0))} | "
            f"{int(counts.get('no_candidate_in_available_source_axis_catalogues', 0))} |"
        )
    stage_rows_text = "\n".join(stage_rows)
    text = f"""# 369-Axis Global Source Coverage and Semantic-Match Audit

## Scope

This audit compares the **{len(targets)} scientifically selected direct prediction targets** with every available source-native axis in the global atlas. It covers **{len(sources):,} source-axis records** across **{source_registry.source_key.nunique()} registered sources**. Sources with no locally extracted catalogue are reported as acquisition gaps rather than as composition absence.

The audit has two intentionally distinct signals:

1. **Exact-name/registered-alias candidate:** normalized source label matches a panel label or retained alias.
2. **Multilingual semantic retrieval:** the nearest `{TOP_K}` targets for every source axis and the nearest `{TARGET_TO_SOURCE_TOP_K}` source axes for every target, from `{SEMANTIC_MODEL}`, using original label, definition, component group, chemical class and INFOODS tag where supplied.

Semantic retrieval is only a discovery step. A candidate is counted as a same-analyte review lead only if it also has one of: an exact retained alias, a matching explicit structural/vitamer signature, or strong name agreement with shared identity-bearing tokens. The audit rejects a semantic neighbour when its label discloses a different positional isomer, fatty-acid isomer state, glycoside/acylated form, vitamer, or aggregate definition. Neither signal establishes chemical identity, and every candidate has `numerical_value_merge_permitted = false`.

## Is the Target Panel Scientifically Complete?

Within the declared scope, yes: nutrition core targets cover matrix/proximate composition, macronutrient expressions, vitamins/vitamers, minerals, indispensable and other amino acids, fatty-acid forms and sterols. The extension covers defined food metabolites across the major specialist composition classes. This does **not** mean every target has a compatible numerical label in every source, and it does not claim that the complete food metabolome is finite or fully observed.

Direct prediction deliberately excludes energy, activity equivalents, calculated-by-difference expressions, non-specific colorimetric totals, contaminants and detected-only associations. Those are retained outside the direct target loss because treating them as molecular mass labels would introduce invalid equivalences.

## Target-Coverage Results

| Coverage status | Target axes |
|---|---:|
{status_rows}

`numeric_exact_name_candidate_present_requires_identity_check` is the strongest catalogue-level signal, not a completed merge. It only means that at least one source with local numeric values uses an exact retained label. A value can be pooled only after its chemical identity, expression, unit, denominator and analytical method have been checked.

`numeric_registered_source_code_candidate_requires_mapping_check` is separate: a source-native code was retained in the panel's provenance, but the source codebook and its measurement denominator still require verification. It is not counted as an exact natural-language name match.

## Coverage by Training Stage

| Training stage | Targets | Numeric name/alias lead | Numeric source-code lead | Numeric semantic lead | Catalogue-only lead | No safe lead |
|---|---:|---:|---:|---:|---:|---:|
{stage_rows_text}

The panel remains a scientific target specification, not a declaration that every target is ready for numerical training. In particular, the remaining Stage 1 gaps concentrate in narrow fatty-acid isomers and explicitly defined carbohydrate/fibre expressions; the Stage 2 gaps concentrate in specialist food-metabolome classes. These gaps should drive targeted catalogue acquisition or source-codebook review, not fuzzy value imputation.

## Source Catalogue Coverage

| Source | Region | Catalogued axes | Numeric axes | Candidate targets |
|---|---|---:|---:|---:|
{source_rows}

## Candidate Decision Rules

- **Exact name**: manually confirm definition, chemical form, method and denominator. Do not merge solely on spelling.
- **Possible aggregate/member or chemical-form relation**: add a relation edge such as `aggregate_of` or `chemical_form_of`; keep separate numerical axes unless a standards-backed conversion is valid.
- **Derived/activity expression**: preserve as provenance or an auxiliary derived output; do not merge into direct-mass targets.
- **High semantic neighbour only**: review with a chemical authority (INFOODS, ChEBI, LIPID MAPS, source definition). It is deliberately not a merge instruction.

## Deliverables

- `target_axis_coverage.csv`: one coverage row for each of 369 targets.
- `semantic_axis_candidates.csv.gz`: retained same-analyte review leads, with identity evidence and a no-merge decision.
- `semantic_review_queue.csv.gz`: every same-analyte lead requiring source/chemical identity review before any value mapping.
- `source_catalogue_coverage.csv`: all registered sources, including catalogue-acquisition gaps.
- `unresolved_target_semantic_neighbours.csv`: discovery-only nearby numeric source labels for target axes with no safe same-analyte lead; this is not a mapping table.
- `coverage_manifest.json`: model, thresholds and input version information.
"""
    path.write_text(text, encoding="utf-8")


def _write_report_zh(targets: pd.DataFrame, sources: pd.DataFrame, source_registry: pd.DataFrame,
                     candidates: pd.DataFrame, coverage: pd.DataFrame, source_coverage: pd.DataFrame,
                     path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = coverage.coverage_status.value_counts()
    local_sources = int((source_coverage.catalogued_source_axis_count > 0).sum())
    numeric_exact = int(counts.get("numeric_exact_name_candidate_present_requires_identity_check", 0))
    numeric_code = int(counts.get("numeric_registered_source_code_candidate_requires_mapping_check", 0))
    numeric_fuzzy = int(counts.get("numeric_high_semantic_candidate_requires_identity_check", 0))
    catalogue_only = int(counts.get("catalogue_exact_name_candidate_but_numeric_value_not_confirmed", 0)) + int(
        counts.get("catalogue_high_semantic_candidate_requires_manual_review", 0)
    )
    no_candidate = int(counts.get("no_candidate_in_available_source_axis_catalogues", 0))
    stage_rows: list[str] = []
    for stage, subset in coverage.groupby("recommended_training_stage", sort=True):
        values = subset.coverage_status.value_counts()
        stage_rows.append(
            f"| {stage} | {len(subset)} | "
            f"{int(values.get('numeric_exact_name_candidate_present_requires_identity_check', 0))} | "
            f"{int(values.get('numeric_registered_source_code_candidate_requires_mapping_check', 0))} | "
            f"{int(values.get('numeric_high_semantic_candidate_requires_identity_check', 0))} | "
            f"{int(values.get('catalogue_exact_name_candidate_but_numeric_value_not_confirmed', 0) + values.get('catalogue_high_semantic_candidate_requires_manual_review', 0))} | "
            f"{int(values.get('no_candidate_in_available_source_axis_catalogues', 0))} |"
        )
    stage_rows_text = "\n".join(stage_rows)
    text = f"""# 369 个预测轴的全球来源覆盖与语义名称审核

## 范围与结论

本审核将最终的 **{len(targets)} 个直接预测轴** 与全球图谱中 **{len(sources):,} 个来源原生轴**进行比对。图谱登记 {source_registry.source_key.nunique()} 个来源，其中 {local_sources} 个已有本地可机读的轴目录。没有本地目录的来源被标为**获取缺口**，不能据此认定其不含相应食品成分。

最终面板在预先声明的营养核心与食品代谢组扩展范围内是完整的：它涵盖食品基质、宏量营养表达、维生素及其化学形式、矿物质、氨基酸、脂肪酸和甾醇，以及有明确结构的食品来源代谢物。这里的“完整”不表示每一来源都能提供每一轴的兼容数值，也不表示食品代谢组已经被穷尽。

## 覆盖结果

- **{numeric_exact} 个轴**：至少一个有本地数值的来源以精确保留名称或已登记别名出现。
- **{numeric_code} 个轴**：仅通过已保留的来源原生代码找到候选；需要核对该来源的官方代码表、定义和分母，不能视为自然语言精确匹配。
- **{numeric_fuzzy} 个轴**：没有精确保留名称，但存在“结构编号/维生素形式一致”或“强词面一致”的有数值候选；仍须人工确认。
- **{catalogue_only} 个轴**：在目录中找到名称候选，但该目录未确认有可用于食品层面的数值。
- **{no_candidate} 个轴**：在当前已获取的来源目录中没有同一分析物候选；这是定向补数或补目录的优先清单，而不是删除该科学目标的理由。

## 按训练阶段的覆盖

| 训练阶段 | 目标轴 | 有数值的名称/别名候选 | 有数值的来源代码候选 | 有数值的语义候选 | 仅目录候选 | 无安全候选 |
|---|---:|---:|---:|---:|---:|---:|
{stage_rows_text}

369 个轴仍是科学目标定义，不代表每一个都已经可以进入数值训练。Stage 1 的剩余缺口主要是极细的脂肪酸异构体，以及定义明确的碳水化合物/膳食纤维表达；Stage 2 的缺口主要位于需要专门检测的食品代谢物类别。这些缺口应指导定向获取专家数据库或审核来源代码表，不能用模糊匹配或插补来填补。

## 语义匹配如何避免错误合并

多语言句向量模型只负责从全部来源名称中找出最接近的候选。一个候选只有在满足以下至少一项时，才会被计入上述覆盖：

1. 来源名称与目标的规范名称或已登记别名完全一致；
2. 脂肪酸碳数/双键/omega 编号，或维生素/甲基萘醌编号一致；
3. 在名称变体中同时具有高词面相似度和共享的身份词。

审核会排除或单列以下不能合并的情况：不同位置异构体（如 `3,4-` 与 `4,5-`）、不同 cis/trans/共轭状态、不同糖苷或酰化形式、不同维生素化学形式、总量/子项关系、活性当量和计算表达。即使名称很相似，这些情况也不能成为同一回归轴，更不能直接合并数值。

## 合并决定

本次没有自动合并任何来源数值。每一候选均保存 `numerical_value_merge_permitted = false`。下一步必须针对候选逐项核对：化学身份、分析定义、单位、分母（鲜重/干重/脂肪酸百分比等）与方法。只有这些条件兼容，才可以把来源值映射到同一最终预测轴；不兼容的记录应保留为独立的 `chemical_form_of`、`aggregate_of` 或 `method_variant_of` 关系。

## 文件

- `target_axis_coverage.csv`：每个目标轴的覆盖状态。
- `semantic_axis_candidates.csv.gz`：通过身份证据门槛的同一分析物候选。
- `semantic_review_queue.csv.gz`：需要人工作身份与表达审核的全量候选。
- `source_catalogue_coverage.csv`：48 个来源的目录状态与候选覆盖。
- `unresolved_target_semantic_neighbours.csv`：对当前无安全候选的目标轴给出最接近的有数值来源名称，仅用于人工定位，不是映射或合并依据。
- `coverage_manifest.json`：模型、阈值和输入版本。
"""
    path.write_text(text, encoding="utf-8")


def audit_semantic_axis_coverage(
    *,
    target_registry_path: Path,
    atlas_dir: Path,
    output_dir: Path,
    report_dir: Path,
    cache_dir: Path,
    model_name: str = SEMANTIC_MODEL,
) -> dict[str, Any]:
    if not target_registry_path.exists():
        raise FileNotFoundError(f"Prediction-axis registry not found: {target_registry_path}")
    targets = pd.read_csv(target_registry_path, keep_default_na=False)
    sources = pd.read_csv(atlas_dir / "source_axis_catalog.csv.gz", keep_default_na=False, low_memory=False)
    source_registry = pd.read_csv(atlas_dir / "source_registry.csv", keep_default_na=False)
    required_targets = {"target_axis_id", "canonical_name", "aliases_for_review", "axis_family", "chemical_identity"}
    required_sources = {"source_axis_id", "source_key", "original_name", "value_availability"}
    if missing := required_targets - set(targets.columns):
        raise ValueError(f"Target registry missing required columns: {sorted(missing)}")
    if missing := required_sources - set(sources.columns):
        raise ValueError(f"Source axis catalogue missing required columns: {sorted(missing)}")
    targets = targets.reset_index(drop=True)
    sources = sources.reset_index(drop=True)

    target_embeddings = _encode(
        [_target_text(row) for _, row in targets.iterrows()],
        cache_dir=cache_dir, cache_name="target_axis_embeddings.npy", model_name=model_name,
    )
    source_embeddings = _encode(
        [_source_text(row) for _, row in sources.iterrows()],
        cache_dir=cache_dir, cache_name="source_axis_embeddings.npy", model_name=model_name,
    )
    candidates = _candidate_frame(sources, targets, source_embeddings, target_embeddings)
    coverage = _target_coverage(targets, candidates)
    source_coverage = _source_coverage(source_registry, sources, candidates)
    unresolved_neighbours = _unresolved_target_neighbours(
        sources, targets, coverage, source_embeddings, target_embeddings,
    )
    review_queue = candidates.sort_values(
        ["review_priority", "semantic_cosine_similarity", "source_has_numeric_values"],
        ascending=[True, False, False],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(coverage, output_dir / "target_axis_coverage.csv")
    candidates.to_csv(output_dir / "semantic_axis_candidates.csv.gz", index=False, compression="gzip")
    review_queue.to_csv(output_dir / "semantic_review_queue.csv.gz", index=False, compression="gzip")
    write_csv(unresolved_neighbours, output_dir / "unresolved_target_semantic_neighbours.csv")
    write_csv(source_coverage, output_dir / "source_catalogue_coverage.csv")
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_model": model_name,
        "source_to_target_top_k": TOP_K,
        "target_to_source_top_k": TARGET_TO_SOURCE_TOP_K,
        "high_semantic_review_threshold": HIGH_SEMANTIC_REVIEW_THRESHOLD,
        "lexical_similarity_threshold": MIN_LEXICAL_SUPPORT,
        "informative_token_overlap_threshold": MIN_TOKEN_SUPPORT,
        "target_count": int(len(targets)),
        "source_axis_count": int(len(sources)),
        "source_registry_count": int(len(source_registry)),
        "candidate_count": int(len(candidates)),
        "review_queue_count": int(len(review_queue)),
        "unresolved_target_count": int(coverage.coverage_status.eq("no_candidate_in_available_source_axis_catalogues").sum()),
        "unresolved_target_neighbour_count": int(len(unresolved_neighbours)),
        "numeric_value_merge_permitted": False,
    }
    write_json(manifest, output_dir / "coverage_manifest.json")
    _write_report(targets, sources, source_registry, candidates, coverage, source_coverage,
                  report_dir / "SEMANTIC_AXIS_COVERAGE_AUDIT.md")
    _write_report_zh(targets, sources, source_registry, candidates, coverage, source_coverage,
                     report_dir / "SEMANTIC_AXIS_COVERAGE_AUDIT_ZH.md")
    return manifest

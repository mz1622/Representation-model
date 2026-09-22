"""Global, provenance-preserving audit for a frozen prediction-axis panel.

The module intentionally has a narrow boundary:

* it maps source-native component expressions to an already frozen target panel;
* it reads every locally usable numerical source with a reviewed adapter;
* it only creates exact-name food candidate groups after all source food rows exist;
* it never pools, selects, averages, or imputes numerical measurements.

Unavailable, PDF-only, or terms-restricted sources remain in the source ledger with
an explicit reason.  Their absence from the numerical layer is therefore auditable,
rather than looking like an absence of geographic coverage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .frozen_food_audit import (
    _category_metadata_coverage,
    _food_group_distribution,
    _source_distribution,
    build_exact_name_food_groups,
    exact_label_key,
    summarize_exact_name_axis_cells,
)
from .util import normalize_text, sha256_file, stable_id, write_csv, write_json


GLOBAL_AUDIT_VERSION = "global_frozen_prediction_panel_food_audit_v8"
_CROSSWALK_LOOKUP_CACHE: dict[int, dict[str, Any]] = {}
ACQUIRED_SOURCE_REGISTRY = Path("data/raw/source_acquisition_2026_09_16/acquired_source_registry.csv")

# The v3 staging directory predates the global registry and uses shorter
# directory names.  This is a source-identity alias, not a food/value merge.
STAGING_TO_REGISTRY_KEY = {
    "afcd": "afcd_release_3",
    "ciqual": "ciqual_2025",
    "cnf": "cnf",
    "cofid": "cofid_2021",
    "fndds": "usda_fndds_2021_2023",
    "foodb": "foodb_2020",
    "frida": "frida_6_1",
    "norway": "norwegian_fcdb",
    "usda_foundation": "usda_foundation",
    "usda_sr_legacy": "usda_sr_legacy",
}

# A limited mapping for standard INFOODS labels that occur as *column codes*
# in the downloaded source tables.  Every entry denotes the same direct
# component expression; calculated-by-difference and activity-equivalent
# forms are deliberately omitted.
STANDARD_TAG_ALIASES = {
    "WATER": "Water",
    "ASH": "Ash",
    "PROT": "Protein, total",
    "PROCNT": "Protein, total",
    "PROTCNT": "Protein, total",
    "FAT": "Fat, total",
    "FAT-": "Fat, total",
    "FATCE": "Fat, total",
    "CHOAVL": "Available carbohydrate",
    "FIBTG": "Dietary fibre, total",
    "ALC": "Ethanol",
    "CA": "Calcium",
    "FE": "Iron",
    "MG": "Magnesium",
    "P": "Phosphorus",
    "K": "Potassium",
    "NA": "Sodium",
    "ZN": "Zinc",
    "CU": "Copper",
    "MN": "Manganese",
    "SE": "Selenium, Se",
    "RETOL": "Retinol",
    "CARTA": "Carotene, alpha",
    "CARTB": "Carotene, beta",
    "CRYPXB": "Cryptoxanthin, beta",
    "LYCPN": "Lycopene",
    "LUTN": "Lutein",
    "LUTNZEA": "Lutein + zeaxanthin",
    "VITD": "Vitamin D (D2 + D3)",
    "CHOCAL": "Cholecalciferol (vitamin D3)",
    "THIA": "Thiamin (vitamin B1)",
    "RIBF": "Riboflavin (vitamin B2)",
    "NIA": "Niacin",
    "PANTAC": "Pantothenic acid (vitamin B5)",
    "BIOT": "Biotin",
    "FOL": "Folate, total",
    "FOLFD": "Folate, food",
    "FOLAC": "Folic acid",
    "VITB12": "Cobalamin (vitamin B12)",
    "VITC": "Vitamin C",
    "CHOL-": "Cholesterol",
    "CHOLE": "Cholesterol",
    "FASAT": "Fatty acids, total saturated",
    "FAMS": "Fatty acids, total monounsaturated",
    "FAPU": "Fatty acids, total polyunsaturated",
    "FATRN": "Fatty acids, total trans",
    "F4D0": "Butyric acid (4:0)",
    "F6D0": "Hexanoic acid (6:0)",
    "F8D0": "Octanoic acid (8:0)",
    "F10D0": "Decanoic acid (10:0)",
    "F12D0": "Dodecanoic acid (12:0)",
    "F13D0": "Tridecanoic acid (13:0)",
    "F14D0": "Tetradecanoic acid (14:0)",
    "F15D0": "Pentadecanoic acid (15:0)",
    "F16D0": "Hexadecanoic acid (16:0)",
    "F17D0": "Heptadecanoic acid (17:0)",
    "F18D0": "Octadecanoic acid (18:0)",
    "F20D0": "Eicosanoic acid (20:0)",
    "F22D0": "Docosanoic acid (22:0)",
    "F24D0": "Tetracosanoic acid (24:0)",
    "F14D1": "Tetradecenoic acid (14:1; isomer unspecified)",
    "F15D1": "Pentadecenoic acid (15:1; isomer unspecified)",
    "F17D1": "Heptadecenoic acid (17:1; isomer unspecified)",
    "F20D1": "Eicosenoic acid (20:1; isomer unspecified)",
    "F18D2CN6": "Linoleic acid (18:2n-6)",
    "F18D3CN3": "Alpha-linolenic acid (18:3n-3)",
    "F20D4N6": "Arachidonic acid (20:4n-6)",
    "F20D5N3": "Eicosapentaenoic acid (EPA; 20:5n-3)",
    "F22D5N3": "Docosapentaenoic acid (DPA; 22:5n-3)",
    "F22D6N3": "Docosahexaenoic acid (DHA; 22:6n-3)",
    "ILE": "Isoleucine",
    "LEU": "Leucine",
    "LYS": "Lysine",
    "MET": "Methionine",
    "CYS": "Cystine",
    "PHE": "Phenylalanine",
    "TYR": "Tyrosine",
    "THR": "Threonine",
    "TRP": "Tryptophan",
    "VAL": "Valine",
    "ARG": "Arginine",
    "HIS": "Histidine",
    "ALA": "Alanine",
    "ASP": "Aspartic acid",
    "GLU": "Glutamic acid",
    "GLY": "Glycine",
    "PRO": "Proline",
    "SER": "Serine",
    "STARCH": "Starch",
    "SUGAR": "Sugars, total",
    "FRUS": "Fructose",
    "GLUS": "Glucose",
    "LACS": "Lactose",
    "MALS": "Maltose",
}

SOURCE_METADATA_COLUMNS = [
    "scientific_name", "food_group", "food_subgroup", "food_type", "part", "maturity",
    "processing", "cooking", "preservation", "physical_state", "packing_medium",
    "geography", "cultivar", "recipe_status", "brand_status", "edible_status",
    # Preserve authority and source-native identifiers for hierarchy review.
    # Their presence does not itself merge foods or create a hierarchy edge.
    "foodon_id", "foodex2_code", "langual_code", "taxonomy_id",
]


@dataclass
class SourceBundle:
    """Source-native food, component, and measurement records.

    Every measurement is retained at its source cell granularity.  `direct_mass`
    means only that a record is a finite numerical mass value on a stated
    100-g edible/fresh-weight denominator, not that it is an independent assay.
    """

    foods: pd.DataFrame
    components: pd.DataFrame
    measurements: pd.DataFrame
    provenance: dict[str, Any]


def _load_source_registry(root: Path) -> pd.DataFrame:
    """Load the immutable inventory plus separately versioned acquired releases."""
    registry_path = root / "data/processed/global_food_metabolome_axis_atlas_v1/source_registry.csv"
    registry = pd.read_csv(registry_path, keep_default_na=False)
    supplement_path = root / ACQUIRED_SOURCE_REGISTRY
    if supplement_path.exists():
        supplement = pd.read_csv(supplement_path, keep_default_na=False)
        registry = pd.concat([registry, supplement], ignore_index=True, sort=False)
    if registry.source_key.duplicated().any():
        duplicate = registry.loc[registry.source_key.duplicated(keep=False), "source_key"].tolist()
        raise ValueError(f"Source registry keys must be unique; duplicates={duplicate}")
    return registry


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _as_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.number)):
        result = float(value)
        return result if np.isfinite(result) else None
    text = _text(value).replace("\u00a0", " ")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _parse_value(value: Any) -> tuple[float | None, str, bool]:
    """Parse a table cell without converting trace/censored values to zero."""
    raw = _text(value)
    if not raw or raw.casefold() in {"na", "n/a", "nan", "n.i.", "-", "--", "…"}:
        return None, "missing_or_not_reported", False
    if raw.casefold() in {"tr", "trace", "spur", "spuren", "<", "<lod"} or raw.startswith("<"):
        return None, "trace_or_censored", False
    bracketed = (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("(") and raw.endswith(")"))
    inner = raw[1:-1].strip() if bracketed else raw
    number = _as_number(inner)
    if number is None:
        return None, "unparsed_non_numeric", False
    if number < 0:
        return number, "invalid_negative", False
    if bracketed:
        return number, "bracketed_or_parenthesized_estimate", False
    return number, "explicit_zero" if number == 0 else "observed", True


def _unit_factor(unit: str) -> float | None:
    text = _text(unit).casefold().replace("μ", "µ").replace("mcg", "µg")
    if "kilogram" in text:
        return 1000.0
    if "milligram" in text:
        return 1e-3
    if "microgram" in text:
        return 1e-6
    if "gram" in text:
        return 1.0
    if re.search(r"(?:^|[^a-z])kg(?:$|[^a-z])", text):
        return 1000.0
    if "mg" in text:
        return 1e-3
    if "µg" in text or "ug" in text:
        return 1e-6
    if re.search(r"(?:^|[^a-z])g(?:$|[^a-z])", text) or text in {"g", "gram"}:
        return 1.0
    return None


def _header_unit(value: Any) -> str:
    text = _text(value)
    match = re.search(r"\(([^()]*)\)\s*$", text)
    if match:
        return match.group(1).strip()
    if re.search(r"(?:^|\s)(?:mcg|µg|ug|mg|g)(?:\s|$)", text, flags=re.IGNORECASE):
        return text
    return ""


def _column_code(value: Any) -> str:
    text = _text(value)
    text = re.sub(r"\([^()]*\)", "", text).strip()
    text = text.split("/")[0].strip()
    return re.sub(r"\s+", "", text).upper()


def _source_value_column_codes(value: Any) -> list[str]:
    """Return source-native component-code candidates for a value column.

    INFOODS workbooks commonly encode a quantity in the header, for example
    ``WATER(G)`` or ``PHYTC- (MG)``. Their component dictionaries store the
    same identifier without that final unit suffix. This is a deterministic
    source-format normalization, not a semantic match.
    """
    raw = _text(value).upper()
    if not raw:
        return []
    without_trailing_unit = re.sub(
        r"\s*\(\s*(?:KG|G|MG|MCG|UG|µG|KJ|KCAL)\s*\)\s*$", "", raw
    ).strip()
    candidates = [raw, without_trailing_unit, re.sub(r"\s+", "", without_trailing_unit)]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _make_food(source_key: str, food_id: Any, name: Any, **metadata: Any) -> dict[str, Any]:
    source_food_id = _text(food_id)
    # International workbooks can reuse a local food code in several country
    # sheets. Keep the raw code, but scope the source-native observation ID by
    # its table and original row so those records cannot collide or be silently
    # merged. Reused food codes can legitimately denote distinct samples.
    source_table = _text(metadata.get("source_table", ""))
    source_row = metadata.get("source_row", "")
    return {
        "food_observation_id": stable_id(
            "global_food_observation", source_key, source_table, source_food_id, source_row
        ),
        "source_key": source_key,
        "source_food_id": source_food_id,
        "original_name": _text(name),
        "original_name_local": _text(metadata.pop("original_name_local", "")),
        **{column: _text(metadata.pop(column, "")) for column in SOURCE_METADATA_COLUMNS},
        "source_record_url": _text(metadata.pop("source_record_url", "")),
        "source_lineage": _text(metadata.pop("source_lineage", "")),
        "source_table": _text(metadata.pop("source_table", "")),
        "source_row": source_row,
    }


def _make_component(source_key: str, component_id: Any, name: Any, *, unit: Any = "",
                    infoods_tag: Any = "", definition: Any = "", table: Any = "") -> dict[str, Any]:
    component_id = _text(component_id)
    return {
        "component_observation_id": stable_id("global_component_observation", source_key, component_id, name, unit, table),
        "source_key": source_key,
        "source_component_id": component_id,
        "original_name": _text(name),
        "original_unit": _text(unit),
        "infoods_tag": _text(infoods_tag).upper(),
        "source_definition": _text(definition),
        "source_table": _text(table),
    }


def _make_measurement(source_key: str, food_id: Any, component: dict[str, Any], value: Any, *, unit: Any,
                      table: str, source_row: int | str, source_column: int | str,
                      basis: str, source_value_origin: str = "source_reported",
                      direct_basis: bool = True, note: str = "") -> dict[str, Any]:
    numeric, status, direct_numeric = _parse_value(value)
    factor = _unit_factor(unit)
    normalized = numeric * factor if numeric is not None and factor is not None else np.nan
    direct_mass = bool(
        direct_numeric
        and direct_basis
        and factor is not None
        and np.isfinite(normalized)
        and 0 <= normalized <= 100
    )
    measurement_id = stable_id(
        "global_measurement", source_key, food_id, component["component_observation_id"], table,
        source_row, source_column,
    )
    return {
        "measurement_id": measurement_id,
        "source_key": source_key,
        "food_observation_id": stable_id(
            "global_food_observation", source_key, table, _text(food_id), source_row
        ),
        "component_observation_id": component["component_observation_id"],
        "source_measurement_locator": f"{table}:row={source_row};column={source_column}",
        "raw_value": _text(value),
        "numeric_value": numeric,
        "raw_unit": _text(unit),
        "raw_basis": basis,
        "value_status": status,
        "normalized_value_g_per_100g": normalized,
        "conversion_factor_to_g_per_100g": factor if factor is not None else np.nan,
        "conversion_status": "converted_exact_mass" if factor is not None and direct_basis else "not_direct_fresh_weight_mass",
        "measurement_modality": "mass_fraction_fresh_weight" if factor is not None and direct_basis else "other_or_unverified_modality",
        "source_value_origin": source_value_origin,
        "direct_mass_label_candidate": direct_mass,
        "source_note": _text(note),
    }


def _target_name_index(panel: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in panel.itertuples(index=False):
        result[normalize_text(row.canonical_name)] = row.target_axis_id
        for field in ("original_axis_names", "aliases_for_review"):
            value = _text(getattr(row, field))
            for alias in re.split(r"\s*[;|]\s*", value):
                key = normalize_text(alias)
                if key:
                    result.setdefault(key, row.target_axis_id)
    return result


def _candidate_score(source_name: str, source_tag: str, candidate: pd.Series) -> float:
    """Resolve only an unambiguous candidate emitted by the prior exact review."""
    score = 0.0
    source_key = normalize_text(source_name)
    target_key = normalize_text(candidate.target_canonical_name)
    candidate_source_key = normalize_text(candidate.source_original_name)
    if source_key and source_key == target_key:
        score += 100.0
    if source_key and source_key == candidate_source_key:
        score += 45.0
    tag = _text(source_tag).upper()
    if tag and tag == _text(candidate.source_infoods_tag).upper():
        score += 25.0
    # A deterministic lexical tie-breaker for competing exact-source-axis
    # candidates, such as FASAT/FAMS/FAPU. It never creates a new candidate.
    source_tokens = set(source_key.split())
    target_tokens = set(target_key.split())
    if source_tokens or target_tokens:
        score += len(source_tokens & target_tokens) / max(1, len(source_tokens | target_tokens))
    return score


def build_global_axis_crosswalk(root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """Map all atlas source axes to frozen targets only when exact review permits it."""
    source_axes = pd.read_csv(
        root / "data/processed/global_food_metabolome_axis_atlas_v1/source_axis_catalog.csv.gz",
        keep_default_na=False,
    )
    candidates = pd.read_csv(
        root / "data/processed/final_prediction_axis_semantic_coverage_v1/semantic_axis_candidates.csv.gz",
        keep_default_na=False,
    )
    candidates = candidates[
        candidates.exact_name_or_registered_alias.astype(str).str.casefold().eq("true")
    ].copy()
    target_names = panel.set_index("target_axis_id").canonical_name.to_dict()
    direct_name_index = _target_name_index(panel)
    rows: list[dict[str, Any]] = []

    for axis in source_axes.itertuples(index=False):
        source_key = _text(axis.source_key)
        component_id = _text(axis.source_component_id)
        axis_candidates = candidates[
            candidates.source_axis_id.eq(axis.source_axis_id)
            | ((candidates.source_key.eq(source_key)) & candidates.source_component_id.eq(component_id))
        ].copy()
        if axis_candidates.empty:
            name_target = direct_name_index.get(normalize_text(axis.original_name), "")
            tag_target_name = STANDARD_TAG_ALIASES.get(_text(axis.infoods_tag).upper(), "")
            tag_target = direct_name_index.get(normalize_text(tag_target_name), "")
            target_ids = {target for target in (name_target, tag_target) if target}
            mapping_status = "mapped_registered_exact_name_or_standard_tag" if len(target_ids) == 1 else (
                "ambiguous_exact_mapping" if len(target_ids) > 1 else "not_mapped_exactly"
            )
            target_axis_id = next(iter(target_ids), "") if len(target_ids) == 1 else ""
            evidence = "frozen_target_registered_name_or_curated_INFOODS_tag" if target_axis_id else ""
            candidate_target_ids = ";".join(sorted(target_ids))
        else:
            scored = axis_candidates.copy()
            scored["score"] = scored.apply(
                lambda candidate: _candidate_score(axis.original_name, axis.infoods_tag, candidate), axis=1
            )
            grouped = scored.groupby("target_axis_id", as_index=False).score.max().sort_values(
                ["score", "target_axis_id"], ascending=[False, True]
            )
            best_score = float(grouped.iloc[0].score)
            second_score = float(grouped.iloc[1].score) if len(grouped) > 1 else -np.inf
            if len(grouped) == 1 or best_score > second_score:
                target_axis_id = _text(grouped.iloc[0].target_axis_id)
                mapping_status = "mapped_prior_exact_review_unambiguous"
                evidence = "prior_exact_name_or_registered_alias_review"
            else:
                target_axis_id = ""
                mapping_status = "ambiguous_prior_exact_candidates"
                evidence = ""
            candidate_target_ids = ";".join(sorted(grouped.target_axis_id.astype(str).unique()))
        rows.append({
            "source_axis_id": axis.source_axis_id,
            "source_key": source_key,
            "source_component_id": component_id,
            "source_original_name": axis.original_name,
            "source_infoods_tag": axis.infoods_tag,
            "source_raw_unit": axis.raw_unit,
            "source_raw_denominator": axis.raw_denominator,
            "target_axis_id": target_axis_id,
            "target_axis_name": target_names.get(target_axis_id, ""),
            "mapping_status": mapping_status,
            "mapping_evidence": evidence,
            "candidate_target_axis_ids": candidate_target_ids,
        })
    result = pd.DataFrame(rows)
    if result.source_axis_id.duplicated().any():
        raise ValueError("Global axis crosswalk has duplicate source-axis IDs.")
    return result


def _axis_lookup(crosswalk: pd.DataFrame, source_key: str, component_id: str,
                 original_name: str, infoods_tag: str) -> tuple[str, str, str]:
    """Resolve a parsed component through the source-axis crosswalk without fuzzy matching."""
    lookup = _crosswalk_lookup_index(crosswalk)
    source_lookup = lookup["by_source"].get(source_key)
    if source_lookup is None:
        candidates = pd.DataFrame(columns=crosswalk.columns)
    else:
        row_positions = set(source_lookup["by_component_id"].get(_text(component_id), ()))
        row_positions.update(source_lookup["by_original_name"].get(normalize_text(original_name), ()))
        row_positions.update(source_lookup["by_infoods_tag"].get(_text(infoods_tag).upper(), ()))
        candidates = source_lookup["rows"].iloc[sorted(row_positions)].copy() if row_positions else pd.DataFrame(columns=crosswalk.columns)
    mapped = candidates[candidates.target_axis_id.ne("")].copy()
    if not mapped.empty:
        scores = []
        for row in mapped.itertuples(index=False):
            score = 0
            if _text(row.source_component_id) == _text(component_id):
                score += 100
            if normalize_text(row.source_original_name) == normalize_text(original_name):
                score += 40
            if _text(infoods_tag) and _text(row.source_infoods_tag).upper() == _text(infoods_tag).upper():
                score += 20
            scores.append(score)
        mapped["lookup_score"] = scores
        best = mapped.sort_values(["lookup_score", "target_axis_id"], ascending=[False, True])
        winner = best.iloc[0]
        if len(best) == 1 or winner.lookup_score > best.iloc[1].lookup_score:
            return _text(winner.target_axis_id), _text(winner.target_axis_name), _text(winner.mapping_evidence)

    # Final fallback is a documented standard component tag, not a semantic
    # similarity claim. It is useful where a workbook abbreviates a code that
    # differs from the atlas dictionary's component identifier.
    tag_target_name = STANDARD_TAG_ALIASES.get(_text(infoods_tag).upper(), "")
    if tag_target_name:
        targets = lookup["target_ids_by_name"].get(normalize_text(tag_target_name), ())
        if len(targets) == 1:
            return _text(next(iter(targets))), tag_target_name, "curated_standard_INFOODS_tag"
    return "", "", ""


def _crosswalk_lookup_index(crosswalk: pd.DataFrame) -> dict[str, Any]:
    """Build exact source-axis indexes once instead of repeatedly scanning a source table."""
    cache_key = id(crosswalk)
    cached = _CROSSWALK_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    by_source: dict[str, dict[str, Any]] = {}
    for source_key, group in crosswalk.groupby("source_key", sort=False):
        rows = group.reset_index(drop=True)
        by_component_id: dict[str, list[int]] = defaultdict(list)
        by_original_name: dict[str, list[int]] = defaultdict(list)
        by_infoods_tag: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(rows.itertuples(index=False)):
            by_component_id[_text(row.source_component_id)].append(position)
            by_original_name[normalize_text(row.source_original_name)].append(position)
            tag = _text(row.source_infoods_tag).upper()
            if tag:
                by_infoods_tag[tag].append(position)
        by_source[_text(source_key)] = {
            "rows": rows,
            "by_component_id": by_component_id,
            "by_original_name": by_original_name,
            "by_infoods_tag": by_infoods_tag,
        }

    target_ids_by_name: dict[str, set[str]] = defaultdict(set)
    for row in crosswalk.loc[crosswalk.target_axis_id.ne("")].itertuples(index=False):
        target_ids_by_name[normalize_text(row.target_axis_name)].add(_text(row.target_axis_id))
    cached = {"by_source": by_source, "target_ids_by_name": target_ids_by_name}
    _CROSSWALK_LOOKUP_CACHE[cache_key] = cached
    return cached


def _attach_target_mapping(bundle: SourceBundle, crosswalk: pd.DataFrame) -> SourceBundle:
    # Some registered, machine-readable sources contain no expression that is
    # eligible under the frozen panel's exact-mapping policy. They remain in
    # the source ledger with their food records and zero mapped values.
    if bundle.components.empty:
        return SourceBundle(
            foods=bundle.foods,
            components=bundle.components,
            measurements=bundle.measurements,
            provenance=bundle.provenance,
        )
    mapping_source_key = bundle.provenance["source_registry_key"]
    component_target_rows = []
    for row in bundle.components.itertuples(index=False):
        target_axis_id, target_axis_name, evidence = _axis_lookup(
            crosswalk, mapping_source_key, row.source_component_id, row.original_name, row.infoods_tag
        )
        component_target_rows.append({
            "component_observation_id": row.component_observation_id,
            "target_axis_id": target_axis_id,
            "target_axis_name": target_axis_name,
            "mapping_evidence": evidence,
            "mapping_status": "mapped_exact_or_registered" if target_axis_id else "not_mapped_exactly",
        })
    mapping = pd.DataFrame(component_target_rows)
    components = bundle.components.merge(mapping, on="component_observation_id", validate="one_to_one")
    if bundle.measurements.empty:
        return SourceBundle(foods=bundle.foods, components=components, measurements=bundle.measurements, provenance=bundle.provenance)
    measurements = bundle.measurements.merge(
        mapping[["component_observation_id", "target_axis_id", "target_axis_name", "mapping_evidence", "mapping_status"]],
        on="component_observation_id", validate="many_to_one",
    )
    return SourceBundle(foods=bundle.foods, components=components, measurements=measurements, provenance=bundle.provenance)


def _load_existing_staging(staging_dir: Path, crosswalk: pd.DataFrame) -> list[SourceBundle]:
    bundles: list[SourceBundle] = []
    for source_dir in sorted(path for path in staging_dir.iterdir() if path.is_dir()):
        source_key = source_dir.name
        food_path = source_dir / "food_observation.csv.gz"
        component_path = source_dir / "component_observation.csv.gz"
        if not food_path.exists() or not component_path.exists():
            raise FileNotFoundError(f"Incomplete staging source: {source_dir}")
        foods = pd.read_csv(food_path, keep_default_na=False, na_values=[""], low_memory=False)
        components = pd.read_csv(component_path, keep_default_na=False, na_values=[""], low_memory=False)
        food_columns = ["food_observation_id", "source_key", "source_food_id", "original_name", "original_name_local", *SOURCE_METADATA_COLUMNS, "source_record_url", "source_lineage_id"]
        for column in food_columns:
            if column not in foods:
                foods[column] = ""
        foods = foods[food_columns].rename(columns={"source_lineage_id": "source_lineage"})
        foods["source_table"] = "staging_catalogue"
        foods["source_row"] = ""
        component_columns = ["component_observation_id", "source_key", "source_component_id", "original_name", "original_unit", "infoods_tag", "source_definition"]
        for column in component_columns:
            if column not in components:
                components[column] = ""
        components = components[component_columns]
        components["source_table"] = "staging_catalogue"
        parts = []
        for path in sorted(source_dir.glob("measurement_part_*.csv.gz")):
            for chunk in pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False, chunksize=250_000):
                parts.append(chunk)
        if not parts:
            bundles.append(SourceBundle(
                foods=foods,
                components=components,
                measurements=pd.DataFrame(),
                provenance={
                    "source_key": source_key,
                    "source_registry_key": STAGING_TO_REGISTRY_KEY.get(source_key, source_key),
                    "ingestion_status": "catalogued_source_without_measurement_parts",
                    "input_paths": [str(food_path), str(component_path)],
                },
            ))
            continue
        raw = pd.concat(parts, ignore_index=True)
        required = ["measurement_id", "source_key", "food_observation_id", "component_observation_id", "raw_value", "numeric_value", "raw_unit", "raw_basis", "value_status", "normalized_value_g_per_100g", "conversion_factor", "conversion_status", "measurement_modality"]
        missing = sorted(set(required) - set(raw.columns))
        if missing:
            raise ValueError(f"Staging measurement schema missing {missing}: {source_dir}")
        measurements = raw[required].rename(columns={"conversion_factor": "conversion_factor_to_g_per_100g"})
        direct = (
            pd.to_numeric(measurements.normalized_value_g_per_100g, errors="coerce").notna()
            & measurements.value_status.astype(str).isin(["observed", "explicit_zero"])
            & measurements.conversion_status.eq("converted_exact_mass")
            & measurements.measurement_modality.eq("mass_fraction_fresh_weight")
        )
        measurements["source_measurement_locator"] = "source staging measurement record"
        measurements["source_value_origin"] = raw.get("value_origin", pd.Series("source_reported", index=raw.index)).fillna("source_reported")
        # A source-native record with an explicit fresh-weight mass basis is a
        # mathematically compatible reconstruction label regardless of whether
        # the source describes it as reported, linked, or calculated.  Keep
        # ``source_value_origin`` as provenance metadata; it must not silently
        # turn a valid mg/100 g -> g/100 g conversion into missing data.
        measurements["direct_mass_label_candidate"] = direct
        measurements["source_note"] = raw.get("method_expression", pd.Series("", index=raw.index)).fillna("")
        bundles.append(SourceBundle(
            foods=foods,
            components=components,
            measurements=measurements,
            provenance={
                "source_key": source_key,
                "source_registry_key": STAGING_TO_REGISTRY_KEY.get(source_key, source_key),
                "ingestion_status": "integrated_source_native_staging_all_components",
                "input_paths": [str(food_path), str(component_path), *[str(path) for path in sorted(source_dir.glob("measurement_part_*.csv.gz"))]],
            },
        ))
    return bundles


def _component_index(components: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["source_component_id"]: row for row in components}


def _bundle_from_wafct(path: Path) -> SourceBundle:
    source_key, table = "wafct_2019", "05 NV_sum_57 (per 100g EP)"
    dictionary = pd.read_excel(path, sheet_name="02 Components", header=0, keep_default_na=False)
    dictionary = dictionary.iloc[1:].copy()
    components: list[dict[str, Any]] = []
    definition_by_tag: dict[str, tuple[str, str, str]] = {}
    for _, row in dictionary.iterrows():
        tag = _text(row.get("INFOODS tagname", ""))
        name = _text(row.get("Component in English", ""))
        unit = _text(row.get("Unit", ""))
        if not tag or not name:
            continue
        definition_by_tag[tag] = (name, unit, _text(row.get("Analytical/determination method/definition in English", "")))
    raw = pd.read_excel(path, sheet_name=table, header=None, keep_default_na=False)
    headers, tags = raw.iloc[0].tolist(), raw.iloc[2].tolist()
    component_by_column: dict[int, dict[str, Any]] = {}
    for index, tag_value in enumerate(tags):
        tag = _text(tag_value)
        if not tag or index < 7:
            continue
        name, unit, definition = definition_by_tag.get(tag, (_text(headers[index]), _header_unit(headers[index]), ""))
        component = _make_component(source_key, tag, name, unit=unit, infoods_tag=tag, definition=definition, table=table)
        components.append(component)
        component_by_column[index] = component
    foods, measurements = [], []
    group = ""
    for row_index in range(3, len(raw)):
        row = raw.iloc[row_index]
        food_code = _text(row.iloc[0])
        if not re.fullmatch(r"\d{2}_\d{3}", food_code):
            if food_code:
                group = food_code
            continue
        source_reference = _text(row.iloc[4])
        origin = "recipe_calculated" if "calc. from recipe" in source_reference.casefold() else "source_reported"
        food = _make_food(
            source_key, food_code, row.iloc[1], original_name_local=row.iloc[2], scientific_name=row.iloc[3],
            food_group=group, recipe_status="recipe_calculated" if origin != "source_reported" else "not_reported",
            edible_status="edible_portion", source_lineage=source_reference, source_table=table, source_row=row_index + 1,
        )
        foods.append(food)
        for column, component in component_by_column.items():
            measurements.append(_make_measurement(
                source_key, food_code, component, row.iloc[column], unit=component["original_unit"],
                table=table, source_row=row_index + 1, source_column=column + 1, basis="per 100 g edible portion",
                source_value_origin=origin, direct_basis=True, note=source_reference,
            ))
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components).drop_duplicates("component_observation_id"), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source", "input_paths": [str(path)],
    })


def _bangladesh_component_dictionary(path: Path) -> dict[str, tuple[str, str]]:
    raw = pd.read_excel(path, sheet_name="Components", header=None, keep_default_na=False)
    rows: dict[str, tuple[str, str]] = {}
    for _, row in raw.iloc[6:].iterrows():
        name, unit, tag = (_text(row.iloc[i]) if len(row) > i else "" for i in range(3))
        if tag and name and tag.casefold() not in {"proximates", "minerals", "vitamins"}:
            rows[tag.upper()] = (name, unit)
    return rows


def _bundle_from_bangladesh(path: Path) -> SourceBundle:
    source_key = "bangladesh_fct_2013"
    dictionary = _bangladesh_component_dictionary(path)
    foods: list[dict[str, Any]] = []
    component_by_key: dict[str, dict[str, Any]] = {}
    measurements: list[dict[str, Any]] = []
    table_specs = [
        ("UserDB_Main_table", 0, 1, 0),
        ("Annex_Amio acids", 0, 1, 0),
        ("Annex_ Fatty acids", 0, 1, 0),
        ("Annex_Antinutrients", 0, 1, 0),
        ("Annex_Sugar", 0, 1, 0),
    ]
    for sheet, header_row, name_column, _ in table_specs:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, keep_default_na=False)
        headers = raw.iloc[header_row].tolist()
        units_row = raw.iloc[header_row + 1].tolist() if len(raw) > header_row + 1 else [""] * len(headers)
        columns: dict[int, dict[str, Any]] = {}
        for column in range(2, len(headers)):
            code = _column_code(headers[column])
            if not code or code in {"PROT", "WATER", "FATCE"} and sheet != "UserDB_Main_table":
                # These duplicated supporting columns identify neither an
                # additional analyte nor an independent profile measurement.
                continue
            name, unit = dictionary.get(code, (code, _header_unit(headers[column]) or _header_unit(units_row[column])))
            if not unit:
                unit = _header_unit(headers[column]) or _text(units_row[column])
            if not _unit_factor(unit):
                continue
            component = _make_component(source_key, code, name, unit=unit, infoods_tag=code, table=sheet)
            component_by_key[component["component_observation_id"]] = component
            columns[column] = component
        group = ""
        for row_index in range(header_row + 2, len(raw)):
            row = raw.iloc[row_index]
            food_code = _text(row.iloc[0])
            if not re.fullmatch(r"\d{2}_\d{4}", food_code):
                if food_code:
                    group = food_code
                continue
            name = _text(row.iloc[name_column])
            if not name:
                continue
            source_reference = _text(row.iloc[4]) if sheet == "UserDB_Main_table" and len(row) > 4 else ""
            origin = "recipe_calculated" if "recipe calculation" in source_reference.casefold() else "source_reported"
            # The main table and each annex reuse the local food code.  Keep
            # their source-native rows separate; exact-name grouping happens
            # only after all sources and tables have been integrated.
            foods.append(_make_food(
                source_key, food_code, name,
                original_name_local=row.iloc[2] if sheet == "UserDB_Main_table" and len(row) > 2 else "",
                scientific_name=row.iloc[3] if sheet == "UserDB_Main_table" and len(row) > 3 else "",
                food_group=group, recipe_status="recipe_calculated" if origin != "source_reported" else "not_reported",
                edible_status="edible_portion", source_lineage=source_reference, source_table=sheet, source_row=row_index + 1,
            ))
            for column, component in columns.items():
                measurements.append(_make_measurement(
                    source_key, food_code, component, row.iloc[column], unit=component["original_unit"],
                    table=sheet, source_row=row_index + 1, source_column=column + 1,
                    basis="per 100 g edible portion", source_value_origin=origin, direct_basis=True, note=source_reference,
                ))
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(component_by_key.values()), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source", "input_paths": [str(path)],
    })


SMILING_LAYOUTS = {
    "smiling_cambodia_2013": ("smiling_cambodia_values.xlsx", "User FCT", 1, None, "ID Code", "Food Name in English", "Local Food Name", "Scientific Name", ""),
    "smiling_indonesia_2013": ("smiling_indonesia_values.xlsx", "FCT_INA_05082013_FINAL", 0, 1, "FOOD CODE", "FOOD_NAME_ENGLISH", "FOOD_NAME_LOCAL", "", "FOOD_GRP_Optifood"),
    "smiling_laos_2013": ("smiling_laos_values.xlsx", "FCTB", 0, None, "LAO_FOOD_CODE", "FOOD_NAME_ENGLISH", "FOOD_NAME_LOCAL", "SCIENTIIFIC_NAME", "FOOD_GROUP"),
    "smiling_thailand_2013": ("smiling_thailand_values.xlsx", "user FCT", 1, 2, "Code", "Corrected food name", "Food Name Thai", "Scientific name", "Group"),
    "smiling_vietnam_2013": ("smiling_vietnam_values.xlsx", "WP3 FCT", 1, None, "Code", "FOOD_NAME_ENGLISH", "FOOD_NAME_LOCAL", "", "FOOD_GROUP"),
}


def _find_column(headers: list[Any], label: str) -> int | None:
    wanted = normalize_text(label)
    for index, header in enumerate(headers):
        if normalize_text(header) == wanted:
            return index
    return None


def _bundle_from_smiling(path: Path, source_key: str) -> SourceBundle:
    _, sheet, header_row, unit_row, id_label, name_label, local_label, scientific_label, group_label = SMILING_LAYOUTS[source_key]
    raw = pd.read_excel(path, sheet_name=sheet, header=None, keep_default_na=False)
    headers = raw.iloc[header_row].tolist()
    units = raw.iloc[unit_row].tolist() if unit_row is not None else [""] * len(headers)
    food_id_column = _find_column(headers, id_label)
    name_column = _find_column(headers, name_label)
    local_column = _find_column(headers, local_label) if local_label else None
    scientific_column = _find_column(headers, scientific_label) if scientific_label else None
    group_column = _find_column(headers, group_label) if group_label else None
    if food_id_column is None or name_column is None:
        raise ValueError(f"SMILING {source_key} layout is missing required food columns")
    metadata_columns = {column for column in [food_id_column, name_column, local_column, scientific_column, group_column] if column is not None}
    components: list[dict[str, Any]] = []
    component_by_column: dict[int, dict[str, Any]] = {}
    for column, header in enumerate(headers):
        if column in metadata_columns:
            continue
        code = _column_code(header)
        unit = _header_unit(header) or _text(units[column])
        if not code or not _unit_factor(unit):
            continue
        component = _make_component(source_key, code, code, unit=unit, infoods_tag=code, table=sheet)
        components.append(component)
        component_by_column[column] = component
    foods, measurements = [], []
    for row_index in range(header_row + 1, len(raw)):
        if unit_row is not None and row_index == unit_row:
            continue
        row = raw.iloc[row_index]
        food_id, name = _text(row.iloc[food_id_column]), _text(row.iloc[name_column])
        if not food_id or not name:
            continue
        foods.append(_make_food(
            source_key, food_id, name,
            original_name_local=row.iloc[local_column] if local_column is not None else "",
            scientific_name=row.iloc[scientific_column] if scientific_column is not None else "",
            food_group=row.iloc[group_column] if group_column is not None else "",
            edible_status="edible_portion", source_table=sheet, source_row=row_index + 1,
        ))
        for column, component in component_by_column.items():
            measurements.append(_make_measurement(
                source_key, food_id, component, row.iloc[column], unit=component["original_unit"],
                table=sheet, source_row=row_index + 1, source_column=column + 1,
                basis="per 100 g edible portion", direct_basis=True,
            ))
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components).drop_duplicates("component_observation_id"), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source", "input_paths": [str(path)],
    })


def _component_sheet_dictionary(path: Path, *, header_row: int, name_column: int, id_column: int,
                                unit_column: int, tag_column: int | None = None) -> dict[str, tuple[str, str, str]]:
    raw = pd.read_excel(path, sheet_name="Components", header=None, keep_default_na=False)
    result: dict[str, tuple[str, str, str]] = {}
    for _, row in raw.iloc[header_row + 1:].iterrows():
        identifier = _text(row.iloc[id_column]) if len(row) > id_column else ""
        name = _text(row.iloc[name_column]) if len(row) > name_column else ""
        unit = _text(row.iloc[unit_column]) if len(row) > unit_column else ""
        tag = _text(row.iloc[tag_column]) if tag_column is not None and len(row) > tag_column else identifier
        if identifier and name:
            result[identifier.upper()] = (name, unit, tag.upper())
    return result


def _bundle_from_infoods_workbook(path: Path, source_key: str, *, crosswalk: pd.DataFrame, component_header_row: int,
                                  component_name_column: int, component_id_column: int,
                                  component_unit_column: int, component_tag_column: int | None,
                                  food_sheet_pattern: str, food_name_column: str,
                                  food_id_column: str, local_name_column: str = "",
                                  group_column: str = "", scientific_column: str = "") -> SourceBundle:
    dictionary = _component_sheet_dictionary(
        path, header_row=component_header_row, name_column=component_name_column,
        id_column=component_id_column, unit_column=component_unit_column, tag_column=component_tag_column,
    )
    workbook = pd.ExcelFile(path)
    food_sheets = [sheet for sheet in workbook.sheet_names if re.match(food_sheet_pattern, sheet)]
    foods, components, measurements = [], [], []
    component_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for sheet in food_sheets:
        frame = pd.read_excel(path, sheet_name=sheet, header=0, keep_default_na=False)
        if food_id_column not in frame.columns or food_name_column not in frame.columns:
            continue
        metadata = {food_id_column, food_name_column, local_name_column, group_column, scientific_column, "Country, region", "Processing", "Cultivar/Variety/Accession Name", "Type", "n", "Comments on data processing/methods"}
        for row_index, row in frame.iterrows():
            food_id, name = _text(row.get(food_id_column, "")), _text(row.get(food_name_column, ""))
            if not re.fullmatch(r"\d{7,8}", food_id) or not name:
                continue
            note = _text(row.get("Comments on data processing/methods", ""))
            # The source itself documents fresh-weight conversions on some
            # records. Keep these records but do not silently elevate an
            # individually described conversion to a direct assay label.
            direct_basis = not bool(re.search(r"\bper\s+dm\b|dry\s+matter|\bdw\b", note, flags=re.IGNORECASE))
            foods.append(_make_food(
                source_key, food_id, name,
                original_name_local=row.get(local_name_column, "") if local_name_column else "",
                scientific_name=row.get(scientific_column, "") if scientific_column else "",
                food_group=row.get(group_column, "") if group_column else sheet,
                processing=row.get("Processing", ""), geography=row.get("Country, region", ""),
                cultivar=row.get("Cultivar/Variety/Accession Name", ""), food_type=row.get("Type", ""),
                edible_status="source_declared_edible_portion", source_table=sheet, source_row=row_index + 2,
                source_lineage=note,
            ))
            for column in frame.columns:
                if column in metadata:
                    continue
                codes = _source_value_column_codes(column)
                code = next((candidate for candidate in codes if candidate in dictionary), "")
                if not code:
                    continue
                name_component, unit, tag = dictionary[code]
                if not unit:
                    continue
                raw_value = row[column]
                if not _text(raw_value):
                    continue
                cache_key = (code, unit)
                component = component_cache.get(cache_key)
                if component is None:
                    component = _make_component(source_key, code, name_component, unit=unit, infoods_tag=tag, table="Components")
                    component_cache[cache_key] = component
                    components.append(component)
                measurements.append(_make_measurement(
                    source_key, food_id, component, raw_value, unit=unit, table=sheet,
                    source_row=row_index + 2, source_column=list(frame.columns).index(column) + 1,
                    basis="source workbook per 100 g edible portion", direct_basis=direct_basis, note=note,
                ))
    return SourceBundle(pd.DataFrame(foods).drop_duplicates("food_observation_id"), pd.DataFrame(components), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source", "input_paths": [str(path)],
    })


def _bundle_from_lesotho(path: Path) -> SourceBundle:
    source_key, table = "lesotho_fct_2006", "source DB original"
    frame = pd.read_excel(path, sheet_name=table, header=0, keep_default_na=False)
    metadata_columns = {"code", "progress", "type", "type.1", "priorityclass", "foodname", "LONG FOODNAME", "scientific name", "EDIBLE", "FCT SOURCE ", "FDNUMBER SOURCE", "comments", "DEN", "XN", "XFA", "yield"}
    components, measurements, foods = [], [], []
    component_by_column: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        if column in metadata_columns or str(column).startswith("Unnamed"):
            continue
        match = re.match(r"^(.+?)-+(g|mg|mcg|µg)$", _text(column), flags=re.IGNORECASE)
        if not match:
            continue
        code, unit = match.group(1).upper(), match.group(2)
        component = _make_component(source_key, code, code, unit=unit, infoods_tag=code, table=table)
        component_by_column[column] = component
        components.append(component)
    for row_index, row in frame.iterrows():
        food_id, name = _text(row.get("code", "")), _text(row.get("foodname", ""))
        if not re.fullmatch(r"\d{6}", food_id) or not name:
            continue
        source_reference = _text(row.get("FCT SOURCE ", ""))
        foods.append(_make_food(
            source_key, food_id, name, scientific_name=row.get("scientific name", ""),
            food_group=food_id[:2], food_type=row.get("type", ""), edible_status="source_declared_edible_portion",
            source_lineage=source_reference, source_table=table, source_row=row_index + 2,
        ))
        for column, component in component_by_column.items():
            measurements.append(_make_measurement(
                source_key, food_id, component, row[column], unit=component["original_unit"], table=table,
                source_row=row_index + 2, source_column=list(frame.columns).index(column) + 1,
                basis="source table per 100 g edible portion", direct_basis=True, note=source_reference,
            ))
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components).drop_duplicates("component_observation_id"), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source_original_layer", "input_paths": [str(path)],
    })


def _bundle_from_swiss(path: Path) -> SourceBundle:
    source_key = "swiss_fcdb_7_1"
    codes = pd.read_excel(path, sheet_name="Nutrient codes", header=2, keep_default_na=False)
    code_dictionary: dict[str, tuple[str, str]] = {}
    code_by_label: dict[str, str] = {}
    for _, row in codes.iterrows():
        code = _text(row.iloc[0]).upper()
        name = _text(row.iloc[3])
        unit = _header_unit(name)
        if code and name and unit:
            code_dictionary[code] = (re.sub(r"\s*\([^()]*\)\s*$", "", name), unit)
            for label in row.iloc[3:7].tolist():
                cleaned = re.sub(r"\s*\([^()]*\)\s*$", "", _text(label)).strip()
                if cleaned:
                    code_by_label[normalize_text(cleaned)] = code
    foods, components, measurements = [], [], []
    component_cache: dict[str, dict[str, Any]] = {}
    for sheet, brand_status in [("Aliments génériques", "not_reported"), ("Produits de marque", "branded_product")]:
        frame = pd.read_excel(path, sheet_name=sheet, header=2, keep_default_na=False)
        columns = list(frame.columns)
        for row_index, row in frame.iterrows():
            food_id, name = _text(row.get("ID", "")), _text(row.get("Nom", ""))
            if not food_id or not name:
                continue
            foods.append(_make_food(
                source_key, f"{sheet}:{food_id}", name, food_group=row.get("Catégorie", ""),
                brand_status=brand_status, edible_status="edible_portion", source_table=sheet, source_row=row_index + 4,
            ))
            for column_index, column in enumerate(columns):
                header = _text(column)
                if not _header_unit(header):
                    continue
                # Swiss data columns occur as value / derivation / source triplets.
                if column_index + 2 >= len(columns) or not _text(columns[column_index + 1]).startswith("Dérivation de la valeur"):
                    continue
                name_component = re.sub(r"\s*\([^()]*\)\s*$", "", header).strip()
                code = code_by_label.get(normalize_text(name_component), name_component.upper())
                unit = _header_unit(header)
                if not unit:
                    continue
                raw_value = row.iloc[column_index]
                if not _text(raw_value):
                    continue
                component = component_cache.get(code)
                if component is None:
                    component = _make_component(source_key, code, name_component, unit=unit, infoods_tag=code, table="Nutrient codes")
                    component_cache[code] = component
                    components.append(component)
                derivation = _text(row.iloc[column_index + 1])
                origin = "recipe_or_calculated" if re.search(r"calcul|estimation", derivation, flags=re.IGNORECASE) else "source_reported"
                measurements.append(_make_measurement(
                    source_key, f"{sheet}:{food_id}", component, raw_value, unit=unit, table=sheet,
                    source_row=row_index + 4, source_column=column_index + 1,
                    basis="per 100 g edible portion", source_value_origin=origin, direct_basis=True, note=derivation,
                ))
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components), pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_source", "input_paths": [str(path)],
    })


def _bundle_from_mext(root: Path) -> SourceBundle:
    source_key = "mext_japan_2023"
    base = root / "reports/validation_mext_evidence_2026_09_06"
    foods_raw = pd.read_csv(base / "mext_food_provenance.csv.gz", keep_default_na=False)
    values = pd.read_csv(base / "mext_raw_cell_semantics.csv.gz", keep_default_na=False)
    components_raw = pd.read_csv(base / "mext_component_evidence_registry.csv", keep_default_na=False)
    main_foods = foods_raw[foods_raw.table.eq("main")].drop_duplicates("food_id")
    foods = pd.DataFrame([_make_food(
        source_key, row.food_id, row.original_food_name_ja, food_group=row.food_group_id,
        recipe_status="recipe_or_estimation_note" if bool(row.recipe_or_estimation_note) else "not_reported",
        geography="Japan", edible_status="edible_portion", source_lineage=row.source_note_ja,
        source_table="main", source_row=row.excel_row,
    ) for row in main_foods.itertuples(index=False)])
    main_food_observation_id_by_source_id = dict(
        zip(foods.source_food_id.astype(str), foods.food_observation_id.astype(str))
    )
    components = pd.DataFrame([_make_component(
        source_key, row.component_key, row.original_name_ja or row.tag, unit=row.raw_unit,
        infoods_tag=row.tag, definition=row.definition_review, table=row.table,
    ) for row in components_raw.itertuples(index=False)])
    component_by_key = _component_index(components.to_dict("records"))
    main_ids = set(main_foods.food_id.astype(str))
    measurements = []
    for row in values.itertuples(index=False):
        if _text(row.food_id) not in main_ids or row.component_key not in component_by_key:
            continue
        component = component_by_key[row.component_key]
        direct_basis = row.modality == "mass_per_100g_edible_food"
        origin = "recipe_or_estimation" if bool(row.parenthesized_estimate) else "source_reported"
        measurement = _make_measurement(
            source_key, row.food_id, component, row.raw_token, unit=row.raw_unit, table=row.table,
            source_row=row.excel_row, source_column=row.excel_column, basis=row.basis,
            source_value_origin=origin, direct_basis=direct_basis, note=row.food_specific_definition_hold,
        )
        # MEXT stores the food catalogue in `main` and detailed analytes in
        # separate worksheets. Retain the latter in the locator, but attach
        # each cell to its canonical source-native main-food record.
        measurement["food_observation_id"] = main_food_observation_id_by_source_id[_text(row.food_id)]
        measurements.append(measurement)
    return SourceBundle(foods, components, pd.DataFrame(measurements), {
        "source_key": source_key, "source_registry_key": source_key,
        "ingestion_status": "integrated_previously_audited_machine_readable_source", "input_paths": [str(base)],
    })


EFSA_2013_INFOODS_TAGS = {
    "Calcium (Ca)": "CA",
    "Copper (Cu)": "CU",
    "Magnesium (Mg)": "MG",
    "Phosphorus (P)": "P",
    "Potassium (K)": "K",
    "Total Selenium": "SE",
    "Total iron": "FE",
    "Zinc (Zn)": "ZN",
    "riboflavin": "RIBF",
    "thiamin": "THIA",
    "vitamin B-12": "VITB12",
}


def _bundle_from_efsa_eu_fcdb_2013(root: Path) -> SourceBundle:
    """Read EFSA's CC-BY archived workbook without elevating it to an assay source.

    The release contains harmonized national submissions. It is useful
    source-specific coverage, but its records are intentionally marked as an
    official compilation, not independent direct model labels.
    """
    source_key = "efsa_eu_fcdb_2013"
    path = root / "data/raw/source_acquisition_2026_09_16/efsa_eu_fcdb_2013/Food_composition_dataset.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing downloaded EFSA 2013 workbook: {path}")
    frame = pd.read_excel(path, sheet_name="COMPO7", keep_default_na=False)
    required = {
        "FOOD_ID", "code", "COUNTRY", "efsaprodcode2_recoded", "level1", "level2",
        "NUTRIENT_ID", "NUTRIENT_TEXT", "UNIT", "LEVEL",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"EFSA 2013 workbook is missing required columns: {missing}")

    components: list[dict[str, Any]] = []
    component_by_id: dict[str, dict[str, Any]] = {}
    for row in frame[["NUTRIENT_ID", "NUTRIENT_TEXT", "UNIT"]].drop_duplicates().itertuples(index=False):
        component_id = _text(row.NUTRIENT_ID)
        if component_id in component_by_id:
            continue
        component = _make_component(
            source_key,
            component_id,
            row.NUTRIENT_TEXT,
            unit=row.UNIT,
            infoods_tag=EFSA_2013_INFOODS_TAGS.get(_text(row.NUTRIENT_TEXT), ""),
            definition="EFSA 2013 harmonized food-composition expression; selected vitamins and minerals",
            table="COMPO7",
        )
        component_by_id[component_id] = component
        components.append(component)

    # FOOD_ID is a row identifier in this long-format release, not a food
    # identifier shared across nutrient rows. FoodEx2 code plus the released
    # food label and hierarchy facets is the narrowest source-supported food
    # observation key. Keep different facet combinations separate.
    food_key_columns = ["COUNTRY", "code", "efsaprodcode2_recoded", "level1", "level2", "level3"]
    foods: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for food_index, (_, group) in enumerate(frame.groupby(food_key_columns, sort=False, dropna=False), start=2):
        first = group.iloc[0]
        source_food_id = ":".join((
            _text(first.COUNTRY), _text(first.code), _text(first.efsaprodcode2_recoded),
        ))
        food = _make_food(
            source_key,
            source_food_id,
            first.efsaprodcode2_recoded,
            food_group=first.level1,
            food_subgroup=first.level2,
            food_type="EFSA FoodEx2 harmonized food",
            geography=first.COUNTRY,
            foodex2_code=first.code,
            edible_status="source_declared_100_g_food_basis",
            source_lineage="EFSA 2013 harmonized national FCDB submission; original provider is retained only at country level in this release",
            source_record_url="https://doi.org/10.5281/zenodo.438313",
            source_table="COMPO7",
            source_row=food_index,
        )
        foods.append(food)
        for row_index, row in group.iterrows():
            component = component_by_id[_text(row.NUTRIENT_ID)]
            measurement = _make_measurement(
                source_key,
                source_food_id,
                component,
                row.LEVEL,
                unit=row.UNIT,
                table="COMPO7",
                # The released long table can contain more than one source
                # value for a country/FoodEx2/component cell. Preserve every
                # row rather than choosing a within-source winner.
                source_row=row_index + 2,
                source_column=_text(row.NUTRIENT_ID),
                basis="per 100 g food; basis as supplied by the EFSA harmonized release",
                source_value_origin="official_harmonized_compilation",
                direct_basis=True,
                note=f"Country={_text(row.COUNTRY)}; EFSA FoodEx2={_text(row.code)}; source row={row_index + 2}",
            )
            measurement["food_observation_id"] = food["food_observation_id"]
            measurements.append(measurement)
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components), pd.DataFrame(measurements), {
        "source_key": source_key,
        "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_machine_readable_harmonized_compilation",
        "input_paths": [str(path)],
    })


def _bundle_from_bls_4_0(root: Path) -> SourceBundle:
    """Read the official CC-BY BLS 4.0 archive without flattening its provenance.

    Each BLS component has a companion ``Datenherkunft`` and reference column.
    Formula, recipe, aggregation, pattern, rescaling, and logical-assumption
    values remain source-native evidence but are not direct mass-label
    candidates.  The raw BLS archive remains the immutable complete release;
    this adapter exposes only source axes with an existing exact frozen-panel
    mapping for the current numerical atlas.
    """
    source_key = "bls_4_0"
    archive = root / "data/raw/expansion_evidence_2026_09_07/bls/archive_00.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing downloaded BLS 4.0 archive: {archive}")

    with ZipFile(archive) as bundle:
        component_member = next(
            (name for name in bundle.namelist() if name.endswith("BLS_4_0_Components_DE_EN.xlsx")),
            None,
        )
        data_member = next(
            (name for name in bundle.namelist() if name.endswith("BLS_4_0_Daten_2025_DE.xlsx")),
            None,
        )
        if component_member is None or data_member is None:
            raise ValueError("BLS 4.0 archive is missing its component dictionary or food-value workbook")
        component_frame = pd.read_excel(BytesIO(bundle.read(component_member)), keep_default_na=False)
        data_frame = pd.read_excel(BytesIO(bundle.read(data_member)), keep_default_na=False)

    component_code = "Nährstoffcode / Component code"
    german_name = "Nährstoffbezeichnung"
    english_name = "Component name"
    unit_column = "Einheit / Unit"
    group_column = "Component group"
    formula_column = "Formelanwendung / Formula application"
    required_component_columns = {
        component_code, german_name, english_name, unit_column, group_column, formula_column,
    }
    missing_component_columns = sorted(required_component_columns - set(component_frame.columns))
    if missing_component_columns:
        raise ValueError(f"BLS component dictionary is missing required columns: {missing_component_columns}")
    required_data_columns = {"BLS Code", "Lebensmittelbezeichnung", "Food name"}
    missing_data_columns = sorted(required_data_columns - set(data_frame.columns))
    if missing_data_columns:
        raise ValueError(f"BLS food-value workbook is missing required columns: {missing_data_columns}")

    component_rows = component_frame[component_frame[component_code].map(_text).ne("")].copy()
    components: list[dict[str, Any]] = []
    component_by_code: dict[str, dict[str, Any]] = {}
    value_column_by_code: dict[str, str] = {}
    for _, row in component_rows.iterrows():
        code = _text(row[component_code])
        matching_value_columns = [
            column for column in data_frame.columns
            if column.startswith(f"{code} ") and "[" in column and column.endswith("]")
        ]
        if len(matching_value_columns) != 1:
            raise ValueError(
                f"Expected one BLS value column for component {code!r}; found {matching_value_columns}"
            )
        definition = "; ".join(part for part in (
            _text(row[english_name]),
            _text(row[group_column]),
            _text(row[formula_column]),
        ) if part)
        component = _make_component(
            source_key,
            code,
            row[german_name],
            unit=row[unit_column],
            definition=definition,
            table="BLS_4_0_Daten_2025_DE",
        )
        components.append(component)
        component_by_code[code] = component
        value_column_by_code[code] = matching_value_columns[0]

    food_table = "BLS_4_0_Daten_2025_DE"
    foods: list[dict[str, Any]] = []
    food_observation_id_by_code: dict[str, str] = {}
    for row_index, row in data_frame.iterrows():
        food_code = _text(row["BLS Code"])
        if not food_code:
            raise ValueError(f"BLS workbook row {row_index + 2} has no BLS Code")
        if food_code in food_observation_id_by_code:
            raise ValueError(f"BLS workbook repeats BLS Code {food_code!r}")
        food = _make_food(
            source_key,
            food_code,
            row["Food name"],
            original_name_local=row["Lebensmittelbezeichnung"],
            food_type="BLS food record",
            geography="Germany",
            edible_status="source_declared_per_100_g_food",
            source_record_url="https://doi.org/10.25826/Data20251217-134202-0",
            source_lineage="German Nutrient Database BLS 4.0 (2025); Max Rubner-Institut",
            source_table=food_table,
            source_row=row_index + 2,
        )
        foods.append(food)
        food_observation_id_by_code[food_code] = food["food_observation_id"]

    derived_origins = {
        "Rezeptberechnung", "Formelberechnung", "Musterberechnung", "Reskalierung",
        "Aggregation", "Logische Null", "Logische Annahme",
    }
    measurements: list[dict[str, Any]] = []
    for row_index, row in data_frame.iterrows():
        food_code = _text(row["BLS Code"])
        for code, component in component_by_code.items():
            value_column = value_column_by_code[code]
            raw_value = row[value_column]
            if not _text(raw_value):
                continue
            origin = _text(row.get(f"{code} Datenherkunft", ""))
            reference = _text(row.get(f"{code} Referenz", ""))
            source_value_origin = (
                "source_derived_or_calculated" if origin in derived_origins else "source_reported"
            )
            measurement = _make_measurement(
                source_key,
                food_code,
                component,
                raw_value,
                unit=component["original_unit"],
                table=food_table,
                source_row=row_index + 2,
                source_column=value_column,
                basis="per 100 g food as supplied by BLS 4.0",
                source_value_origin=source_value_origin,
                direct_basis=True,
                note=f"BLS data origin={origin}; reference={reference}",
            )
            measurement["food_observation_id"] = food_observation_id_by_code[food_code]
            measurements.append(measurement)
    return SourceBundle(pd.DataFrame(foods), pd.DataFrame(components), pd.DataFrame(measurements), {
        "source_key": source_key,
        "source_registry_key": source_key,
        "ingestion_status": "integrated_downloaded_open_cc_by_4_machine_readable",
        "input_paths": [str(archive)],
    })


def _available_external_bundles(root: Path, crosswalk: pd.DataFrame) -> list[SourceBundle]:
    downloads = root / "data/raw/global_fcdb_inventory_2026_09_10/downloads"
    bundles = [
        _bundle_from_mext(root),
        _bundle_from_efsa_eu_fcdb_2013(root),
        _bundle_from_bls_4_0(root),
        _bundle_from_wafct(downloads / "wafct_2019.xlsx"),
        _bundle_from_bangladesh(downloads / "bangladesh_fct_2013.xlsx"),
        _bundle_from_lesotho(downloads / "lesotho_fct_2006.xlsm"),
        _bundle_from_swiss(downloads / "swiss_fcdb_v7_1.xlsx"),
    ]
    for source_key, layout in SMILING_LAYOUTS.items():
        bundles.append(_bundle_from_smiling(downloads / layout[0], source_key))
    bundles.extend([
        _bundle_from_infoods_workbook(
            downloads / "anfood_2_0.xlsx", "anfood_2_0", crosswalk=crosswalk, component_header_row=3, component_name_column=1,
            component_id_column=0, component_unit_column=2, component_tag_column=0, food_sheet_pattern=r"^\d{2} ",
            food_name_column="Foodname in English", food_id_column="Food Item ID", local_name_column="Foodname in own language",
            group_column="Country, region", scientific_column="Scientific name",
        ),
        _bundle_from_infoods_workbook(
            downloads / "biofoodcomp_4_0.xlsx", "biofoodcomp_4_0", crosswalk=crosswalk, component_header_row=2, component_name_column=1,
            component_id_column=0, component_unit_column=2, component_tag_column=0, food_sheet_pattern=r"^\d{2}",
            food_name_column="Foodname in English", food_id_column="Food Item ID", local_name_column="Foodname in own language",
            group_column="Country, region", scientific_column="Species/Subspecies",
        ),
        _bundle_from_infoods_workbook(
            downloads / "phyfoodcomp_1_0.xlsx", "phyfoodcomp_1_0", crosswalk=crosswalk, component_header_row=3, component_name_column=1,
            component_id_column=2, component_unit_column=3, component_tag_column=2, food_sheet_pattern=r"^\d{2} ",
            food_name_column="Food name in English", food_id_column="Food item ID", local_name_column="Food name in own language",
            group_column="Food Group", scientific_column="Species/Subspecies",
        ),
    ])
    return bundles


def _extend_crosswalk_with_observed_source_axes(
    crosswalk: pd.DataFrame, bundles: list[SourceBundle]
) -> pd.DataFrame:
    """Register axes from newly acquired releases before attaching values.

    The frozen crosswalk predates newly acquired source releases. This function
    adds only source-native component rows encountered in downloaded files;
    it maps them only through the same exact-name or curated INFOODS rules as
    the frozen crosswalk. It never performs a fuzzy or semantic merge.
    """
    additions: list[dict[str, Any]] = []
    existing = set(zip(crosswalk.source_key.astype(str), crosswalk.source_component_id.astype(str)))
    for bundle in bundles:
        registry_key = _text(bundle.provenance["source_registry_key"])
        for component in bundle.components.itertuples(index=False):
            component_id = _text(component.source_component_id)
            key = (registry_key, component_id)
            if key in existing:
                continue
            original_name = _text(component.original_name)
            tag = _text(component.infoods_tag)
            target_id, target_name, evidence = _axis_lookup(
                crosswalk, registry_key, component_id, original_name, tag
            )
            additions.append({
                "source_axis_id": stable_id("observed_source_axis", registry_key, component_id, original_name),
                "source_key": registry_key,
                "source_component_id": component_id,
                "source_original_name": original_name,
                "source_infoods_tag": tag,
                "source_raw_unit": _text(component.original_unit),
                "source_raw_denominator": "per 100 g food as supplied by source",
                "target_axis_id": target_id,
                "target_axis_name": target_name,
                "mapping_status": (
                    "mapped_exact_observed_source_axis" if target_id else "not_mapped_exactly_observed_source_axis"
                ),
                "mapping_evidence": evidence,
                "candidate_target_axis_ids": target_id,
            })
            existing.add(key)
    if not additions:
        return crosswalk
    return pd.concat([crosswalk, pd.DataFrame(additions)], ignore_index=True, sort=False)


def _source_ingestion_ledger(registry: pd.DataFrame, bundles: list[SourceBundle], crosswalk: pd.DataFrame) -> pd.DataFrame:
    by_registry = {bundle.provenance["source_registry_key"]: bundle for bundle in bundles}
    rows = []
    for row in registry.itertuples(index=False):
        source_key = _text(row.source_key)
        bundle = by_registry.get(source_key)
        exact_axis_count = int(crosswalk.loc[
            crosswalk.source_key.eq(source_key) & crosswalk.target_axis_id.ne(""), "target_axis_id"
        ].nunique())
        if bundle is not None:
            raw_measurements = bundle.measurements
            numeric = (
                pd.to_numeric(raw_measurements.get("numeric_value", pd.Series(dtype=float)), errors="coerce").notna()
                & raw_measurements.get("value_status", pd.Series(dtype=str)).isin(["observed", "explicit_zero"])
            )
            source_measurement_records = int(len(raw_measurements))
            source_numeric_records = int(numeric.sum())
            source_numeric_foods = int(raw_measurements.loc[numeric, "food_observation_id"].nunique()) if source_numeric_records else 0
            if source_numeric_records:
                status = "integrated_source_native_numeric_values"
            elif source_measurement_records:
                status = "integrated_source_value_semantics_without_numeric_values"
            else:
                status = bundle.provenance["ingestion_status"]
            reason = ""
            raw_foods = int(bundle.foods.food_observation_id.nunique())
            mapped_numeric = numeric & raw_measurements.get("target_axis_id", pd.Series(dtype=str)).ne("")
            mapped_foods = int(raw_measurements.loc[mapped_numeric, "food_observation_id"].nunique()) if mapped_numeric.any() else 0
            mapped_records = int(mapped_numeric.sum())
        else:
            access = _text(row.access_status)
            status = "registered_not_numerically_integrated"
            if "pdf" in access.casefold():
                reason = "official PDF retained; audited table extraction is required before numerical ingestion"
            elif "permission" in access.casefold() or "terms" in access.casefold() or "paid" in access.casefold():
                reason = "access or redistribution terms require review before numerical ingestion"
            elif not _text(row.local_raw_path):
                reason = "no locally readable numerical artifact is available"
            else:
                reason = "local artifact needs a source-specific food-value adapter"
            raw_foods = mapped_foods = mapped_records = source_measurement_records = source_numeric_records = source_numeric_foods = 0
        rows.append({
            "source_key": source_key, "source_name": row.name, "region": row.region,
            "countries_or_coverage": row.countries_or_coverage, "access_status": row.access_status,
            "source_version": row.source_version, "local_raw_path": row.local_raw_path,
            "ingestion_status": status, "ingestion_reason": reason,
            "raw_food_observations_loaded": raw_foods,
            "source_measurement_records_loaded": source_measurement_records,
            "source_numeric_measurements_loaded": source_numeric_records,
            "food_observations_with_source_numeric_values": source_numeric_foods,
            "food_observations_with_mapped_prediction_axis": mapped_foods,
            "mapped_numeric_measurements_retained": mapped_records,
            "frozen_prediction_axes_with_exact_crosswalk": exact_axis_count,
        })
    return pd.DataFrame(rows)


def _global_report(path: Path, manifest: dict[str, Any], source_ledger: pd.DataFrame,
                   source_distribution: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    integrated = source_ledger[source_ledger.ingestion_status.str.startswith("integrated")].copy()
    pending = source_ledger[~source_ledger.ingestion_status.str.startswith("integrated")].copy()
    lines = [
        f"# Global Frozen-Prediction-Axis Food Audit ({manifest['audit_version']})",
        "",
        "## Scope",
        "",
        "This audit re-runs source-axis integration before food-name grouping. It uses the immutable 369-axis prediction panel and every locally usable, machine-readable numerical source registered in the global source inventory. It does not pool values, choose a source winner, create a canonical food profile, split data, or train a model.",
        "",
        f"- Registered sources: {manifest['registered_sources']}",
        f"- Numerically integrated sources: {manifest['numerically_integrated_sources']}",
        f"- Integrated food observations: {manifest['food_observations']:,}",
        f"- Exact-name candidate groups: {manifest['exact_name_food_groups']:,}",
        f"- Mapped numerical source measurements retained: {manifest['mapped_numeric_measurements']:,}",
        f"- Direct-mass label candidates retained: {manifest['direct_mass_label_candidates']:,}",
        f"- Frozen targets observed in at least one retained record: {manifest['observed_frozen_targets']}",
        f"- Cross-source value conflicts preserved: {manifest['cross_source_axis_conflicts_preserved']:,}",
        "",
        "## Source Inclusion",
        "",
        "`integrated_*` means that an official, locally readable food-value table was parsed source-natively. It does not mean every value is an independent chemical assay. PDF-only, access-restricted, web-only, or adapter-pending sources remain visible below but contribute no numerical rows.",
        "",
        source_ledger.to_csv(index=False),
        "",
        "## Exact Food-Name Grouping",
        "",
        "Only Unicode normalization, case folding, and whitespace normalization are applied to food names. No translation, fuzzy match, taxonomy inference, or parent-child relation is used. Measurements from name-matched records remain separate. A conflict is reported when two sources provide non-equal g/100 g values for the same exact-name group and target axis.",
        "",
        "## Integrated Source Distribution",
        "",
        source_distribution.to_csv(index=False),
        "",
        "## Pending Sources",
        "",
        pending[["source_key", "region", "ingestion_status", "ingestion_reason"]].to_csv(index=False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_global_prediction_food_audit(*, root: Path, output_dir: Path, report_dir: Path) -> dict[str, Any]:
    """Create a new global audit, leaving prior frozen/audit outputs untouched."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing global audit output: {output_dir}")
    panel_path = root / "data/processed/frozen_prediction_axis_panel_v1/frozen_prediction_axis_registry.csv"
    panel = pd.read_csv(panel_path, keep_default_na=False)
    registry = _load_source_registry(root)
    crosswalk = build_global_axis_crosswalk(root, panel)

    bundles = _load_existing_staging(root / "data/processed/scientific_food_composition_v3/staging", crosswalk)
    bundles.extend(_available_external_bundles(root, crosswalk))
    crosswalk = _extend_crosswalk_with_observed_source_axes(crosswalk, bundles)
    attached = [_attach_target_mapping(bundle, crosswalk) for bundle in bundles]
    source_ledger = _source_ingestion_ledger(registry, attached, crosswalk)

    all_foods = pd.concat([bundle.foods for bundle in attached], ignore_index=True)
    if all_foods.food_observation_id.duplicated().any():
        raise ValueError("Source-native food observation IDs collided during global integration")
    all_components = pd.concat([bundle.components for bundle in attached], ignore_index=True)
    if all_components.component_observation_id.duplicated().any():
        raise ValueError("Source-native component observation IDs collided during global integration")
    all_measurements = pd.concat([bundle.measurements for bundle in attached], ignore_index=True)
    if all_measurements.measurement_id.duplicated().any():
        raise ValueError("Source-native measurement IDs collided during global integration")
    food_ids = set(all_foods.food_observation_id)
    orphan_measurements = all_measurements[~all_measurements.food_observation_id.isin(food_ids)]
    if not orphan_measurements.empty:
        sample = orphan_measurements[
            ["source_key", "food_observation_id", "source_measurement_locator"]
        ].head(10).to_dict("records")
        raise ValueError(
            "Measurement table contains food observations absent from the global food table; "
            f"count={len(orphan_measurements)}, sample={sample}"
        )

    all_foods, exact_groups = build_exact_name_food_groups(all_foods)
    all_measurements = all_measurements.merge(
        all_foods[["food_observation_id", "exact_name_group_id", "exact_food_name_key"]],
        on="food_observation_id", validate="many_to_one",
    )
    numeric_measurements = all_measurements.loc[
        pd.to_numeric(all_measurements.numeric_value, errors="coerce").notna()
        & all_measurements.value_status.isin(["observed", "explicit_zero"])
    ].copy()
    mapped_measurements = numeric_measurements[numeric_measurements.target_axis_id.ne("")].copy()
    direct = mapped_measurements[mapped_measurements.direct_mass_label_candidate.eq(True)].copy()
    cell_summary = summarize_exact_name_axis_cells(direct) if not direct.empty else pd.DataFrame()
    conflicts = cell_summary[cell_summary.cell_status.eq("cross_source_value_disagreement_preserved")].copy() if not cell_summary.empty else pd.DataFrame()
    source_distribution = _source_distribution(all_foods, mapped_measurements)
    food_group_distribution = _food_group_distribution(all_foods, mapped_measurements)
    category_coverage = _category_metadata_coverage(all_foods)

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(panel, output_dir / "frozen_prediction_axis_registry.csv")
    write_csv(crosswalk, output_dir / "global_source_axis_to_frozen_prediction_axis.csv.gz")
    write_csv(source_ledger, output_dir / "global_source_ingestion_ledger.csv")
    write_csv(all_foods, output_dir / "food_observation_to_exact_name_group.csv.gz")
    write_csv(exact_groups, output_dir / "exact_name_food_groups.csv")
    write_csv(all_components, output_dir / "integrated_source_component_observations.csv.gz")
    write_csv(all_measurements, output_dir / "source_native_measurements.csv.gz")
    write_csv(mapped_measurements, output_dir / "mapped_numeric_target_measurements.csv.gz")
    write_csv(direct, output_dir / "direct_mass_label_candidate_measurements.csv.gz")
    write_csv(cell_summary, output_dir / "exact_name_food_axis_cell_summary_direct_mass.csv.gz")
    write_csv(conflicts, output_dir / "cross_source_axis_conflicts_preserved.csv.gz")
    write_csv(source_distribution, output_dir / "source_distribution.csv")
    write_csv(food_group_distribution, output_dir / "food_group_distribution_by_source.csv")
    write_csv(category_coverage, output_dir / "food_category_metadata_coverage.csv")

    manifest = {
        "audit_version": GLOBAL_AUDIT_VERSION,
        "status": "global_source_axis_integration_then_exact_food_name_grouping_no_pooling_no_splits_no_model_training",
        "frozen_panel_path": str(panel_path),
        "frozen_panel_sha256": sha256_file(panel_path),
        "frozen_prediction_axes": int(len(panel)),
        "registered_sources": int(len(registry)),
        "numerically_integrated_sources": int(source_ledger.ingestion_status.str.startswith("integrated").sum()),
        "food_observations": int(len(all_foods)),
        "exact_name_food_groups": int(len(exact_groups)),
        "exact_name_groups_with_multiple_observations": int(exact_groups.food_observation_count.gt(1).sum()),
        "source_native_measurement_records": int(len(all_measurements)),
        "source_native_numeric_measurements": int(len(numeric_measurements)),
        "mapped_numeric_measurements": int(len(mapped_measurements)),
        "direct_mass_label_candidates": int(len(direct)),
        "observed_frozen_targets": int(mapped_measurements.target_axis_id.nunique()),
        "direct_mass_frozen_targets": int(direct.target_axis_id.nunique()),
        "cross_source_axis_conflicts_preserved": int(len(conflicts)),
        "axis_mapping_policy": "prior exact source-axis review, frozen registered aliases, or curated standard INFOODS tag only; no embedding or fuzzy match accepted",
        "food_name_grouping_policy": "Unicode/case/whitespace normalized original name only; all source records and values remain separate",
        "value_conflict_policy": "retain every source-native value; report disagreement; do not select, average, or pool",
        "unavailable_source_policy": "retain every registered source in global_source_ingestion_ledger with a reason when numerical ingestion is not currently lawful or technically defensible",
    }
    write_json(manifest, output_dir / "audit_manifest.json")
    _global_report(report_dir / "GLOBAL_FOOD_AUDIT_V2.md", manifest, source_ledger, source_distribution, conflicts)
    return manifest

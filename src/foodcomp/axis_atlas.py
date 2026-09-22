"""Provenance-first global food-metabolome axis atlas.

This module deliberately operates before food matching, value conversion,
deduplication, partitioning, or model training.  A source axis is a source's
own analytical or calculated expression.  It is never deleted because of
coverage and its numeric values are never pooled here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import re
import zipfile

import pandas as pd

from .util import normalize_text, stable_id, write_csv, write_json


INFOODS_URL = "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"
FAO_COMPOSITION_URL = "https://www.fao.org/food-composition/overview-2/en"
FAO_MATCHING_URL = "https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf"
NIH_DRI_URL = "https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/"
FOODB_URL = "https://foodb.ca/"
CHEBI_URL = "https://www.ebi.ac.uk/chebi/"
LIPID_MAPS_URL = "https://www.lipidmaps.org/"


SOURCE_AXIS_COLUMNS = [
    "source_axis_id", "source_key", "source_name", "source_version", "source_region",
    "source_scope", "license_or_terms", "source_access_status", "source_dependency",
    "source_component_id", "original_name", "original_name_local", "source_definition",
    "source_component_group", "source_chemical_class", "raw_unit", "raw_denominator",
    "analytical_method_or_expression", "table_locator", "source_record_url",
    "value_availability", "catalog_extraction_status", "infoods_tag", "eurofir_id",
    "foodb_public_id", "inchi", "inchikey", "smiles", "cas_number", "formula",
    "molecular_mass", "external_identifiers", "source_field_integrity_note",
]

CONCEPT_COLUMNS = [
    "axis_concept_id", "canonical_name", "canonical_name_basis", "authority_namespace",
    "authority_id", "definition", "chemical_identity", "food_composition_role",
    "nutritional_role", "measurement_modality", "scientific_inclusion_tier",
    "chemical_class", "authority_reference", "evidence_status", "review_priority",
    "source_axis_count", "source_count", "source_keys", "value_available_source_count",
]

RELATION_COLUMNS = [
    "relation_id", "subject_id", "object_id", "relation_type", "relation_status",
    "evidence_basis", "evidence_url", "note",
]

REQUIREMENT_COLUMNS = [
    "requirement_id", "canonical_name", "scientific_group", "component_kind",
    "normative_status", "rationale", "primary_authority", "authority_url",
    "preferred_infoods_tags", "related_forms_note",
]


@dataclass(frozen=True)
class Requirement:
    identifier: str
    name: str
    group: str
    kind: str
    rationale: str
    tags: tuple[str, ...] = ()
    forms: str = ""


def _requirement(identifier: str, name: str, group: str, kind: str,
                 rationale: str, *tags: str, forms: str = "") -> Requirement:
    return Requirement(identifier, name, group, kind, rationale, tuple(tags), forms)


# The requirements register is normative: it is not derived from how many
# observations happen to occur in current source files.  Chemical forms are
# retained separately in the source catalogue rather than collapsed here.
MUST_HAVE_REQUIREMENTS = [
    _requirement("water", "Water", "Food matrix and proximate composition", "food_matrix",
                 "Required to describe fresh-weight composition, hydration, and convert compatible dry-basis records.", "WATER"),
    _requirement("ash", "Ash", "Food matrix and proximate composition", "food_matrix",
                 "Required proximate expression for inorganic residue and mass-balance quality checks.", "ASH"),
    _requirement("protein_total", "Protein, total", "Macronutrient", "nutrient_expression",
                 "Dietary protein is a core macronutrient; nitrogen-conversion definition must remain explicit.", "PROT", "PROCNT"),
    _requirement("fat_total", "Fat, total", "Macronutrient", "nutrient_expression",
                 "Total fat is a core macronutrient and aggregate lipid expression.", "FAT"),
    _requirement("available_carbohydrate", "Available carbohydrate", "Macronutrient", "nutrient_expression",
                 "Available carbohydrate is the nutrition-relevant carbohydrate expression; by-difference values remain a distinct derivation.", "CHOAVL", "CHOAVLDF"),
    _requirement("dietary_fibre_total", "Dietary fibre, total", "Macronutrient", "method_defined_expression",
                 "Dietary fibre is a core nutrition quantity, but method-specific definitions cannot be silently pooled.", "FIBTG"),
    _requirement("alcohol_ethanol", "Ethanol", "Macronutrient-related", "chemical_entity",
                 "Alcohol contributes dietary energy and exposure; retain as a separately defined food constituent.", "ALC"),
    _requirement("energy", "Food energy", "Derived nutrition expression", "derived_expression",
                 "Energy is essential for dietary assessment but is a calculated physical/nutritional expression, not a chemical analyte.", "ENERC", "ENERC_KCAL"),
    _requirement("linoleic_acid", "Linoleic acid (18:2 n-6)", "Essential fatty acid", "chemical_entity",
                 "An essential omega-6 fatty acid; isomer and total-expression distinctions must be retained.", "F18D2N6"),
    _requirement("alpha_linolenic_acid", "Alpha-linolenic acid (18:3 n-3)", "Essential fatty acid", "chemical_entity",
                 "An essential omega-3 fatty acid; do not merge with generic 18:3 or gamma-linolenic acid.", "F18D3N3"),
    _requirement("vitamin_a", "Vitamin A activity and provitamin-A forms", "Essential micronutrient", "nutrient_family",
                 "Vitamin A adequacy requires activity expressions and separately recorded retinoids/provitamin carotenoids.", "VITA", "RETOL", "CARTB", forms="RAE and individual retinoid/carotenoid forms are distinct expressions."),
    _requirement("thiamin", "Thiamin (vitamin B1)", "Essential micronutrient", "chemical_entity", "Essential vitamin.", "THIA"),
    _requirement("riboflavin", "Riboflavin (vitamin B2)", "Essential micronutrient", "chemical_entity", "Essential vitamin.", "RIBF"),
    _requirement("niacin", "Niacin (vitamin B3) and niacin equivalents", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; niacin-equivalent calculations are derived expressions.", "NIA", "NIAEQ"),
    _requirement("pantothenic_acid", "Pantothenic acid (vitamin B5)", "Essential micronutrient", "chemical_entity", "Essential vitamin.", "PANTAC"),
    _requirement("vitamin_b6", "Vitamin B6 and vitamers", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; pyridoxine-family forms and total B6 are not interchangeable without definition.", "VITB6"),
    _requirement("biotin", "Biotin (vitamin B7)", "Essential micronutrient", "chemical_entity", "Essential vitamin.", "BIOT"),
    _requirement("folate", "Folate and folate equivalents", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; food folate, folic acid, and DFE remain separate expressions.", "FOL", "FOLDFE"),
    _requirement("vitamin_b12", "Vitamin B12 and cobalamin forms", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; measured cobalamins and added forms require explicit form definitions.", "VITB12"),
    _requirement("vitamin_c", "Vitamin C and redox forms", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; ascorbic and dehydroascorbic forms remain distinguishable.", "VITC"),
    _requirement("vitamin_d", "Vitamin D and vitamers", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; D2, D3, and total vitamin D are separate analytical expressions.", "VITD"),
    _requirement("vitamin_e", "Vitamin E activity and tocopherol/tocotrienol forms", "Essential micronutrient", "nutrient_family",
                 "Essential vitamin; alpha-TE is an activity expression, not a molecular identity.", "VITE"),
    _requirement("vitamin_k", "Vitamin K and phylloquinone/menaquinone forms", "Essential micronutrient", "nutrient_family",
                 "Vitamin K nutritional family with chemically distinct K forms.", "VITK"),
    _requirement("choline", "Choline", "Essential nutrient", "chemical_entity", "Choline has an established dietary reference intake.", "CHOLN"),
]

for tag, name in [
    ("CA", "Calcium"), ("P", "Phosphorus"), ("MG", "Magnesium"), ("K", "Potassium"),
    ("NA", "Sodium"), ("CL", "Chloride"), ("FE", "Iron"), ("ZN", "Zinc"),
    ("CU", "Copper"), ("MN", "Manganese"), ("SE", "Selenium"), ("I", "Iodine"),
    ("MO", "Molybdenum"), ("CR", "Chromium"), ("F", "Fluoride"),
]:
    MUST_HAVE_REQUIREMENTS.append(
        _requirement(tag.lower(), name, "Essential mineral or trace element", "elemental_nutrient",
                     "Dietary reference nutrient or trace-element expression; elemental assay form must remain explicit.", tag)
    )

for tag, name in [
    ("HIS", "Histidine"), ("ILE", "Isoleucine"), ("LEU", "Leucine"), ("LYS", "Lysine"),
    ("MET", "Methionine"), ("CYS", "Cysteine"), ("PHE", "Phenylalanine"), ("TYR", "Tyrosine"),
    ("THR", "Threonine"), ("TRP", "Tryptophan"), ("VAL", "Valine"),
]:
    MUST_HAVE_REQUIREMENTS.append(
        _requirement(tag.lower(), name, "Protein nutritional quality", "amino_acid",
                     "Indispensable amino acid or accepted paired precursor in protein-quality assessment.", tag)
    )


NAME_TO_REQUIREMENT = {
    "water": "water", "moisture": "water", "ash": "ash", "protein": "protein_total",
    "proteins": "protein_total", "protein total": "protein_total", "total lipid fat": "fat_total",
    "fat": "fat_total", "fat total lipids": "fat_total", "dietary fibre": "dietary_fibre_total",
    "fiber dietary": "dietary_fibre_total", "fibre total dietary": "dietary_fibre_total",
    "calcium": "ca", "phosphorus": "p", "magnesium": "mg", "potassium": "k", "sodium": "na",
    "chloride": "cl", "iron": "fe", "zinc": "zn", "copper": "cu", "manganese": "mn",
    "selenium": "se", "iodine": "i", "molybdenum": "mo", "chromium": "cr", "fluoride": "f",
    "thiamin": "thiamin", "thiamine": "thiamin", "riboflavin": "riboflavin",
    "pantothenic acid": "pantothenic_acid", "biotin": "biotin", "choline": "choline",
    "linoleic acid": "linoleic_acid", "alpha linolenic acid": "alpha_linolenic_acid",
    "histidine": "his", "isoleucine": "ile", "leucine": "leu", "lysine": "lys",
    "methionine": "met", "cysteine": "cys", "phenylalanine": "phe", "tyrosine": "tyr",
    "threonine": "thr", "tryptophan": "trp", "valine": "val",
}
TAG_TO_REQUIREMENT = {
    tag.upper(): requirement.identifier
    for requirement in MUST_HAVE_REQUIREMENTS
    for tag in requirement.tags
}
REQUIREMENT_BY_ID = {row.identifier: row for row in MUST_HAVE_REQUIREMENTS}


SAFETY_PATTERN = re.compile(
    r"\b(aflatoxin|pesticide|herbicide|fungicide|insecticide|cadmium|lead|mercury|arsenic|"
    r"polychlorinated|dioxin|furan|acrylamide|polycyclic aromatic|pah|perfluoro|pfas|"
    r"plasticizer|phthalate|bisphenol|mycotoxin|contaminant|residue)\b", re.IGNORECASE,
)
DERIVED_PATTERN = re.compile(r"\b(by difference|calculated|atwater|equivalent|activity equivalent|rae|dfe|alpha[- ]?te|niacin equivalent|per cent|percentage)\b", re.IGNORECASE)
METADATA_PATTERN = re.compile(r"\b(refuse|edible portion|yield factor|retention factor|coefficient|sample count|moisture ratio)\b", re.IGNORECASE)
INCHI_PATTERN = re.compile(r"^InChI=(?:1|1S)/")
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
NUMERIC_VALUE_AVAILABILITY = {"food_value_table_present", "food_numeric_measurement_present"}


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write the atlas tables with real gzip encoding when requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def _compact(value: Any) -> str:
    return normalize_text(_clean(value)).replace("_", " ")


def _unit_modality(unit: str, denominator: str) -> str:
    text = f"{unit} {denominator}".casefold().replace("μ", "µ")
    if any(token in text for token in ("kcal", "kj", "kilocal", "kiloj")):
        return "energy"
    if any(token in text for token in ("rae", "dfe", "alpha-te", "α-te", "ne", "iu")):
        return "activity_equivalent"
    if any(token in text for token in ("mmol", "µmol", "umol", "mol/")):
        return "molar_concentration"
    if "%" in text or "per 100 g fatty acid" in text or "100g fa" in text:
        return "relative_fraction"
    if "dry" in text:
        return "mass_fraction_dry_weight"
    if "ml" in text or "litre" in text or "liter" in text:
        return "mass_per_volume"
    if any(token in text for token in ("mg", "µg", "ug", "mcg", "gram", " g")):
        return "mass_fraction_or_amount"
    return "unspecified_or_nonquantitative"


def _source_path(root: Path, source_key: str) -> Path | None:
    overrides = {
        "foodb_2020": root / "foodb_2020_04_07_csv",
        "usda_foundation": root / "data/raw/usda/foundation_2026_04_30",
        "usda_sr_legacy": root / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04",
        "cnf": root / "data/raw/cnf_2026/extracted",
        "usda_fndds_2021_2023": root / "data/raw/usda/fndds_2021_2023/FoodData_Central_survey_food_csv_2024-10-31",
        "ciqual_2025": root / "data/raw/candidates/ciqual_2025/Table_Ciqual_2025_ENG.xlsx",
        "frida_6_1": root / "data/raw/candidates/frida_6_1/FCDB_6.1_Dataset.xlsx",
        "cofid_2021": root / "data/raw/candidates/cofid_2021/cofid_2021.xlsx",
        "norwegian_fcdb": root / "data/raw/expansion_2026_09_06/norway/nutrients.json",
        "swiss_fcdb_7_1": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/swiss_fcdb_v7_1.xlsx",
        "bls_4_0": root / "data/raw/expansion_evidence_2026_09_07/bls/archive_00.zip",
        "afcd_release_3": root / "data/raw/expansion_2026_09_06/afcd/03_AFCD Release 3 - Nutrient details.xlsx",
        "mext_japan_2023": root / "reports/validation_mext_evidence_2026_09_06/mext_component_evidence_registry.csv",
        "bangladesh_fct_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/bangladesh_fct_2013.xlsx",
        "wafct_2019": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/wafct_2019.xlsx",
        "lesotho_fct_2006": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/lesotho_fct_2006.xlsm",
        "anfood_2_0": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/anfood_2_0.xlsx",
        "biofoodcomp_4_0": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/biofoodcomp_4_0.xlsx",
        "phyfoodcomp_1_0": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/phyfoodcomp_1_0.xlsx",
        "smiling_cambodia_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/smiling_cambodia_values.xlsx",
        "smiling_indonesia_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/smiling_indonesia_values.xlsx",
        "smiling_laos_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/smiling_laos_values.xlsx",
        "smiling_thailand_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/smiling_thailand_values.xlsx",
        "smiling_vietnam_2013": root / "data/raw/global_fcdb_inventory_2026_09_10/downloads/smiling_vietnam_values.xlsx",
    }
    if source_key in overrides and overrides[source_key].exists():
        return overrides[source_key]
    return None


def _artifact_sha256(path: Path) -> str:
    """Hash a raw file or an ordered raw-artifact tree without modifying it."""
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(item.relative_to(path.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reported_version(row: pd.Series, local_path: Path | None) -> tuple[str, str]:
    """Record a release label without confusing a local download date with version."""
    label = _clean(row.get("name", ""))
    labels = re.findall(r"\b\d{4}(?:[-/]\d{2,4})?\b|\b\d{1,3}(?:\.\d+){1,3}\b", label)
    if labels:
        return "; ".join(dict.fromkeys(labels)), "parsed_from_inherited_source_label; official release verification pending"
    return "not_recorded", "not recorded in inherited registry; requires source-level verification"


def enrich_source_registry(registry: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Add release/provenance fields while preserving every inherited record."""
    rows = []
    for row in registry.to_dict(orient="records"):
        source_key = _clean(row["source_key"])
        local_path = _source_path(root, source_key)
        version, version_status = _reported_version(pd.Series(row), local_path)
        rows.append({
            **row,
            "source_version": version,
            "source_version_status": version_status,
            "source_dependency": _clean(row.get("measurement_provenance", "")),
            "source_dependency_note": _clean(row.get("limitations", "")),
            "resolved_local_raw_path": str(local_path.relative_to(root)) if local_path else "",
            "raw_artifact_sha256": _artifact_sha256(local_path) if local_path else "",
            "raw_artifact_hash_scope": (
                "single_file" if local_path and local_path.is_file() else "ordered_directory_tree"
                if local_path else "not_locally_available"
            ),
        })
    return pd.DataFrame(rows)


def _record(source: pd.Series, *, component_id: Any, name: Any, local_name: Any = "",
            definition: Any = "", unit: Any = "", denominator: Any = "",
            group: Any = "", chemical_class: Any = "", method: Any = "",
            locator: Any = "", value_availability: str = "catalogue_definition_only",
            status: str = "extracted", infoods_tag: Any = "", eurofir_id: Any = "",
            foodb_public_id: Any = "", inchi: Any = "", inchikey: Any = "",
            smiles: Any = "", cas_number: Any = "", formula: Any = "",
            molecular_mass: Any = "", external_identifiers: Any = "",
            source_field_integrity_note: Any = "", url: Any = "") -> dict[str, Any]:
    key = _clean(source.source_key)
    component = _clean(component_id)
    return {
        # A source can legitimately reuse a tag or code for parallel expressions
        # (for example kJ and kcal energy).  Preserve both raw-axis rows.
        "source_axis_id": stable_id("source_axis", key, component, _clean(name), _clean(unit), _clean(denominator), _clean(locator)),
        "source_key": key,
        "source_name": _clean(source["name"]),
        "source_version": _clean(source.get("source_version", source.get("version", ""))),
        "source_region": _clean(source.get("region", "")),
        "source_scope": _clean(source.get("scope", "")),
        "license_or_terms": _clean(source.get("license_or_terms", "")),
        "source_access_status": _clean(source.get("access_status", "")),
        "source_dependency": _clean(source.get("source_dependency", source.get("limitations", ""))),
        "source_component_id": component,
        "original_name": _clean(name),
        "original_name_local": _clean(local_name),
        "source_definition": _clean(definition),
        "source_component_group": _clean(group),
        "source_chemical_class": _clean(chemical_class),
        "raw_unit": _clean(unit),
        "raw_denominator": _clean(denominator),
        "analytical_method_or_expression": _clean(method),
        "table_locator": _clean(locator),
        "source_record_url": _clean(url),
        "value_availability": value_availability,
        "catalog_extraction_status": status,
        "infoods_tag": _clean(infoods_tag),
        "eurofir_id": _clean(eurofir_id),
        "foodb_public_id": _clean(foodb_public_id),
        "inchi": _clean(inchi),
        "inchikey": _clean(inchikey),
        "smiles": _clean(smiles),
        "cas_number": _clean(cas_number),
        "formula": _clean(formula),
        "molecular_mass": _clean(molecular_mass),
        "external_identifiers": _clean(external_identifiers),
        "source_field_integrity_note": _clean(source_field_integrity_note),
    }


def _nutrient_csv(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path / "nutrient.csv", low_memory=False)
    return [_record(source, component_id=row.id, name=row.name, unit=row.unit_name,
                    locator="nutrient.csv", value_availability="food_value_table_present")
            for row in frame.itertuples(index=False)]


def _cnf(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path / "Nutrient_Name.csv", low_memory=False)
    return [_record(source, component_id=row.Nutrient_Code, name=row.Nutrient_Name_EN,
                    local_name=row.Nutrient_Name_FR, unit=row.Nutrient_Unit,
                    infoods_tag=row.Tagname, locator="Nutrient_Name.csv",
                    value_availability="food_value_table_present")
            for row in frame.itertuples(index=False)]


def _foodb(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    nutrients = pd.read_csv(path / "Nutrient.csv", low_memory=False)
    # A FooDB food--compound association is not necessarily a quantitative
    # measurement.  Classify source IDs by actual numeric content fields before
    # exposing availability in the axis catalogue.  This uses no values and
    # does not aggregate food measurements.
    availability: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    content_path = path / "Content.csv"
    for chunk in pd.read_csv(
        content_path,
        usecols=["source_id", "source_type", "orig_content", "orig_min", "orig_max", "standard_content", "orig_unit"],
        chunksize=250_000,
        low_memory=False,
    ):
        for kind, group in chunk.groupby("source_type", dropna=False):
            kind_key = _clean(kind).casefold()
            source_ids = group.source_id.astype("string")
            availability[kind_key]["associated"].update(source_ids.dropna().astype(str))
            numeric = (
                pd.to_numeric(group.standard_content, errors="coerce").notna()
                | pd.to_numeric(group.orig_content, errors="coerce").notna()
            )
            bounded = (
                pd.to_numeric(group.orig_min, errors="coerce").notna()
                | pd.to_numeric(group.orig_max, errors="coerce").notna()
            )
            availability[kind_key]["numeric"].update(source_ids[numeric].dropna().astype(str))
            availability[kind_key]["range_only"].update(source_ids[~numeric & bounded].dropna().astype(str))
            availability[kind_key]["association_only"].update(source_ids[~numeric & ~bounded].dropna().astype(str))

    def availability_label(kind: str, source_id: Any) -> str:
        source_id_text = str(source_id)
        states = availability[kind.casefold()]
        if source_id_text in states["numeric"]:
            return "food_numeric_measurement_present"
        if source_id_text in states["range_only"]:
            return "food_range_only_measurement_present"
        if source_id_text in states["associated"]:
            return "food_association_without_quantified_value"
        return "catalogue_definition_only"

    rows = []
    for row in nutrients.itertuples(index=False):
        rows.append(_record(source, component_id=f"Nutrient:{row.id}", name=row.name,
                            definition=row.description, group="FooDB nutrient", foodb_public_id=row.public_id,
                            locator="Nutrient.csv", value_availability=availability_label("nutrient", row.id),
                            status="extracted"))
    compounds = pd.read_csv(path / "Compound.csv", low_memory=False)
    for row in compounds.itertuples(index=False):
        chemical = " > ".join(part for part in (_clean(row.kingdom), _clean(row.superklass), _clean(row.klass), _clean(row.subklass)) if part)
        # The local FooDB 2020 Compound.csv export has a documented positional
        # field-label drift after `state`.  We validate values by syntax rather
        # than trusting its column headings: description=CAS, cas_number=SMILES,
        # moldb_inchikey=InChI, moldb_inchi=monoisotopic mass,
        # moldb_smiles=InChIKey, and moldb_mono_mass=IUPAC name.
        source_inchi = _clean(row.moldb_inchikey)
        source_inchikey = _clean(row.moldb_smiles).upper()
        source_cas = _clean(row.description)
        source_smiles = _clean(row.cas_number)
        source_mass = _clean(row.moldb_inchi)
        source_iupac = _clean(row.moldb_mono_mass)
        inchi = source_inchi if INCHI_PATTERN.match(source_inchi) else ""
        inchikey = source_inchikey if INCHIKEY_PATTERN.match(source_inchikey) else ""
        cas_number = source_cas if CAS_PATTERN.match(source_cas) else ""
        identifiers = "; ".join(part for part in [
            f"FooDB:{_clean(row.public_id)}",
            f"FooDB_IUPAC:{source_iupac}" if source_iupac else "",
            f"FooDB_raw_InChI:{source_inchi}" if source_inchi and not inchi else "",
            f"FooDB_raw_InChIKey:{source_inchikey}" if source_inchikey and not inchikey else "",
        ] if part)
        rows.append(_record(source, component_id=f"Compound:{row.id}", name=row.name,
                            definition=row.annotation_quality, group="FooDB compound", chemical_class=chemical,
                            foodb_public_id=row.public_id, inchi=inchi, inchikey=inchikey,
                            smiles=source_smiles, cas_number=cas_number, molecular_mass=source_mass,
                            external_identifiers=identifiers,
                            source_field_integrity_note=(
                                "FooDB 2020 Compound.csv field labels after state were corrected by "
                                "syntactic validation; original bytes remain in the immutable raw file."
                            ),
                            locator="Compound.csv",
                            value_availability=availability_label("compound", row.id),
                            status="extracted", url=f"https://foodb.ca/compounds/{row.public_id}"))
    return rows


def _ciqual(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="INFOODS codes")
    rows = []
    for index, row in frame.iterrows():
        tag = _clean(row.get("INFDSTAG", ""))
        name = _clean(row.get("const_nom_eng", ""))
        if not tag and not name:
            continue
        rows.append(_record(source, component_id=tag or f"row:{index + 1}", name=name,
                            local_name=row.get("const_nom_fr", ""), infoods_tag=tag,
                            locator="Table_Ciqual_2025_ENG.xlsx:INFOODS codes",
                            value_availability="food_value_table_present"))
    return rows


def _frida(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_excel(path, sheet_name="Parameter")
    rows = []
    for index, row in frame.iterrows():
        name = _clean(row.get("ParameterName", ""))
        if not name:
            continue
        rows.append(_record(source, component_id=row.get("ParameterID", f"row:{index + 1}"), name=name,
                            local_name=row.get("ParameterNavn", ""), unit=row.get("Unit", ""),
                            group=row.get("ParameterGroupID", ""), locator="FCDB_6.1_Dataset.xlsx:Parameter",
                            value_availability="food_value_table_present"))
    return rows


def _norway(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("nutrients", []):
        rows.append(_record(source, component_id=row.get("nutrientId", ""), name=row.get("name", ""),
                            definition=row.get("euroFirName", ""), unit=row.get("unit", ""),
                            eurofir_id=row.get("euroFirId", ""), locator="nutrients.json",
                            value_availability="food_value_table_present", url=row.get("uri", "")))
    return rows


def _mext(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(_record(source, component_id=row.component_key, name=row.tag or row.original_name_ja,
                            local_name=row.original_name_ja, definition=row.definition_review,
                            unit=row.raw_unit, denominator=row.basis, group=row.table,
                            locator=f"MEXT audit:{row.table}", value_availability="food_value_table_present",
                            status="extracted_from_official_table_audit", infoods_tag=row.tag,
                            url=row.definition_evidence_url))
    return rows


def _smiling(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    """Read the official SMILING component description dictionary."""
    workbook = pd.ExcelFile(path)
    component_sheet = next(
        sheet for sheet in workbook.sheet_names
        if "component" in sheet.casefold() or "tagname" in sheet.casefold()
    )
    raw = pd.read_excel(path, sheet_name=component_sheet, header=None, nrows=30)
    header_index = next(
        index for index in range(len(raw))
        if (
            "component name" in " | ".join(_clean(value).casefold() for value in raw.iloc[index].tolist())
            or (
                "nutrients" in " | ".join(_clean(value).casefold() for value in raw.iloc[index].tolist())
                and "tagname" in " | ".join(_clean(value).casefold() for value in raw.iloc[index].tolist())
            )
        )
    )
    frame = pd.read_excel(path, sheet_name=component_sheet, header=header_index)
    columns = {str(column).casefold(): column for column in frame.columns}
    name_column = next(
        column for key, column in columns.items()
        if "component name" in key or key.strip() == "nutrients"
    )
    tag_column = next((column for key, column in columns.items() if "tagname" in key), None)
    unit_column = next((column for key, column in columns.items() if key.strip() == "unit"), None)
    definition_column = next((column for key, column in columns.items() if "definition" in key), None)
    rows = []
    for index, row in frame.iterrows():
        name = _clean(row.get(name_column, ""))
        if not name:
            continue
        raw_tag = _clean(row.get(tag_column, "")) if tag_column else ""
        tag_match = re.match(r"^([A-Za-z0-9_:-]+)", raw_tag)
        infoods_tag = tag_match.group(1).upper() if tag_match else ""
        rows.append(_record(
            source, component_id=raw_tag or f"row:{index + header_index + 2}", name=name,
            unit=row.get(unit_column, "") if unit_column else "", infoods_tag=infoods_tag,
            definition=row.get(definition_column, "") if definition_column else "",
            locator=f"{path.name}:{component_sheet}", value_availability="food_value_table_present",
        ))
    return _deduplicate_source_records(rows)


def _lesotho(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    """Read the reference-DB column dictionary without treating recipe rows as axes."""
    frame = pd.read_excel(path, sheet_name="reference DB")
    metadata = {"code", "progress", "type", "priorityclass", "foodname", "long foodname", "scientific name", "edible"}
    rows = []
    for position, header in enumerate(frame.columns):
        label = _clean(header)
        if not label or _compact(label) in metadata or label.casefold().startswith("unnamed"):
            continue
        match = re.match(r"^([A-Za-z0-9_]+)--?(.+)$", label)
        component = match.group(1) if match else label
        unit = match.group(2) if match else ""
        rows.append(_record(
            source, component_id=component, name=label, unit=unit,
            locator="lesotho_fct_2006.xlsm:reference DB", value_availability="food_value_table_present",
            status="extracted_column_dictionary",
        ))
    return _deduplicate_source_records(rows)


def _component_workbook(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    """Extract dictionary rows from common INFOODS-style component sheets."""
    workbook = pd.ExcelFile(path)
    rows: list[dict[str, Any]] = []
    for sheet in workbook.sheet_names:
        if not any(token in sheet.casefold() for token in ("component", "nutrient", "parameter", "detail")):
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=30)
        header_index = None
        for index in range(len(raw)):
            line = " | ".join(_clean(value).casefold() for value in raw.iloc[index].tolist())
            if ("component" in line or "parameter" in line or "nutrient" in line) and ("unit" in line or "infoods" in line or "eurofir" in line):
                header_index = index
                break
        if header_index is None:
            continue
        frame = pd.read_excel(path, sheet_name=sheet, header=header_index)
        columns = {str(column).casefold(): column for column in frame.columns}
        name_column = next((columns[key] for key in columns if "component" in key or "parameter" in key or "nutrient" in key), None)
        if name_column is None:
            continue
        unit_column = next((columns[key] for key in columns if key.strip() in {"unit", "units", "unité"}), None)
        tag_column = next((columns[key] for key in columns if "infoods" in key), None)
        eurofir_column = next((columns[key] for key in columns if "eurofir" in key), None)
        definition_column = next((columns[key] for key in columns if "definition" in key or "description" in key or "method" in key), None)
        for index, row in frame.iterrows():
            name = _clean(row.get(name_column, ""))
            if not name or name.casefold() in {"nan", "component", "nutrient"}:
                continue
            tag = _clean(row.get(tag_column, "")) if tag_column else ""
            unit = _clean(row.get(unit_column, "")) if unit_column else ""
            component = tag or f"{sheet}:row:{index + header_index + 2}"
            rows.append(_record(source, component_id=component, name=name, unit=unit,
                                definition=row.get(definition_column, "") if definition_column else "",
                                infoods_tag=tag, eurofir_id=row.get(eurofir_column, "") if eurofir_column else "",
                                locator=f"{path.name}:{sheet}", value_availability="food_value_table_present"))
    return _deduplicate_source_records(rows)


def _cofid(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    workbook = pd.ExcelFile(path)
    rows: list[dict[str, Any]] = []
    metadata = {"food code", "food name", "description", "group", "previous", "main data references", "notes"}
    for sheet in workbook.sheet_names:
        if not re.match(r"^1\.\d+", sheet):
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
        header_index = next((i for i in range(len(raw)) if any(_compact(v) == "food code" for v in raw.iloc[i].tolist())), None)
        if header_index is None:
            continue
        headers = pd.read_excel(path, sheet_name=sheet, header=header_index, nrows=1).columns.tolist()
        for position, header in enumerate(headers):
            name = _clean(header)
            if not name or _compact(name) in metadata or name.casefold().startswith("unnamed"):
                continue
            rows.append(_record(source, component_id=f"{sheet}:column:{position + 1}", name=name,
                                locator=f"cofid_2021.xlsx:{sheet}:column:{position + 1}",
                                value_availability="food_value_table_present", status="extracted_column_dictionary"))
    return _deduplicate_source_records(rows)


def _afcd(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    workbook = pd.ExcelFile(path)
    rows: list[dict[str, Any]] = []
    for sheet in workbook.sheet_names:
        if sheet.casefold() in {"contents"}:
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=20)
        header_index = next((i for i in range(len(raw)) if any(_compact(v) == "component" for v in raw.iloc[i].tolist())), None)
        if header_index is None:
            continue
        frame = pd.read_excel(path, sheet_name=sheet, header=header_index)
        columns = {str(c).casefold(): c for c in frame.columns}
        component = next((columns[k] for k in columns if k.strip() == "component"), None)
        if component is None:
            continue
        tag = next((columns[k] for k in columns if "infoods" in k), None)
        unit = next((columns[k] for k in columns if k.strip() in {"unit", "units"}), None)
        description = next((columns[k] for k in columns if "description" in k), None)
        method = next((columns[k] for k in columns if "analytical method" in k), None)
        for index, row in frame.iterrows():
            name = _clean(row.get(component, ""))
            if not name:
                continue
            raw_tag = _clean(row.get(tag, "")) if tag else ""
            rows.append(_record(source, component_id=raw_tag or f"{sheet}:row:{index + header_index + 2}", name=name,
                                unit=row.get(unit, "") if unit else "", infoods_tag=raw_tag,
                                definition=row.get(description, "") if description else "",
                                method=row.get(method, "") if method else "", group=sheet,
                                locator=f"{path.name}:{sheet}", value_availability="food_value_table_present"))
    return _deduplicate_source_records(rows)


def _bls(source: pd.Series, path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("Components_DE_EN.xlsx"))
        with archive.open(member) as handle:
            frame = pd.read_excel(handle)
    rows = []
    for index, row in frame.iterrows():
        fields = {str(key).casefold(): value for key, value in row.items()}
        name = next((_clean(value) for key, value in fields.items() if "english" in key or "bezeichnung" in key), "")
        identifier = next((_clean(value) for key, value in fields.items() if "component" in key or key in {"id", "nr"}), f"row:{index + 1}")
        unit = next((_clean(value) for key, value in fields.items() if "unit" in key), "")
        if name:
            rows.append(_record(source, component_id=identifier, name=name, unit=unit,
                                locator=f"{path.name}:{member}", value_availability="food_value_table_present"))
    return _deduplicate_source_records(rows)


def _deduplicate_source_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only remove literal repeat rows emitted by one dictionary extractor."""
    seen, output = set(), []
    for row in records:
        key = (row["source_key"], row["source_component_id"], row["original_name"], row["raw_unit"], row["table_locator"])
        if key not in seen:
            output.append(row)
            seen.add(key)
    return output


def extract_source_axes(source: pd.Series, root: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Return source-native axes with a truthful acquisition state."""
    path = _source_path(root, _clean(source.source_key))
    if path is None:
        return [], "catalogue_pending", "No locally readable official component dictionary is available for this registry entry."
    try:
        source_key = _clean(source.source_key)
        if source_key in {"usda_foundation", "usda_sr_legacy", "usda_fndds_2021_2023"}:
            rows = _nutrient_csv(source, path)
        elif source_key == "cnf":
            rows = _cnf(source, path)
        elif source_key == "foodb_2020":
            rows = _foodb(source, path)
        elif source_key == "ciqual_2025":
            rows = _ciqual(source, path)
        elif source_key == "frida_6_1":
            rows = _frida(source, path)
        elif source_key == "cofid_2021":
            rows = _cofid(source, path)
        elif source_key == "norwegian_fcdb":
            rows = _norway(source, path)
        elif source_key == "mext_japan_2023":
            rows = _mext(source, path)
        elif source_key.startswith("smiling_"):
            rows = _smiling(source, path)
        elif source_key == "lesotho_fct_2006":
            rows = _lesotho(source, path)
        elif source_key == "afcd_release_3":
            rows = _afcd(source, path)
        elif source_key == "bls_4_0":
            rows = _bls(source, path)
        else:
            rows = _component_workbook(source, path)
        return rows, "extracted" if rows else "local_artifact_needs_adapter", ""
    except (FileNotFoundError, KeyError, StopIteration, ValueError, zipfile.BadZipFile, OSError) as error:
        return [], "local_artifact_needs_adapter", f"{type(error).__name__}: {error}"


def _requirement_match(row: pd.Series) -> str:
    tag = _clean(row.infoods_tag).upper()
    if tag in TAG_TO_REQUIREMENT:
        return TAG_TO_REQUIREMENT[tag]
    name = _compact(row.original_name)
    if name in NAME_TO_REQUIREMENT:
        return NAME_TO_REQUIREMENT[name]
    # Explicitly curated exact strings. Substring matching is not used for identity.
    exact = {
        "vitamin c": "vitamin_c", "ascorbic acid": "vitamin_c",
        "vitamin d": "vitamin_d", "vitamin d3": "vitamin_d",
        "vitamin e": "vitamin_e", "vitamin k": "vitamin_k",
        "vitamin b6": "vitamin_b6", "folate": "folate", "folic acid": "folate",
        "vitamin b12": "vitamin_b12", "cobalamin": "vitamin_b12",
        "niacin": "niacin", "vitamin a": "vitamin_a", "retinol": "vitamin_a",
        "beta carotene": "vitamin_a", "carbohydrate available": "available_carbohydrate",
        "ethanol": "alcohol_ethanol", "alcohol": "alcohol_ethanol",
        "energy": "energy", "food energy": "energy",
    }
    return exact.get(name, "")


def _classify_axis(row: pd.Series) -> dict[str, str]:
    name = _clean(row.original_name)
    definition = _clean(row.source_definition)
    combined = f"{name} {definition}"
    requirement_id = _requirement_match(row)
    modality = _unit_modality(_clean(row.raw_unit), _clean(row.raw_denominator))
    if METADATA_PATTERN.search(combined):
        return dict(
            requirement_id="", chemical_identity="nonchemical_metadata", food_composition_role="food_or_measurement_metadata",
            nutritional_role="not_a_nutrient", measurement_modality=modality,
            scientific_inclusion_tier="retain_nonchemical_expression", review_priority="low",
        )
    if SAFETY_PATTERN.search(combined):
        return dict(
            requirement_id="", chemical_identity="defined_or_source_defined_safety_analyte", food_composition_role="exposure_or_safety_chemical",
            nutritional_role="not_a_nutrient", measurement_modality=modality,
            scientific_inclusion_tier="separate_exposure_safety_layer", review_priority="medium",
        )
    if DERIVED_PATTERN.search(combined) or modality in {"energy", "activity_equivalent"}:
        return dict(
            requirement_id=requirement_id, chemical_identity="derived_or_activity_expression", food_composition_role="derived_nutrition_expression",
            nutritional_role="nutrition_relevant_expression", measurement_modality=modality,
            scientific_inclusion_tier="must_have_nutrition_core" if requirement_id else "retain_nonchemical_expression",
            review_priority="high" if requirement_id else "medium",
        )
    if requirement_id:
        requirement = REQUIREMENT_BY_ID[requirement_id]
        role = "food_matrix" if requirement.kind == "food_matrix" else "essential_nutrient" if "Essential" in requirement.group or requirement_id in {"protein_total", "fat_total", "available_carbohydrate", "dietary_fibre_total"} else "nutrient_related_component"
        return dict(
            requirement_id=requirement_id, chemical_identity="defined_chemical_entity" if requirement.kind in {"chemical_entity", "amino_acid", "elemental_nutrient"} else "defined_aggregate_or_nutrient_expression",
            food_composition_role=role, nutritional_role=requirement.group.casefold().replace(" ", "_"),
            measurement_modality=modality, scientific_inclusion_tier="must_have_nutrition_core", review_priority="high",
        )
    if _clean(row.inchi) or _clean(row.inchikey) or _clean(row.foodb_public_id):
        return dict(
            requirement_id="", chemical_identity=(
                "defined_molecular_entity" if (_clean(row.inchi) or _clean(row.inchikey))
                else "stable_source_defined_entity"
            ),
            food_composition_role="food_derived_metabolite", nutritional_role="nonessential_bioactive_or_food_metabolite",
            measurement_modality=modality, scientific_inclusion_tier="include_capable_food_metabolome", review_priority="medium",
        )
    if _clean(row.infoods_tag) or _clean(row.eurofir_id):
        return dict(
            requirement_id="", chemical_identity="standardized_food_component_expression", food_composition_role="food_composition_component",
            nutritional_role="nutrition_or_food_chemistry_review_required", measurement_modality=modality,
            scientific_inclusion_tier="include_capable_pending_semantic_review", review_priority="medium",
        )
    return dict(
        requirement_id="", chemical_identity="stable_source_axis_pending_resolution", food_composition_role="unresolved_food_composition_axis",
        nutritional_role="unresolved", measurement_modality=modality,
        scientific_inclusion_tier="retain_pending_manual_review", review_priority="high",
    )


def _concept_identity(row: pd.Series, classification: dict[str, str]) -> tuple[str, str, str, str, str]:
    tag = _clean(row.infoods_tag).upper()
    if tag:
        return f"INFOODS:{tag}", "INFOODS", tag, "INFOODS tagname", "authority_standard"
    inchi = _clean(row.inchi)
    if inchi and INCHI_PATTERN.match(inchi):
        return (
            f"INCHI:{stable_id('inchi', inchi)}", "InChI", inchi,
            "Syntactically validated source-provided InChI", "structure_defined",
        )
    inchikey = _clean(row.inchikey).upper()
    if inchikey:
        return f"INCHIKEY:{inchikey}", "InChIKey", inchikey, "Syntactically validated source-provided InChIKey", "structure_defined"
    return (
        f"SOURCE_AXIS:{row.source_axis_id}", "SOURCE", row.source_axis_id,
        "Stable source-axis identity without a cross-source assertion", "source_defined_only",
    )


def _relation(subject_id: str, object_id: str, relation_type: str, status: str,
              basis: str, url: str, note: str) -> dict[str, str]:
    """Build a non-numeric semantic graph edge.

    Graph relations state an interpretation constraint.  They are deliberately
    not an instruction to merge, transform, or impute measurements.
    """
    return {
        "relation_id": stable_id("axis_relation", subject_id, object_id, relation_type),
        "subject_id": subject_id,
        "object_id": object_id,
        "relation_type": relation_type,
        "relation_status": status,
        "evidence_basis": basis,
        "evidence_url": url,
        "note": note,
    }


def _semantic_relations(series: pd.Series, classification: dict[str, str],
                        concept_id: str) -> list[dict[str, str]]:
    """Return deliberately conservative component-expression relationships."""
    tag = _clean(series.infoods_tag).upper()
    rows: list[dict[str, str]] = []
    aggregate_families = {
        "FAT": "CHEMICAL_FAMILY:total_lipids",
        "FASAT": "CHEMICAL_FAMILY:fatty_acids",
        "FAMS": "CHEMICAL_FAMILY:fatty_acids",
        "FAPU": "CHEMICAL_FAMILY:fatty_acids",
        "FATRN": "CHEMICAL_FAMILY:fatty_acids",
        "FIBTG": "CHEMICAL_FAMILY:dietary_fibre",
        "SUGAR": "CHEMICAL_FAMILY:sugars",
    }
    if tag in aggregate_families:
        rows.append(_relation(
            concept_id, aggregate_families[tag], "aggregate_of", "curated_expression_relation",
            "INFOODS tagname identifies an aggregate food-component expression.", INFOODS_URL,
            "Aggregate and component-level measurements remain separate axes; this edge does not authorise arithmetic reconciliation.",
        ))

    form_families = {
        "RETOL": "REQUIREMENT:vitamin_a", "CARTB": "REQUIREMENT:vitamin_a",
        "VITD2": "REQUIREMENT:vitamin_d", "VITD3": "REQUIREMENT:vitamin_d",
        "PYRX": "REQUIREMENT:vitamin_b6", "PYRD": "REQUIREMENT:vitamin_b6",
        "PYRDX": "REQUIREMENT:vitamin_b6", "FOL": "REQUIREMENT:folate",
        "FOLAC": "REQUIREMENT:folate", "VITK1": "REQUIREMENT:vitamin_k",
    }
    if tag in form_families:
        rows.append(_relation(
            concept_id, form_families[tag], "chemical_form_of", "curated_family_relation",
            "INFOODS tagname and dietary-reference nutrient-family definition.", NIH_DRI_URL,
            "Chemical form and nutritional family are linked but not collapsed into one molecular axis.",
        ))

    # These expressions arise from distinct analytical or calculated
    # definitions.  The graph records the constraint instead of resolving it.
    if tag == "FATCE":
        rows.append(_relation(
            concept_id, "INFOODS:FAT", "method_variant_of", "curated_method_relation",
            "INFOODS component-expression convention.", INFOODS_URL,
            "Continuous-extraction fat is a method-specific total-fat expression and is not pooled here.",
        ))
    if tag == "FIBC":
        rows.append(_relation(
            concept_id, "INFOODS:FIBTG", "not_comparable_to", "curated_definition_constraint",
            "INFOODS distinguishes crude fibre from total dietary fibre.", INFOODS_URL,
            "Crude fibre must not be silently substituted for total dietary fibre.",
        ))
    if tag == "CHOCDF":
        rows.append(_relation(
            concept_id, "REQUIREMENT:available_carbohydrate", "not_comparable_to", "curated_definition_constraint",
            "FAO/INFOODS food-component definitions distinguish carbohydrate by difference from available carbohydrate.", FAO_MATCHING_URL,
            "Calculated carbohydrate by difference is retained as an expression, not treated as available carbohydrate.",
        ))

    if classification["requirement_id"]:
        requirement = REQUIREMENT_BY_ID[classification["requirement_id"]]
        target = f"REQUIREMENT:{requirement.identifier}"
        if target != concept_id:
            if classification["food_composition_role"] == "derived_nutrition_expression":
                relation_type = "activity_equivalent_of"
            elif requirement.kind in {"chemical_entity", "amino_acid", "elemental_nutrient"}:
                relation_type = "chemical_form_of"
            else:
                relation_type = "aggregate_of"
            basis = (
                "INFOODS tagname and normative requirement registry."
                if tag else "Source name/definition candidate matched to normative requirement; human semantic review pending."
            )
            rows.append(_relation(
                concept_id, target, relation_type, "requirement_assignment",
                basis, INFOODS_URL if tag else NIH_DRI_URL,
                "This records nutritional-role assignment only. It does not assert cross-source analytical equivalence or permit value pooling.",
            ))
    return rows


def build_concepts_and_relations(source_axes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = []
    relation_rows = []
    for row in source_axes.itertuples(index=False):
        series = pd.Series(row._asdict())
        classification = _classify_axis(series)
        concept_id, namespace, authority_id, basis, evidence = _concept_identity(series, classification)
        mapped.append({**series.to_dict(), **classification, "axis_concept_id": concept_id,
                       "authority_namespace": namespace, "authority_id": authority_id,
                       "canonical_name_basis": basis, "evidence_status": evidence})
        relation_rows.append(_relation(
            series.source_axis_id, concept_id, "exact_analyte", "source_to_concept_mapping", basis,
            INFOODS_URL if namespace == "INFOODS" else (
                _clean(series.source_record_url) or (FOODB_URL if _clean(series.foodb_public_id) else "")
            ),
            "Maps a source-native axis to an atlas concept; does not combine any numeric values.",
        ))
        text = f"{_clean(series.original_name)} {_clean(series.source_definition)}"
        if DERIVED_PATTERN.search(text):
            target = f"REQUIREMENT:{classification['requirement_id']}" if classification["requirement_id"] else "DERIVATION:source_defined_calculation"
            if target != concept_id:
                relation_rows.append(_relation(
                    concept_id, target, "calculated_from", "expression_annotation",
                    "Source label/definition identifies a calculated or activity-equivalent expression.", FAO_MATCHING_URL,
                    "This is a semantic relation, not a numerical conversion instruction.",
                ))
        relation_rows.extend(_semantic_relations(series, classification, concept_id))
    mapped_frame = pd.DataFrame(mapped)
    concepts = []
    for concept_id, group in mapped_frame.groupby("axis_concept_id", sort=True):
        first = group.iloc[0]
        requirement = REQUIREMENT_BY_ID.get(_clean(first.requirement_id))
        canonical_name = requirement.name if requirement else _clean(first.original_name)
        chemical_class = next((value for value in group.source_chemical_class if _clean(value)), "")
        reference = INFOODS_URL if _clean(first.authority_namespace) == "INFOODS" else (
            FOODB_URL if any(group.foodb_public_id.astype(str).str.len().gt(0)) else FAO_COMPOSITION_URL
        )
        concepts.append({
            "axis_concept_id": concept_id, "canonical_name": canonical_name,
            "canonical_name_basis": _clean(first.canonical_name_basis), "authority_namespace": _clean(first.authority_namespace),
            "authority_id": _clean(first.authority_id), "definition": next((value for value in group.source_definition if _clean(value)), ""),
            "chemical_identity": _clean(first.chemical_identity), "food_composition_role": _clean(first.food_composition_role),
            "nutritional_role": _clean(first.nutritional_role), "measurement_modality": _clean(first.measurement_modality),
            "scientific_inclusion_tier": _clean(first.scientific_inclusion_tier), "chemical_class": chemical_class,
            "authority_reference": reference, "evidence_status": _clean(first.evidence_status),
            "review_priority": _clean(first.review_priority), "source_axis_count": len(group),
            "source_count": group.source_key.nunique(), "source_keys": ";".join(sorted(group.source_key.unique())),
            "value_available_source_count": group.loc[
                group.value_availability.isin(NUMERIC_VALUE_AVAILABILITY), "source_key"
            ].nunique(),
        })
    relations = pd.DataFrame(relation_rows).drop_duplicates(RELATION_COLUMNS)
    return mapped_frame, pd.DataFrame(concepts, columns=CONCEPT_COLUMNS), relations


def requirement_registry() -> pd.DataFrame:
    return pd.DataFrame([{
        "requirement_id": requirement.identifier, "canonical_name": requirement.name,
        "scientific_group": requirement.group, "component_kind": requirement.kind,
        "normative_status": "must_have_nutrition_core", "rationale": requirement.rationale,
        "primary_authority": "National Academies DRI / NIH ODS; FAO/INFOODS food-composition framework",
        "authority_url": f"{NIH_DRI_URL} | {FAO_COMPOSITION_URL}",
        "preferred_infoods_tags": ";".join(requirement.tags), "related_forms_note": requirement.forms,
    } for requirement in MUST_HAVE_REQUIREMENTS], columns=REQUIREMENT_COLUMNS)


def axis_evidence_ledger(mapped_axes: pd.DataFrame) -> pd.DataFrame:
    """Expose every candidate identity and the remaining review obligation.

    This is an evidence ledger, not a statement that an automatic candidate
    has received expert confirmation. The raw source axis remains the source
    of record even where a standardized identifier is present.
    """
    rows = []
    for row in mapped_axes.itertuples(index=False):
        namespace = _clean(row.authority_namespace)
        if namespace == "INFOODS":
            definition_url = INFOODS_URL
        elif namespace in {"InChI", "InChIKey"}:
            definition_url = CHEBI_URL
        elif _clean(row.foodb_public_id):
            definition_url = _clean(row.source_record_url) or FOODB_URL
        else:
            definition_url = _clean(row.source_record_url) or FAO_COMPOSITION_URL
        source_defined_only = _clean(row.evidence_status) == "source_defined_only"
        unresolved = _clean(row.scientific_inclusion_tier) in {
            "retain_pending_manual_review", "include_capable_pending_semantic_review",
        }
        if namespace in {"INFOODS", "InChI", "InChIKey"}:
            uncertainty = "identifier mapping is syntactically or standards supported; human semantic review not yet recorded"
        elif source_defined_only:
            uncertainty = "stable source identity only; no cross-source equivalence asserted"
        else:
            uncertainty = "curated project requirement mapping; human review not yet recorded"
        rows.append({
            "source_axis_id": row.source_axis_id,
            "axis_concept_id": row.axis_concept_id,
            "source_key": row.source_key,
            "source_component_id": row.source_component_id,
            "original_name": row.original_name,
            "source_definition": row.source_definition,
            "candidate_chemical_identity": row.chemical_identity,
            "authority_namespace": namespace,
            "authority_id": row.authority_id,
            "infoods_tag": row.infoods_tag,
            "eurofir_id": row.eurofir_id,
            "foodb_public_id": row.foodb_public_id,
            "inchi": row.inchi,
            "inchikey": row.inchikey,
            "cas_number": row.cas_number,
            "external_identifiers": row.external_identifiers,
            "canonical_name_basis": row.canonical_name_basis,
            "authority_definition_url": definition_url,
            "evidence_status": row.evidence_status,
            "human_review_status": "not_human_reviewed",
            "uncertainty_statement": uncertainty,
            "manual_review_required": bool(unresolved or source_defined_only),
            "source_field_integrity_note": row.source_field_integrity_note,
        })
    return pd.DataFrame(rows)


def source_acquisition_registry(registry: pd.DataFrame, outcomes: dict[str, tuple[str, str, int]]) -> pd.DataFrame:
    records = []
    for row in registry.itertuples(index=False):
        status, detail, count = outcomes.get(row.source_key, ("not_attempted", "", 0))
        records.append({
            "source_key": row.source_key, "source_name": row.name, "region": row.region,
            "source_version": row.source_version, "source_version_status": row.source_version_status,
            "access_status": row.access_status, "license_or_terms": row.license_or_terms,
            "official_url": row.official_url, "download_url": row.download_url,
            "axis_catalog_status": status, "extracted_axis_count": count,
            "status_detail": detail, "local_raw_path": row.resolved_local_raw_path,
            "raw_artifact_sha256": row.raw_artifact_sha256,
            "raw_artifact_hash_scope": row.raw_artifact_hash_scope,
        })
    return pd.DataFrame(records)


def gap_report(requirements: pd.DataFrame, mapped_axes: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    known_sources = registry.source_key.tolist()
    rows = []
    for requirement in requirements.itertuples(index=False):
        matching = mapped_axes[mapped_axes.requirement_id.eq(requirement.requirement_id)]
        contributing = set(matching.source_key)
        for source_key in known_sources:
            source_match = matching[matching.source_key.eq(source_key)]
            rows.append({
                "requirement_id": requirement.requirement_id, "canonical_name": requirement.canonical_name,
                "scientific_group": requirement.scientific_group, "source_key": source_key,
                "axis_present_in_catalog": bool(len(source_match)),
                "food_value_table_present": bool(source_match.value_availability.isin(NUMERIC_VALUE_AVAILABILITY).any()),
                "matching_source_axis_ids": ";".join(source_match.source_axis_id.tolist()),
                "matching_source_names": " | ".join(source_match.original_name.tolist()),
                "global_catalogue_source_count": len(contributing),
                "interpretation": "Coverage is descriptive only and does not affect normative inclusion.",
            })
    return pd.DataFrame(rows)


def write_atlas_report(report_path: Path, *, registry: pd.DataFrame, acquisition: pd.DataFrame,
                       source_axes: pd.DataFrame, concepts: pd.DataFrame, requirements: pd.DataFrame,
                       gap: pd.DataFrame, relations: pd.DataFrame) -> None:
    status_counts = acquisition.groupby("axis_catalog_status").source_key.nunique().sort_values(ascending=False)
    regional_counts = registry.groupby("region").source_key.nunique().sort_values(ascending=False)
    tier_counts = concepts.groupby("scientific_inclusion_tier").axis_concept_id.nunique().sort_values(ascending=False)
    role_counts = concepts.groupby("food_composition_role").axis_concept_id.nunique().sort_values(ascending=False)
    relation_counts = relations.groupby("relation_type").relation_id.nunique().sort_values(ascending=False)
    covered = gap.groupby("requirement_id").axis_present_in_catalog.any()
    core_covered = int(covered.sum())
    lines = [
        "# Global Food Metabolome Axis Atlas v1",
        "",
        "## Scope and boundary",
        "",
        f"This atlas inventories the source-native component axes registered in {len(registry)} global food-composition and specialist sources. It is prior to food matching, value conversion, numerical aggregation, split construction, and model training.",
        "",
        "No component is discarded because it is rare, appears in one source, lacks a current label, or is not currently suitable for masked prediction. Coverage is reported only as a data-gap diagnostic.",
        "",
        "The atlas follows compatible international guidance rather than claiming a single global standard: [FAO/INFOODS component identifiers](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/), [FAO/INFOODS food matching](https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf), [FAO food-composition scope](https://www.fao.org/food-composition/overview-2/en), dietary-reference definitions from [NIH ODS](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/), and structural identities from [ChEBI](https://www.ebi.ac.uk/chebi/), [LIPID MAPS](https://www.lipidmaps.org/), and source databases such as [FooDB](https://foodb.ca/).",
        "",
        "## Source acquisition",
        "",
        f"- Registry sources: {len(registry):,}",
        f"- Extracted source-native axes: {len(source_axes):,}",
        f"- Atlas concepts: {len(concepts):,}",
        "- Source catalog states:",
        *[f"  - `{state}`: {count}" for state, count in status_counts.items()],
        "- Registry sources by declared region:",
        *[f"  - `{region}`: {count}" for region, count in regional_counts.items()],
        "",
        "Each `source_registry.csv` row contains the inherited source label, access/terms, official and download routes, resolved local raw path where available, and a SHA-256 hash of the raw file or ordered raw-artifact tree. A source without a locally readable dictionary remains in the acquisition ledger. It is not represented by invented axes and it is not silently excluded.",
        "",
        "## Scientific axis layers",
        "",
        "1. **Must-have nutrition core.** Normative requirements based on dietary-reference nutrient categories and FAO/INFOODS food-composition practice. This includes food matrix/proximate expressions, macronutrients, essential fatty acids, indispensable amino acids, vitamins, minerals, trace elements, and choline. Energy and activity equivalents are kept as derived nutritional expressions, not molecular entities.",
        "2. **Include-capable food metabolome.** Structurally defined food-derived molecules such as organic acids, sterols, carotenoids, polyphenols, flavonoids, alkaloids, terpenes, glucosinolates, organosulfur compounds, and other metabolite classes. They are not made equivalent to essential nutrients.",
        "3. **Exposure and safety.** Contaminants, residues, toxins, processing products, and packaging migrants remain a distinct chemical layer.",
        "4. **Non-chemical or derived expressions.** Calculated-by-difference values, activity equivalents, energy, coefficients, edible portion, yield, retention, and similar expressions are retained with semantic relations rather than treated as chemical analytes.",
        "",
        "### Why the layers are separate",
        "",
        "Chemical identity, food-composition role, nutritional role, and measurement modality are stored independently. For example, retinol is a chemical form within the vitamin-A nutritional family; RAE is an activity-equivalent expression rather than a molecule; carbohydrate by difference is a calculated expression rather than available carbohydrate; and a pesticide residue can be a well-defined chemical but belongs to the exposure/safety layer. Collapsing these distinctions would create invalid targets for any later foundation-model task.",
        "",
        "The 50 `must_have_nutrition_core` requirements are normative: water/ash and proximate composition, macronutrient expressions, essential fatty acids, protein-quality amino acids, vitamin families/forms, minerals/trace elements, choline, energy, and activity equivalents. They are included because they describe food composition and dietary relevance, not because a current file has many records.",
        "",
        "### Concept counts by inclusion tier",
        "",
        "| Tier | Concepts |",
        "|---|---:|",
        *[f"| {tier} | {count:,} |" for tier, count in tier_counts.items()],
        "",
        "### Concept counts by composition role",
        "",
        "| Role | Concepts |",
        "|---|---:|",
        *[f"| {role} | {count:,} |" for role, count in role_counts.items()],
        "",
        "### Graph relationship counts",
        "",
        "| Relation | Edges | Meaning |",
        "|---|---:|---|",
        *[f"| {relation} | {count:,} | Source/semantic relation only; never a numerical merge. |" for relation, count in relation_counts.items()],
        "",
        "## Normative core coverage gap",
        "",
        f"The requirements register contains {len(requirements)} scientifically required nutrition-core axes. {core_covered} currently have at least one source-axis candidate match in the local atlas. This indicates catalogue availability only; it does not establish compatible units, analytical definitions, measurements, or cross-source poolability. Any absent requirement would be a future acquisition/semantic-review gap, not an exclusion.",
        "",
        "## Evidence and comparability policy",
        "",
        "- A raw source axis is immutable: its source name, definition, unit, denominator, table location, and source identity remain preserved.",
        "- INFOODS tags can establish standardized food-component expressions. A syntactically valid InChI or InChIKey can establish a candidate structural identity when supplied by a source. Names alone do not establish cross-source equivalence.",
        "- `exact_analyte`, `aggregate_of`, `chemical_form_of`, `activity_equivalent_of`, `calculated_from`, `method_variant_of`, and `not_comparable_to` are graph relations. They never trigger numerical pooling in this atlas.",
        "- `axis_evidence_ledger.csv.gz` records the candidate identifier, mapping basis, authority link, residual uncertainty, and whether human review is still required for every source-native axis.",
        "- FAO/INFOODS requires checking identity, definition, unit, denominator, and analytical expression before treating values as comparable. See the standards and food-matching guidance.",
        "- The FooDB 2020 Compound.csv export has a positional field-label drift after `state`. The adapter performs syntax-validated interpretation for CAS, SMILES, InChI, InChIKey, molecular mass, and IUPAC fields; the immutable raw artifact and field-integrity note remain linked from each FooDB axis.",
        "",
        "## Deliberate exclusions from this phase",
        "",
        "No food-level data are deduplicated, no units are converted, no source is ranked for numerical conflict resolution, no train/validation partition is produced, and no model or baseline is trained.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_atlas_readme(output_dir: Path) -> None:
    """Document the atlas tables at the point of creation."""
    lines = [
        "# Global Food Metabolome Axis Atlas v1",
        "",
        "This directory is a source-native component atlas. It is intentionally **not** a food-level training matrix and contains no train/validation/test partition, pooled value, unit conversion, food deduplication, numerical conflict resolution, or model result.",
        "",
        "## Tables",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `source_registry.csv` | Immutable registry coverage plus source release status, licensing/access metadata, resolved raw path, and raw-artifact SHA-256. |",
        "| `source_axis_acquisition.csv` | Acquisition state for each registered source. `catalogue_pending` means no catalog was invented. |",
        "| `source_axis_catalog.csv.gz` | Complete raw axis catalog for locally readable source dictionaries. Raw names, definitions, units, denominators, source identifiers, methods, and field-integrity notes remain preserved. |",
        "| `axis_evidence_ledger.csv.gz` | Candidate identities, identifier evidence, authority links, uncertainty and human-review state for every source-native axis. |",
        "| `source_axis_to_concept.csv.gz` | Source-axis-to-concept mapping and the four independent classifications. This table never pools values. |",
        "| `axis_concept_registry.csv.gz` | Canonical concept layer with authority/stable identifiers and provenance counts. |",
        "| `axis_relation_graph.csv.gz` | Explicit `exact_analyte`, form, aggregate, calculated-expression, method, and non-comparability edges. |",
        "| `must_have_requirement_registry.csv` | Normative nutrition-core requirements, independent of observation counts. |",
        "| `must_have_gap_report.csv.gz` | Source-by-requirement catalogue availability diagnostic; it is not evidence of numerical compatibility. |",
        "| `unresolved_axis_review_queue.csv.gz` | Entries whose chemical identity or semantic assignment still requires manual review. |",
        "",
        "## Invariants",
        "",
        "- Missing, zero, censored values, ranges, and numeric measurements are not handled here because this phase does not ingest or transform food-level measurements.",
        "- Identical names never establish cross-source equivalence. Only an INFOODS expression or syntactically valid structural identifier can create a candidate shared concept; even then, unit, denominator, method and definition still govern later numerical compatibility.",
        "- Inclusion tiers are based on food/nutrition science and not on source count, observation count, or model performance.",
        "- Sources lacking a legally accessible or locally readable catalogue remain registered and pending. Their axes are not inferred.",
        "",
        "## Rebuild",
        "",
        "Run `python scripts/build_global_food_metabolome_axis_atlas.py` from the repository root. The script reads only registered local raw artifacts and does not download around access restrictions.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_axis_atlas(root: Path, output_dir: Path, report_dir: Path) -> dict[str, int]:
    registry_path = root / "data/raw/global_fcdb_inventory_2026_09_10/source_registry.csv"
    registry = enrich_source_registry(pd.read_csv(registry_path, keep_default_na=False), root)
    if registry.source_key.duplicated().any():
        raise ValueError("Global source registry contains duplicate source_key values.")
    all_rows: list[dict[str, Any]] = []
    outcomes: dict[str, tuple[str, str, int]] = {}
    for _, source in registry.iterrows():
        rows, status, detail = extract_source_axes(source, root)
        all_rows.extend(rows)
        outcomes[_clean(source.source_key)] = (status, detail, len(rows))
    source_axes = pd.DataFrame(all_rows, columns=SOURCE_AXIS_COLUMNS)
    if source_axes.empty:
        raise ValueError("No source axes were extracted; verify local raw paths and source adapters.")
    if source_axes.source_axis_id.duplicated().any():
        raise ValueError("Source-axis identifiers are not unique.")
    mapped_axes, concepts, relations = build_concepts_and_relations(source_axes)
    requirements = requirement_registry()
    acquisition = source_acquisition_registry(registry, outcomes)
    gap = gap_report(requirements, mapped_axes, registry)
    evidence = axis_evidence_ledger(mapped_axes)
    review_ids = set(evidence.loc[evidence.manual_review_required, "source_axis_id"])
    unresolved = mapped_axes[mapped_axes.source_axis_id.isin(review_ids)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(registry, output_dir / "source_registry.csv")
    write_csv(acquisition, output_dir / "source_axis_acquisition.csv")
    _write_table(source_axes, output_dir / "source_axis_catalog.csv.gz")
    _write_table(mapped_axes, output_dir / "source_axis_to_concept.csv.gz")
    _write_table(evidence, output_dir / "axis_evidence_ledger.csv.gz")
    _write_table(concepts, output_dir / "axis_concept_registry.csv.gz")
    _write_table(relations, output_dir / "axis_relation_graph.csv.gz")
    write_csv(requirements, output_dir / "must_have_requirement_registry.csv")
    _write_table(gap, output_dir / "must_have_gap_report.csv.gz")
    _write_table(unresolved, output_dir / "unresolved_axis_review_queue.csv.gz")
    write_json({
        "dataset": "global_food_metabolome_axis_atlas_v1",
        "purpose": "Source-native axis atlas before value harmonization, deduplication, splitting, or training.",
        "source_registry_count": int(len(registry)), "source_axis_count": int(len(source_axes)),
        "axis_concept_count": int(len(concepts)), "must_have_requirement_count": int(len(requirements)),
        "unresolved_axis_count": int(len(unresolved)), "axis_evidence_record_count": int(len(evidence)),
        "invariants": [
            "No source numeric values are pooled.", "No food rows are matched or deduplicated.",
            "No inclusion tier uses observation count or source count as a selection criterion.",
            "Coverage tables are descriptive gap analyses only.",
        ],
    }, output_dir / "manifest.json")
    report_kwargs = dict(registry=registry, acquisition=acquisition, source_axes=source_axes,
                         concepts=concepts, requirements=requirements, gap=gap, relations=relations)
    write_atlas_report(report_dir / "global_food_metabolome_atlas.md", **report_kwargs)
    write_atlas_report(report_dir / "GLOBAL_FOOD_METABOLOME_AXIS_ATLAS_ZH.md", **report_kwargs)
    write_atlas_readme(output_dir)
    return {
        "sources": len(registry), "source_axes": len(source_axes), "concepts": len(concepts),
        "requirements": len(requirements), "unresolved": len(unresolved),
    }

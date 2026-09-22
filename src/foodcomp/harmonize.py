"""Conservative identity harmonization, aggregation, filtering and splitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import (
    AS_OF_DATE,
    AMINO_ACID_NAMES,
    AMINO_ACID_TAGS,
    DATASET_VERSION,
    FAMILY_HOLDOUT_CONCEPT_TARGET,
    MINERAL_TAGS,
    NUTRITIONAL_ROLE_BY_TAG,
    PROCESSING_TERMS,
    RANDOM_SEED,
    VITAMIN_TAG_PREFIXES,
)
from .ontology import (
    ancestor_names,
    normalize_ontology_id,
    ontology_label_index,
    parse_foodon,
    parse_infoods_tagnames,
    parse_obo,
)
from .util import normalize_text, read_component_csv, stable_id, write_csv, write_json
from .source_policy import apply_source_policy, validation_reference_mask


QUALITY_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
SUPPLEMENT_PATTERN = re.compile(r"\b(?:supplement|multivitamin|tablet|capsule|formula powder)\b", re.I)
INEDIBLE_PATTERN = re.compile(r"\b(?:inedible|refuse|shell only|bone only|peel only)\b", re.I)
FOOD_IDENTITY_FACETS = (
    "scientific_name", "part", "maturity", "processing", "cooking",
    "preservation", "physical_state", "packing_medium", "cultivar", "geography",
)


class UnionFind:
    def __init__(self, items: Iterable[str], facets: dict[str, dict[str, str]] | None = None):
        self.parent = {item: item for item in items}
        self.facets = facets or {item: {} for item in self.parent}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return True
        left_facets = self.facets.get(a, {})
        right_facets = self.facets.get(b, {})
        for key in set(left_facets) | set(right_facets):
            x, y = left_facets.get(key, ""), right_facets.get(key, "")
            if x and y and x != y:
                return False
        winner, loser = min(a, b), max(a, b)
        self.parent[loser] = winner
        merged = dict(left_facets if winner == a else right_facets)
        other = right_facets if winner == a else left_facets
        for key, value in other.items():
            if value and not merged.get(key):
                merged[key] = value
        self.facets[winner] = merged
        self.facets.pop(loser, None)
        return True


def read_staging(staging_dir: Path, *, source_admission_policy: str | None = None,
                 policy_ledger_dir: Path | None = None,
                 component_definitions: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    food_paths = sorted(staging_dir.glob("*/food_observation.csv.gz"))
    component_paths = sorted(staging_dir.glob("*/component_observation.csv.gz"))
    measurement_paths = sorted(staging_dir.glob("*/measurement_part_*.csv.gz"))
    if not food_paths or not component_paths or not measurement_paths:
        raise FileNotFoundError(f"Staging tables are incomplete under {staging_dir}")
    foods = pd.concat([pd.read_csv(path, low_memory=False) for path in food_paths], ignore_index=True)
    components = pd.concat([read_component_csv(path) for path in component_paths], ignore_index=True)
    eligible_parts = []
    disposition_parts = []
    manifest_rows = []
    policy_summaries = []
    for path in measurement_paths:
        frame = pd.read_csv(path, low_memory=False)
        for column in ("main_value_eligible", "independent_evidence", "strict_validation_eligible"):
            if column in frame:
                frame[column] = frame[column].astype("string").str.casefold().eq("true").fillna(False).astype(bool)
        if source_admission_policy is not None:
            from .sources import gate_component_expressions
            frame = apply_source_policy(frame, source_admission_policy)
            source_components = components[components.source_key.eq(path.parent.name)]
            frame = gate_component_expressions(frame, source_components, component_definitions or {})
            frame["validation_reference_eligible"] &= frame.main_value_eligible
            if policy_ledger_dir is not None:
                ledger_columns = [
                    "measurement_id", "source_key", "food_observation_id", "component_observation_id",
                    "source_reference", "method_expression", "quality_tier", "independent_evidence",
                    "value_status", "conversion_status", "normalized_value_g_per_100g",
                    "source_admission_policy", "source_trusted", "source_gate_previous_eligible",
                    "source_gate_previous_validation_status",
                    "source_gate_previous_validation_eligible", "source_gate_previous_exclusion_reason",
                    "main_value_eligible", "strict_validation_eligible", "validation_reference_eligible",
                    "source_policy_decision", "source_policy_reason", "source_independence_status",
                    "source_value_expression", "validation_evidence_basis", "exclusion_reason",
                ]
                write_csv(frame[ledger_columns], policy_ledger_dir / path.parent.name / path.name)
                group_columns = ["source_key", "source_gate_previous_eligible", "main_value_eligible",
                                 "source_gate_previous_validation_eligible", "validation_reference_eligible",
                                 "source_policy_decision", "source_policy_reason", "exclusion_reason"]
                policy_summaries.append(frame.groupby(group_columns, dropna=False).size().reset_index(name="measurement_count"))
        flag = frame["main_value_eligible"]
        if flag.dtype != bool:
            flag = flag.astype(str).str.casefold().eq("true")
        disposition_columns = [
            "source_key", "data_layer", "quality_tier", "value_status",
            "conversion_status", "main_value_eligible", "exclusion_reason",
        ]
        disposition_parts.append(frame.groupby(disposition_columns, dropna=False).size().reset_index(name="measurement_count"))
        if flag.any():
            eligible_parts.append(frame[flag].copy())
        manifest_rows.append({"relative_partition": str(path.relative_to(staging_dir.parent.parent.parent)), "source_key": path.parent.name, "rows": len(frame), "main_eligible_rows": int(flag.sum())})
    measurements = pd.concat(eligible_parts, ignore_index=True)
    disposition = pd.concat(disposition_parts, ignore_index=True).groupby(disposition_columns, dropna=False)["measurement_count"].sum().reset_index()
    if policy_summaries:
        summary = pd.concat(policy_summaries, ignore_index=True).groupby(group_columns, dropna=False)["measurement_count"].sum().reset_index()
        write_csv(summary, policy_ledger_dir / "source_policy_disposition.csv")
    return foods, components, measurements, disposition, pd.DataFrame(manifest_rows)


def _clean_identifier(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return "" if text.casefold() in {"", "nan", "none"} else text


def _formal_food_lineage(value: Any) -> str:
    """Return only identifiers that can support food identity or split blocking."""
    text = _clean_identifier(value)
    if text.upper().startswith("FOODB_CITATION:"):
        return ""
    return text


def _component_variant(name: str) -> str:
    normalized = normalize_text(name)
    if "label" in normalized or "declaration" in normalized:
        return "label_expression"
    if "from amino acid" in normalized:
        return "calculated_from_constituents"
    if "by difference" in normalized or "difference" in normalized:
        return "calculated_by_difference"
    if any(token in normalized for token in ("equivalent", "rae", "dfe", "alpha te", "from tryptophan", "vitamin a activities", "vitamin e activities")):
        return "biological_equivalent"
    return "standard_expression"


def _authority_key(row: pd.Series, valid_infoods_tags: set[str]) -> tuple[str, str, str]:
    tag = _clean_identifier(row.get("infoods_tag")).upper()
    # Both official tag definitions are identical in the INFOODS registry.
    # Keep the raw supplied tag in component_observation for provenance.
    if tag == "PROT":
        tag = "PROCNT"
    chebi = normalize_ontology_id(row.get("chebi_id"), "CHEBI")
    inchikey = _clean_identifier(row.get("inchikey")).upper()
    eurofir = _clean_identifier(row.get("eurofir_component_id")).upper()
    variant = _component_variant(str(row.get("original_name", "")))
    if tag in valid_infoods_tags:
        return "INFOODS", tag, variant
    original_usda_expression = (row.get("source_key") == "foodb" and
                                str(row.get("source_definition", "")).startswith("FooDB Content.orig_source_id = USDA nutrient number "))
    if row.get("source_key") in {"usda_sr_legacy", "usda_foundation", "fndds", "cnf"} or original_usda_expression:
        definition = _clean_identifier(row.get("source_definition"))
        match = re.search(r"(?:number|code)\s+([0-9]+(?:\.[0-9]+)?)", definition, flags=re.I)
        if match:
            return "USDA_CNF_NUTRIENT_NBR", match.group(1), variant
    if chebi:
        return "CHEBI", chebi, "chemical_identity"
    if re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey):
        return "INCHIKEY", inchikey, "chemical_identity"
    if eurofir:
        return "EUROFIR_EFSA", eurofir, variant
    return "SOURCE", f"{row['source_key']}:{row['source_component_id']}", variant


def _preferred_label(
    group: pd.DataFrame,
    chebi_terms: dict[str, dict[str, Any]],
    infoods_terms: dict[str, dict[str, str]],
    authority_ns: str,
    authority_id: str,
) -> tuple[str, str]:
    if authority_ns == "INFOODS" and authority_id in infoods_terms:
        label = infoods_terms[authority_id].get("preferred_name", "").strip()
        if label:
            return label, "FAO/INFOODS official Tagname description"
    if authority_ns == "CHEBI" and authority_id in chebi_terms:
        return chebi_terms[authority_id].get("name", authority_id), "ChEBI preferred name"
    labels = []
    for value in group["original_name"].dropna().astype(str):
        label = re.sub(r"\s*\([^()]*?(?:/100|mg|µg|ug|kcal|kj)[^()]*\)\s*$", "", value, flags=re.I).strip()
        if label:
            labels.append(label)
    if not labels:
        return authority_id, "authority identifier"
    normalized_counts = Counter(normalize_text(label) for label in labels)
    winner_norm, _ = normalized_counts.most_common(1)[0]
    candidates = sorted((x for x in labels if normalize_text(x) == winner_norm), key=lambda x: (len(x), x.casefold()))
    return candidates[0], "modal official source label"


def _role_from_component(tag: str, name: str, group: str) -> tuple[str, str, str, str]:
    tag = tag.upper()
    normalized = normalize_text(f"{name} {group}")
    clean_name = normalize_text(name)
    if tag == "CHOLN" or clean_name in {"choline", "choline total"}:
        return "essential_nutrient_choline", "NIH essential nutrient choline", "https://ods.od.nih.gov/factsheets/Choline-HealthProfessional/", "authority_supported"
    if clean_name in {"protein", "total protein", "fat total", "total fat", "carbohydrate", "dietary fibre", "dietary fiber"}:
        return "macronutrient", "Official source proximate definition; DRI role", "https://nap.nationalacademies.org/collection/57/dietary-reference-intakes", "source_definition_role_pending_expert_review"
    if tag in NUTRITIONAL_ROLE_BY_TAG:
        return NUTRITIONAL_ROLE_BY_TAG[tag], "FAO/INFOODS component role with DRI terminology", "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/", "authority_supported"
    mineral_name = any(token in normalized for token in ("calcium", "iron", "magnesium", "phosphorus", "potassium", "sodium", "zinc", "copper", "manganese", "selenium", "iodine", "molybdenum", "chloride"))
    if tag in MINERAL_TAGS or mineral_name:
        status = "authority_supported" if tag in MINERAL_TAGS else "name_rule_pending_expert_review"
        return "micronutrient_mineral", "National Academies/NIH mineral nutrient role", "https://nap.nationalacademies.org/collection/57/dietary-reference-intakes", status
    vitamin_name = "vitamin" in normalized or any(token in normalized for token in ("thiamin", "riboflavin", "niacin", "folate", "biotin", "pantothenic", "cobalamin", "tocopherol"))
    if tag.startswith(VITAMIN_TAG_PREFIXES) or vitamin_name:
        status = "authority_supported" if tag.startswith(VITAMIN_TAG_PREFIXES) else "name_rule_pending_expert_review"
        return "micronutrient_vitamin", "National Academies/NIH vitamin nutrient role", "https://ods.od.nih.gov/factsheets/list-all/", status
    if tag.startswith(("FA", "FASAT", "FAMS", "FAPU")) or "fatty acid" in normalized:
        status = "authority_supported" if tag else "name_rule_pending_expert_review"
        return "nutrient_constituent_fatty_acid", "INFOODS/LIPID MAPS chemical form", "https://www.lipidmaps.org/resources/education/classification", status
    if tag in AMINO_ACID_TAGS or tag.startswith("AA") or clean_name in AMINO_ACID_NAMES or "amino acid" in normalized:
        status = "authority_supported" if tag else "name_rule_pending_expert_review"
        return "nutrient_constituent_amino_acid", "INFOODS/CDNO chemical form", "https://www.cropontology.org/ontology/CDNO", status
    if any(token in normalized for token in ("sugar", "glucose", "fructose", "sucrose", "starch", "oligosaccharide")):
        return "nutrient_constituent_carbohydrate", "INFOODS/CDNO chemical form", "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/", "name_rule_pending_expert_review" if not tag else "authority_supported"
    return "other_food_component", "Chemical identity only; no nutritional-role assertion", "https://www.ebi.ac.uk/chebi/", "no_nutritional_role_asserted"


def _component_family(tag: str, name: str, role: str) -> str:
    normalized = normalize_text(name)
    tag = tag.upper()
    if tag == "CHOLN" or role == "essential_nutrient_choline":
        return "choline_family"
    if role == "macronutrient" or tag.startswith(("FAT", "FA")) or "fatty acid" in normalized:
        if tag.startswith(("FAT", "FA")) or "fat" in normalized:
            return "lipid_and_fatty_acid_family"
    if tag in AMINO_ACID_TAGS or normalized in AMINO_ACID_NAMES or role == "nutrient_constituent_amino_acid" or tag.startswith(("PRO", "AA")) or "protein" in normalized or "amino acid" in normalized:
        return "protein_and_amino_acid_family"
    if tag.startswith(("CHO", "SUGAR", "FIB", "STAR")) or any(x in normalized for x in ("carbohydrate", "sugar", "starch", "fiber", "fibre")):
        return "carbohydrate_family"
    if role == "micronutrient_vitamin":
        vitamin_families = {
            "a": (("VITA", "RETOL", "CART"), ("retinol", "carotene", "cryptoxanthin")),
            "b1": (("THIA",), ("thiamin",)), "b2": (("RIBF",), ("riboflavin",)),
            "b3": (("NIA",), ("niacin", "nicotin")), "b5": (("PANTAC",), ("pantothen",)),
            "b6": (("VITB6",), ("vitamin b6", "pyridox")), "b7": (("BIOT",), ("biotin",)),
            "b9": (("FOL",), ("folate", "folic")), "b12": (("VITB12",), ("vitamin b12", "cobalamin")),
            "c": (("VITC",), ("vitamin c", "ascorb")), "d": (("VITD",), ("vitamin d", "calciferol")),
            "e": (("VITE", "TOCP", "TOCT"), ("vitamin e", "tocopherol", "tocotrienol")),
            "k": (("VITK",), ("vitamin k", "phylloquinone", "menaquinone")),
        }
        for family, (prefixes, terms) in vitamin_families.items():
            if tag.startswith(prefixes) or any(term in normalized for term in terms):
                return "vitamin:" + family
        return "vitamin_unresolved:" + (tag or stable_id("label", normalized))
    if role == "micronutrient_mineral":
        return "mineral:" + (tag or normalized.split()[0])
    return "component:" + (tag or stable_id("label", normalized))


def build_component_concepts(
    components: pd.DataFrame,
    root: Path,
    infoods_terms: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = components.reset_index(drop=True)
    if infoods_terms is None:
        infoods_terms = parse_infoods_tagnames(root / "data/raw/reference/infoods_tagnames_2022")
    chebi_terms = parse_obo(root / "data/raw/ontology/chebi_lite.obo.gz")
    cdno_terms = parse_obo(root / "data/raw/ontology/cdno.obo")
    cdno_labels = ontology_label_index(cdno_terms)
    valid_infoods_tags = set(infoods_terms)
    keys = components.apply(lambda row: _authority_key(row, valid_infoods_tags), axis=1, result_type="expand")
    keys.columns = ["authority_namespace", "authority_id", "expression_variant"]
    working = pd.concat([components.reset_index(drop=True), keys], axis=1)
    working["component_concept_id"] = [stable_id("component", ns, ident, variant) for ns, ident, variant in keys.itertuples(index=False, name=None)]
    concept_rows = []
    evidence_rows = []
    for concept_id, group in working.groupby("component_concept_id", sort=True):
        authority_ns = group["authority_namespace"].iloc[0]
        authority_id = group["authority_id"].iloc[0]
        variant = group["expression_variant"].iloc[0]
        name, name_basis = _preferred_label(group, chebi_terms, infoods_terms, authority_ns, authority_id)
        tag_values = sorted({_clean_identifier(x).upper() for x in group["infoods_tag"] if _clean_identifier(x)})
        tag = authority_id if authority_ns == "INFOODS" else tag_values[0] if len(tag_values) == 1 else ""
        source_group = "; ".join(sorted({_clean_identifier(x) for x in group["source_component_group"] if _clean_identifier(x)}))
        role, role_basis, role_url, role_status = _role_from_component(tag, name, source_group)
        chebi_ids = sorted({normalize_ontology_id(x, "CHEBI") for x in group["chebi_id"] if normalize_ontology_id(x, "CHEBI")})
        chebi_id = chebi_ids[0] if len(chebi_ids) == 1 else ""
        chemical_path = ""
        if chebi_id:
            chemical_path = " > ".join(ancestor_names(chebi_id, chebi_terms, max_depth=4))
        if not chemical_path:
            source_classes = sorted({_clean_identifier(x) for x in group["source_chemical_class"] if _clean_identifier(x)})
            chemical_path = source_classes[0] if len(source_classes) == 1 else ""
        exact_cdno = cdno_labels.get(normalize_text(name), [])
        cdno_id = exact_cdno[0] if len(exact_cdno) == 1 else ""
        identity_status = "authority_verified" if authority_ns in {"INFOODS", "CHEBI", "INCHIKEY", "EUROFIR_EFSA", "USDA_CNF_NUTRIENT_NBR"} else "stable_source_identity_pending_cross_database_review"
        official_definition = infoods_terms.get(authority_id, {}).get("definition", "") if authority_ns == "INFOODS" else ""
        concept_rows.append({
            "component_concept_id": concept_id, "canonical_name": name,
            "authority_namespace": authority_ns, "authority_id": authority_id,
            "infoods_tag": tag, "chebi_id": chebi_id, "cdno_id": cdno_id,
            "expression_variant": variant, "definition": official_definition or _first_nonempty(group["source_definition"]),
            "chemical_class": chemical_path or "unresolved_chemical_class",
            "nutritional_role": role, "nutritional_role_basis": role_basis,
            "nutritional_role_evidence_url": role_url,
            "nutritional_role_review_status": role_status,
            "measurement_modality": "pending_measurement_linkage",
            "component_family": _component_family(tag, name, role),
            "identity_status": identity_status, "canonical_name_basis": name_basis,
            "source_count": group["source_key"].nunique(), "source_keys": ";".join(sorted(group["source_key"].unique())),
        })
        for row in group.itertuples(index=False):
            evidence_rows.append({
                "evidence_id": stable_id("evidence", row.component_observation_id),
                "entity_type": "component", "observation_id": row.component_observation_id,
                "source_key": row.source_key, "original_name": row.original_name,
                "canonical_id": concept_id, "canonical_name": name,
                "query": f"{row.original_name} {row.infoods_tag or ''} {row.chebi_id or ''}".strip(),
                "authority_namespace": authority_ns, "authority_id": authority_id,
                "evidence_url": _component_evidence_url(authority_ns, authority_id),
                "decision": "canonicalized_by_formal_identifier" if authority_ns != "SOURCE" else "kept_source_specific",
                "confidence": "high" if identity_status == "authority_verified" else "medium",
                "review_status": "machine_verified_identifier" if identity_status == "authority_verified" else "expert_review_required",
                "review_reason": name_basis,
            })
    concepts = pd.DataFrame(concept_rows)
    mapping = working[["component_observation_id", "component_concept_id", "authority_namespace", "authority_id", "expression_variant"]].copy()
    candidates = _component_name_candidates(working, concepts)
    expert = _expert_review_sample(pd.DataFrame(evidence_rows), entity_type="component")
    return concepts, mapping, pd.DataFrame(evidence_rows), pd.concat([candidates, expert], ignore_index=True, sort=False)


def _first_nonempty(values: pd.Series) -> str:
    cleaned = [_clean_identifier(value) for value in values]
    return next((value for value in cleaned if value), "")


def _component_evidence_url(namespace: str, authority_id: str) -> str:
    if namespace == "INFOODS":
        return "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"
    if namespace == "CHEBI":
        return f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={authority_id}"
    if namespace == "INCHIKEY":
        return f"https://pubchem.ncbi.nlm.nih.gov/#query={authority_id}"
    if namespace == "EUROFIR_EFSA":
        return "https://www.efsa.europa.eu/en/data-report/food-classification"
    if namespace == "USDA_CNF_NUTRIENT_NBR":
        return "https://fdc.nal.usda.gov/data-documentation.html"
    return ""


def _component_name_candidates(working: pd.DataFrame, concepts: pd.DataFrame) -> pd.DataFrame:
    source_specific = working[working["authority_namespace"].eq("SOURCE")].copy()
    authority = working[~working["authority_namespace"].eq("SOURCE")].copy()
    by_name: dict[str, set[str]] = defaultdict(set)
    for row in authority.itertuples(index=False):
        by_name[normalize_text(row.original_name)].add(row.component_concept_id)
    rows = []
    concept_name = concepts.set_index("component_concept_id")["canonical_name"].to_dict()
    for row in source_specific.itertuples(index=False):
        candidates = by_name.get(normalize_text(row.original_name), set())
        for candidate in sorted(candidates):
            rows.append({
                "review_type": "component_exact_name_candidate", "entity_type": "component",
                "observation_id": row.component_observation_id, "current_concept_id": row.component_concept_id,
                "candidate_concept_id": candidate, "original_name": row.original_name,
                "candidate_name": concept_name.get(candidate, ""), "similarity": 1.0,
                "automatic_merge": False, "review_status": "expert_review_required",
                "reason": "Exact display-name match is not sufficient without compatible analytical definition and denominator.",
            })
    return pd.DataFrame(rows)


def _compatible_food_facets(left: pd.Series, right: pd.Series) -> bool:
    for column in FOOD_IDENTITY_FACETS:
        a, b = normalize_text(left.get(column)), normalize_text(right.get(column))
        if a and b and a != b:
            return False
    return True


def _food_family_label(row: pd.Series) -> str:
    scientific = normalize_text(row.get("scientific_name"))
    if scientific:
        words = scientific.split()
        return "taxon:" + " ".join(words[:2])
    name = str(row.get("canonical_name") or row.get("original_name") or "")
    first_clause = normalize_text(name.split(",", 1)[0])
    tokens = [token for token in first_clause.split() if token not in PROCESSING_TERMS]
    if not tokens:
        tokens = normalize_text(name).split()
    group = normalize_text(row.get("food_group"))
    return f"lexical:{group}:{' '.join(tokens[:2])}"


def build_food_concepts(foods: pd.DataFrame, root: Path, frozen_mapping: pd.DataFrame | None = None,
                        frozen_concepts: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    working = foods.copy()
    working["normalized_name"] = working["original_name"].map(normalize_text)
    working["normalized_lineage"] = working["source_lineage_id"].map(_formal_food_lineage)
    working["exclusion_flag"] = ""
    supplement = working["original_name"].fillna("").str.contains(SUPPLEMENT_PATTERN)
    inedible = working["original_name"].fillna("").str.contains(INEDIBLE_PATTERN)
    known_brand = working["brand_status"].fillna("").eq("label_value_present")
    working.loc[supplement, "exclusion_flag"] = "supplement"
    working.loc[inedible, "exclusion_flag"] = "inedible_or_refuse"
    working.loc[known_brand, "exclusion_flag"] = "brand_or_label_only"

    ids = working["food_observation_id"].astype(str).tolist()
    facet_map = {
        str(row.food_observation_id): {
            column: normalize_text(getattr(row, column))
            for column in FOOD_IDENTITY_FACETS
            if normalize_text(getattr(row, column))
        }
        for row in working.itertuples(index=False)
    }
    frozen_lookup = {} if frozen_mapping is None else frozen_mapping.set_index("food_observation_id")["food_concept_id"].to_dict()
    for observation, concept in frozen_lookup.items():
        if observation in facet_map:
            facet_map[observation]["frozen_concept_id"] = concept
    uf = UnionFind(ids, facet_map)
    merge_ledger = []
    if frozen_mapping is not None:
        present = frozen_mapping[frozen_mapping["food_observation_id"].isin(uf.parent)]
        for frozen_id, members in present.groupby("food_concept_id"):
            observations = members["food_observation_id"].tolist()
            for observation in observations[1:]:
                if not uf.union(observations[0], observation):
                    raise ValueError(f"Frozen concept {frozen_id} contains conflicting explicit facets")
    for lineage, group in working[working["normalized_lineage"].ne("")].groupby("normalized_lineage"):
        members = group["food_observation_id"].astype(str).tolist()
        for member in members[1:]:
            merged = uf.union(members[0], member)
            reason = "shared_formal_source_lineage" if merged else "shared_lineage_conflicting_facets"
            merge_ledger.append(_food_merge_row(members[0], member, reason, lineage, merged))
    for normalized_name, group in working[working["normalized_name"].ne("")].groupby("normalized_name"):
        rows = [row for _, row in group.iterrows()]
        for index, row in enumerate(rows):
            for candidate in rows[:index]:
                left = str(row["food_observation_id"])
                right = str(candidate["food_observation_id"])
                merged = _compatible_food_facets(row, candidate) and uf.union(left, right)
                reason = "exact_name_compatible_facets" if merged else "exact_name_conflicting_facets"
                merge_ledger.append(_food_merge_row(left, right, reason, normalized_name, merged))

    working["root"] = working["food_observation_id"].astype(str).map(uf.find)
    working["food_concept_id"] = working.groupby("root")["normalized_lineage"].transform(lambda x: stable_id("food", next((v for v in x if v), ""), x.name))
    if frozen_lookup:
        for _, members in working.groupby("root"):
            previous = {frozen_lookup[x] for x in members["food_observation_id"] if x in frozen_lookup}
            if len(previous) > 1:
                raise ValueError("A new identity link would merge distinct frozen food concepts")
            if previous:
                working.loc[members.index, "food_concept_id"] = next(iter(previous))
        roots_per_id = working.groupby("food_concept_id")["root"].nunique()
        if roots_per_id.gt(1).any():
            raise ValueError("A frozen identity now has conflicting facets; adjudicate before rebuilding")
    # The transform above intentionally keys on lineage when present, but x.name
    # is the union root and keeps unlinked exact-name concepts stable.
    concept_rows = []
    for concept_id, group in working.groupby("food_concept_id", sort=True):
        names = group["original_name"].dropna().astype(str)
        canonical_name = sorted(names, key=lambda x: (-len(normalize_text(x)), x.casefold()))[0] if len(names) else concept_id
        foodon_ids = sorted({normalize_ontology_id(x, "FOODON") for x in group["foodon_id"] if normalize_ontology_id(x, "FOODON")})
        exclusions = sorted({_clean_identifier(x) for x in group["exclusion_flag"] if _clean_identifier(x)})
        row = {
            "food_concept_id": concept_id, "canonical_name": canonical_name,
            "scientific_name": _first_nonempty(group["scientific_name"]),
            "food_group": _first_nonempty(group["food_group"]), "food_subgroup": _first_nonempty(group["food_subgroup"]),
            "food_type": _first_nonempty(group["food_type"]), "foodon_id": foodon_ids[0] if len(foodon_ids) == 1 else "",
            "part": _first_nonempty(group["part"]), "processing": _first_nonempty(group["processing"]),
            "source_count": group["source_key"].nunique(), "source_keys": ";".join(sorted(group["source_key"].unique())),
            "source_lineage_ids": ";".join(sorted({_clean_identifier(x) for x in group["normalized_lineage"] if _clean_identifier(x)})),
            "normalized_source_names": ";".join(sorted({_clean_identifier(x) for x in group["normalized_name"] if _clean_identifier(x)})),
            "observation_count": len(group), "ml_exclusion_reason": ";".join(exclusions),
            "identity_status": "authority_linked" if len(foodon_ids) == 1 else "exact_name_or_lineage_concept",
        }
        row["family_label"] = _food_family_label(pd.Series(row))
        for facet in FOOD_IDENTITY_FACETS:
            row[facet] = _first_nonempty(group[facet])
        row["family_cluster_id"] = stable_id("family", row["family_label"])
        concept_rows.append(row)
    concepts = pd.DataFrame(concept_rows)
    mapping = working[["food_observation_id", "food_concept_id", "normalized_name", "normalized_lineage", "exclusion_flag"]].copy()
    candidates = _food_fuzzy_candidates(concepts)
    if frozen_concepts is None:
        concepts = _assign_split_family_blocks(concepts, mapping, candidates)
    else:
        from .frozen_release import attach_new_family_blocks
        concepts = attach_new_family_blocks(concepts, mapping, candidates, frozen_concepts)
    hierarchy = _food_hierarchy(concepts, root)
    return concepts, mapping, pd.concat([pd.DataFrame(merge_ledger), candidates], ignore_index=True, sort=False), hierarchy


def _assign_split_family_blocks(concepts: pd.DataFrame, mapping: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Create conservative blocks without treating near names as the same food."""
    result = concepts.copy()
    ids = result["food_concept_id"].astype(str).tolist()
    uf = UnionFind(ids)
    for _, group in result.groupby("family_label"):
        members = group["food_concept_id"].astype(str).tolist()
        for member in members[1:]:
            uf.union(members[0], member)
    linked = mapping[mapping["normalized_lineage"].fillna("").ne("")]
    for _, group in linked.groupby("normalized_lineage"):
        members = sorted(set(group["food_concept_id"].astype(str)))
        for member in members[1:]:
            uf.union(members[0], member)
    if not candidates.empty:
        near = candidates[candidates["similarity"].fillna(0).ge(0.9)]
        for row in near.itertuples(index=False):
            uf.union(str(row.left_concept_id), str(row.right_concept_id))
    roots = result["food_concept_id"].astype(str).map(uf.find)
    result["family_cluster_id"] = roots.map(lambda value: stable_id("split-family", value))
    result["family_block_policy"] = "lexical_family_plus_shared_lineage_plus_high_similarity_split_block"
    return result


def _food_merge_row(left: str, right: str, reason: str, evidence: str, merged: bool) -> dict[str, Any]:
    return {
        "review_type": "food_identity_decision", "left_observation_id": left,
        "right_observation_id": right, "reason": reason, "evidence": evidence,
        "automatic_merge": merged, "review_status": "machine_rule_applied" if merged else "expert_review_required",
    }


def _food_fuzzy_candidates(concepts: pd.DataFrame) -> pd.DataFrame:
    names = sorted((normalize_text(row.canonical_name), row.food_concept_id, row.canonical_name, row.family_cluster_id) for row in concepts.itertuples(index=False) if normalize_text(row.canonical_name))
    rows = []
    for index, (name, concept_id, display, family) in enumerate(names):
        for other_name, other_id, other_display, other_family in names[max(0, index - 4):index]:
            if concept_id == other_id or name == other_name:
                continue
            score = SequenceMatcher(None, name, other_name).ratio()
            if score >= 0.9:
                rows.append({
                    "review_type": "food_similar_name_candidate", "left_concept_id": concept_id,
                    "right_concept_id": other_id, "left_name": display, "right_name": other_display,
                    "similarity": score, "same_family_cluster": family == other_family,
                    "automatic_merge": False, "review_status": "expert_review_required",
                    "reason": "Fuzzy similarity proposes review only; facet compatibility has not been established.",
                })
    return pd.DataFrame(rows)


def _food_hierarchy(concepts: pd.DataFrame, root: Path) -> dict[str, Any]:
    foodon_path = root / "data/raw/ontology/foodon.owl"
    foodon = parse_foodon(foodon_path) if foodon_path.exists() else {}
    edges = []
    concept_by_foodon = {row.foodon_id: row.food_concept_id for row in concepts.itertuples(index=False) if row.foodon_id}
    for foodon_id, concept_id in concept_by_foodon.items():
        for parent in foodon.get(foodon_id, {}).get("parents", []):
            edges.append({
                "child_food_concept_id": concept_id,
                "parent_food_concept_id": concept_by_foodon.get(parent, ""),
                "parent_authority_id": parent, "relation": "is_a",
                "authority": "FoodOn", "numeric_inheritance_allowed": False,
            })
    return {
        "dataset_version": DATASET_VERSION,
        "policy": "Parent-child relations organize identity and split blocking only; composition values are never inherited.",
        "concepts": concepts[["food_concept_id", "canonical_name", "foodon_id", "family_cluster_id"]].to_dict("records"),
        "edges": edges,
    }


def assign_partitions(concepts: pd.DataFrame, mapping: pd.DataFrame, foods: pd.DataFrame, eligible_food_ids: set[str]) -> pd.DataFrame:
    result = concepts[concepts["food_concept_id"].isin(eligible_food_ids)].copy()
    source_map = mapping.merge(foods[["food_observation_id", "source_key"]], on="food_observation_id", how="left")
    foundation_rows = source_map[source_map["source_key"].eq("usda_foundation")]
    foundation = set(foundation_rows["food_concept_id"])
    foundation_lineages = set(foundation_rows["normalized_lineage"].dropna().astype(str)) - {""}
    lineage_linked = set(source_map.loc[source_map["normalized_lineage"].isin(foundation_lineages), "food_concept_id"])
    result["source_holdout"] = result["food_concept_id"].isin(foundation | lineage_linked)
    result["source_holdout_reason"] = np.where(
        result["food_concept_id"].isin(foundation),
        "USDA Foundation Food observation",
        np.where(result["food_concept_id"].isin(lineage_linked), "shared formal lineage with USDA Foundation Food", ""),
    )
    source_holdout_families = set(result.loc[result["source_holdout"], "family_cluster_id"])
    result["source_holdout_family_overlap"] = (
        result["family_cluster_id"].isin(source_holdout_families) & ~result["source_holdout"]
    )
    # Keep the source and family benchmarks distinct. Foods related to a
    # Foundation Food may train the source-transfer task, but those families
    # cannot also be sampled into the unseen-family panel.
    candidate = result[
        ~result["source_holdout"]
        & ~result["source_holdout_family_overlap"]
        & result["ml_exclusion_reason"].fillna("").eq("")
    ]
    family_sizes = candidate.groupby("family_cluster_id")["food_concept_id"].nunique().to_dict()
    ordered = sorted(family_sizes, key=lambda key: hashlib.sha256(f"{RANDOM_SEED}:{key}".encode()).hexdigest())
    target = min(FAMILY_HOLDOUT_CONCEPT_TARGET, candidate["food_concept_id"].nunique())
    selected: set[str] = set()
    count = 0
    for family in ordered:
        family_size = family_sizes[family]
        if count + family_size > target:
            continue
        selected.add(family)
        count += family_size
        if count == target:
            break
    if count != target:
        raise ValueError(
            f"Could not construct the locked {target}-concept family holdout "
            f"without splitting a family block; selected {count}."
        )
    result["family_holdout"] = result["family_cluster_id"].isin(selected)
    result["partition"] = np.where(
        result["source_holdout"] | result["family_holdout"],
        "validation",
        "train",
    )
    result["validation_panel"] = np.select(
        [
            result["source_holdout"] & result["family_holdout"],
            result["source_holdout"],
            result["family_holdout"],
        ],
        ["source_holdout+family_holdout", "source_holdout", "family_holdout"],
        default="",
    )
    result["benchmark_eligible"] = True
    result["cv_fold"] = result["family_cluster_id"].map(lambda x: int(hashlib.sha256(f"fold:{RANDOM_SEED}:{x}".encode()).hexdigest()[:8], 16) % 5)
    result.loc[result["partition"].eq("validation"), "cv_fold"] = pd.NA
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * weights.sum(), side="left")])


def _aggregate_group(group: pd.DataFrame) -> dict[str, Any]:
    if len(group) == 1:
        row = group.iloc[0]
        value = float(row["normalized_value_g_per_100g"])
        n = pd.to_numeric(row["sample_count"], errors="coerce")
        known_n = pd.notna(n) and n > 0
        result = {
            "canonical_value_g_per_100g": value, "aggregation_status": "accepted",
            "aggregation_method": "single_source_record", "best_quality_tier": row["quality_tier"],
            "selected_measurement_count": 1, "contributing_source_count": 1,
            "independent_source_count": int(row["independent_evidence"] is True or row["independent_evidence"] == True),
            "total_sample_count": float(n) if known_n else np.nan,
            "records_with_missing_sample_count": int(not known_n),
            "log_value_span": 0.0, "heterogeneity_i2": np.nan,
            "max_min_ratio": 1.0 if value > 0 else np.nan, "zero_positive_conflict": False,
            "measurement_ids": str(row["measurement_id"]), "source_keys": row["source_key"],
        }
        result.update({f"raw_{stat}_g_per_100g": value for stat in ("mean", "median", "q1", "q3", "min", "max")})
        return result
    ranks = group["quality_tier"].map(QUALITY_RANK).fillna(99)
    best_rank = ranks.min()
    selected = group[ranks.eq(best_rank)].copy()
    values = selected["normalized_value_g_per_100g"].astype(float).to_numpy()
    logs = np.log1p(values)
    reported_counts = pd.to_numeric(selected["sample_count"], errors="coerce")
    counts = reported_counts.fillna(1).clip(lower=1).to_numpy(float)
    raw_se = pd.to_numeric(selected["standard_error"], errors="coerce").to_numpy(float)
    factors = pd.to_numeric(selected["conversion_factor"], errors="coerce").to_numpy(float)
    se_log = raw_se * factors / (1.0 + values)
    valid_se = np.isfinite(se_log) & (se_log > 0)
    method = "sample_size_weighted_median_log_scale" if reported_counts.notna().all() else "weighted_median_log_scale_missing_n_equal_record_weight"
    i2 = np.nan
    if valid_se.sum() >= 2:
        y = logs[valid_se]
        variances = se_log[valid_se] ** 2
        fixed_w = 1.0 / variances
        fixed_mean = np.sum(fixed_w * y) / np.sum(fixed_w)
        q = float(np.sum(fixed_w * (y - fixed_mean) ** 2))
        df = len(y) - 1
        c = float(np.sum(fixed_w) - np.sum(fixed_w**2) / np.sum(fixed_w))
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        random_w = 1.0 / (variances + tau2)
        center_log = float(np.sum(random_w * y) / np.sum(random_w))
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        method = "dersimonian_laird_random_effects_log_scale"
    else:
        center_log = _weighted_median(logs, counts)
    span = float(logs.max() - logs.min()) if len(logs) > 1 else 0.0
    independent_sources = selected["source_key"].nunique()
    relative_ratio = float(values.max() / values.min()) if values.min() > 0 else np.nan
    zero_positive_conflict = independent_sources >= 2 and np.any(values == 0) and np.any(values > 0)
    # This is a review trigger, not a claim that a threefold biological
    # difference is impossible. Unlike log1p span it also detects trace axes.
    high_heterogeneity = (np.isfinite(i2) and i2 > 0.75) or (
        independent_sources >= 2 and np.isfinite(relative_ratio) and relative_ratio > 3.0
    ) or zero_positive_conflict
    canonical_value = float(np.expm1(center_log)) if not high_heterogeneity else np.nan
    if np.isfinite(canonical_value) and 100.0 < canonical_value <= 100.0 + 1e-6:
        canonical_value = 100.0
    return {
        "canonical_value_g_per_100g": canonical_value,
        "aggregation_status": "expert_review_required_high_heterogeneity" if high_heterogeneity else "accepted",
        "aggregation_method": method, "best_quality_tier": min(selected["quality_tier"], key=lambda x: QUALITY_RANK.get(x, 99)),
        "selected_measurement_count": len(selected), "contributing_source_count": selected["source_key"].nunique(),
        "independent_source_count": selected.loc[selected["independent_evidence"].eq(True), "source_key"].nunique(),
        "total_sample_count": float(reported_counts.sum()) if reported_counts.notna().all() else np.nan,
        "records_with_missing_sample_count": int(reported_counts.isna().sum()), "raw_mean_g_per_100g": float(np.mean(values)),
        "raw_median_g_per_100g": float(np.median(values)),
        "raw_q1_g_per_100g": float(np.quantile(values, 0.25)), "raw_q3_g_per_100g": float(np.quantile(values, 0.75)),
        "raw_min_g_per_100g": float(np.min(values)), "raw_max_g_per_100g": float(np.max(values)),
        "log_value_span": span, "heterogeneity_i2": i2,
        "max_min_ratio": relative_ratio, "zero_positive_conflict": zero_positive_conflict,
        "measurement_ids": ";".join(selected["measurement_id"].astype(str)),
        "source_keys": ";".join(sorted(selected["source_key"].unique())),
    }


def aggregate_profiles(
    measurements: pd.DataFrame,
    food_mapping: pd.DataFrame,
    component_mapping: pd.DataFrame,
    partitioned_foods: pd.DataFrame,
    frozen_validation_observations: set[str] | None = None,
    *,
    source_priority: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = measurements[measurements["main_value_eligible"].fillna(False)].copy()
    eligible = eligible.merge(food_mapping[["food_observation_id", "food_concept_id", "exclusion_flag"]], on="food_observation_id", how="inner")
    eligible = eligible.merge(component_mapping[["component_observation_id", "component_concept_id"]], on="component_observation_id", how="inner")
    eligible = eligible.merge(partitioned_foods[["food_concept_id", "partition", "validation_panel", "source_holdout", "cv_fold", "family_cluster_id"]], on="food_concept_id", how="inner")
    eligible = eligible[eligible["exclusion_flag"].fillna("").eq("") & eligible["normalized_value_g_per_100g"].notna()].copy()
    if "strict_validation_eligible" in eligible or "validation_reference_eligible" in eligible:
        validation_ok = validation_reference_mask(eligible)
        eligible = eligible[eligible["partition"].eq("train") | validation_ok].copy()
    if frozen_validation_observations is not None:
        eligible = eligible[eligible["partition"].eq("train") | eligible["food_observation_id"].isin(frozen_validation_observations)].copy()
    selection_audit = None
    if source_priority is not None:
        from .cell_selection import select_cell_records
        # Holdout eligibility must precede ranking, so an ineligible preferred
        # record cannot displace the admissible reference label.
        eligible = eligible[~eligible.source_holdout.astype(bool) | eligible.source_key.eq("usda_foundation")]
        eligible, selection_audit = select_cell_records(eligible, source_priority)
    rows = []
    conflicts = []
    for (food_id, component_id), group in eligible.groupby(["food_concept_id", "component_concept_id"], sort=True):
        if bool(group["source_holdout"].iloc[0]):
            foundation_only = group[group["source_key"].eq("usda_foundation")]
            if foundation_only.empty:
                continue
            group = foundation_only
        aggregated = _aggregate_group(group)
        base = group.iloc[0]
        row = {
            "food_concept_id": food_id, "component_concept_id": component_id,
            "partition": base["partition"], "validation_panel": base["validation_panel"],
            "cv_fold": base["cv_fold"], "family_cluster_id": base["family_cluster_id"], **aggregated,
        }
        rows.append(row)
        if aggregated["aggregation_status"] != "accepted":
            conflicts.append({**row, "review_reason": "Independent best-tier measurements exceed the prespecified heterogeneity review trigger; no canonical label is released until expert adjudication."})
    profiles = pd.DataFrame(rows)
    if selection_audit is not None and not profiles.empty:
        profiles = profiles.merge(selection_audit, on=["food_concept_id", "component_concept_id"], validate="one_to_one")
        profiles["aggregation_method"] = "source_priority_single_record"
    return profiles, pd.DataFrame(conflicts)


def classify_training_roles(
    profiles: pd.DataFrame,
    components: pd.DataFrame,
    partitioned_foods: pd.DataFrame,
    component_mapping: pd.DataFrame,
    component_observations: pd.DataFrame,
    measurements: pd.DataFrame,
    food_mapping: pd.DataFrame,
    food_observations: pd.DataFrame,
    *,
    minimum_train: int = 100,
    minimum_validation: int = 30,
    require_verified_identity: bool = True,
    exclude_expression_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minimum_train < 1 or minimum_validation < 1:
        raise ValueError("Target support thresholds must be positive.")
    accepted = profiles[profiles["aggregation_status"].eq("accepted") & profiles["canonical_value_g_per_100g"].notna()].copy()
    train_support = accepted[accepted["partition"].eq("train")].groupby("component_concept_id")["food_concept_id"].nunique()
    validation_support = accepted[accepted["partition"].eq("validation")].groupby(
        "component_concept_id"
    )["food_concept_id"].nunique()

    eligible_component_observations = set(measurements.loc[measurements["main_value_eligible"].fillna(False), "component_observation_id"])
    comp_source = component_mapping[component_mapping["component_observation_id"].isin(eligible_component_observations)].merge(component_observations[["component_observation_id", "source_key"]], on="component_observation_id", how="left")
    food_source = food_mapping.merge(food_observations[["food_observation_id", "source_key"]], on="food_observation_id", how="left")
    food_source = food_source.merge(partitioned_foods[["food_concept_id", "partition"]], on="food_concept_id", how="inner")
    capability_counts = {}
    for component_id, group in comp_source.groupby("component_concept_id"):
        sources = set(group["source_key"].dropna())
        capability_counts[component_id] = food_source[food_source["source_key"].isin(sources)]["food_concept_id"].nunique()
    observed_counts = accepted.groupby("component_concept_id")["food_concept_id"].nunique().to_dict()

    registry = components.set_index("component_concept_id").copy()
    registry["train_count"] = train_support.reindex(registry.index).fillna(0).astype(int)
    registry["validation_count"] = validation_support.reindex(registry.index).fillna(0).astype(int)
    registry["observed_concept_count"] = pd.Series(observed_counts).reindex(registry.index).fillna(0).astype(int)
    registry["capable_concept_count"] = pd.Series(capability_counts).reindex(registry.index).fillna(0).astype(int)
    registry["capability_adjusted_coverage"] = registry["observed_concept_count"] / registry["capable_concept_count"].replace(0, np.nan)
    registry["measurement_modality"] = np.where(
        registry["observed_concept_count"].gt(0),
        "mass_fraction_fresh_weight",
        "not_represented_in_main_mass_modality",
    )
    train_values = accepted[accepted["partition"].eq("train")].groupby("component_concept_id")["canonical_value_g_per_100g"]
    stats = train_values.agg(train_raw_min="min", train_raw_max="max", train_raw_mean="mean", train_raw_sd="std", train_raw_median="median", train_raw_q1=lambda x: x.quantile(0.25), train_raw_q3=lambda x: x.quantile(0.75))
    log_stats = train_values.agg(
        train_log_median=lambda x: float(np.median(np.log1p(x))),
        train_log_iqr=lambda x: float(np.quantile(np.log1p(x), 0.75) - np.quantile(np.log1p(x), 0.25)),
        train_log_mad=lambda x: float(np.median(np.abs(np.log1p(x) - np.median(np.log1p(x))))),
    )
    stats = stats.join(log_stats)
    stats["train_robust_scale"] = np.maximum(stats["train_log_iqr"] / 1.349, stats["train_log_mad"] * 1.4826)
    registry = registry.join(stats, how="left")
    identity_ok = registry["identity_status"].isin(["authority_verified", "stable_source_identity_pending_cross_database_review"])
    if not require_verified_identity:
        identity_ok = pd.Series(True, index=registry.index)
    above_two = registry["capability_adjusted_coverage"].ge(0.02)
    enough = registry["train_count"].ge(minimum_train) & registry["validation_count"].ge(minimum_validation)
    nondegenerate = registry["train_robust_scale"].fillna(0).gt(1e-8)
    non_target_expression = registry["expression_variant"].isin(["label_expression", "biological_equivalent"])
    if not exclude_expression_labels:
        non_target_expression = pd.Series(False, index=registry.index)
    context_expression = registry["expression_variant"].isin(["calculated_from_constituents", "calculated_by_difference"])
    registry["training_role"] = np.select(
        [~non_target_expression & ~context_expression & above_two & enough & identity_ok & nondegenerate, ~non_target_expression & above_two & identity_ok],
        ["maskable_target", "context_only"], default="excluded",
    )
    registry.loc[context_expression & above_two & identity_ok, "training_role"] = "context_only"
    registry["training_exclusion_reason"] = np.select(
        [non_target_expression, ~identity_ok, registry["capability_adjusted_coverage"].lt(0.02), above_two & ~enough, above_two & enough & ~nondegenerate],
        ["non_mass_or_label_expression", "unresolved_component_identity", "coverage_below_2_percent", "insufficient_train_or_validation_power", "degenerate_train_distribution"], default="",
    )
    unsupported_context = registry["training_role"].eq("context_only") & registry["train_count"].eq(0)
    registry.loc[unsupported_context, "training_role"] = "excluded"
    registry.loc[unsupported_context, "training_exclusion_reason"] = "no_training_observations_archival_only"
    if not require_verified_identity or not exclude_expression_labels:
        registry.loc[registry.observed_concept_count.eq(0), "training_exclusion_reason"] = "no_accepted_mass_observations"
        registry.loc[context_expression & registry.training_role.eq("context_only"), "training_exclusion_reason"] = "calculated_expression_context_only"
    sensitivity = []
    for threshold in (0.01, 0.02, 0.05):
        passed = registry["capability_adjusted_coverage"].ge(threshold)
        sensitivity.append({
            "coverage_threshold": threshold, "component_count": int(passed.sum()),
            "authority_verified_count": int((passed & registry["identity_status"].eq("authority_verified")).sum()),
            "maskable_support_count": int((passed & enough & identity_ok).sum()),
        })
    return registry.reset_index(), pd.DataFrame(sensitivity)


def build_profile_matrix(profiles: pd.DataFrame, foods: pd.DataFrame, components: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    food_ids = foods["food_concept_id"].astype(str).tolist()
    component_ids = components.loc[components["training_role"].ne("excluded"), "component_concept_id"].astype(str).tolist()
    food_index = {key: index for index, key in enumerate(food_ids)}
    component_index = {key: index for index, key in enumerate(component_ids)}
    values = np.full((len(food_ids), len(component_ids)), np.nan, dtype=np.float32)
    accepted = profiles[profiles["aggregation_status"].eq("accepted")]
    for row in accepted.itertuples(index=False):
        i, j = food_index.get(row.food_concept_id), component_index.get(row.component_concept_id)
        if i is not None and j is not None:
            values[i, j] = row.canonical_value_g_per_100g
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, values=values, observed=np.isfinite(values), food_ids=np.asarray(food_ids), component_ids=np.asarray(component_ids))
    return {"shape": list(values.shape), "observed": int(np.isfinite(values).sum()), "missing": int(np.isnan(values).sum()), "path": output_path.name}


def _expert_review_sample(evidence: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    high_risk = evidence["review_status"].eq("expert_review_required")
    random_tenth = evidence["evidence_id"].map(lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 10 == 0)
    result = evidence[high_risk | random_tenth].copy()
    result["review_type"] = f"{entity_type}_expert_review_sample"
    result["selection_reason"] = np.where(high_risk[high_risk | random_tenth], "all_high_risk", "deterministic_10_percent_high_confidence_audit")
    return result


def build_food_evidence_ledger(foods: pd.DataFrame, mapping: pd.DataFrame, concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    concept_fields = concepts[["food_concept_id", "canonical_name", "foodon_id", "identity_status"]].rename(
        columns={"foodon_id": "canonical_foodon_id"}
    )
    linked = foods.merge(mapping[["food_observation_id", "food_concept_id"]], on="food_observation_id", how="left").merge(
        concept_fields, on="food_concept_id", how="left"
    )
    rows = []
    for row in linked.itertuples(index=False):
        authority = "FoodOn" if _clean_identifier(row.canonical_foodon_id) else row.source_key
        authority_id = _clean_identifier(row.canonical_foodon_id) or str(row.source_food_id)
        high = authority == "FoodOn"
        rows.append({
            "evidence_id": stable_id("evidence", row.food_observation_id), "entity_type": "food",
            "observation_id": row.food_observation_id, "source_key": row.source_key,
            "original_name": row.original_name, "canonical_id": row.food_concept_id,
            "canonical_name": row.canonical_name,
            "query": " ".join(str(x) for x in (row.original_name, row.scientific_name, row.food_group) if pd.notna(x) and str(x).strip()),
            "authority_namespace": authority, "authority_id": authority_id,
            "evidence_url": "https://foodon.org/" if high else _clean_identifier(row.source_record_url),
            "decision": "retained_source_name_and_facets",
            "confidence": "high" if high else "medium",
            "review_status": "machine_verified_identifier" if high else "expert_review_required",
            "review_reason": "Direct FoodOn identifier supplied by source." if high else "Stable official source identity retained; scientific display-name rewrite requires human verification.",
        })
    ledger = pd.DataFrame(rows)
    return ledger, _expert_review_sample(ledger, "food")


def run_harmonization(root: Path, staging_dir: Path, output_dir: Path, audit_dir: Path,
                      *, frozen_release: Path | None = None, dataset_version: str = DATASET_VERSION,
                      as_of_date: str = AS_OF_DATE, source_admission_policy: str | None = None) -> None:
    infoods_terms = parse_infoods_tagnames(root / "data/raw/reference/infoods_tagnames_2022")
    foods, component_observations, measurements, disposition, measurement_manifest = read_staging(
        staging_dir, source_admission_policy=source_admission_policy,
        policy_ledger_dir=audit_dir / "source_policy" if source_admission_policy else None,
        component_definitions=infoods_terms,
    )
    print(f"Loaded staging: foods={len(foods):,}, components={len(component_observations):,}, main-eligible measurements={len(measurements):,}", flush=True)
    component_concepts, component_mapping, component_evidence, component_review = build_component_concepts(component_observations, root, infoods_terms)
    frozen_mapping = None if frozen_release is None else pd.read_csv(frozen_release / "food_observation_to_concept.csv.gz")
    frozen_concepts = None if frozen_release is None else pd.read_csv(frozen_release / "food_concept.csv.gz")
    food_concepts, food_mapping, food_review, hierarchy = build_food_concepts(foods, root, frozen_mapping, frozen_concepts)
    hierarchy["dataset_version"] = dataset_version
    food_evidence, food_evidence_review = build_food_evidence_ledger(foods, food_mapping, food_concepts)

    initial_eligible = measurements[measurements["main_value_eligible"].fillna(False)]
    eligible_obs_ids = set(initial_eligible["food_observation_id"].astype(str))
    eligible_concepts = set(food_mapping.loc[food_mapping["food_observation_id"].astype(str).isin(eligible_obs_ids) & food_mapping["exclusion_flag"].fillna("").eq(""), "food_concept_id"])
    frozen_val_obs = None
    if frozen_release is None:
        partitioned = assign_partitions(food_concepts, food_mapping, foods, eligible_concepts)
    else:
        from .frozen_release import assign_frozen_partitions
        previous = pd.read_csv(frozen_release / "ml_partition.csv")
        partitioned, split_exclusions = assign_frozen_partitions(food_concepts, food_mapping, foods, eligible_concepts, previous)
        frozen_ids = set(previous.loc[previous["partition"].eq("validation"), "food_concept_id"])
        frozen_val_obs = set(frozen_mapping.loc[frozen_mapping["food_concept_id"].isin(frozen_ids), "food_observation_id"])
        write_csv(split_exclusions, audit_dir / "frozen_split_exclusion_ledger.csv")
    profiles, conflicts = aggregate_profiles(measurements, food_mapping, component_mapping, partitioned, frozen_val_obs)
    component_registry, sensitivity = classify_training_roles(
        profiles, component_concepts, partitioned, component_mapping,
        component_observations, measurements, food_mapping, foods,
    )

    target_ids = set(component_registry.loc[component_registry["training_role"].eq("maskable_target"), "component_concept_id"])
    target_profiles = profiles[profiles["component_concept_id"].isin(target_ids) & profiles["aggregation_status"].eq("accepted")]
    family_counts = target_profiles.merge(component_registry[["component_concept_id", "component_family"]], on="component_concept_id", how="left").groupby("food_concept_id")["component_family"].nunique()
    partitioned["mask_family_count"] = partitioned["food_concept_id"].map(family_counts).fillna(0).astype(int)
    partitioned["text_task_eligible"] = partitioned["food_concept_id"].isin(set(target_profiles["food_concept_id"]))
    partitioned["reconstruction_task_eligible"] = partitioned["mask_family_count"].ge(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(foods, output_dir / "food_observation.csv.gz")
    write_csv(food_concepts, output_dir / "food_concept.csv.gz")
    write_csv(food_mapping, output_dir / "food_observation_to_concept.csv.gz")
    write_csv(component_observations, output_dir / "component_observation.csv.gz")
    write_csv(component_registry, output_dir / "component_concept.csv.gz")
    write_csv(component_mapping, output_dir / "component_observation_to_concept.csv.gz")
    write_csv(measurements, output_dir / "measurement_main_eligible.csv.gz")
    write_csv(measurement_manifest, output_dir / "measurement_partition_manifest.csv")
    write_csv(profiles, output_dir / "canonical_profile.csv.gz")
    write_csv(partitioned, output_dir / "ml_partition.csv")
    write_json(hierarchy, output_dir / "food_hierarchy.json")
    matrix_summary = build_profile_matrix(profiles, partitioned, component_registry, output_dir / "canonical_profile_matrix.npz")

    write_csv(component_evidence, audit_dir / "component_evidence_ledger.csv.gz")
    write_csv(component_review, audit_dir / "component_review_queue.csv")
    write_csv(pd.DataFrame(infoods_terms.values()), audit_dir / "infoods_tag_authority_registry.csv")
    write_csv(food_evidence, audit_dir / "food_evidence_ledger.csv.gz")
    write_csv(food_evidence_review, audit_dir / "food_evidence_review_queue.csv.gz")
    write_csv(food_review, audit_dir / "food_identity_and_review_ledger.csv.gz")
    food_exclusions = food_mapping[food_mapping["exclusion_flag"].fillna("").ne("")].merge(
        foods[["food_observation_id", "source_key", "source_food_id", "original_name"]],
        on="food_observation_id", how="left",
    )
    write_csv(food_exclusions, audit_dir / "food_exclusion_ledger.csv")
    write_csv(conflicts, audit_dir / "unresolved_measurement_conflicts.csv")
    write_csv(sensitivity, audit_dir / "coverage_threshold_sensitivity.csv")
    write_csv(
        component_registry[
            ["component_concept_id", "canonical_name", "authority_namespace", "authority_id", "definition",
             "chemical_class", "nutritional_role", "measurement_modality", "training_role",
             "training_exclusion_reason", "train_count", "validation_count", "capability_adjusted_coverage"]
        ],
        audit_dir / "component_training_decision_ledger.csv.gz",
    )
    write_csv(disposition, audit_dir / "source_measurement_disposition.csv")
    write_csv(partitioned.groupby(["partition", "validation_panel"], dropna=False).size().reset_index(name="food_concept_count"), audit_dir / "partition_summary.csv")

    summary = {
        "dataset_version": dataset_version, "built_as_of": as_of_date,
        "food_observations": len(foods), "food_concepts": len(food_concepts),
        "ml_food_concepts": len(partitioned), "component_observations": len(component_observations),
        "component_concepts": len(component_registry), "staged_measurements": int(disposition["measurement_count"].sum()),
        "main_eligible_measurements": len(measurements),
        "accepted_profiles": int(profiles["aggregation_status"].eq("accepted").sum()),
        "unresolved_conflicts": len(conflicts),
        "maskable_targets": int(component_registry["training_role"].eq("maskable_target").sum()),
        "context_only_components": int(component_registry["training_role"].eq("context_only").sum()),
        "excluded_components": int(component_registry["training_role"].eq("excluded").sum()),
        "train_food_concepts": int(partitioned["partition"].eq("train").sum()),
        "validation_food_concepts": int(partitioned["partition"].eq("validation").sum()),
        "family_holdout_food_concepts": int(partitioned["validation_panel"].eq("family_holdout").sum()),
        "source_holdout_food_concepts": int(partitioned["validation_panel"].eq("source_holdout").sum()),
        "source_family_overlap_train_food_concepts": int(
            (partitioned["partition"].eq("train") & partitioned["source_holdout_family_overlap"]).sum()
        ),
        "matrix": matrix_summary,
        "release_gate": "candidate_only_pending_domain_expert_review_and_dual_reviewer_scoping_screening",
    }
    if source_admission_policy:
        summary["source_admission_policy"] = source_admission_policy
        summary["validation_label_interpretation"] = "curated_reference_prediction_not_universally_analytical_ground_truth"
    write_json(summary, output_dir / "dataset_summary.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--staging-dir", type=Path, default=Path("data/processed/scientific_food_composition_v1/staging"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/scientific_food_composition_v1/release"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audits/scientific_food_composition_v1/dataset"))
    args = parser.parse_args()
    root = args.root.resolve()
    run_harmonization(root, root / args.staging_dir, root / args.output_dir, root / args.audit_dir)


if __name__ == "__main__":
    main()

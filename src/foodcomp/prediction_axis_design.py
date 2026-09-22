"""Build an auditable VMH-aligned prediction-axis design.

This module is intentionally separate from the global axis atlas.  It uses the
current Virtual Metabolic Human (VMH) nutrient catalogue as a nutritional
reference panel, then produces *candidate* lexical links to source-native
axes.  It does not combine food records, component values, units, analytical
definitions, or measurement expressions.  A fuzzy match is evidence to
review, never permission to pool values or assert chemical equivalence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .util import stable_id, write_csv, write_json


VMH_NUTRIENT_URL = "https://www.vmh.life/_api/nutrients/?page_size=1000"
VMH_REFERENCE_URL = "https://www.vmh.life/_api/docs/"
VMH_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6323901/"
INFOODS_URL = "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"
NIH_DRI_URL = "https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/"
FAO_URL = "https://www.fao.org/food-composition/overview-2/en"


DERIVED_OR_LABEL_PATTERNS = (
    "by difference",
    "energy",
    "equivalent",
    " iu",
    "added",
    "adjusted protein",
    "calculated from sodium",
    "whole amount of salt",
    "mineral content",
    "glycerol + lipids",
    "purin-n",
    "poly-uric acid",
)

# These aliases are deliberately narrow.  They describe terminology, rather
# than asserting that two analytically different expressions are interchangeable.
EXACT_NAME_ALIASES = {
    "saccharose": "sucrose",
    "vitamin a retinol": "retinol",
    "vitamin b1 thiamin": "thiamin",
    "vitamin b2 riboflavin": "riboflavin",
    "vitamin b5 pantothenic acid": "pantothenic acid",
    "vitamin b12 cobalamin": "cobalamin",
    "vitamin c ascorbic acid": "ascorbic acid",
    "vitamin d cholecalciferol": "cholecalciferol",
    "vitamin d3 cholecalciferol": "cholecalciferol",
    "vitamin k phyllochinon": "phylloquinone",
    "vitamin k phylloquinone": "phylloquinone",
    "vitamin c total ascorbic acid": "ascorbic acid",
    "vitamin e alpha tocopherol": "alpha tocopherol",
    "vitamin e tocopherol": "alpha tocopherol",
    "whole folate content": "folate total",
    "proteins": "protein total",
    "protein": "protein total",
    "lipids": "fat total",
    "total lipid": "fat total",
    "total lipid fat": "fat total",
    "fat total": "fat total",
    "carbohydrates": "carbohydrate total",
    "total carbohydrate": "carbohydrate total",
    "dietary fibers": "dietary fibre total",
    "fiber total dietary": "dietary fibre total",
    "alcohol": "ethanol",
    "alcohol ethyl": "ethanol",
    "glucose dextrose": "glucose",
    "potassium k": "potassium",
    "phosphorus p": "phosphorus",
    "magnesium mg": "magnesium",
    "manganese mn": "manganese",
    "zinc zn": "zinc",
    "copper cu": "copper",
    "iron fe": "iron",
    "calcium ca": "calcium",
    "sodium na": "sodium",
    "butyric acid butanoic acid": "4 0",
    "caproic acid hexanoic acid": "6 0",
    "caprylic acid octanoic acid": "8 0",
    "decanoic acid capric acid": "10 0",
    "lauric acid": "12 0",
    "myristic acid tetradecanoic acid": "14 0",
    "pentadecanoic acid": "15 0",
    "margaric acid": "17 0",
    "palmitic acid": "16 0",
    "stearic acid octadecanoic acid": "18 0",
    "arachidic acid eicosanoic acid": "20 0",
    "behenic acid docosanoic acid": "22 0",
    "lignoceric acid tetracosanoic acid": "24 0",
    "eicosapentaenoic acid": "20 5 n 3 epa",
    "docosapentaenoic acid dpa": "22 5 n 3 dpa",
    "docosahexaenoic acid dha": "22 6 n 3 dha",
    "alpha linolenic acid": "18 3 n 3 ala",
    "alpha linolenic acid all cis 9 12 15 octadecatrienoic acid": "18 3 n 3 ala",
    "18 3 n 3 c c c ala": "18 3 n 3 ala",
}

DISPLAY_ALIASES = {
    "protein total": "Protein, total",
    "fat total": "Fat, total",
    "carbohydrate total": "Carbohydrate, total",
    "dietary fibre total": "Dietary fibre, total",
    "ethanol": "Ethanol",
    "retinol": "Retinol",
    "thiamin": "Thiamin (vitamin B1)",
    "riboflavin": "Riboflavin (vitamin B2)",
    "pantothenic acid": "Pantothenic acid (vitamin B5)",
    "cobalamin": "Cobalamin (vitamin B12)",
    "ascorbic acid": "Ascorbic acid (vitamin C)",
    "cholecalciferol": "Cholecalciferol (vitamin D3)",
    "phylloquinone": "Phylloquinone (vitamin K1)",
}

MASS_UNITS = {"g", "mg", "ug", "microg", "mcg", "µg"}


# The core VMH catalogue has broad nutrient-form coverage but intentionally
# does not represent the dietary phytochemical space.  These are chemically
# specified food metabolites selected by food-chemistry class, not by their
# current row frequency.  They form a reviewable extension target panel; an
# individual entry can remain a data gap until a compatible numeric expression
# is demonstrated.
METABOLOME_EXTENSION_SEEDS = [
    # Organic acids and nitrogenous food metabolites.
    ("Acetic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "acetic acid", "Common food organic acid."),
    ("Citric acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "citric acid", "Tricarboxylic acid abundant in plant foods."),
    ("Lactic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "lactic acid", "Fermentation-related food organic acid."),
    ("Malic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "malic acid", "Food organic acid."),
    ("Succinic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "succinic acid", "Food organic acid."),
    ("Fumaric acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "fumaric acid", "Food organic acid."),
    ("Tartaric acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "tartaric acid", "Food organic acid."),
    ("Oxalic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "oxalic acid", "Food organic acid and antinutrient-relevant analyte."),
    ("Quinic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "quinic acid", "Plant-derived food organic acid."),
    ("Shikimic acid", "Organic acids and small food metabolites", "food_metabolite", "chemical_entity", "shikimic acid", "Plant shikimate-pathway metabolite."),
    ("Taurine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "taurine", "Food-derived amino-sulfonic acid."),
    ("Creatine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "creatine", "Food-derived guanidino compound."),
    ("L-Carnitine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "l carnitine | carnitine", "Food-derived quaternary ammonium compound."),
    ("Gamma-aminobutyric acid", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "gamma aminobutyric acid | gaba", "Food amino-acid-derived metabolite."),
    ("Glutathione", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "glutathione", "Food tripeptide redox metabolite."),
    ("Spermidine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "spermidine", "Food polyamine."),
    ("Spermine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "spermine", "Food polyamine."),
    ("Putrescine", "Nitrogenous food metabolites", "food_metabolite", "chemical_entity", "putrescine", "Food polyamine."),
    # Carotenoids which are not all represented in VMH's nutrient list.
    ("Violaxanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "violaxanthin", "Dietary xanthophyll."),
    ("Neoxanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "neoxanthin", "Dietary xanthophyll."),
    ("Antheraxanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "antheraxanthin", "Dietary xanthophyll."),
    ("Capsanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "capsanthin", "Dietary xanthophyll."),
    ("Capsorubin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "capsorubin", "Dietary xanthophyll."),
    ("Astaxanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "astaxanthin", "Dietary xanthophyll."),
    ("Canthaxanthin", "Carotenoids and other isoprenoid food metabolites", "food_metabolite", "chemical_entity", "canthaxanthin", "Dietary xanthophyll."),
    # Phenolic acids, flavonoids, isoflavones, anthocyanidins, and stilbenes.
    ("Caffeic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "caffeic acid", "Hydroxycinnamic acid."),
    ("Chlorogenic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "chlorogenic acid", "Caffeoylquinic acid food polyphenol."),
    ("Ferulic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "ferulic acid", "Hydroxycinnamic acid."),
    ("p-Coumaric acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "p coumaric acid | coumaric acid", "Hydroxycinnamic acid."),
    ("Sinapic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "sinapic acid", "Hydroxycinnamic acid."),
    ("Gallic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "gallic acid", "Hydroxybenzoic acid."),
    ("Ellagic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "ellagic acid", "Ellagitannin-derived polyphenol."),
    ("Protocatechuic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "protocatechuic acid", "Hydroxybenzoic acid."),
    ("Vanillic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "vanillic acid", "Hydroxybenzoic acid."),
    ("Syringic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "syringic acid", "Hydroxybenzoic acid."),
    ("Rosmarinic acid", "Polyphenols and phenolic acids", "food_metabolite", "chemical_entity", "rosmarinic acid", "Food phenolic ester."),
    ("Apigenin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "apigenin", "Flavone."),
    ("Luteolin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "luteolin", "Flavone."),
    ("Quercetin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "quercetin", "Flavonol."),
    ("Kaempferol", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "kaempferol", "Flavonol."),
    ("Myricetin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "myricetin", "Flavonol."),
    ("Catechin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "catechin", "Flavan-3-ol."),
    ("Epicatechin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "epicatechin", "Flavan-3-ol."),
    ("Epigallocatechin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "epigallocatechin", "Flavan-3-ol."),
    ("Epicatechin gallate", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "epicatechin gallate", "Gallated flavan-3-ol."),
    ("Epigallocatechin gallate", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "epigallocatechin gallate", "Gallated flavan-3-ol."),
    ("Naringenin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "naringenin", "Flavanone."),
    ("Hesperetin", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "hesperetin", "Flavanone."),
    ("Eriodictyol", "Flavonoids and related polyphenols", "food_metabolite", "chemical_entity", "eriodictyol", "Flavanone."),
    ("Genistein", "Isoflavones", "food_metabolite", "chemical_entity", "genistein", "Isoflavone."),
    ("Daidzein", "Isoflavones", "food_metabolite", "chemical_entity", "daidzein", "Isoflavone."),
    ("Glycitein", "Isoflavones", "food_metabolite", "chemical_entity", "glycitein", "Isoflavone."),
    ("Cyanidin", "Anthocyanidins", "food_metabolite", "chemical_entity", "cyanidin", "Anthocyanidin."),
    ("Delphinidin", "Anthocyanidins", "food_metabolite", "chemical_entity", "delphinidin", "Anthocyanidin."),
    ("Pelargonidin", "Anthocyanidins", "food_metabolite", "chemical_entity", "pelargonidin", "Anthocyanidin."),
    ("Malvidin", "Anthocyanidins", "food_metabolite", "chemical_entity", "malvidin", "Anthocyanidin."),
    ("Peonidin", "Anthocyanidins", "food_metabolite", "chemical_entity", "peonidin", "Anthocyanidin."),
    ("Resveratrol", "Stilbenes", "food_metabolite", "chemical_entity", "resveratrol", "Food stilbene."),
    # Glucosinolates and other characteristic organosulfur food metabolites.
    ("Glucoraphanin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucoraphanin", "Glucosinolate."),
    ("Sinigrin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "sinigrin", "Glucosinolate."),
    ("Glucobrassicin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucobrassicin", "Glucosinolate."),
    ("Gluconasturtiin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "gluconasturtiin", "Glucosinolate."),
    ("Progoitrin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "progoitrin", "Glucosinolate."),
    ("Glucotropaeolin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucotropaeolin", "Glucosinolate."),
    ("Glucoiberin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucoiberin", "Glucosinolate."),
    ("Glucosiberin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucosiberin", "Glucosinolate."),
    ("Glucoerucin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "glucoerucin", "Glucosinolate."),
    ("Alliin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "alliin", "Allium organosulfur precursor."),
    ("Allicin", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "allicin", "Allium organosulfur compound."),
    ("Diallyl sulfide", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "diallyl sulfide", "Allium organosulfur compound."),
    ("Diallyl disulfide", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "diallyl disulfide", "Allium organosulfur compound."),
    ("Diallyl trisulfide", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "diallyl trisulfide", "Allium organosulfur compound."),
    ("Sulforaphane", "Glucosinolates and organosulfur metabolites", "food_metabolite", "chemical_entity", "sulforaphane", "Isothiocyanate derived from glucoraphanin."),
    # Non-nutrient bioactives with established food occurrence.
    ("Theophylline", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "theophylline", "Methylxanthine."),
    ("Trigonelline", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "trigonelline", "Food alkaloid."),
    ("Capsaicin", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "capsaicin", "Capsicum alkaloid-like food bioactive."),
    ("Piperine", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "piperine", "Piper alkaloid-like food bioactive."),
    ("Curcumin", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "curcumin", "Curcuminoid food polyphenol."),
    ("[6]-Gingerol", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "6 gingerol | gingerol", "Ginger phenolic ketone."),
    ("[6]-Shogaol", "Alkaloids and other food bioactives", "food_metabolite", "chemical_entity", "6 shogaol | shogaol", "Ginger dehydration product."),
]

# VMH is the breadth reference, not a replacement for dietary-reference
# requirements. These direct mass-expression targets retain known core gaps
# explicitly instead of dropping them because the current VMH list is silent.
NUTRITION_CORE_ADDITIONS = [
    ("Available carbohydrate", "Macronutrients", "macronutrient", "defined_aggregate", "available carbohydrate", "A defined digestible-carbohydrate expression; never merge with carbohydrate by difference."),
    ("Chromium", "Minerals and trace elements", "micronutrient", "element_or_ion", "chromium", "Dietary trace-element requirement with no direct current VMH target entry."),
    ("Molybdenum", "Minerals and trace elements", "micronutrient", "element_or_ion", "molybdenum", "Dietary trace-element requirement with no direct current VMH target entry."),
]


@dataclass(frozen=True)
class FuzzyMatch:
    target_axis_id: str
    score: float
    sequence_score: float
    token_score: float
    number_score: float
    shared_tokens: str


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_axis_name(value: Any) -> str:
    """Normalise spelling for candidate generation without chemical inference."""
    text = _clean(value).casefold()
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("phyllochinon", "phylloquinone")
    text = text.replace("pyridoxin", "pyridoxine")
    text = text.replace("saccharose", "sucrose")
    text = re.sub(r"\bvitamin\s+b\s*([0-9]+)\b", r"vitamin b\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return EXACT_NAME_ALIASES.get(text, text)


def _tokens(value: str) -> set[str]:
    return {
        token for token in value.split()
        if len(token) > 1 and token not in {"acid", "vitamin", "total", "fatty", "food", "content"}
    }


def _number_signature(value: str) -> set[str]:
    """Capture chain-length or vitamin-number clues, without declaring identity."""
    return set(re.findall(r"\b\d{1,2}(?:\s+\d+)?\b|\b(?:b|d)\s*\d+\b", value))


def _unit_relation(source_unit: str, target_units: set[str]) -> str:
    source = _normalise_axis_name(source_unit)
    targets = {_normalise_axis_name(unit) for unit in target_units if _clean(unit)}
    if not source or not targets:
        return "unit_not_stated_in_one_or_both_catalogues"
    if source in MASS_UNITS and targets & MASS_UNITS:
        return "both_report_mass_units; conversion_not_performed_here"
    if source in targets:
        return "same_raw_unit"
    return "unit_or_expression_review_required"


def fetch_vmh_catalog(path: Path, *, refresh: bool = False) -> dict[str, Any]:
    """Download the public VMH nutrient catalogue once and retain raw bytes."""
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("results"), list):
            raise ValueError(f"Cached VMH catalogue is malformed: {path}")
        return payload

    request = Request(
        VMH_NUTRIENT_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "FoodNutritionResearch/1.0 (+axis catalogue audit)",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read()
    except HTTPError as error:
        raise RuntimeError(f"VMH nutrient catalogue request failed with HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Could not download the public VMH nutrient catalogue: {error.reason}") from error
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload.get("results"), list):
        raise ValueError("VMH nutrient response has no results list.")
    if int(payload.get("count", -1)) != len(payload["results"]):
        raise ValueError(
            "VMH nutrient response is paginated or incomplete; expected one full page. "
            f"count={payload.get('count')}, results={len(payload['results'])}."
        )
    payload["_local_retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["_local_source_url"] = VMH_NUTRIENT_URL
    payload["_local_sha256"] = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _is_derived_or_label_expression(row: pd.Series) -> bool:
    text = _clean(row.description).casefold()
    unit = _clean(row.unit).casefold()
    return (
        any(pattern in text for pattern in DERIVED_OR_LABEL_PATTERNS)
        or bool(re.search(r"\b(?:rae|dfe)\b", text))
        or unit.strip() == "iu"
    )


def _semantic_group(row: pd.Series, key: str) -> tuple[str, str, str, str]:
    """Return axis family, nutritional role, chemical-status, and mask family."""
    category = _clean(row.category).casefold()
    subcategory = _clean(row.subcategory).casefold()
    name = _clean(row.description).casefold()
    if key in {"protein total", "fat total", "carbohydrate total"}:
        return "Macronutrients", "macronutrient", "defined_aggregate", "macronutrient"
    if category == "lipids":
        if "fatty acid" in subcategory:
            return "Fatty-acid and lipid chemical forms", "nutrient_chemical_form_or_food_metabolite", "chemical_or_defined_aggregate", "lipid_fatty_acid"
        if "phytosterol" in subcategory or "cholesterol" in subcategory:
            return "Sterols and sterol expressions", "food_metabolite", "chemical_or_defined_aggregate", "lipid_sterol"
        return "Lipid aggregate expressions", "nutrient_related_aggregate", "defined_aggregate", "lipid_aggregate"
    if category == "proteins":
        if "amino acid" in subcategory:
            return "Amino acids and protein-related metabolites", "nutrient_chemical_form_or_food_metabolite", "chemical_entity", "amino_acid"
        return "Protein aggregate expressions", "nutrient_related_aggregate", "defined_aggregate", "protein_aggregate"
    if category == "vitamins":
        return "Vitamins, vitamers, and provitamins", "micronutrient_or_nutrient_chemical_form", "chemical_or_defined_aggregate", "vitamin"
    if category == "minerals and trace elements":
        if "ash" in subcategory or name == "ash":
            return "Food matrix and proximate composition", "food_matrix", "defined_aggregate", "food_matrix"
        return "Minerals and trace elements", "micronutrient", "element_or_ion", "mineral"
    if category in {"dietary fibers", "dietary fibres"}:
        if key == "dietary fibre total":
            return "Macronutrients", "macronutrient_related", "defined_aggregate", "macronutrient"
        return "Dietary-fibre chemical forms", "food_metabolite_or_nutrient_chemical_form", "chemical_or_defined_aggregate", "dietary_fibre"
    if category == "carbohydrates":
        return "Carbohydrate forms and small organic metabolites", "food_metabolite_or_nutrient_chemical_form", "chemical_or_defined_aggregate", "carbohydrate"
    if category == "other":
        if key == "ethanol":
            return "Macronutrient-related alcohol", "macronutrient_related", "chemical_entity", "macronutrient"
        if key == "water":
            return "Food matrix and proximate composition", "food_matrix", "chemical_entity", "food_matrix"
        return "Other food metabolites", "food_metabolite", "chemical_or_defined_aggregate", "other_metabolite"
    return "Review-required nutrition expression", "nutrition_or_food_chemistry_review_required", "uncertain", "review_required"


def _display_name(key: str, rows: pd.DataFrame) -> str:
    if key in DISPLAY_ALIASES:
        return DISPLAY_ALIASES[key]
    # Source descriptions are the official VMH display labels.  Use the shortest
    # one only for a human-readable label; preserve every original label below.
    choices = sorted({_clean(value) for value in rows.description if _clean(value)}, key=lambda value: (len(value), value))
    return choices[0] if choices else key


def build_vmh_target_registry(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 200+ target panel while preserving excluded expressions separately."""
    rows = pd.DataFrame(payload["results"]).copy()
    required = {"nut_no", "description", "unit", "category", "subcategory", "mets"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"VMH catalogue missing expected fields: {sorted(missing)}")
    rows["canonical_key"] = rows.description.map(_normalise_axis_name)
    rows["selection_status"] = rows.apply(
        lambda row: "derived_or_label_expression_not_direct_target"
        if _is_derived_or_label_expression(row) else "direct_prediction_candidate", axis=1,
    )

    targets: list[dict[str, Any]] = []
    for key, group in rows[rows.selection_status.eq("direct_prediction_candidate")].groupby("canonical_key", sort=True):
        representative = group.iloc[0]
        axis_family, nutritional_role, chemical_status, mask_family = _semantic_group(representative, key)
        raw_names = sorted({_clean(value) for value in group.description if _clean(value)})
        aliases = sorted({key, *(_normalise_axis_name(value) for value in raw_names)})
        vmh_codes = sorted({_clean(value) for value in group.nut_no if _clean(value)})
        units = sorted({_clean(value) for value in group.unit if _clean(value)})
        met_ids = sorted({item for values in group.mets for item in (values or [])})
        targets.append({
            "target_axis_id": stable_id("prediction_target", "vmh", key),
            "target_version": "vmh_aligned_prediction_panel_v1",
            "canonical_name": _display_name(key, group),
            "canonical_key": key,
            "axis_family": axis_family,
            "nutritional_role": nutritional_role,
            "chemical_status": chemical_status,
            "training_intent": "direct_prediction_candidate",
            "mask_family_for_future_benchmark": mask_family,
            "vmh_category": _clean(representative.category),
            "vmh_subcategories": " | ".join(sorted({_clean(value) for value in group.subcategory if _clean(value)})),
            "vmh_raw_axis_count": len(group),
            "vmh_nutrient_codes": " | ".join(vmh_codes),
            "vmh_original_name_variants": " | ".join(raw_names),
            "vmh_units": " | ".join(units),
            "vmh_metabolite_ids": " | ".join(met_ids),
            "target_aliases": " | ".join(aliases),
            "target_origin": "VMH public nutrient catalogue",
            "target_definition": (
                "VMH-aligned food-composition expression. Exact analytical definition, chemical form, "
                "unit, denominator, and source method must remain explicit in any numerical dataset."
            ),
            "scientific_basis": "VMH Nutrition nutrient catalogue; FAO/INFOODS component-expression framework; NIH dietary-reference scope where applicable.",
            "authority_urls": " | ".join([VMH_REFERENCE_URL, INFOODS_URL, NIH_DRI_URL, FAO_URL]),
            "source_match_count_exact": 0,
            "source_match_count_fuzzy_candidate": 0,
        })
    present_keys = {row["canonical_key"] for row in targets}
    for name, family, role, chemical_status, aliases, rationale in NUTRITION_CORE_ADDITIONS:
        key = _normalise_axis_name(name)
        if key in present_keys:
            continue
        targets.append({
            "target_axis_id": stable_id("prediction_target", "nutrition_core_addition", key),
            "target_version": "vmh_aligned_prediction_panel_v1",
            "canonical_name": name,
            "canonical_key": key,
            "axis_family": family,
            "nutritional_role": role,
            "chemical_status": chemical_status,
            "training_intent": "direct_prediction_candidate_pending_catalogue_match",
            "mask_family_for_future_benchmark": "macronutrient" if family == "Macronutrients" else "mineral",
            "vmh_category": "",
            "vmh_subcategories": "",
            "vmh_raw_axis_count": 0,
            "vmh_nutrient_codes": "",
            "vmh_original_name_variants": "",
            "vmh_units": "",
            "vmh_metabolite_ids": "",
            "target_aliases": " | ".join(sorted({_normalise_axis_name(item) for item in aliases.split(" | ")})),
            "target_origin": "Normative nutrition-core addition",
            "target_definition": rationale + " Require a source-specific quantified, compatible mass expression before use as a numerical training label.",
            "scientific_basis": "National Academies/NIH dietary-reference nutrition scope; FAO/INFOODS component-expression framework.",
            "authority_urls": " | ".join([NIH_DRI_URL, INFOODS_URL, FAO_URL]),
            "source_match_count_exact": 0,
            "source_match_count_fuzzy_candidate": 0,
        })
        present_keys.add(key)
    for name, family, role, chemical_status, aliases, rationale in METABOLOME_EXTENSION_SEEDS:
        key = _normalise_axis_name(name)
        if key in present_keys:
            continue
        alias_values = sorted({_normalise_axis_name(item) for item in aliases.split(" | ") if _normalise_axis_name(item)})
        targets.append({
            "target_axis_id": stable_id("prediction_target", "food_metabolome_extension", key),
            "target_version": "vmh_aligned_prediction_panel_v1",
            "canonical_name": name,
            "canonical_key": key,
            "axis_family": family,
            "nutritional_role": role,
            "chemical_status": chemical_status,
            "training_intent": "metabolome_extension_prediction_candidate",
            "mask_family_for_future_benchmark": family.casefold().replace(" ", "_").replace("-", "_"),
            "vmh_category": "",
            "vmh_subcategories": "",
            "vmh_raw_axis_count": 0,
            "vmh_nutrient_codes": "",
            "vmh_original_name_variants": "",
            "vmh_units": "",
            "vmh_metabolite_ids": "",
            "target_aliases": " | ".join(alias_values),
            "target_origin": "Food-chemistry metabolome extension",
            "target_definition": rationale + " Require a source-specific quantified, compatible mass expression before use as a numerical training label.",
            "scientific_basis": "FAO food-composition scope includes phytochemicals and bioactives; chemical identity must be reviewed against ChEBI/FooDB or an equivalent authority.",
            "authority_urls": " | ".join([FAO_URL, "https://www.ebi.ac.uk/chebi/", "https://foodb.ca/"]),
            "source_match_count_exact": 0,
            "source_match_count_fuzzy_candidate": 0,
        })
        present_keys.add(key)
    targets_frame = pd.DataFrame(targets).sort_values(["axis_family", "canonical_name"]).reset_index(drop=True)
    excluded = rows[rows.selection_status.ne("direct_prediction_candidate")].copy()
    excluded["exclusion_reason"] = (
        "Derived, activity-equivalent, energy, calculated, added, or label-specific expression; retain in provenance "
        "but do not present as a direct chemical or composition prediction target."
    )
    return targets_frame, excluded


def _build_target_indices(targets: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    tokens: dict[str, set[str]] = defaultdict(set)
    lookup: dict[str, dict[str, Any]] = {}
    for row in targets.to_dict(orient="records"):
        target_id = row["target_axis_id"]
        lookup[target_id] = row
        names = [
            row["canonical_key"],
            *row["vmh_original_name_variants"].split(" | "),
            *row["target_aliases"].split(" | "),
        ]
        for name in names:
            normalised = _normalise_axis_name(name)
            if not normalised:
                continue
            aliases[normalised].add(target_id)
            for token in _tokens(normalised):
                tokens[token].add(target_id)
    return aliases, tokens, lookup


def _fuzzy_candidates(source_name: str, candidates: set[str], lookup: dict[str, dict[str, Any]]) -> list[FuzzyMatch]:
    normalised = _normalise_axis_name(source_name)
    source_tokens = _tokens(normalised)
    source_numbers = _number_signature(normalised)
    matches: list[FuzzyMatch] = []
    for target_id in candidates:
        target = lookup[target_id]
        target_key = target["canonical_key"]
        target_tokens = _tokens(target_key)
        sequence = SequenceMatcher(a=normalised, b=target_key).ratio()
        union = source_tokens | target_tokens
        token_score = len(source_tokens & target_tokens) / len(union) if union else 0.0
        target_numbers = _number_signature(target_key)
        if source_numbers and target_numbers:
            number_score = 1.0 if source_numbers == target_numbers else 0.0
        elif source_numbers or target_numbers:
            number_score = 0.0
        else:
            number_score = 0.5
        score = 0.55 * sequence + 0.30 * token_score + 0.15 * number_score
        if score >= 0.74:
            matches.append(FuzzyMatch(
                target_axis_id=target_id,
                score=score,
                sequence_score=sequence,
                token_score=token_score,
                number_score=number_score,
                shared_tokens=" | ".join(sorted(source_tokens & target_tokens)),
            ))
    return sorted(matches, key=lambda match: (-match.score, match.target_axis_id))[:3]


def _identity_caution(source_name: str, target_key: str) -> str:
    """Highlight lexical patterns that commonly conceal a non-identical analyte."""
    source = _normalise_axis_name(source_name)
    source_numbers = _number_signature(source)
    target_numbers = _number_signature(target_key)
    cautions: list[str] = []
    if source_numbers and target_numbers and source_numbers != target_numbers:
        cautions.append("numeric_form_or_homologue_differs")
    source_stereo = bool(re.search(r"^(?:d|l)\s|\b(?:d|l)\s+[a-z]", source))
    target_stereo = bool(re.search(r"^(?:d|l)\s|\b(?:d|l)\s+[a-z]", target_key))
    if source_stereo != target_stereo:
        cautions.append("stereochemical_form_not_confirmed")
    if re.match(r"^\d{1,2}\s+[a-z]", source) or re.match(r"^\d{1,2}\s+[a-z]", target_key):
        cautions.append("positional_or_isomeric_form_not_confirmed")
    return " | ".join(cautions) if cautions else "none_detected_from_name_only"


def build_fuzzy_match_candidates(source_axes: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Make name-similarity candidates; never return an asserted equivalence."""
    alias_index, token_index, target_lookup = _build_target_indices(targets)
    records: list[dict[str, Any]] = []
    for source in source_axes.itertuples(index=False):
        source_name = _clean(source.original_name)
        if any(pattern in source_name.casefold() for pattern in DERIVED_OR_LABEL_PATTERNS):
            continue
        normalised = _normalise_axis_name(source_name)
        if not normalised:
            continue
        exact = alias_index.get(normalised, set())
        if exact:
            candidates = [FuzzyMatch(target_id, 1.0, 1.0, 1.0, 1.0, "exact canonical alias") for target_id in sorted(exact)]
            match_method = "exact_canonical_alias_candidate"
        else:
            source_tokens = _tokens(normalised)
            candidate_ids: set[str] = set()
            for token in source_tokens:
                candidate_ids.update(token_index.get(token, set()))
            candidates = _fuzzy_candidates(source_name, candidate_ids, target_lookup)
            match_method = "fuzzy_lexical_candidate"
        for rank, match in enumerate(candidates, start=1):
            target = target_lookup[match.target_axis_id]
            records.append({
                "candidate_id": stable_id("fuzzy_axis_candidate", source.source_axis_id, match.target_axis_id),
                "source_axis_id": source.source_axis_id,
                "source_key": source.source_key,
                "source_name": source.source_name,
                "source_component_id": source.source_component_id,
                "source_original_name": source_name,
                "source_definition": _clean(source.source_definition),
                "source_component_group": _clean(source.source_component_group),
                "source_chemical_class": _clean(source.source_chemical_class),
                "source_infoods_tag": _clean(source.infoods_tag),
                "source_inchi": _clean(source.inchi),
                "source_unit": _clean(source.raw_unit),
                "source_denominator": _clean(source.raw_denominator),
                "source_value_availability": _clean(source.value_availability),
                "target_axis_id": match.target_axis_id,
                "target_canonical_name": target["canonical_name"],
                "target_axis_family": target["axis_family"],
                "target_vmh_codes": target["vmh_nutrient_codes"],
                "candidate_rank_for_source_axis": rank,
                "match_method": match_method,
                "similarity_score_0_to_1": round(match.score, 6),
                "sequence_similarity_0_to_1": round(match.sequence_score, 6),
                "token_jaccard_0_to_1": round(match.token_score, 6),
                "number_signature_agreement_0_to_1": round(match.number_score, 6),
                "shared_specific_tokens": match.shared_tokens,
                "identity_caution_from_name_only": _identity_caution(source_name, target["canonical_key"]),
                "unit_relation": _unit_relation(source.raw_unit, set(target["vmh_units"].split(" | "))),
                "asserted_relation": "candidate_name_relation_only",
                "numerical_value_merge_permitted": False,
                "review_status": (
                    "definition_and_expression_review_required" if match_method.startswith("exact")
                    else "chemical_identity_definition_and_expression_review_required"
                ),
                "review_reason": (
                    "Lexical terminology agrees, but analytical definition, chemical form, unit, denominator, and method "
                    "still require verification before asserting equivalence."
                    if match_method.startswith("exact") else
                    "Fuzzy lexical similarity is a discovery signal only; it cannot establish chemical, aggregate, or method equivalence."
                ),
            })
    return pd.DataFrame(records)


def _target_match_counts(targets: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    result = targets.copy()
    if candidates.empty:
        return result
    exact = candidates[candidates.match_method.eq("exact_canonical_alias_candidate")].groupby("target_axis_id").source_axis_id.nunique()
    fuzzy = candidates[candidates.match_method.eq("fuzzy_lexical_candidate")].groupby("target_axis_id").source_axis_id.nunique()
    result["source_match_count_exact"] = result.target_axis_id.map(exact).fillna(0).astype(int)
    result["source_match_count_fuzzy_candidate"] = result.target_axis_id.map(fuzzy).fillna(0).astype(int)
    return result


def _write_report(path: Path, *, payload: dict[str, Any], targets: pd.DataFrame,
                  excluded: pd.DataFrame, candidates: pd.DataFrame, source_axis_count: int) -> None:
    raw = pd.DataFrame(payload["results"])
    category_counts = raw.groupby(raw.category.fillna("Unspecified")).size().sort_values(ascending=False)
    family_counts = targets.groupby(["axis_family", "nutritional_role"]).size().sort_values(ascending=False)
    match_counts = candidates.groupby("match_method").size().sort_values(ascending=False) if not candidates.empty else pd.Series(dtype="int64")
    review_count = int((
        candidates.match_method.eq("fuzzy_lexical_candidate")
        | candidates.identity_caution_from_name_only.ne("none_detected_from_name_only")
        | candidates.unit_relation.eq("unit_or_expression_review_required")
    ).sum()) if not candidates.empty else 0
    total = len(targets)
    lines = [
        "# VMH-Aligned Nutrition Foundation Model Prediction-Axis Design",
        "",
        "## Decision",
        "",
        f"The public VMH nutrient API returned **{payload['count']} raw nutrient records** in this cached snapshot. "
        f"After retaining only direct food-composition, nutrient-chemical-form, and food-metabolite expressions, and "
        f"consolidating only narrow terminology aliases, this design contains **{total} candidate prediction axes**. "
        "The panel therefore exceeds the requested 200-axis scope without treating energy, activity equivalents, "
        "calculated-by-difference values, or added/label-specific fields as molecular composition targets.",
        "",
        "This is a **target-design and match-discovery artifact**, not a merged training matrix. A prediction target is "
        "not automatically a trainable numerical label: unit conversion, denominator, analytical definition, censoring, "
        "food identity, and source-quality review still occur later.",
        "",
        "## VMH Reference Catalogue",
        "",
        "VMH exposes its current Nutrition nutrient list through a public endpoint. Its 2018 database paper reported "
        "150 nutritional constituents at that time; the current endpoint is the operative reference for this design and "
        "contains 285 records. Both the raw JSON snapshot and its SHA-256 are retained in the output manifest.",
        "",
        "| VMH category | Raw records |",
        "|---|---:|",
        *[f"| {category} | {count:,} |" for category, count in category_counts.items()],
        "",
        "## Selection Principles",
        "",
        "1. **Only three aggregate macronutrient targets** are used: total protein, total fat, and total carbohydrate when directly defined. Available carbohydrate is retained separately because it has a different analytical definition; dietary fibre and ethanol are nutrient-related expressions, while water and ash are food-matrix targets.",
        "2. **Micronutrients and chemical forms are retained separately**: vitamins, vitamers/provitamins, minerals, trace elements, essential fatty acids, amino acids, individual fatty-acid forms, sterols, sugar forms and fibre forms are legitimate composition targets when their measurement definitions are preserved.",
        "3. **Aggregates are not silently treated as molecules**. Totals (for example saturated fatty acids or total sugars) are retained as defined aggregate expressions. A later benchmark must mask their family jointly so an unmasked total cannot algebraically reveal a masked member, or vice versa.",
        "4. **Derived or label expressions are excluded from direct prediction**: energy, IU/activity equivalents, DFE/RAE-like equivalents, carbohydrate by difference, adjusted protein, and added nutrient labels remain provenance or auxiliary derived expressions.",
        "5. **A bounded specialist-metabolome extension is explicit**. Polyphenols, glucosinolates, organosulfurs, carotenoids, organic acids, polyamines, and selected food bioactives are candidate targets because they are defined food chemicals, not because they are common in a particular source. Each remains a data gap until a compatible quantified source expression is verified; FooDB association-only rows are never numerical labels.",
        "",
        "## Candidate Prediction Panel",
        "",
        "| Axis family | Nutritional role | Candidate axes |",
        "|---|---|---:|",
        *[f"| {family} | {role} | {count:,} |" for (family, role), count in family_counts.items()],
        "",
        "The complete axis-level list, VMH source IDs, original name variants, units, metabolite identifiers, future mask family, and lexical source-match counts is in `prediction_axis_registry.csv`.",
        "",
        "## Fuzzy Cross-Source Match Discovery",
        "",
        f"{source_axis_count:,} source-native axes were compared against the VMH-aligned panel. The candidate table deliberately "
        "does not merge axes or values. Exact normalized terminology and fuzzy lexical similarity are both reviewable evidence, "
        "not final equivalence assertions.",
        "",
        "| Candidate method | Candidate links |",
        "|---|---:|",
        *[f"| {method} | {count:,} |" for method, count in match_counts.items()],
        "",
        "Every candidate stores similarity components, shared tokens, apparent unit relationship, source definition, INFOODS tag, "
        "and the explicit rule `numerical_value_merge_permitted = false`. Review may assert only an exact analyte, chemical form, "
        "aggregate, method-variant, or not-comparable relation after checking definition and measurement expression.",
        "",
        f"`fuzzy_match_review_queue.csv.gz` contains **{review_count:,}** links requiring priority review because they are fuzzy, "
        "have an apparent stereochemical/positional/homologue caution, or have a unit/expression mismatch.",
        "",
        "## Excluded VMH Records",
        "",
        f"{len(excluded):,} VMH raw records are retained in `vmh_excluded_non_direct_expressions.csv` but omitted from direct target selection. "
        "Their exclusion is semantic rather than a judgment of data quality or scientific relevance.",
        "",
        "## Scientific Scope",
        "",
        "The panel follows the VMH nutritional catalogue for breadth while retaining FAO/INFOODS' distinction between component identity "
        "and analytical expression. Dietary-reference authorities define the macronutrient/micronutrient scope; they do not make all "
        "vitamin forms or food metabolites numerically interchangeable. This preserves a usable 200+ target design without creating "
        "invalid pooled labels.",
        "",
        "Sources: [VMH API documentation](https://www.vmh.life/_api/docs/); "
        "[VMH database paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323901/); "
        "[FAO/INFOODS component identifiers](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/); "
        "[NIH dietary-reference information](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_target_panel(path: Path, targets: pd.DataFrame) -> None:
    """Write every selected target in a readable, grouped form alongside CSV."""
    lines = [
        "# VMH-Aligned Prediction Axis Panel",
        "",
        "This file lists every candidate target axis. A source match count reports lexical candidate links only; it is not evidence that a compatible numerical training label is already available.",
        "",
    ]
    for family, group in targets.groupby("axis_family", sort=True):
        lines.extend([
            f"## {family}",
            "",
            "| Canonical axis | Role | Origin | Exact-name source candidates | Fuzzy source candidates |",
            "|---|---|---|---:|---:|",
        ])
        for row in group.sort_values("canonical_name").itertuples(index=False):
            lines.append(
                f"| {row.canonical_name} | {row.nutritional_role} | {row.target_origin} | "
                f"{row.source_match_count_exact} | {row.source_match_count_fuzzy_candidate} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prediction_axis_design(
    *, atlas_dir: Path, output_dir: Path, report_dir: Path, vmh_catalog_path: Path,
    refresh_vmh: bool = False,
) -> dict[str, int]:
    """Build the VMH-aligned target design and non-assertive fuzzy candidates."""
    source_path = atlas_dir / "source_axis_catalog.csv.gz"
    if not source_path.exists():
        raise FileNotFoundError(f"Atlas source-axis catalogue is required: {source_path}")
    payload = fetch_vmh_catalog(vmh_catalog_path, refresh=refresh_vmh)
    targets, excluded = build_vmh_target_registry(payload)
    if len(targets) < 200:
        raise AssertionError(f"Prediction panel unexpectedly has fewer than 200 axes: {len(targets)}")
    source_axes = pd.read_csv(source_path, low_memory=False)
    candidates = build_fuzzy_match_candidates(source_axes, targets)
    targets = _target_match_counts(targets, candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(targets, output_dir / "prediction_axis_registry.csv")
    write_csv(excluded, output_dir / "vmh_excluded_non_direct_expressions.csv")
    candidates.to_csv(
        output_dir / "source_axis_fuzzy_match_candidates.csv.gz", index=False, compression="gzip",
    )
    review_queue = candidates[
        candidates.match_method.eq("fuzzy_lexical_candidate")
        | candidates.identity_caution_from_name_only.ne("none_detected_from_name_only")
        | candidates.unit_relation.eq("unit_or_expression_review_required")
    ].copy()
    review_queue.to_csv(
        output_dir / "fuzzy_match_review_queue.csv.gz", index=False, compression="gzip",
    )
    write_csv(
        candidates.groupby(["target_axis_id", "match_method"], as_index=False).size()
        if not candidates.empty else pd.DataFrame(columns=["target_axis_id", "match_method", "size"]),
        output_dir / "fuzzy_match_summary_by_target.csv",
    )
    manifest = {
        "artifact": "vmh_aligned_prediction_axis_design_v1",
        "purpose": "Scientific target design and non-assertive fuzzy cross-source match discovery; no numerical merge or model training.",
        "vmh_source_url": VMH_NUTRIENT_URL,
        "vmh_catalogue_local_path": str(vmh_catalog_path),
        "vmh_raw_record_count": int(payload["count"]),
        "direct_prediction_axis_count": int(len(targets)),
        "derived_or_label_expression_count": int(len(excluded)),
        "source_axis_count_compared": int(len(source_axes)),
        "fuzzy_candidate_link_count": int(len(candidates)),
        "fuzzy_candidate_review_queue_count": int(len(review_queue)),
        "fuzzy_match_policy": "Candidates only. No candidate asserts equivalence or permits numerical value pooling.",
    }
    write_json(manifest, output_dir / "manifest.json")
    _write_report(
        report_dir / "VMH_ALIGNED_PREDICTION_AXIS_DESIGN.md",
        payload=payload, targets=targets, excluded=excluded, candidates=candidates,
        source_axis_count=len(source_axes),
    )
    _write_target_panel(report_dir / "PREDICTION_AXIS_PANEL.md", targets)
    return manifest

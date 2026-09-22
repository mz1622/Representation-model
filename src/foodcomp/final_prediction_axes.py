"""Build a scientifically bounded food-composition prediction-axis panel.

This module is a *semantic target-design* step.  It never pools measurements,
does not infer that similar source labels are equal, and does not decide that a
target is numerically trainable.  It selects chemically defined food
composition targets from nutrition and food-chemistry evidence, independently
of their present coverage in any one database.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .prediction_axis_design import _normalise_axis_name
from .util import stable_id, write_csv, write_json


PANEL_VERSION = "scientific_food_composition_prediction_panel_v1"

AUTHORITIES = {
    "FAO_INFOODS": {
        "label": "FAO/INFOODS food-composition framework and component identifiers",
        "url": "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/",
    },
    "FAO_OVERVIEW": {
        "label": "FAO food-composition overview",
        "url": "https://www.fao.org/food-composition/overview-2/en",
    },
    "NIH_DRI": {
        "label": "NIH Office of Dietary Supplements: Dietary Reference Intakes",
        "url": "https://ods.od.nih.gov/HealthInformation/nutrientrecommendations/",
    },
    "WHO_FAO_MICRO": {
        "label": "WHO/FAO: Vitamin and mineral requirements in human nutrition",
        "url": "https://www.who.int/publications/i/item/9241546123",
    },
    "FAO_PROTEIN": {
        "label": "FAO: Dietary protein quality evaluation in human nutrition",
        "url": "https://www.fao.org/4/i3124e/i3124e.pdf",
    },
    "FAO_FATS": {
        "label": "FAO/WHO: Fats and fatty acids in human nutrition",
        "url": "https://www.fao.org/4/i1953e/i1953e00.pdf",
    },
    "FDC": {
        "label": "USDA FoodData Central Foundation Foods documentation",
        "url": "https://fdc.nal.usda.gov/Foundation_Foods_Documentation/",
    },
    "LIPID_MAPS": {
        "label": "LIPID MAPS classification system",
        "url": "https://lipidmaps.org/databases/lmsd",
    },
    "PHENOL_EXPLORER": {
        "label": "Phenol-Explorer food polyphenol composition database",
        "url": "https://phenol-explorer.eu/compounds",
    },
    "USDA_FLAVONOID": {
        "label": "USDA Database for the Flavonoid Content of Selected Foods",
        "url": "https://www.ars.usda.gov/research/publications/publication/?seqNo115=271905",
    },
    "USDA_ISOFLAVONE": {
        "label": "USDA Database for the Isoflavone Content of Selected Foods, Release 2.0",
        "url": "https://www.ars.usda.gov/ARSUserFiles/80400525/data/isoflav/isoflav_r2.pdf",
    },
    "USDA_PA": {
        "label": "USDA Database for the Proanthocyanidin Content of Selected Foods",
        "url": "https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/methods-and-application-of-food-composition-laboratory/mafcl-site-pages/proanthocyanidin/",
    },
    "CHEBI": {
        "label": "ChEBI chemical-entity ontology",
        "url": "https://www.ebi.ac.uk/chebi/",
    },
}


# These are label expressions or broad classes without a unique compositional
# identity. They stay in the global atlas/provenance layer, not this direct
# target panel.  A separately defined aggregate can still be a direct target.
EXCLUDE_DRAFT_KEYS = {
    "dispensable amino acids",
    "indispensable amino acids",
    "disaccharide",
    "monosaccharides",
    "poly hexose",
    "poly pentose",
    "polysaccharide",
    "resorbable oligosaccharides",
    "sugar",
    "sugar monosaccaride disaccaride",
    "total sugar alcohols",
    "organic acids",
    "long chain fatty acids",
    "medium chain fatty acids",
    "short chain fatty acids",
    "omega 3 fatty acids",
    "omega 6 fatty acids",
    "monounsaturated fatty acids",
    "polyunsaturated fatty acids",
    "saturated fatty acids",
    "vitamin a carotene",
    "vitamin b 12",
    "vitamin b 6",
    "vitamin e tocopherol",
    "fluoride f",
    "fluorine",
}

EXCLUDE_DRAFT_DISPLAY_NAMES = {
    # "Vitamin E - Tocopherol" does not state which tocopherol form is meant.
    # Alpha, beta, gamma and delta forms are kept separately where available.
    "Vitamin E - Tocopherol",
}


# Only exact nomenclature equivalents are merged.  Positional, isomeric,
# stereochemical and method-expression differences intentionally remain apart.
EXACT_SYNONYM_CANONICAL = {
    "20 5 n 3 epa": "Eicosapentaenoic acid (EPA; 20:5n-3)",
    "eicosapentaenoic acid all cis 5 8 11 14 17 eicosapentaenoic acid": "Eicosapentaenoic acid (EPA; 20:5n-3)",
    "22 5 n 3 dpa": "Docosapentaenoic acid (DPA; 22:5n-3)",
    "docosapentaenoic acid dpa": "Docosapentaenoic acid (DPA; 22:5n-3)",
    "22 6 n 3 dha": "Docosahexaenoic acid (DHA; 22:6n-3)",
    "docosahexaenoic acid dha": "Docosahexaenoic acid (DHA; 22:6n-3)",
    "20 4 n 6": "Arachidonic acid (20:4n-6)",
    "arachidonic acid all cis 5 8 11 14 eicosatetraenoic acid": "Arachidonic acid (20:4n-6)",
    "18 2 n 6 c c": "Linoleic acid (18:2n-6)",
    "linoleic acid all cis 9 12 octadecadienoic acid": "Linoleic acid (18:2n-6)",
    "18 3 n 3 c c c ala": "Alpha-linolenic acid (18:3n-3)",
    "alpha linolenic acid": "Alpha-linolenic acid (18:3n-3)",
    "eicosenoic acid": "Eicosenoic acid (20:1; isomer unspecified)",
    "20 1": "Eicosenoic acid (20:1; isomer unspecified)",
    "eicosatrienoic acid": "Eicosatrienoic acid (20:3; isomer unspecified)",
    "20 3 undifferentiated": "Eicosatrienoic acid (20:3; isomer unspecified)",
    "heptadecenoic acid": "Heptadecenoic acid (17:1; isomer unspecified)",
    "17 1": "Heptadecenoic acid (17:1; isomer unspecified)",
    "pentadecenoic acid": "Pentadecenoic acid (15:1; isomer unspecified)",
    "15 1": "Pentadecenoic acid (15:1; isomer unspecified)",
    "tetradecenoic acid": "Tetradecenoic acid (14:1; isomer unspecified)",
    "14 1": "Tetradecenoic acid (14:1; isomer unspecified)",
    "vitamin b6 pyridoxin": "Pyridoxine (vitamin B6)",
    "pyridoxine": "Pyridoxine (vitamin B6)",
    "cobalamin vitamin b12": "Cobalamin (vitamin B12)",
    "vitamin b 6 pyridoxin": "Pyridoxine (vitamin B6)",
    "vitamin b 6 pyridoxine": "Pyridoxine (vitamin B6)",
}

SCIENTIFIC_EXACT_CANONICAL = {
    # These are explicit chemical-name equivalents, not fuzzy name matches.
    "chlorogenic acid": "5-Caffeoylquinic acid",
    "rutin": "Quercetin 3-O-rutinoside",
    "glucosinalbin": "Sinalbin",
}


# Natural-language scientific labels for chain notation. The notation remains
# in parentheses because it carries analytically essential information.
DISPLAY_CANONICAL = {
    "4 0": "Butyric acid (4:0)",
    "6 0": "Hexanoic acid (6:0)",
    "8 0": "Octanoic acid (8:0)",
    "10 0": "Decanoic acid (10:0)",
    "12 0": "Dodecanoic acid (12:0)",
    "13 0": "Tridecanoic acid (13:0)",
    "14 0": "Tetradecanoic acid (14:0)",
    "15 0": "Pentadecanoic acid (15:0)",
    "16 0": "Hexadecanoic acid (16:0)",
    "17 0": "Heptadecanoic acid (17:0)",
    "18 0": "Octadecanoic acid (18:0)",
    "20 0": "Eicosanoic acid (20:0)",
    "21 5": "Heneicosapentaenoic acid (21:5)",
    "22 0": "Docosanoic acid (22:0)",
    "24 0": "Tetracosanoic acid (24:0)",
    "16 1 c": "cis-Hexadecenoic acid (16:1 cis)",
    "16 1 t": "trans-Hexadecenoic acid (16:1 trans)",
    "16 1 undifferentiated": "Hexadecenoic acid (16:1; cis/trans unspecified)",
    "18 1 c": "cis-Octadecenoic acid (18:1 cis)",
    "18 1 t": "trans-Octadecenoic acid (18:1 trans)",
    "18 1 undifferentiated": "Octadecenoic acid (18:1; cis/trans unspecified)",
    "18 1 11 t 18 1t n 7": "trans-Vaccenic acid (18:1 trans-11)",
    "18 2 clas": "Conjugated linoleic acids (18:2 CLA isomers)",
    "18 2 i": "Isomeric linoleic acids (18:2)",
    "18 2 t not further defined": "trans-Linoleic acid (18:2; isomer unspecified)",
    "18 2 t t": "trans,trans-Linoleic acid (18:2 trans,trans)",
    "18 2 undifferentiated": "Octadecadienoic acid (18:2; isomer unspecified)",
    "18 3 n 6 c c c": "Gamma-linolenic acid (18:3n-6)",
    "18 3 undifferentiated": "Octadecatrienoic acid (18:3; isomer unspecified)",
    "18 3i": "Isomeric linolenic acids (18:3)",
    "18 4": "Octadecatetraenoic acid (18:4; isomer unspecified)",
    "20 2 n 6 c c": "cis,cis-Eicosadienoic acid (20:2n-6)",
    "20 3 n 3": "Eicosatrienoic acid (20:3n-3)",
    "20 3 n 6": "Eicosatrienoic acid (20:3n-6)",
    "20 4 undifferentiated": "Eicosatetraenoic acid (20:4; isomer unspecified)",
    "22 1 c": "cis-Docosenoic acid (22:1 cis)",
    "22 1 t": "trans-Docosenoic acid (22:1 trans)",
    "22 1 undifferentiated": "Docosenoic acid (22:1; cis/trans unspecified)",
    "22 4": "Docosatetraenoic acid (22:4; isomer unspecified)",
    "24 1 c": "cis-Tetracosenoic acid (24:1 cis)",
}


FAMILY_METADATA = {
    "Food matrix and proximate composition": (
        "food_matrix_or_proximate", "food matrix", "Stage 1 nutrition composition", "food_matrix",
        "Directly measured food-matrix or proximate composition expression.", "FAO_INFOODS | FDC",
    ),
    "Macronutrients": (
        "defined_nutritional_aggregate", "macronutrient", "Stage 1 nutrition composition", "macronutrient",
        "Direct nutritional aggregate. The analytical or calculation definition must stay attached to every value.", "FAO_INFOODS | NIH_DRI | FDC",
    ),
    "Carbohydrate forms and small organic metabolites": (
        "chemical_entity_or_defined_aggregate", "nutrient_related", "Stage 1 nutrition composition", "carbohydrate",
        "Direct sugar, sugar-alcohol, starch or other specified carbohydrate expression.", "FAO_INFOODS | NIH_DRI",
    ),
    "Dietary-fibre chemical forms": (
        "chemical_entity_or_defined_aggregate", "nutrient_related", "Stage 1 nutrition composition", "dietary_fibre",
        "Dietary-fibre constituent or explicitly defined fibre fraction.", "FAO_INFOODS | NIH_DRI",
    ),
    "Minerals and trace elements": (
        "element_or_ion", "essential_micronutrient", "Stage 1 nutrition composition", "mineral",
        "Element or ion with a nutritional role; speciation remains a separate future expression when measured.", "WHO_FAO_MICRO | NIH_DRI | FAO_INFOODS",
    ),
    "Vitamins, vitamers, and provitamins": (
        "chemical_entity_or_defined_vitamer_aggregate", "essential_micronutrient", "Stage 1 nutrition composition", "vitamin",
        "Vitamin, provitamin, or chemically specified vitamer. Activity equivalents are not direct targets.", "WHO_FAO_MICRO | NIH_DRI | FAO_INFOODS",
    ),
    "Amino acids and protein-related metabolites": (
        "chemical_entity", "amino_acid_composition", "Stage 1 nutrition composition", "amino_acid",
        "Individual amino acid or directly measured protein-related chemical; not an amino-acid score.", "FAO_PROTEIN | FAO_INFOODS",
    ),
    "Fatty-acid and lipid chemical forms": (
        "chemical_entity_or_defined_lipid_aggregate", "fatty_acid_composition", "Stage 1 nutrition composition", "fatty_acid",
        "Individual fatty-acid form or explicitly defined fatty-acid aggregate. Structural/isomeric definitions are retained.", "FAO_FATS | LIPID_MAPS | FAO_INFOODS",
    ),
    "Sterols and sterol expressions": (
        "chemical_entity_or_defined_aggregate", "food_metabolite", "Stage 1 nutrition composition", "sterol",
        "Sterol molecule or explicitly defined sterol aggregate.", "LIPID_MAPS | FAO_INFOODS",
    ),
    "Macronutrient-related alcohol": (
        "chemical_entity", "nutrient_related", "Stage 1 nutrition composition", "ethanol",
        "Ethanol is a directly measurable food constituent and is retained separately from energy.", "FAO_INFOODS | FDC",
    ),
    "Organic acids and small food metabolites": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "organic_acid",
        "Structurally specified, food-occurring small molecule.", "FAO_INFOODS | CHEBI",
    ),
    "Nitrogenous food metabolites": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "nitrogenous_metabolite",
        "Structurally specified food-derived nitrogenous metabolite.", "FAO_INFOODS | CHEBI",
    ),
    "Carotenoids and other isoprenoid food metabolites": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "carotenoid",
        "Structurally specified dietary carotenoid or related isoprenoid.", "FAO_INFOODS | CHEBI",
    ),
    "Polyphenols and phenolic acids": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "phenolic_acid",
        "Structurally specified food polyphenol or phenolic acid, not a total-phenols assay.", "PHENOL_EXPLORER | CHEBI",
    ),
    "Flavonoids and related polyphenols": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "flavonoid",
        "Structurally specified flavonoid or chemical form; glycosides are separate from aglycones.", "PHENOL_EXPLORER | USDA_FLAVONOID | CHEBI",
    ),
    "Anthocyanidins": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "anthocyanin",
        "Structurally specified anthocyanidin or anthocyanin chemical form.", "PHENOL_EXPLORER | CHEBI",
    ),
    "Isoflavones": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "isoflavone",
        "Structurally specified isoflavone or related phytoestrogen.", "USDA_ISOFLAVONE | PHENOL_EXPLORER | CHEBI",
    ),
    "Stilbenes": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "stilbene",
        "Structurally specified food stilbene.", "PHENOL_EXPLORER | CHEBI",
    ),
    "Glucosinolates and organosulfur metabolites": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "organosulfur_or_glucosinolate",
        "Structurally specified glucosinolate, isothiocyanate, or organosulfur food metabolite.", "FAO_INFOODS | CHEBI",
    ),
    "Alkaloids and other food bioactives": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "alkaloid_or_bioactive",
        "Structurally specified alkaloid or non-nutrient food bioactive.", "FAO_INFOODS | CHEBI",
    ),
    "Proanthocyanidin fractions": (
        "defined_polymerization_aggregate", "food_metabolite", "Stage 2 food metabolome", "proanthocyanidin",
        "USDA-defined proanthocyanidin polymerization fraction; it is an aggregate, not one molecule.", "USDA_PA | PHENOL_EXPLORER",
    ),
    "Lignans and related phenylpropanoids": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "lignan",
        "Structurally specified plant lignan or phenylpropanoid food metabolite.", "PHENOL_EXPLORER | CHEBI",
    ),
    "Nucleotides and purine-related food metabolites": (
        "chemical_entity", "food_metabolite", "Stage 2 food metabolome", "nucleotide_or_purine",
        "Structurally specified nucleotide or purine-related food metabolite.", "FAO_INFOODS | CHEBI",
    ),
}


# The previous candidate panel had a residual catch-all family.  It is never
# carried into the final panel: a direct target needs a scientific class.
REVIEWED_DRAFT_FAMILY_OVERRIDES = {
    "Caffeine": "Alkaloids and other food bioactives",
    "Theobromine": "Alkaloids and other food bioactives",
    "Uric acid": "Nucleotides and purine-related food metabolites",
}

NUTRITIONAL_ROLE_BY_FAMILY = {
    "Food matrix and proximate composition": "food_matrix",
    "Macronutrients": "macronutrient_or_defined_nutrition_expression",
    "Macronutrient-related alcohol": "nutrient_related_nonessential_constituent",
    "Carbohydrate forms and small organic metabolites": "nutrient_related_carbohydrate_form",
    "Dietary-fibre chemical forms": "nutrient_related_fibre_form",
    "Minerals and trace elements": "essential_or_conditionally_essential_micronutrient",
    "Vitamins, vitamers, and provitamins": "essential_micronutrient_or_chemical_form",
    "Amino acids and protein-related metabolites": "indispensable_or_dispensable_amino_acid_form",
    "Fatty-acid and lipid chemical forms": "essential_or_nonessential_fatty_acid_form",
    "Sterols and sterol expressions": "lipid_constituent_or_defined_aggregate",
}


# Additions are specifically outside the VMH-centred draft.  Each is a defined
# analyte or a documented analytical fraction, never a fuzzy category label.
RESEARCH_EXTENSION: dict[str, list[str]] = {
    "Carotenoids and other isoprenoid food metabolites": [
        "Lutein", "Zeaxanthin", "Alpha-cryptoxanthin", "Phytoene", "Phytofluene",
        "Beta-apo-8'-carotenal", "Beta-apo-8'-carotenoic acid",
    ],
    "Organic acids and small food metabolites": [
        "Formic acid", "Propionic acid", "Benzoic acid", "Gluconic acid",
        "Pyruvic acid", "Phenyllactic acid", "2-Methylbutyric acid", "Isobutyric acid",
        "Isovaleric acid",
    ],
    "Nitrogenous food metabolites": [
        "Histamine", "Tyramine", "Cadaverine", "Agmatine", "Choline", "Acetylcholine",
        "Beta-alanine", "Anserine", "Carnosine",
    ],
    "Sterols and sterol expressions": [
        "Ergosterol", "Brassicasterol", "Delta-5-avenasterol", "Delta-7-stigmastenol",
    ],
    "Polyphenols and phenolic acids": [
        "3-Caffeoylquinic acid", "4-Caffeoylquinic acid",
        "3,5-Dicaffeoylquinic acid", "3,4-Dicaffeoylquinic acid", "4,5-Dicaffeoylquinic acid",
        "Caftaric acid", "Chicoric acid", "p-Hydroxybenzoic acid", "Gentisic acid",
        "2,5-Dihydroxybenzoic acid", "3,4-Dihydroxybenzoic acid", "3,5-Dihydroxybenzoic acid",
        "Methyl gallate", "Ethyl gallate", "Salicylic acid",
    ],
    "Flavonoids and related polyphenols": [
        "Isorhamnetin", "Fisetin", "Morin", "Galangin", "Tamarixetin", "Rhamnetin",
        "Gallocatechin", "Catechin gallate", "Epigallocatechin gallate", "Gallocatechin gallate",
        "Hesperidin", "Naringin", "Narirutin", "Eriocitrin", "Diosmin", "Diosmetin",
        "Chrysin", "Baicalein", "Tangeretin", "Tectochrysin", "Orientin", "Vitexin",
        "Quercetin 3-O-glucoside", "Quercetin 3-O-rutinoside",
        "Kaempferol 3-O-glucoside", "Kaempferol 3-O-rutinoside",
    ],
    "Anthocyanidins": [
        "Petunidin", "Cyanidin 3-O-glucoside", "Cyanidin 3-O-rutinoside",
        "Delphinidin 3-O-glucoside", "Malvidin 3-O-glucoside", "Pelargonidin 3-O-glucoside",
        "Peonidin 3-O-glucoside", "Petunidin 3-O-glucoside",
    ],
    "Isoflavones": [
        "Biochanin A", "Formononetin", "Coumestrol", "Daidzin", "Genistin", "Glycitin",
    ],
    "Stilbenes": ["Piceid", "Pterostilbene"],
    "Lignans and related phenylpropanoids": [
        "Secoisolariciresinol", "Matairesinol", "Pinoresinol", "Lariciresinol", "Sesamin", "Sesamolin",
    ],
    "Proanthocyanidin fractions": [
        "Proanthocyanidins, monomers", "Proanthocyanidins, dimers", "Proanthocyanidins, trimers",
        "Proanthocyanidins, 4-6 mers", "Proanthocyanidins, 7-10 mers", "Proanthocyanidins, polymers",
    ],
    "Glucosinolates and organosulfur metabolites": [
        "Gluconapin", "Glucobrassicanapin", "Glucocheirolin", "Glucoalyssin", "Glucoberteroin",
        "Glucocapparin", "Glucoibervirin", "Glucoraphasatin", "Sinalbin",
        "S-Allyl-L-cysteine", "Allyl isothiocyanate", "Benzyl isothiocyanate", "Phenethyl isothiocyanate",
    ],
    "Alkaloids and other food bioactives": [
        "Theobromine", "Theacrine", "Hordenine", "Nicotine", "Solanine", "Chaconine",
    ],
    "Nucleotides and purine-related food metabolites": [
        "Adenosine 5'-monophosphate", "Inosine 5'-monophosphate", "Guanosine 5'-monophosphate",
        "Adenosine", "Inosine", "Hypoxanthine", "Xanthine",
    ],
}


def _authority_text(keys: str) -> tuple[str, str]:
    values = [key.strip() for key in keys.split("|")]
    return (
        " | ".join(AUTHORITIES[key]["label"] for key in values),
        " | ".join(AUTHORITIES[key]["url"] for key in values),
    )


def _draft_to_rows(draft: pd.DataFrame) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in draft.to_dict(orient="records"):
        if item["canonical_name"] in EXCLUDE_DRAFT_DISPLAY_NAMES:
            continue
        key = _normalise_axis_name(item["canonical_name"])
        if key in EXCLUDE_DRAFT_KEYS:
            continue
        canonical = SCIENTIFIC_EXACT_CANONICAL.get(
            key,
            EXACT_SYNONYM_CANONICAL.get(key, DISPLAY_CANONICAL.get(key, item["canonical_name"])),
        )
        groups[canonical].append(item)

    rows: list[dict[str, Any]] = []
    for canonical, items in sorted(groups.items()):
        families = {item["axis_family"] for item in items}
        if len(families) != 1:
            raise ValueError(f"Unexpected cross-family exact synonym group for {canonical}: {families}")
        family = REVIEWED_DRAFT_FAMILY_OVERRIDES.get(canonical, next(iter(families)))
        if family == "Other food metabolites":
            raise ValueError(
                f"Direct target {canonical!r} remains in an undefined catch-all family; "
                "assign a reviewed chemical class before rebuilding."
            )
        metadata = FAMILY_METADATA[family]
        original_names = sorted({item["canonical_name"] for item in items})
        aliases = sorted({alias.strip() for item in items for alias in str(item["target_aliases"]).split("|") if alias.strip()})
        basis, urls = _authority_text(metadata[5])
        rows.append({
            "target_axis_id": stable_id("food_prediction_axis", PANEL_VERSION, canonical),
            "canonical_name": canonical,
            "original_axis_names": " | ".join(original_names),
            "axis_family": family,
            "chemical_identity": metadata[0],
            "food_composition_role": metadata[1],
            "nutritional_role": NUTRITIONAL_ROLE_BY_FAMILY.get(family, "nonessential_food_metabolite"),
            "measurement_modality_required": "direct quantified mass expression; denominator and method retained",
            "selection_tier": "nutrition_core" if metadata[2].startswith("Stage 1") else "food_metabolome_extension",
            "recommended_training_stage": metadata[2],
            "mask_family": metadata[3],
            "selection_reason": metadata[4],
            "scientific_basis": basis,
            "authority_urls": urls,
            "source_basis": "VMH-aligned draft, retained only after semantic review",
            "source_match_count_exact_candidate": int(sum(int(item.get("source_match_count_exact", 0)) for item in items)),
            "source_match_count_fuzzy_candidate": int(sum(int(item.get("source_match_count_fuzzy_candidate", 0)) for item in items)),
            "aliases_for_review": " | ".join(aliases),
            "direct_prediction_target": True,
            "numeric_pooling_permitted": False,
            "notes": "Original labels remain in provenance. Any numerical merge requires identity, expression, unit, denominator and method review.",
        })
    return rows


def _extension_to_rows(existing_names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, names in RESEARCH_EXTENSION.items():
        metadata = FAMILY_METADATA[family]
        basis, urls = _authority_text(metadata[5])
        selection_tier = "nutrition_core" if metadata[2].startswith("Stage 1") else "food_metabolome_extension"
        for name in names:
            if name in existing_names:
                continue
            rows.append({
                "target_axis_id": stable_id("food_prediction_axis", PANEL_VERSION, name),
                "canonical_name": name,
                "original_axis_names": "",
                "axis_family": family,
                "chemical_identity": metadata[0],
                "food_composition_role": metadata[1],
                "nutritional_role": NUTRITIONAL_ROLE_BY_FAMILY.get(family, "nonessential_food_metabolite"),
                "measurement_modality_required": "direct quantified mass expression; denominator and method retained",
                "selection_tier": selection_tier,
                "recommended_training_stage": metadata[2],
                "mask_family": metadata[3],
                "selection_reason": metadata[4] + " Included from specialist food-composition literature rather than from VMH breadth alone.",
                "scientific_basis": basis,
                "authority_urls": urls,
                "source_basis": "Food-chemistry extension from authoritative specialist database scope",
                "source_match_count_exact_candidate": 0,
                "source_match_count_fuzzy_candidate": 0,
                "aliases_for_review": _normalise_axis_name(name),
                "direct_prediction_target": True,
                "numeric_pooling_permitted": False,
                "notes": "A source must provide a compatible quantitative expression before this semantic target becomes a numerical label.",
            })
    return rows


def _excluded_expression_registry() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "expression": "Energy (kJ or kcal)",
            "reason": "Derived nutritional expression, not a chemical entity; retain as a separate derived output if needed.",
            "authority_urls": AUTHORITIES["FAO_INFOODS"]["url"],
        },
        {
            "expression": "IU, RAE, DFE, NE, alpha-TE, and other activity equivalents",
            "reason": "Biological activity conversion depends on chemical form; do not pool or model as one mass analyte.",
            "authority_urls": AUTHORITIES["FAO_INFOODS"]["url"] + " | " + AUTHORITIES["NIH_DRI"]["url"],
        },
        {
            "expression": "Carbohydrate by difference or calculated carbohydrate",
            "reason": "Calculated residual, not an independently measured chemical composition target.",
            "authority_urls": AUTHORITIES["FDC"]["url"] + " | " + AUTHORITIES["FAO_INFOODS"]["url"],
        },
        {
            "expression": "Total phenolics by colorimetric assay",
            "reason": "Method-dependent response, not a defined chemical entity or compositional total.",
            "authority_urls": AUTHORITIES["PHENOL_EXPLORER"]["url"],
        },
        {
            "expression": "Contaminants, pesticide residues, heavy metals, processing contaminants and additives",
            "reason": "Safety/exposure layer; scientifically important but not part of the nutrition/metabolome prediction target panel.",
            "authority_urls": AUTHORITIES["FAO_INFOODS"]["url"],
        },
        {
            "expression": "Flavour descriptors, sensory terms, detected-only associations and predicted FooDB entries",
            "reason": "Not compatible quantitative food-composition labels.",
            "authority_urls": AUTHORITIES["FDC"]["url"],
        },
    ])


def _write_report(panel: pd.DataFrame, excluded: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(panel)
    core = int(panel.selection_tier.eq("nutrition_core").sum())
    extension = int(panel.selection_tier.eq("food_metabolome_extension").sum())
    family_counts = panel.groupby(["recommended_training_stage", "axis_family"]).size().reset_index(name="axes")
    family_rows = "\n".join(
        f"| {row.recommended_training_stage} | {row.axis_family} | {int(row.axes)} |"
        for row in family_counts.itertuples(index=False)
    )
    source_rows = "\n".join(
        f"| {item['label']} | {item['url']} |"
        for item in AUTHORITIES.values()
    )
    text = f"""# Final Scientific Food-Composition Prediction-Axis Panel

## Decision

This panel contains **{total} direct prediction targets**: **{core} nutrition-composition core** axes and **{extension} food-metabolome extension** axes. It is not a claim to enumerate the complete food metabolome. No such closed list exists: food chemical diversity is large, method-dependent, and still incompletely measured. Instead, this is a bounded, scientifically defensible target domain for a nutrition foundation model: all entries are either a chemically specified food constituent or a tightly defined aggregate expression used in food composition.

The design is deliberately **not frequency-selected** and is not limited to VMH. VMH was used only as one breadth reference. The final extension adds specialist food-composition chemistry supported by Phenol-Explorer and USDA's flavonoid, isoflavone and proanthocyanidin databases. FAO/INFOODS explicitly treats phytochemicals, bioactive compounds, anti-nutrients and toxic components as distinguishable food-composition classes; this panel includes the first two when the analyte is structurally defined, and separates safety/exposure chemicals from nutrition targets. [FAO food-composition overview]({AUTHORITIES['FAO_INFOODS']['url']})

## What Is a Direct Target?

A target must satisfy all of the following semantic conditions:

1. It is a chemical entity, element/ion, or explicitly defined aggregate with a stable name and definition.
2. It can in principle be represented as a **direct quantified mass expression** for a stated food denominator. This panel does not convert or pool values.
3. Its class is scientifically meaningful for nutrition, food composition or food metabolomics, rather than being a label claim, sensory descriptor or database status.
4. Its original source expression, method, denominator and chemical form remain attached in any future numerical dataset.

Consequently, target selection and numerical trainability are separate decisions. A target with no compatible numerical records is a documented data gap, not evidence that the chemical is unimportant. A name-similarity match is never evidence that two source values can be merged.

## Scientific Basis

- **Nutrition core.** The macronutrient, vitamin and mineral scope follows dietary-reference and food-composition standards. NIH describes the Dietary Reference Intake framework; WHO/FAO provide international vitamin and mineral requirements; FAO identifies indispensable amino-acid composition and essential fatty acids as nutritional concerns. [NIH DRI information]({AUTHORITIES['NIH_DRI']['url']}), [WHO/FAO vitamins and minerals]({AUTHORITIES['WHO_FAO_MICRO']['url']}), [FAO protein quality]({AUTHORITIES['FAO_PROTEIN']['url']}), [FAO/WHO fats]({AUTHORITIES['FAO_FATS']['url']}).
- **Component expression.** FAO/INFOODS component identifiers distinguish an analyte from its expression and method. USDA Foundation Foods similarly documents that proximate values and certain nutrients may be calculated or analytical, with source metadata retained. [FAO/INFOODS]({AUTHORITIES['FAO_INFOODS']['url']}), [USDA Foundation Foods documentation]({AUTHORITIES['FDC']['url']}).
- **Food metabolome extension.** Phenol-Explorer provides food-composition data for 501 polyphenols across six classes; USDA maintains dedicated flavonoid, isoflavone and proanthocyanidin databases. This supports retaining individual structural forms and defined polymerization fractions rather than one vague "polyphenol" number. [Phenol-Explorer]({AUTHORITIES['PHENOL_EXPLORER']['url']}), [USDA flavonoids]({AUTHORITIES['USDA_FLAVONOID']['url']}), [USDA isoflavones]({AUTHORITIES['USDA_ISOFLAVONE']['url']}), [USDA proanthocyanidins]({AUTHORITIES['USDA_PA']['url']}).
- **Chemical classification.** Lipid targets use the LIPID MAPS classification hierarchy; non-lipid small molecules require a chemical-entity review against ChEBI before numerical integration. [LIPID MAPS]({AUTHORITIES['LIPID_MAPS']['url']}), [ChEBI]({AUTHORITIES['CHEBI']['url']}).

## Final Target Composition

| Recommended stage | Axis family | Axes |
|---|---|---:|
{family_rows}
| **Total** | **All direct targets** | **{total}** |

### Stage 1: Nutrition Composition

Stage 1 covers food matrix/proximate composition, defined macronutrient and carbohydrate expressions, minerals, vitamins/vitamers, amino acids, fatty-acid forms and sterols. This is the shared nutritional composition space: it contains the components with direct nutrition roles and the molecular forms needed to avoid treating their biological equivalents as the same measured substance.

### Stage 2: Food Metabolome Extension

Stage 2 covers structurally defined small food metabolites: organic acids, nitrogenous metabolites, carotenoids, polyphenols, flavonoids, anthocyanins, isoflavones, lignans, proanthocyanidin fractions, glucosinolates, organosulfurs, alkaloids and purine/nucleotide metabolites. It is separate because these components are chemically heterogeneous and often have source-specific analytical coverage. Separation is architectural, not a claim that they are biologically unimportant.

## Deliberate Exclusions From Direct Prediction

| Expression or class | Reason |
|---|---|
"""
    for row in excluded.itertuples(index=False):
        text += f"| {row.expression} | {row.reason} |\n"
    text += f"""

## Required Implementation Guardrails

- Treat missing as **unknown**, never as zero. Preserve zero, trace, censored, range and measured values separately.
- Pool no values merely because this registry lists one canonical target. Require compatible chemical identity, expression, unit, denominator and method.
- Retain defined totals and individual members, but apply family-aware masks. For example, do not reveal total fat or a fatty-acid sum while asking the model to reconstruct its algebraically recoverable members.
- Activity equivalents, energy and calculated residuals may be released as derived outputs but must not enter the direct-mass target loss.
- Do not add a target from a FooDB detected/expected/predicted association without a compatible quantitative source measurement.

## Files

- `final_prediction_axis_registry.csv`: the complete, axis-level final target list.
- `excluded_non_direct_expressions.csv`: important composition expressions retained outside the direct target loss.
- `axis_family_summary.csv`: target counts by scientific family and proposed training stage.
- `panel_manifest.json`: deterministic build metadata and source locations.

## Authority Source Register

| Source | URL |
|---|---|
{source_rows}
"""
    path.write_text(text, encoding="utf-8")


def _write_axis_list(panel: pd.DataFrame, path: Path) -> None:
    """Write the complete target list in a review-friendly, non-tabular form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Prediction Axis List",
        "",
        "This is the human-readable view of `final_prediction_axis_registry.csv`. "
        "Each item is a direct target semantic concept, not permission to pool numerical values across sources.",
        "",
    ]
    for stage, stage_frame in panel.groupby("recommended_training_stage", sort=False):
        lines.extend([f"## {stage}", ""])
        for family, family_frame in stage_frame.groupby("axis_family", sort=True):
            lines.extend([f"### {family} ({len(family_frame)})", ""])
            for row in family_frame.itertuples(index=False):
                lines.append(
                    f"- **{row.canonical_name}**: {row.chemical_identity}; "
                    f"mask family `{row.mask_family}`."
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chinese_report(panel: pd.DataFrame, path: Path) -> None:
    """Write a concise Chinese decision record while keeping chemical labels in English."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(panel)
    core = int(panel.selection_tier.eq("nutrition_core").sum())
    extension = int(panel.selection_tier.eq("food_metabolome_extension").sum())
    summary = panel.groupby(["recommended_training_stage", "axis_family"]).size().reset_index(name="axes")
    rows = "\n".join(
        f"| {row.recommended_training_stage} | {row.axis_family} | {int(row.axes)} |"
        for row in summary.itertuples(index=False)
    )
    text = f"""# 最终食品营养与食品代谢物预测轴面板

## 结论

本版本选择 **{total} 个直接预测轴**：**{core} 个营养组成核心轴**和 **{extension} 个食品代谢物扩展轴**。这不是“完整食物代谢组”的宣称，也不是按当前某个数据库的出现次数筛选的列表。它是一个用于 nutrition foundation model 的、有明确边界的食品组成目标域：每一个直接预测轴必须是明确化学实体、元素/离子，或有严格分析定义的聚合表达。

VMH 只作为营养广度参考，不决定最终清单。面板额外采用 Phenol-Explorer 的食品多酚组成范围，以及 USDA 的黄酮、异黄酮和原花青素专业数据库作为食品化学扩展依据。FAO/INFOODS 明确将宏量/微量营养、phytochemicals、bioactive compounds、anti-nutrients 和 toxic components 区分处理；这里纳入前两类的结构明确成分，安全/暴露化学物保留在独立层，不作为本模型的营养预测目标。[FAO food-composition overview]({AUTHORITIES['FAO_OVERVIEW']['url']})

## 为什么这样分类

1. **Stage 1: nutrition composition** 包括食品基质、宏量和碳水化合物表达、矿物质、维生素及其化学形式、氨基酸、脂肪酸和甾醇。这些是食品营养组成及其分子形式；不能因为它们有相同的生物活性或同属一种营养素而强行合并。
2. **Stage 2: food metabolome** 包括有机酸、含氮代谢物、类胡萝卜素、多酚、黄酮、花青素、异黄酮、木脂素、原花青素聚合度分级、硫代葡萄糖苷/有机硫、植物生物碱和嘌呤/核苷酸。它们是食品中可定义的小分子，但测量方法和来源覆盖更异质，因此与营养核心分阶段建模。
3. **不进入直接回归损失** 的包括能量、IU/RAE/DFE/NE/alpha-TE 等活性当量、by-difference 计算残差、无明确化学身份的总酚测定值、污染物/农残/添加剂和仅检测到而没有可比较定量值的 FooDB 关联。它们可以保存为 provenance 或独立任务，但不能伪装成统一质量浓度的分子标签。

## 分类统计

| 建议训练阶段 | 化学/组成类别 | 轴数 |
|---|---|---:|
{rows}
| **总计** | **直接预测轴** | **{total}** |

## 科学依据

- 宏量营养、维生素、矿物质和微量元素的范围由 NIH Dietary Reference Intakes 及 WHO/FAO 的维生素和矿物质需求文件约束。[NIH DRI]({AUTHORITIES['NIH_DRI']['url']}), [WHO/FAO]({AUTHORITIES['WHO_FAO_MICRO']['url']})
- 氨基酸和脂肪酸以 FAO 对蛋白质质量、必需氨基酸和脂肪酸的定义为依据；脂质的结构分类采用 LIPID MAPS。[FAO protein]({AUTHORITIES['FAO_PROTEIN']['url']}), [FAO fats]({AUTHORITIES['FAO_FATS']['url']}), [LIPID MAPS]({AUTHORITIES['LIPID_MAPS']['url']})
- 多酚及其化学形式以 Phenol-Explorer 的逐化合物食品组成信息为依据；黄酮、异黄酮、原花青素的分析定义由相应 USDA 专业数据库支持。[Phenol-Explorer]({AUTHORITIES['PHENOL_EXPLORER']['url']}), [USDA flavonoids]({AUTHORITIES['USDA_FLAVONOID']['url']}), [USDA isoflavones]({AUTHORITIES['USDA_ISOFLAVONE']['url']}), [USDA proanthocyanidins]({AUTHORITIES['USDA_PA']['url']})
- 任何数值合并前必须核对分析物身份、表达方式、单位、分母和分析方法；FAO/INFOODS 的 component identifier 体系用于保存这些区别。[FAO/INFOODS]({AUTHORITIES['FAO_INFOODS']['url']})

## 使用限制

- `missing` 永远是未知，不能转换为 0；已知 0、trace、`<LOD`、范围和普通数值必须分别保存。
- 该清单的“同一预测轴”仅代表同一目标语义，不自动授权跨数据库的数值合并。
- 总量与其子项需要 family-aware masking，避免未遮蔽的总量代数泄漏出被遮蔽的子项。
- 每个轴在成为训练标签前还必须有兼容的定量来源；没有数据的轴是 data gap，而不是从科学面板删除的理由。

完整机器可读清单在 `final_prediction_axis_registry.csv`，逐轴的人类可读版本在 `FINAL_PREDICTION_AXIS_LIST.md`。
"""
    path.write_text(text, encoding="utf-8")


def build_final_prediction_axis_panel(
    *,
    draft_registry_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    if not draft_registry_path.exists():
        raise FileNotFoundError(f"VMH-aligned draft registry was not found: {draft_registry_path}")
    draft = pd.read_csv(draft_registry_path)
    required = {"canonical_name", "axis_family", "target_aliases"}
    missing = required - set(draft.columns)
    if missing:
        raise ValueError(f"Draft registry missing required fields: {sorted(missing)}")

    base_rows = _draft_to_rows(draft)
    panel = pd.DataFrame(base_rows)
    extension = pd.DataFrame(_extension_to_rows(set(panel.canonical_name)))
    panel = pd.concat([panel, extension], ignore_index=True)
    if panel.canonical_name.duplicated().any():
        duplicated = panel.loc[panel.canonical_name.duplicated(keep=False), "canonical_name"].tolist()
        raise ValueError(f"Final panel has unresolved duplicate canonical names: {duplicated}")
    if panel.target_axis_id.duplicated().any():
        raise ValueError("Final panel has duplicate stable IDs.")
    panel = panel.sort_values(["recommended_training_stage", "axis_family", "canonical_name"]).reset_index(drop=True)
    excluded = _excluded_expression_registry()
    summary = (
        panel.groupby(["selection_tier", "recommended_training_stage", "axis_family", "chemical_identity", "food_composition_role"], dropna=False)
        .size().reset_index(name="axis_count")
        .sort_values(["selection_tier", "recommended_training_stage", "axis_family"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(panel, output_dir / "final_prediction_axis_registry.csv")
    write_csv(excluded, output_dir / "excluded_non_direct_expressions.csv")
    write_csv(summary, output_dir / "axis_family_summary.csv")
    manifest = {
        "panel_version": PANEL_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "draft_registry_path": str(draft_registry_path),
        "direct_prediction_axes": int(len(panel)),
        "nutrition_core_axes": int(panel.selection_tier.eq("nutrition_core").sum()),
        "food_metabolome_extension_axes": int(panel.selection_tier.eq("food_metabolome_extension").sum()),
        "excluded_expression_classes": int(len(excluded)),
        "scientific_design_only": True,
        "numerical_value_merge_permitted": False,
        "source_authorities": AUTHORITIES,
    }
    write_json(manifest, output_dir / "panel_manifest.json")
    _write_report(panel, excluded, report_dir / "FINAL_SCIENTIFIC_PREDICTION_AXIS_PANEL.md")
    _write_axis_list(panel, report_dir / "FINAL_PREDICTION_AXIS_LIST.md")
    _write_chinese_report(panel, report_dir / "FINAL_SCIENTIFIC_PREDICTION_AXIS_PANEL_ZH.md")
    return manifest

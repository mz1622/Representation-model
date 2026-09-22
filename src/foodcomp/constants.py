"""Versioned policy constants for the scientific composition build."""

from __future__ import annotations

DATASET_VERSION = "scientific_food_composition_v1"
AS_OF_DATE = "2026-09-03"
RANDOM_SEED = 42
# Locked v1 benchmark size chosen before model evaluation. USDA Foundation
# contributes 394 source-holdout concepts; 1,032 additional concepts are
# selected as complete family blocks for a 1,426-concept validation release.
FAMILY_HOLDOUT_CONCEPT_TARGET = 1_032

MAIN_BASIS = "g/100 g edible portion, fresh weight"
MAIN_MODALITY = "mass_fraction_fresh_weight"

# Exact conversion factors to grams. A value is converted only when its
# denominator is explicitly compatible with 100 g edible fresh food.
MASS_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "mg": 1e-3,
    "milligram": 1e-3,
    "milligrams": 1e-3,
    "ug": 1e-6,
    "mcg": 1e-6,
    "microgram": 1e-6,
    "micrograms": 1e-6,
}

CENSORED_TOKENS = {
    "tr", "trace", "<lod", "<loq", "lod", "loq", "nd", "not detected",
    "below detection limit", "below quantification limit", "not quantified",
}
MISSING_TOKENS = {"", "-", "--", "n", "na", "n/a", "missing", "unknown"}

# Processing/state terms are retained in the food concept, but removed only
# when constructing conservative split-family blocks.
PROCESSING_TERMS = {
    "raw", "cooked", "boiled", "roasted", "fried", "baked", "grilled",
    "steamed", "dried", "dehydrated", "frozen", "canned", "drained",
    "fresh", "smoked", "pickled", "fermented", "pasteurized", "powdered",
    "ground", "chopped", "peeled", "unpeeled", "with", "without",
}

NUTRITIONAL_ROLE_BY_TAG = {
    "PROCNT": "macronutrient",
    "PROT": "macronutrient",
    "FAT": "macronutrient",
    "FATCE": "macronutrient",
    "CHOCDF": "macronutrient",
    "CHOT": "macronutrient",
    "CHOAVL": "macronutrient",
    "FIBTG": "macronutrient",
    "FIBT": "macronutrient",
    "WATER": "other_nutritional_component",
    "ASH": "other_nutritional_component",
    "ALC": "other_nutritional_component",
    "CHOLN": "essential_nutrient_choline",
}

AMINO_ACID_TAGS = {
    "ALA", "ARG", "ASP", "ASPN", "CYS", "CYST", "GLU", "GLN", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
AMINO_ACID_NAMES = {
    "alanine", "arginine", "aspartic acid", "asparagine", "cysteine", "cystine",
    "cystine plus cysteine", "glutamic acid", "glutamine", "glycine", "histidine",
    "isoleucine", "leucine", "lysine", "methionine", "phenylalanine", "proline",
    "serine", "threonine", "tryptophan", "tyrosine", "valine",
}

VITAMIN_TAG_PREFIXES = (
    "VITA", "RETOL", "CART", "THIA", "RIBF", "NIA", "VITB6",
    "FOL", "VITB12", "VITC", "VITD", "TOCP", "VITE", "VITK",
    "PANTAC", "BIOT",
)

MINERAL_TAGS = {
    "CA", "FE", "MG", "P", "K", "NA", "ZN", "CU", "MN", "SE",
    "I", "CR", "MO", "CLD", "F", "CO",
}

FAO_SCREENING_QUESTIONS = (
    ("Q1", "Values are expressed per 100 g edible fresh weight, with per 100 mL accepted where clearly stated.", 20),
    ("Q2", "Component denominators and SI-compatible units are documented.", 20),
    ("Q3", "INFOODS component identifiers or tagnames are provided.", 20),
    ("Q4", "The sixteen FAO/INFOODS common food components are represented.", 20),
    ("Q5", "Food names and descriptions are sufficiently specified.", 10),
    ("Q6", "Both raw and consumed or cooked foods are represented.", 10),
    ("Q7", "Data sources are documented at food or value level.", 10),
    ("Q8", "The database is available in an importable electronic format.", 10),
)

SOURCE_LAYER = {
    "usda_sr_legacy": "primary_reference",
    "usda_foundation": "locked_source_validation",
    "cnf": "primary_reference",
    "frida": "primary_reference",
    "ciqual": "primary_reference",
    "afcd": "primary_reference",
    "cofid": "primary_reference",
    "foodb": "specialist_and_lineage_archive",
    "fndds": "calculated_dish_auxiliary",
}

SOURCE_INDEPENDENCE_RULES = {
    "usda_sr_legacy": "direct_or_derivation_coded",
    "usda_foundation": "analytical_reference",
    "cnf": "nutrient_source_code",
    "frida": "source_record_and_source_food",
    "ciqual": "value_source_links",
    "afcd": "derivation_field",
    "cofid": "main_data_references",
    "foodb": "citation_lineage",
    "fndds": "derived_auxiliary_only",
}

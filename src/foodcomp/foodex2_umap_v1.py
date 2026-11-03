"""Conservative FoodEx2 level-1 labels for representation visualisation.

The mapping deliberately uses only source-supplied food groups (or their
published category codes).  It never uses food names, nutrient values,
embeddings, or UMAP coordinates to infer a category.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


GRAINS = "Grains and grain-based products"
VEGETABLES = "Vegetables and vegetable products"
STARCHY = "Starchy roots or tubers and products thereof, sugar plants"
LEGUMES = "Legumes, nuts, oilseeds and spices"
FRUIT = "Fruit and fruit products"
MEAT = "Meat and meat products"
FISH = "Fish, seafood, amphibians, reptiles and invertebrates"
DAIRY = "Milk and dairy products"
EGGS = "Eggs and egg products"
SUGAR = "Sugar, confectionery and water-based sweet desserts"
FATS = "Animal and vegetable fats and oils"
JUICES = "Fruit and vegetable juices and nectars"
COFFEE_TEA = "Coffee, cocoa, tea and infusions"
ALCOHOL = "Alcoholic beverages"
COMPOSITE = "Composite dishes"
SEASONING = "Seasoning, sauces and condiments"
WATER = "Water and water-based beverages"
SPECIAL = "Products for non-standard diets, food imitates and food supplements or fortifying agents"
YOUNG = "Food products for young population"
ADDITIVES = "Additives, flavours, baking and processing aids"
HUMAN_MILK = "Human milk"


FOODEX2_LEVEL1 = (
    GRAINS, VEGETABLES, STARCHY, LEGUMES, FRUIT, MEAT, FISH, DAIRY, EGGS,
    SUGAR, FATS, JUICES, COFFEE_TEA, ALCOHOL, COMPOSITE, SEASONING, WATER,
    SPECIAL, YOUNG, ADDITIVES, HUMAN_MILK,
)


DISPLAY_NAMES = {
    GRAINS: "Grains",
    VEGETABLES: "Vegetables",
    STARCHY: "Starchy roots/tubers",
    LEGUMES: "Legumes/nuts/spices",
    FRUIT: "Fruit",
    MEAT: "Meat",
    FISH: "Fish/seafood",
    DAIRY: "Milk/dairy",
    EGGS: "Eggs",
    SUGAR: "Sugar/confectionery",
    FATS: "Fats/oils",
    JUICES: "Fruit/vegetable juices",
    COFFEE_TEA: "Coffee/cocoa/tea",
    ALCOHOL: "Alcoholic beverages",
    COMPOSITE: "Composite dishes",
    SEASONING: "Seasonings/sauces",
    WATER: "Water",
    SPECIAL: "Special-diet products",
    YOUNG: "Young-population foods",
    ADDITIVES: "Additives/processing aids",
    HUMAN_MILK: "Human milk",
}


MEXT = {
    "1": GRAINS, "2": STARCHY, "3": SUGAR, "4": LEGUMES,
    "5": LEGUMES, "6": VEGETABLES, "7": FRUIT, "8": VEGETABLES,
    "9": VEGETABLES, "10": FISH, "11": MEAT, "12": EGGS,
    "13": DAIRY, "14": FATS, "15": SUGAR, "17": SEASONING,
    "18": COMPOSITE,
}


AFCD = {
    "12": GRAINS, "13": GRAINS, "14": FATS, "15": FISH,
    "16": FRUIT, "17": EGGS, "18": MEAT, "19": DAIRY,
    "21": COMPOSITE, "22": LEGUMES, "23": SEASONING,
    "24": VEGETABLES, "25": LEGUMES, "27": SUGAR, "28": SUGAR,
    "29": ALCOHOL, "30": SPECIAL, "32": YOUNG, "33": FISH,
}


COFID_PREFIX = {
    "A": GRAINS, "B": DAIRY, "C": EGGS, "D": VEGETABLES,
    "F": FRUIT, "G": LEGUMES, "H": LEGUMES, "J": FISH,
    "M": MEAT, "O": FATS, "S": SUGAR, "W": SEASONING,
}


LESOTHO = {
    "01": GRAINS, "03": DAIRY, "04": EGGS, "05": VEGETABLES,
    "06": LEGUMES, "07": STARCHY, "09": FATS, "10": SEASONING,
    "11": SUGAR, "12": YOUNG, "13": COFFEE_TEA, "14": ALCOHOL,
}


PHYFOODCOMP = {
    "01": GRAINS, "02": STARCHY, "03": LEGUMES, "04": VEGETABLES,
    "05": FRUIT, "06": LEGUMES, "07": MEAT, "08": FISH,
    "10": FISH, "11": DAIRY, "12": FATS, "14": SUGAR,
    "15": SEASONING, "16": YOUNG, "17": SPECIAL,
    "18": ADDITIVES, "19": COMPOSITE,
}


def normalise_group(value: object) -> str:
    """Normalise a source category label without interpreting food identity."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_EFSA_EXACT = {normalise_group(value): value for value in FOODEX2_LEVEL1}


_TEXT_EXACT = {
    # Categories that identify one FoodEx2 level-1 branch without consulting
    # the individual food name.
    "cereals and cereal products": GRAINS,
    "cereals grains and pasta": GRAINS,
    "cereal grains and pasta": GRAINS,
    "breakfast cereals": GRAINS,
    "baked products": GRAINS,
    "baking goods": GRAINS,
    "grains and grain products": GRAINS,
    "bakery and breakfast cereals": GRAINS,
    "vegetables": VEGETABLES,
    "vegetables and vegetable products": VEGETABLES,
    "vegetables and products": VEGETABLES,
    "vegetable products": VEGETABLES,
    "gourds": VEGETABLES,
    "starchy roots and other starchy plant foods": STARCHY,
    "starchy roots tubers and products": STARCHY,
    "legumes and legume products": LEGUMES,
    "legumes nuts and seeds": LEGUMES,
    "legumes nut seeds": LEGUMES,
    "nut and seed products": LEGUMES,
    "nuts and seeds": LEGUMES,
    "herbs and spices": LEGUMES,
    "spices and herbs": LEGUMES,
    "pulses": LEGUMES,
    "soy": LEGUMES,
    "nuts": LEGUMES,
    "fruits": FRUIT,
    "fruit": FRUIT,
    "beef products": MEAT,
    "pork products": MEAT,
    "poultry products": MEAT,
    "lamb veal and game products": MEAT,
    "lamb veal and game": MEAT,
    "sausages and luncheon meats": MEAT,
    "meat other animal flesh and products": MEAT,
    "finfish and shellfish products": FISH,
    "aquatic foods": FISH,
    "finfish shellfish aquatic animals and products": FISH,
    "milk and milk products": DAIRY,
    "milk and products": DAIRY,
    "milk products": DAIRY,
    "dairy products": DAIRY,
    "eggs": EGGS,
    "eggs and products": EGGS,
    "sweets": SUGAR,
    "confectioneries": SUGAR,
    "sugar and confectionery": SUGAR,
    "sweetened snacks and desserts": SUGAR,
    "added sugars": SUGAR,
    "fats and oils": FATS,
    "fats oil": FATS,
    "added fats": FATS,
    "dishes": COMPOSITE,
    "mixed dishes": COMPOSITE,
    "composites mixed food groups": COMPOSITE,
    "starters and dishes": COMPOSITE,
    "soups sauces and gravies": SEASONING,
    "condiments and spices": SEASONING,
    "baby food": YOUNG,
    "baby foods": YOUNG,
    "babyfoods": YOUNG,
}


def direct_foodex2_label(source_key: object, food_group: object) -> tuple[str, str]:
    """Return ``(level1, provenance)`` from a native source category only."""
    source = str(source_key)
    raw = "" if food_group is None or pd.isna(food_group) else str(food_group).strip()
    group = normalise_group(food_group)
    if not group:
        return "", "unlabelled"

    if source == "efsa_eu_fcdb_2013" and group in _EFSA_EXACT:
        return _EFSA_EXACT[group], "native_foodex2_level1"
    if source == "mext_japan_2023":
        label = MEXT.get(str(int(float(raw))) if re.fullmatch(r"\d+(?:\.0+)?", raw) else raw)
        return (label, "source_code_crosswalk") if label else ("", "unlabelled")
    if source == "afcd":
        label = AFCD.get(raw[:2])
        return (label, "source_code_crosswalk") if label else ("", "unlabelled")
    if source == "cofid":
        label = COFID_PREFIX.get(raw[:1].upper())
        return (label, "source_code_crosswalk") if label else ("", "unlabelled")
    if source == "lesotho_fct_2006":
        label = LESOTHO.get(raw.zfill(2)[:2])
        return (label, "source_code_crosswalk") if label else ("", "unlabelled")
    if source == "phyfoodcomp_1_0":
        label = PHYFOODCOMP.get(raw.zfill(2)[:2])
        return (label, "source_code_crosswalk") if label else ("", "unlabelled")

    if group in _EFSA_EXACT:
        return _EFSA_EXACT[group], "native_group_crosswalk"
    label = _TEXT_EXACT.get(group, "")
    return (label, "native_group_crosswalk") if label else ("", "unlabelled")


def assign_foodex2_labels(profiles: pd.DataFrame) -> pd.DataFrame:
    """Label profiles, then make only unanimous exact-name transfers.

    The returned frame is index-aligned with ``profiles`` and contains no food
    names or model-derived fields.
    """
    direct = [
        direct_foodex2_label(source, group)
        for source, group in zip(profiles["source_key"], profiles["food_group"])
    ]
    result = pd.DataFrame(direct, columns=["foodex2_level1", "label_provenance"], index=profiles.index)
    group_id = profiles["exact_name_group_id"].fillna("").astype(str)
    labelled = result["foodex2_level1"].ne("") & group_id.ne("")
    candidates = pd.DataFrame({"group": group_id[labelled], "label": result.loc[labelled, "foodex2_level1"]})
    unanimous = candidates.groupby("group")["label"].agg(lambda values: values.iloc[0] if values.nunique() == 1 else "")
    # Transfer is a recovery mechanism for a genuinely missing native group,
    # not a way to override a present-but-broad or ambiguous source category.
    blank_native_group = profiles["food_group"].isna() | profiles["food_group"].astype(str).str.strip().eq("")
    missing = (
        result["foodex2_level1"].eq("")
        & blank_native_group
        & group_id.isin(unanimous.index)
    )
    transferred = group_id[missing].map(unanimous).fillna("")
    accepted = missing.copy()
    accepted.loc[missing] = transferred.ne("")
    result.loc[accepted, "foodex2_level1"] = group_id[accepted].map(unanimous)
    result.loc[accepted, "label_provenance"] = "exact_name_unanimous_transfer"
    result["display_category"] = result["foodex2_level1"].map(DISPLAY_NAMES).fillna("Unlabelled")
    return result

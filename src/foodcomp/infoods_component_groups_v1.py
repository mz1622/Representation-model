"""Conservative FAO/INFOODS component groups for sensitivity analysis.

The mapping is deliberately incomplete.  It follows the component hierarchy
published in the INFOODS/EuroFIR workshop report and uses explicit canonical
names from the frozen axis table.  Unlisted axes remain unclassified.
"""
from __future__ import annotations

import re

import pandas as pd


VERSION = "infoods_component_groups_v1"
AUTHORITY = (
    "FAO/INFOODS and EuroFIR AISBL, Report of the INFOODS/EuroFIR Workshop "
    "on Component Nomenclature, Annex 1 component hierarchy"
)
AUTHORITY_URL = (
    "https://www.fao.org/fileadmin/templates/food_composition/documents/"
    "upload/9IFDCReportfinal.pdf"
)
TAGNAME_URL = (
    "https://www.fao.org/infoods/infoods/standards-guidelines/"
    "food-component-identifiers-tagnames/en/"
)


AMINO_ACIDS = {
    "Alanine", "Arginine", "Aspartic acid", "Cystine", "Glutamic acid",
    "Glycine", "Histidine", "Hydroxyproline", "Isoleucine", "Leucine",
    "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine",
    "Threonine", "Tryptophan", "Tyrosine", "Valine",
}
SUGARS = {
    "Fructose", "Galactose", "Glucose", "Lactose", "Maltose", "Sucrose",
    "Sugars, total",
}
DIETARY_FIBRE = {
    "Dietary fibre, total", "Insoluble dietary fibers", "Cellulose", "Lignin",
}
MINERALS = {
    "Calcium", "Chloride", "Chromium", "Copper", "Iron", "Magnesium",
    "Manganese", "Molybdenum", "Phosphorus", "Potassium", "Selenium, Se",
    "Sodium", "Sulfur", "Zinc",
}
ORGANIC_ACIDS = {
    "Acetic acid", "Benzoic acid", "Citric acid", "Lactic acid", "Malic acid",
    "Oxalic acid", "Quinic acid", "Tartaric acid",
}
VITAMINS = {
    "Ascorbic acid (vitamin C)", "Biotin", "Carotene, alpha", "Carotene, beta",
    "Cholecalciferol (vitamin D3)", "Cobalamin (vitamin B12)",
    "Cryptoxanthin, beta", "Folate, food", "Folate, total", "Folic acid",
    "Lutein", "Lutein + zeaxanthin", "Lycopene", "Menaquinone-4", "Niacin",
    "Nicotinic acid", "Pantothenic acid (vitamin B5)", "Retinol",
    "Riboflavin (vitamin B2)", "Thiamin (vitamin B1)", "Tocopherol, beta",
    "Tocopherol, delta", "Tocopherol, gamma", "Tocotrienol, alpha",
    "Tocotrienol, beta", "Tocotrienol, delta", "Tocotrienol, gamma",
    "Vitamin B6 - Pyridoxin", "Vitamin D (D2 + D3)",
    "Vitamin D2 (ergocalciferol)", "Zeaxanthin",
}


def component_group(name: str) -> str | None:
    """Return one selected INFOODS hierarchy group or ``None``."""
    name = str(name)
    if name in AMINO_ACIDS:
        return "Amino acids"
    if name in SUGARS:
        return "Sugars"
    if name in DIETARY_FIBRE:
        return "Dietary fibre"
    if name in MINERALS:
        return "Minerals"
    if name in ORGANIC_ACIDS:
        return "Organic acids"
    if name in VITAMINS:
        return "Vitamins"
    if name.startswith("Fatty acids,") or re.search(r"\(\d{1,2}:\d", name):
        return "Fatty acids"
    if name == "Stearidonic acid":
        return "Fatty acids"
    return None


def group_frame(axes: pd.DataFrame) -> pd.DataFrame:
    """Create an auditable, incomplete axis-to-group table."""
    result = axes[["axis_index", "canonical_name"]].copy()
    result["infoods_group"] = result.canonical_name.map(component_group)
    result["classified"] = result.infoods_group.notna()
    result["authority"] = AUTHORITY
    result["authority_url"] = AUTHORITY_URL
    return result


__all__ = [
    "VERSION", "AUTHORITY", "AUTHORITY_URL", "TAGNAME_URL",
    "component_group", "group_frame",
]

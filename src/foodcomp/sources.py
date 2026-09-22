"""Source-specific readers that preserve original identifiers and semantics."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import (
    COMPONENT_COLUMNS,
    FOOD_COLUMNS,
    MEASUREMENT_COLUMNS,
    apply_value_and_unit_normalization,
    coerce_schema,
)
from .util import normalize_text, require_columns, stable_id, write_csv
from .reference_rules import COFID_ANALYTICAL_REFERENCES, NORWAY_ANALYTICAL_REFERENCES
from .ontology import parse_infoods_tagnames
from .label_evidence import foundation_result_links, gate_reference_evidence, reference_catalogs


@dataclass
class SourceBundle:
    foods: pd.DataFrame
    components: pd.DataFrame
    measurements: pd.DataFrame


def _food_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return coerce_schema(pd.DataFrame(rows), FOOD_COLUMNS)


def _component_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return coerce_schema(pd.DataFrame(rows), COMPONENT_COLUMNS)


def _measurement_frame(rows: pd.DataFrame, implicit_per_100g: bool = False) -> pd.DataFrame:
    rows = coerce_schema(rows, MEASUREMENT_COLUMNS)
    rows = apply_value_and_unit_normalization(rows, implicit_per_100g=implicit_per_100g)
    return coerce_schema(rows, MEASUREMENT_COLUMNS)


def _reason(existing: Any, new_reason: str) -> str:
    old = "" if existing is None or pd.isna(existing) else str(existing).strip()
    return "; ".join(x for x in (old, new_reason) if x)


def _apply_training_gate(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_eligible = frame["main_value_eligible"].fillna(False).copy()
    strict_reference = (
        frame["independent_evidence"].fillna(False)
        & frame["quality_tier"].isin(["A", "B"])
        & frame["data_layer"].isin(["primary_reference", "locked_source_validation"])
    )
    curated_training = (
        frame["data_layer"].eq("curated_reference_training")
        & frame["quality_tier"].eq("C")
        & frame["quality_evidence"].fillna("").ne("")
    )
    eligible = numeric_eligible & (strict_reference | curated_training)
    frame["main_value_eligible"] = eligible
    frame["strict_validation_eligible"] = frame["strict_validation_eligible"].where(
        frame["strict_validation_eligible"].notna(), numeric_eligible & strict_reference
    ).astype(bool) & numeric_eligible
    failures = [
        (frame["value_status"].eq("invalid_mass_fraction_above_100"), "mass_fraction_exceeds_100_g_per_100_g"),
        (~frame["independent_evidence"].fillna(False), "non_independent_or_derived_value"),
        (~frame["quality_tier"].isin(["A", "B"]), "insufficient_value_level_quality"),
        (~frame["data_layer"].isin(["primary_reference", "locked_source_validation"]), "non_primary_data_layer"),
    ]
    for mask, message in failures:
        target = mask & ~eligible & frame["exclusion_reason"].fillna("").eq("")
        frame.loc[target, "exclusion_reason"] = message
    return frame


def _usda_infoods_map(cnf_dir: Path) -> dict[str, str]:
    nutrients = pd.read_csv(cnf_dir / "Nutrient_Name.csv", dtype={"Nutrient_Code": str}, keep_default_na=False, na_values=["", "NULL"])
    return {
        str(code).strip(): str(tag).strip()
        for code, tag in zip(nutrients["Nutrient_Code"], nutrients["Tagname"], strict=True)
        if pd.notna(tag) and str(tag).strip()
    }


def load_usda_sr(root: Path) -> SourceBundle:
    source_key = "usda_sr_legacy"
    base = root / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04"
    food = pd.read_csv(base / "food.csv")
    legacy_ids = pd.read_csv(base / "sr_legacy_food.csv")
    food = food.merge(legacy_ids, on="fdc_id", how="left")
    categories = pd.read_csv(base / "food_category.csv").rename(columns={"id": "food_category_id", "description": "food_group"})
    food = food.merge(categories[["food_category_id", "food_group"]], on="food_category_id", how="left")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.fdc_id),
            "source_key": source_key, "source_food_id": row.fdc_id,
            "original_name": row.description, "food_group": row.food_group,
            "food_type": row.data_type, "recipe_status": "not_reported",
            "brand_status": "not_reported", "edible_status": "edible_food_assumed_by_source",
            "source_lineage_id": f"USDA:NDB:{int(row.NDB_number)}" if pd.notna(row.NDB_number) else f"USDA:FDC:{row.fdc_id}",
            "source_record_url": f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{row.fdc_id}/nutrients",
        }
        for row in food.itertuples(index=False)
    ])

    nutrient = pd.read_csv(base / "nutrient.csv", dtype={"nutrient_nbr": str})
    tags = _usda_infoods_map(root / "data/raw/cnf_2026/extracted")
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.id),
            "source_key": source_key, "source_component_id": row.id,
            "original_name": row.name, "original_unit": row.unit_name,
            "infoods_tag": tags.get(str(row.nutrient_nbr).split(".")[0], ""),
            "source_definition": f"USDA nutrient number {row.nutrient_nbr}",
            "source_component_group": "USDA nutrient",
            "authority_status": "official_source_identifier",
        }
        for row in nutrient.itertuples(index=False)
    ])

    derivation = pd.read_csv(base / "food_nutrient_derivation.csv")
    source = pd.read_csv(base / "food_nutrient_source.csv").rename(columns={"id": "source_id", "description": "source_description"})
    derivation = derivation.merge(source[["source_id", "source_description"]], on="source_id", how="left")
    values = pd.read_csv(base / "food_nutrient.csv", low_memory=False).merge(
        derivation[["id", "code", "description", "source_id", "source_description"]].rename(columns={"id": "derivation_id", "description": "derivation_description"}),
        on="derivation_id", how="left",
    )
    quality = values["source_id"].map({1: "B", 10: "C"}).fillna("D")
    independent = values["source_id"].isin([1, 10])
    assumed_zero = values["source_id"].eq(5) | values["code"].astype(str).str.contains("assumed", case=False, na=False)
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, x) for x in values["id"]],
        "source_key": source_key, "source_measurement_id": values["id"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in values["fdc_id"]],
        "component_observation_id": [stable_id("compobs", source_key, x) for x in values["nutrient_id"]],
        "raw_value": values["amount"], "raw_min": values["min"], "raw_max": values["max"],
        "raw_unit": values["nutrient_id"].map(nutrient.set_index("id")["unit_name"]),
        "raw_basis": "per 100 g edible portion, fresh weight", "analytical_method": pd.NA,
        "method_expression": values["derivation_description"], "sample_count": values["data_points"],
        "standard_error": pd.NA, "quality_tier": quality, "quality_score_available": False,
        "source_reference": values["source_description"], "lineage_source_key": source_key,
        "independent_evidence": independent, "data_layer": "primary_reference",
        "exclusion_reason": "", "range_estimate_flag": False,
    })
    measurements = _measurement_frame(measurements, implicit_per_100g=True)
    measurements.loc[assumed_zero.values, ["value_status", "is_explicit_zero", "main_value_eligible"]] = ["assumed_zero", False, False]
    measurements.loc[assumed_zero.values, "exclusion_reason"] = "assumed_zero_not_measured_zero"
    measurements = _apply_training_gate(measurements)
    measurements = gate_reference_evidence(measurements, {})
    return SourceBundle(foods, components, measurements)


def load_usda_foundation(root: Path) -> SourceBundle:
    source_key = "usda_foundation"
    base = root / "data/raw/usda/foundation_2026_04_30"
    parent_ids = pd.read_csv(base / "foundation_food.csv")["fdc_id"].astype(int)
    foundation_ids = pd.read_csv(base / "foundation_food.csv")
    food = pd.read_csv(base / "food.csv")
    food = food[food["fdc_id"].isin(parent_ids)].copy()
    food = food.merge(foundation_ids, on="fdc_id", how="left")
    categories = pd.read_csv(base / "food_category.csv").rename(columns={"id": "food_category_id", "description": "food_group"})
    food = food.merge(categories[["food_category_id", "food_group"]], on="food_category_id", how="left")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.fdc_id),
            "source_key": source_key, "source_food_id": row.fdc_id, "original_name": row.description,
            "food_group": row.food_group, "food_type": "foundation_food",
            "recipe_status": "not_calculated", "brand_status": "not_reported",
            "edible_status": "edible_food_assumed_by_source",
            "source_lineage_id": f"USDA:NDB:{int(row.NDB_number)}" if pd.notna(row.NDB_number) else f"USDA:FDC:{row.fdc_id}",
            "source_record_url": f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{row.fdc_id}/nutrients",
        }
        for row in food.itertuples(index=False)
    ])
    nutrient = pd.read_csv(base / "nutrient.csv", dtype={"nutrient_nbr": str})
    tags = _usda_infoods_map(root / "data/raw/cnf_2026/extracted")
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.id), "source_key": source_key,
            "source_component_id": row.id, "original_name": row.name, "original_unit": row.unit_name,
            "infoods_tag": tags.get(str(row.nutrient_nbr).split(".")[0], ""),
            "source_definition": f"USDA nutrient number {row.nutrient_nbr}",
            "source_component_group": "USDA nutrient", "authority_status": "official_source_identifier",
        }
        for row in nutrient.itertuples(index=False)
    ])
    values = pd.read_csv(base / "food_nutrient.csv", low_memory=False)
    values = values[values["fdc_id"].isin(parent_ids)].copy()
    method_lookup = foundation_result_links(base).set_index(["parent_fdc_id", "nutrient_id"]).actual_linked_method_labels.to_dict()
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, x) for x in values["id"]],
        "source_key": source_key, "source_measurement_id": values["id"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in values["fdc_id"]],
        "component_observation_id": [stable_id("compobs", source_key, x) for x in values["nutrient_id"]],
        "raw_value": values["amount"], "raw_min": values["min"], "raw_max": values["max"],
        "raw_unit": values["nutrient_id"].map(nutrient.set_index("id")["unit_name"]),
        "raw_basis": "per 100 g edible portion, fresh weight",
        # lab_method_nutrient is a catalogue of possible methods, not a result-to-method link.
        "analytical_method": [method_lookup.get((food, nutrient_id), "") for food, nutrient_id in zip(values.fdc_id, values.nutrient_id, strict=True)],
        "method_expression": "foundation aggregate; methods linked through actual sample results where available",
        "sample_count": values["data_points"], "standard_error": pd.NA,
        "quality_tier": np.where(values["data_points"].fillna(0).gt(0), "A", "B"),
        "quality_score_available": True, "source_reference": "USDA Foundation Foods",
        "lineage_source_key": source_key, "independent_evidence": True,
        "data_layer": "locked_source_validation", "exclusion_reason": "", "range_estimate_flag": False,
        "quality_evidence": "Foundation source metadata; individual methods require sub_sample_result and sample lineage joins",
    })
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=True))
    return SourceBundle(foods, components, measurements)


def _foundation_method_lookup(base: Path) -> dict[int, str]:
    """Possible methods by analyte, not evidence that a food measurement used them."""
    methods = pd.read_csv(base / "lab_method.csv").set_index("id")
    links = pd.read_csv(base / "lab_method_nutrient.csv")
    labels = links.assign(label=links["lab_method_id"].map(methods["description"]).fillna("") + " | " + links["lab_method_id"].map(methods["technique"]).fillna(""))
    return labels.groupby("nutrient_id")["label"].apply(lambda x: "; ".join(sorted(set(v.strip(" |") for v in x if v.strip(" |"))))).to_dict()


def load_fndds(root: Path) -> SourceBundle:
    source_key = "fndds"
    base = root / "data/raw/usda/fndds_2021_2023/FoodData_Central_survey_food_csv_2024-10-31"
    food = pd.read_csv(base / "food.csv")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.fdc_id), "source_key": source_key,
            "source_food_id": row.fdc_id, "original_name": row.description, "food_type": row.data_type,
            "recipe_status": "survey_or_calculated_dish", "brand_status": "not_reported",
            "edible_status": "edible_food_assumed_by_source", "source_lineage_id": f"USDA:FNDDS:{row.fdc_id}",
            "source_record_url": f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{row.fdc_id}/nutrients",
        }
        for row in food.itertuples(index=False)
    ])
    nutrient = pd.read_csv(base / "nutrient.csv", dtype={"nutrient_nbr": str})
    # FNDDS food_nutrient.nutrient_id is the USDA nutrient number, whereas
    # nutrient.id is an internal catalogue row identifier. Joining those two
    # namespaces loses every unit and component link (for example, FNDDS 301
    # is calcium, not nutrient.csv row 301). Keep the official nutrient number
    # as the source component key throughout this adapter.
    nutrient_number = nutrient["nutrient_nbr"].fillna("").astype(str).str.strip()
    nutrient_number = nutrient_number.str.replace(r"\.0$", "", regex=True)
    # Twelve catalogue-only FNDDS component records have no USDA nutrient
    # number and do not occur in food_nutrient. Retain them under a distinct
    # internal namespace so they cannot collide with one another or with a
    # numbered food-value component.
    nutrient["source_nutrient_number"] = nutrient_number.where(
        nutrient_number.ne(""), "internal:" + nutrient["id"].astype(str)
    )
    if nutrient["source_nutrient_number"].duplicated().any():
        duplicate = nutrient.loc[
            nutrient["source_nutrient_number"].duplicated(keep=False),
            ["id", "nutrient_nbr", "name"],
        ].head(10).to_dict("records")
        raise ValueError(f"FNDDS nutrient numbers must be unique; sample duplicates: {duplicate}")
    tags = _usda_infoods_map(root / "data/raw/cnf_2026/extracted")
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.source_nutrient_number),
            "source_key": source_key, "source_component_id": row.source_nutrient_number,
            "original_name": row.name, "original_unit": row.unit_name,
            "infoods_tag": tags.get(row.source_nutrient_number, ""),
            "source_definition": f"USDA nutrient number {row.source_nutrient_number}",
            "source_component_group": "USDA nutrient", "authority_status": "official_source_identifier",
        }
        for row in nutrient.itertuples(index=False)
    ])
    values = pd.read_csv(base / "food_nutrient.csv", low_memory=False)
    values["source_nutrient_number"] = values["nutrient_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    nutrient_by_number = nutrient.set_index("source_nutrient_number")
    unknown_numbers = sorted(set(values["source_nutrient_number"]) - set(nutrient_by_number.index))
    if unknown_numbers:
        raise ValueError(
            "FNDDS food_nutrient contains nutrient numbers absent from nutrient.csv; "
            f"sample={unknown_numbers[:10]}"
        )
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, x) for x in values["id"]], "source_key": source_key,
        "source_measurement_id": values["id"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in values["fdc_id"]],
        "component_observation_id": [
            stable_id("compobs", source_key, x) for x in values["source_nutrient_number"]
        ],
        "raw_value": values["amount"], "raw_min": values["min"], "raw_max": values["max"],
        "raw_unit": values["source_nutrient_number"].map(nutrient_by_number["unit_name"]),
        "raw_basis": "per 100 g edible portion, fresh weight", "method_expression": "survey recipe or derived value",
        "sample_count": values["data_points"], "quality_tier": "D", "quality_score_available": False,
        "source_reference": "USDA FNDDS", "lineage_source_key": "usda_sr_legacy_or_foundation",
        "independent_evidence": False, "data_layer": "calculated_dish_auxiliary",
        "exclusion_reason": "calculated_or_survey_dish_auxiliary_only",
        # FNDDS values are survey recipe / derived-dish expressions. Keep
        # them as source evidence, but do not let a downstream loader mistake
        # a mass-convertible value for an independent primary label.
        "value_origin": "survey_or_calculated_dish",
    })
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=True))
    return SourceBundle(foods, components, measurements)


def load_cnf(root: Path) -> SourceBundle:
    source_key = "cnf"
    base = root / "data/raw/cnf_2026/extracted"
    food = pd.read_csv(base / "Food_Name.csv")
    groups = pd.read_csv(base / "CNF_Food_Group.csv")
    source_food = pd.read_csv(base / "Food_Source.csv")
    food = food.merge(groups, on="CNF_Food_Group_Code", how="left").merge(source_food, on="Food_Source_Code", how="left")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.Food_Code), "source_key": source_key,
            "source_food_id": row.Food_Code, "original_name": row.Food_Description_EN,
            "original_name_local": row.Food_Description_FR, "description": row.Comment_EN,
            "scientific_name": row.ScientificName, "food_group": row.CNF_Food_Group_Description_EN,
            "recipe_status": "not_reported", "brand_status": "not_reported",
            "edible_status": "edible_food_assumed_by_source",
            "source_lineage_id": f"USDA:NDB:{int(float(row.USDA_NDB_Code))}" if pd.notna(row.USDA_NDB_Code) else f"CNF:{row.Food_Code}",
            "source_record_url": "https://food-nutrition.canada.ca/cnf-fce/",
        }
        for row in food.itertuples(index=False)
    ])
    nutrient = pd.read_csv(base / "Nutrient_Name.csv", keep_default_na=False, na_values=["", "NULL"])
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.Nutrient_Code), "source_key": source_key,
            "source_component_id": row.Nutrient_Code, "original_name": row.Nutrient_Name_EN,
            "original_name_local": row.Nutrient_Name_FR, "original_unit": row.Nutrient_Unit,
            "infoods_tag": row.Tagname, "source_definition": f"CNF nutrient code {row.Nutrient_Code}",
            "source_component_group": "CNF nutrient", "authority_status": "official_infoods_tag" if pd.notna(row.Tagname) else "official_source_identifier",
        }
        for row in nutrient.itertuples(index=False)
    ])
    values = pd.read_csv(base / "Nutrient_Amount.csv", low_memory=False)
    nutrient_sources = pd.read_csv(base / "Nutrient_Source.csv").set_index("Nutrient_Source_Code")["Nutrient_Source_Description_EN"]
    quality = values["Nutrient_Source_Code"].map({3: "A", 7: "B", 10: "C", 17: "C"}).fillna("D")
    independent = values["Nutrient_Source_Code"].isin([3, 7, 10, 17])
    lineage = values["Nutrient_Source_Code"].map({0: "usda_sr_legacy", 4: "usda_sr_legacy", 5: "usda_sr_legacy", 82: "frida", 83: "fineli"}).fillna("cnf")
    assumed_zero = values["Nutrient_Source_Code"].eq(12)
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, f, n) for f, n in zip(values["Food_Code"], values["Nutrient_Code"], strict=True)],
        "source_key": source_key,
        "source_measurement_id": values["Food_Code"].astype(str) + ":" + values["Nutrient_Code"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in values["Food_Code"]],
        "component_observation_id": [stable_id("compobs", source_key, x) for x in values["Nutrient_Code"]],
        "raw_value": values["Nutrient_Amount"], "raw_min": pd.NA, "raw_max": pd.NA,
        "raw_unit": values["Nutrient_Code"].map(nutrient.set_index("Nutrient_Code")["Nutrient_Unit"]),
        "raw_basis": "per 100 g edible portion, fresh weight", "analytical_method": pd.NA,
        "method_expression": values["Nutrient_Source_Code"].map(nutrient_sources),
        "sample_count": values["Observations"], "standard_error": values["STD_Error"],
        "quality_tier": quality, "quality_score_available": True,
        "source_reference": values["Nutrient_Source_Code"].map(nutrient_sources),
        "lineage_source_key": lineage, "independent_evidence": independent,
        "data_layer": "primary_reference", "exclusion_reason": "",
    })
    measurements = _measurement_frame(measurements, implicit_per_100g=True)
    measurements.loc[assumed_zero.values, ["value_status", "is_explicit_zero", "main_value_eligible"]] = ["assumed_zero", False, False]
    measurements.loc[assumed_zero.values, "exclusion_reason"] = "assumed_zero_not_measured_zero"
    measurements = _apply_training_gate(measurements)
    return SourceBundle(foods, components, measurements)


def load_frida(root: Path) -> SourceBundle:
    source_key = "frida"
    workbook = root / "data/raw/candidates/frida_6_1/FCDB_6.1_Dataset.xlsx"
    food = pd.read_excel(workbook, sheet_name="Food")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.FoodID), "source_key": source_key,
            "source_food_id": row.FoodID, "original_name": row.FoodName,
            "original_name_local": row.FødevareNavn, "scientific_name": row.TaxonomicName,
            "food_group": row.FoodGroup, "food_type": row.EurofirFoodGroup,
            "foodon_id": row.FoodOntology, "foodex2_code": row.FoodEx2Code,
            "langual_code": row.LangualCode, "taxonomy_id": row.NCBI,
            "recipe_status": "not_reported", "brand_status": "not_reported",
            "edible_status": "edible_food_assumed_by_source", "source_lineage_id": f"FRIDA:{row.FoodID}",
            "source_record_url": "https://frida.fooddata.dk/",
        }
        for row in food.itertuples(index=False)
    ])
    parameter = pd.read_excel(workbook, sheet_name="Parameter", keep_default_na=False, na_values=["", "NULL"])
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.ParameterID), "source_key": source_key,
            "source_component_id": row.ParameterID, "original_name": row.ParameterName,
            "original_name_local": row.ParameterNavn, "original_unit": row.Unit,
            "infoods_tag": row.EurofirComponentID, "eurofir_component_id": row.EFSA_PARAM_Code,
            "chebi_id": row.ChEBI, "pubchem_id": row.PubChem_CID, "cas_number": row.CAS_Nr,
            "formula": row.Formula, "source_component_group": row.ParameterGroupName,
            "source_definition": row.EFSA_PARAM_name,
            "authority_status": "official_infoods_or_eurofir_identifier" if pd.notna(row.EurofirComponentID) else "official_source_identifier",
        }
        for row in parameter.itertuples(index=False)
    ])
    data = pd.read_excel(workbook, sheet_name="Data_Normalised")
    quality = np.where(data["Source"].notna() & data["NumberOfDeterminations"].fillna(0).gt(0), "B", "C")
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, f, p) for f, p in zip(data["FoodID"], data["ParameterID"], strict=True)],
        "source_key": source_key,
        "source_measurement_id": data["FoodID"].astype(str) + ":" + data["ParameterID"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in data["FoodID"]],
        "component_observation_id": [stable_id("compobs", source_key, x) for x in data["ParameterID"]],
        "raw_value": data["ResVal"], "raw_min": data["Min"], "raw_max": data["Max"],
        "raw_unit": data["ParameterID"].map(parameter.set_index("ParameterID")["Unit"]),
        "raw_basis": "as specified by parameter unit", "analytical_method": pd.NA,
        "method_expression": "FRIDA normalized aggregate", "sample_count": data["NumberOfDeterminations"],
        "standard_error": pd.NA, "quality_tier": quality, "quality_score_available": True,
        "source_reference": data["Source"], "lineage_source_key": source_key,
        "independent_evidence": data["Source"].notna(), "data_layer": "primary_reference",
        "exclusion_reason": "",
    })
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=False))
    measurements = gate_reference_evidence(measurements, reference_catalogs(root))
    return SourceBundle(foods, components, measurements)


def _xml_rows(path: Path, element_name: str) -> Iterable[dict[str, Any]]:
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != element_name:
            continue
        row = {child.tag: (child.text or "").strip() for child in element}
        yield row
        element.clear()


def load_ciqual(root: Path) -> SourceBundle:
    source_key = "ciqual"
    base = root / "data/raw/candidates/ciqual_2025"
    food_rows = list(_xml_rows(base / "foods.xml", "ALIM"))
    food = pd.DataFrame(food_rows)
    xlsx_food = pd.read_excel(base / "Table_Ciqual_2025_ENG.xlsx", sheet_name="food composition", usecols=list(range(8)))
    xlsx_food["alim_code"] = xlsx_food["alim_code"].astype(str).str.strip()
    food = food.merge(xlsx_food[["alim_code", "alim_grp_nom_eng", "alim_ssgrp_nom_eng", "alim_ssssgrp_nom_eng"]], on="alim_code", how="left")
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row.alim_code), "source_key": source_key,
            "source_food_id": row.alim_code, "original_name": row.alim_nom_eng,
            "original_name_local": row.alim_nom_fr, "scientific_name": row.alim_nom_sci,
            "food_group": row.alim_grp_nom_eng, "food_subgroup": row.alim_ssgrp_nom_eng,
            "food_type": row.alim_ssssgrp_nom_eng, "recipe_status": "not_reported",
            "brand_status": "not_reported", "edible_status": "edible_food_assumed_by_source",
            "source_lineage_id": f"CIQUAL:{row.alim_code}", "source_record_url": "https://ciqual.anses.fr/",
        }
        for row in food.itertuples(index=False)
    ])
    component = pd.DataFrame(list(_xml_rows(base / "const.xml", "CONST")))
    unit = component["const_nom_eng"].str.extract(r"\(([^()]*/100\s*g[^()]*)\)\s*$", expand=False).fillna("")
    components = _component_frame([
        {
            "component_observation_id": stable_id("compobs", source_key, row.const_code), "source_key": source_key,
            "source_component_id": row.const_code, "original_name": row.const_nom_eng,
            "original_name_local": row.const_nom_fr, "original_unit": unit.iloc[i],
            "infoods_tag": row.code_INFOODS, "source_definition": f"CIQUAL constituent code {row.const_code}",
            "source_component_group": "CIQUAL constituent",
            "authority_status": "official_infoods_tag" if str(row.code_INFOODS).strip() else "official_source_identifier",
        }
        for i, row in enumerate(component.itertuples(index=False))
    ])
    values = pd.DataFrame(list(_xml_rows(base / "compo_2025_11_03.xml", "COMPO")))
    values.insert(0, "row_id", np.arange(len(values), dtype=np.int64))
    confidence = values["code_confiance"].astype(str).str.strip().str.upper()
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, x) for x in values["row_id"]], "source_key": source_key,
        "source_measurement_id": values["row_id"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x.strip()) for x in values["alim_code"]],
        "component_observation_id": [stable_id("compobs", source_key, x.strip()) for x in values["const_code"]],
        "raw_value": values["teneur"], "raw_min": values["min"], "raw_max": values["max"],
        "raw_unit": values["const_code"].str.strip().map(component.assign(const_code=component["const_code"].str.strip()).set_index("const_code")["const_nom_eng"]).str.extract(r"\(([^()]*/100\s*g[^()]*)\)\s*$", expand=False).fillna(""),
        "raw_basis": "as specified by constituent name", "analytical_method": pd.NA,
        "method_expression": "CIQUAL selected aggregate value", "sample_count": pd.NA,
        "standard_error": pd.NA, "quality_tier": confidence.map({"A": "A", "B": "B", "C": "C", "D": "D"}).fillna("D"),
        "quality_score_available": True, "source_reference": values["source_code"],
        "lineage_source_key": source_key, "independent_evidence": confidence.isin(["A", "B", "C"]),
        "data_layer": "primary_reference", "exclusion_reason": "",
    })
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=False))
    measurements = gate_reference_evidence(measurements, reference_catalogs(root))
    return SourceBundle(foods, components, measurements)


def _header_unit(header: Any) -> str:
    text = str(header)
    matches = re.findall(r"\(([^()]*)\)", text)
    return matches[-1].strip() if matches else ""


def _gate_afcd_derivation_conflicts(
    measurements: pd.DataFrame, profile_derivation: pd.Series, details_derivation: pd.Series,
) -> pd.DataFrame:
    """Preserve both official statements; neither table overrides a conflict."""
    profile = profile_derivation.map(normalize_text)
    details = details_derivation.map(normalize_text)
    conflict = profile.ne(details) | details_derivation.isna()
    measurements["method_expression"] = (
        "profile_derivation=" + profile_derivation.fillna("unreported").astype(str)
        + "; food_details_derivation=" + details_derivation.fillna("unreported").astype(str)
    )
    measurements.loc[conflict, ["main_value_eligible", "strict_validation_eligible", "independent_evidence"]] = False
    measurements.loc[conflict, "quality_tier"] = "D"
    measurements.loc[conflict, "data_layer"] = "source_metadata_conflict"
    measurements.loc[conflict, "exclusion_reason"] = measurements.loc[conflict, "exclusion_reason"].map(
        lambda reason: _reason(reason, "AFCD_conflicting_or_missing_food_derivation_metadata")
    )
    return measurements


def load_afcd(root: Path) -> SourceBundle:
    source_key = "afcd"
    workbook = root / "data/raw/candidates/afcd_release_3/nutrient_profiles.xlsx"
    wide = pd.read_excel(workbook, sheet_name="All solids & liquids per 100 g", header=2)
    wide = wide.rename(columns={wide.columns[0]: "Public Food Key", wide.columns[1]: "Classification", wide.columns[2]: "Derivation", wide.columns[3]: "Food Name"})
    details_dir = root / "data/raw/expansion_2026_09_06/afcd"
    nutrient_details = afcd_component_details(details_dir) if details_dir.exists() else {}
    detail_path = details_dir / "00_AFCD Release 3 - Food Details.xlsx"
    food_details = pd.read_excel(detail_path, sheet_name="Food details", header=2).set_index("Public Food Key") if detail_path.exists() else pd.DataFrame()
    if not food_details.index.is_unique:
        raise ValueError("AFCD Food Details contains duplicate Public Food Keys")
    detail_derivation = food_details["Derivation"].to_dict() if not food_details.empty else {}
    foods = _food_frame([
        {
            "food_observation_id": stable_id("foodobs", source_key, row["Public Food Key"]), "source_key": source_key,
            "source_food_id": row["Public Food Key"], "original_name": row["Food Name"],
            "description": str(food_details.loc[row["Public Food Key"], "Food Description"]) if row["Public Food Key"] in food_details.index else "",
            "food_group": row["Classification"], "recipe_status": "calculated_recipe" if "recipe" in {normalize_text(row["Derivation"]), normalize_text(detail_derivation.get(row["Public Food Key"], ""))} else "not_calculated",
            "brand_status": "label_value_present" if normalize_text(row["Derivation"]) == "label data" else "not_reported",
            "edible_status": "edible_food_assumed_by_source", "source_lineage_id": f"AFCD:{row['Public Food Key']}",
            "source_record_url": "https://www.foodstandards.gov.au/science-data/monitoringnutrients/afcd",
        }
        for row in wide.to_dict("records") if pd.notna(row["Public Food Key"])
    ])
    meta = ["Public Food Key", "Classification", "Derivation", "Food Name"]
    value_columns = [c for c in wide.columns if c not in meta and not str(c).startswith("Unnamed")]
    component_rows = []
    for index, column in enumerate(value_columns):
        name = re.sub(r"\s*\([^()]*\)\s*$", "", str(column).replace("\n", " ")).strip()
        detail = nutrient_details.get((normalize_text(name), _unit_key(_header_unit(column))), {})
        component_rows.append({
            "component_observation_id": stable_id("compobs", source_key, index, column), "source_key": source_key,
            "source_component_id": f"R3:{index}", "original_name": name,
            "original_unit": _header_unit(column),
            "infoods_tag": detail.get("infoods_tag", ""),
            "eurofir_component_id": detail.get("eurofir_id", ""),
            "description": detail.get("description", ""),
            "source_definition": str(column).replace("\n", " ") + " | " + detail.get("description", "") + " | Equation: " + detail.get("equation", ""),
            "source_component_group": "AFCD nutrient profile", "authority_status": "official_source_identifier_pending_crosswalk",
        })
    components = _component_frame(component_rows)
    long = wide[meta + value_columns].melt(id_vars=meta, var_name="column_name", value_name="raw_value")
    component_map = {column: stable_id("compobs", source_key, i, column) for i, column in enumerate(value_columns)}
    component_unit = {column: _header_unit(column) for column in value_columns}
    derivation = long["Derivation"].map(normalize_text)
    quality = derivation.map({"analysed": "B", "analytical": "B"}).fillna("D")
    independent = derivation.isin(["analysed", "analytical"])
    measurements = pd.DataFrame({
        "measurement_id": [stable_id("measure", source_key, food, component_map[col]) for food, col in zip(long["Public Food Key"], long["column_name"], strict=True)],
        "source_key": source_key,
        "source_measurement_id": long["Public Food Key"].astype(str) + ":" + long["column_name"].astype(str),
        "food_observation_id": [stable_id("foodobs", source_key, x) for x in long["Public Food Key"]],
        "component_observation_id": long["column_name"].map(component_map),
        "raw_value": long["raw_value"], "raw_min": pd.NA, "raw_max": pd.NA,
        "raw_unit": long["column_name"].map(component_unit), "raw_basis": "per 100 g edible portion, fresh weight",
        "analytical_method": pd.NA, "method_expression": long["Derivation"], "sample_count": pd.NA,
        "standard_error": pd.NA, "quality_tier": quality, "quality_score_available": True,
        "source_reference": "AFCD Release 3", "lineage_source_key": np.where(independent, source_key, "external_or_calculated"),
        "independent_evidence": independent, "data_layer": "primary_reference", "exclusion_reason": "",
    })
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=True))
    zero = measurements["reported_zero"]
    measurements.loc[zero, "zero_semantics"] = "reported_zero_with_unresolved_censoring_or_imputation"
    measurements.loc[zero, "value_status"] = "reported_zero_unresolved"
    measurements.loc[zero, ["is_explicit_zero", "main_value_eligible", "strict_validation_eligible"]] = False
    measurements.loc[zero, "exclusion_reason"] = "AFCD_zero_may_encode_below_LOR_or_imputation"
    if not food_details.empty:
        sampling = food_details["Sampling Details"].fillna("").to_dict()
        foodkey_by_obs = foods.set_index("food_observation_id")["source_food_id"]
        measurements["source_reference"] = measurements["food_observation_id"].map(foodkey_by_obs).map(sampling)
        measurements["quality_evidence"] = "AFCD Food Details derivation and sampling; per-value expert review remains required"
        foods["potential_reference_lineages"] = foods["source_food_id"].map(sampling).fillna("").map(
            lambda text: ";".join("USDA:FDC:" + identifier for identifier in sorted(set(re.findall(r"FDC\s*(?:ID\s*)?[:#]?\s*(\d+)", text, flags=re.I))))
        )
    measurements = _gate_afcd_derivation_conflicts(
        measurements, long["Derivation"], long["Public Food Key"].map(detail_derivation),
    )
    admitted = measurements["main_value_eligible"].eq(True)
    measurements.loc[admitted, "quality_tier"] = "C"
    measurements.loc[admitted, "data_layer"] = "curated_reference_training"
    measurements.loc[admitted, "independent_evidence"] = False
    measurements.loc[admitted, "strict_validation_eligible"] = False
    measurements["value_origin"] = "food_level_derivation_not_individual_assay_certification"
    return SourceBundle(foods, components, measurements)


def _unit_key(value: Any) -> str:
    return str(value).strip().lower().replace("\u00b5", "u").replace("\u03bc", "u").replace(" ", "")


def afcd_component_details(directory: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read source-supplied identifiers using exact source name AND unit."""
    path = directory / "03_AFCD Release 3 - Nutrient details.xlsx"
    excel = pd.ExcelFile(path)
    result = {}
    for sheet in excel.sheet_names:
        if sheet in {"Contents", "Core nutrients"}:
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None).fillna("")
        header = raw.index[raw[0].astype(str).str.strip().eq("Component")]
        if len(header) != 1:
            raise ValueError(f"Expected one component header in {path}: {sheet}")
        raw.columns = raw.iloc[header[0]].astype(str).str.strip()
        for row in raw.iloc[header[0] + 1:].to_dict("records"):
            name, unit = str(row["Component"]).strip(), _unit_key(row["Units"])
            if not name or not unit:
                continue
            if sheet == "Fatty acids" and name.endswith("FD"):
                name = name[:-2]
            key = (normalize_text(name), unit)
            detail = {
                "infoods_tag": str(row["INFOODs tagname"]).strip(),
                "eurofir_id": str(row["EuroFIR Component Name"]).strip(),
                "description": str(row["Description"]).strip(),
                "equation": str(row["Equation (where applicable)"]).strip(),
            }
            if key in result and result[key] != detail:
                raise ValueError(f"Conflicting AFCD metadata for {key}")
            result[key] = detail
    return result


def load_cofid(root: Path) -> SourceBundle:
    source_key = "cofid"
    workbook = root / "data/raw/candidates/cofid_2021/cofid_2021.xlsx"
    excel = pd.ExcelFile(workbook)
    food_records: dict[str, dict[str, Any]] = {}
    component_rows: list[dict[str, Any]] = []
    measurement_parts: list[pd.DataFrame] = []
    raw_tables = {}
    identities = []
    for sheet in excel.sheet_names:
        if not re.match(r"1\.(?:[3-9]|1[0-4])\b", sheet):
            continue
        raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
        if raw.shape[0] < 4 or raw.shape[1] < 8:
            continue
        raw_tables[sheet] = raw
        identities.extend((str(code).strip(), normalize_text(name)) for code, name in raw.iloc[3:, :2].itertuples(index=False, name=None) if pd.notna(code))
    identity_frame = pd.DataFrame(identities, columns=["code", "name"])
    counts = identity_frame.groupby("code").name.nunique()
    ambiguous_codes = set(counts[counts.gt(1)].index)
    for sheet, raw in raw_tables.items():
        headers = raw.iloc[0].fillna("").astype(str).tolist()
        tags = raw.iloc[1].fillna("").astype(str).tolist()
        names = raw.iloc[2].fillna("").astype(str).tolist()
        body = raw.iloc[3:].copy()
        body.columns = [f"c{i}" for i in range(body.shape[1])]
        relative_fat = "100gfa" in normalize_text(sheet).replace(" ", "") or "per 100fa" in normalize_text(sheet)
        for row in body.itertuples(index=False, name=None):
            food_code = row[0]
            if pd.isna(food_code) or not str(food_code).strip():
                continue
            original_code = str(food_code).strip()
            key = _cofid_food_key(original_code, row[1], ambiguous_codes)
            food_records.setdefault(key, {
                "food_observation_id": stable_id("foodobs", source_key, key), "source_key": source_key,
                "source_food_id": original_code, "original_name": row[1], "description": row[2], "food_group": row[3],
                "recipe_status": "not_reported", "brand_status": "not_reported",
                "edible_status": "edible_food_assumed_by_source", "source_lineage_id": f"COFID:{key}",
                "source_record_url": "https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid",
            })
        for index in range(7, raw.shape[1]):
            if not headers[index] and not tags[index] and not names[index]:
                continue
            source_component_id = f"{sheet}:{index}"
            component_id = stable_id("compobs", source_key, source_component_id)
            unit = _header_unit(headers[index])
            component_rows.append({
                "component_observation_id": component_id, "source_key": source_key,
                "source_component_id": source_component_id, "original_name": names[index] or re.sub(r"\([^)]*\)", "", headers[index]).strip(),
                "original_unit": unit, "infoods_tag": tags[index], "source_definition": headers[index],
                "source_component_group": sheet, "authority_status": "official_infoods_tag" if tags[index] else "official_source_identifier_pending_crosswalk",
            })
            subset = body.iloc[:, [0, 1, 2, 5, index]].copy()
            subset.columns = ["food_code", "food_name", "food_description", "reference", "raw_value"]
            food_keys = [_cofid_food_key(code, name, ambiguous_codes) for code, name in zip(subset.food_code, subset.food_name, strict=True)]
            reference_documented = subset["reference"].fillna("").astype(str).str.strip().isin(COFID_ANALYTICAL_REFERENCES)
            calculated = subset["food_description"].fillna("").astype(str).str.contains(r"\b(?:calculated|recipe|estimated|imputed)\b", case=False, regex=True)
            curated = reference_documented & ~calculated
            alcoholic = body.iloc[:, 3].fillna("").astype(str).str.strip().str.startswith("Q")
            part = pd.DataFrame({
                "measurement_id": [stable_id("measure", source_key, sheet, food, source_component_id) for food in food_keys],
                "source_key": source_key, "source_measurement_id": sheet + ":" + subset["food_code"].astype(str) + ":" + str(index) + ":row" + (subset.index + 1).astype(str),
                "food_observation_id": [stable_id("foodobs", source_key, key) for key in food_keys],
                "component_observation_id": component_id, "raw_value": subset["raw_value"], "raw_min": pd.NA, "raw_max": pd.NA,
                "raw_unit": unit, "raw_basis": "per 100 g fatty acids" if relative_fat else np.where(alcoholic, "per 100 mL beverage", "per 100 g edible food, fresh weight"),
                "analytical_method": pd.NA, "method_expression": subset["food_description"], "sample_count": pd.NA,
                "standard_error": pd.NA, "quality_tier": np.where(curated, "C", "D"),
                "quality_score_available": False, "source_reference": subset["reference"],
                "lineage_source_key": source_key, "independent_evidence": False,
                "data_layer": np.where(curated, "curated_reference_training", "reference_review_archive"), "exclusion_reason": "",
                "value_origin": "compiled_food_level_analytical_reference",
                "strict_validation_eligible": False,
                "quality_evidence": np.where(curated, "CoFID 2021 named analytical reference; no individual-value certification", ""),
            })
            measurement_parts.append(part)
    foods = _food_frame(list(food_records.values()))
    components = _component_frame(component_rows).drop_duplicates("component_observation_id")
    measurements = pd.concat(measurement_parts, ignore_index=True)
    measurements = _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=False))
    return SourceBundle(foods, components, measurements)


def _cofid_food_key(code, name, ambiguous_codes: set[str]) -> str:
    """Disambiguate reused official IDs; retain the raw code in source_food_id."""
    code = str(code).strip()
    return f"{code}:{stable_id('name', normalize_text(name))}" if code in ambiguous_codes else code


def _foodb_components(root: Path) -> pd.DataFrame:
    source_key = "foodb"
    nutrient = pd.read_csv(root / "foodb_2020_04_07_csv/Nutrient.csv", low_memory=False)
    nutrient_rows = [
        {
            "component_observation_id": stable_id("compobs", source_key, "Nutrient", row.id), "source_key": source_key,
            "source_component_id": f"Nutrient:{row.id}", "original_name": row.name,
            "description": row.description, "source_definition": row.comments,
            "source_component_group": "FooDB Nutrient", "authority_status": "foodb_source_identifier_pending_authority_crosswalk",
        }
        for row in nutrient.itertuples(index=False)
    ]
    compound = pd.read_csv(root / "foodb_2020_04_07_csv/Compound.csv", low_memory=False)
    external = pd.read_csv(root / "foodb_2020_04_07_csv/CompoundExternalDescriptor.csv", low_memory=False)
    chebi = external[external["external_id"].astype(str).str.startswith("CHEBI:")].drop_duplicates("compound_id").set_index("compound_id")["external_id"]
    lipid = external[external["external_id"].astype(str).str.match(r"^LM[A-Z]{2}\d+")].drop_duplicates("compound_id").set_index("compound_id")["external_id"]
    compound_rows = []
    for row in compound.itertuples(index=False):
        # The 2020 export header is shifted after name. These assignments follow
        # the actual serialized field order, validated against InChI/CAS syntax.
        compound_rows.append({
            "component_observation_id": stable_id("compobs", source_key, "Compound", row.id), "source_key": source_key,
            "source_component_id": f"Compound:{row.id}", "original_name": row.name,
            "description": row.annotation_quality, "chebi_id": chebi.get(row.id, ""),
            "lipidmaps_id": lipid.get(row.id, ""), "inchikey": row.moldb_smiles,
            "cas_number": row.description, "source_chemical_class": " > ".join(str(v) for v in (row.kingdom, row.superklass, row.klass, row.subklass) if pd.notna(v)),
            "source_definition": row.annotation_quality, "source_component_group": "FooDB Compound",
            "authority_status": "authority_identifier_available" if row.id in chebi.index or row.id in lipid.index else "foodb_source_identifier_pending_authority_crosswalk",
        })
    return _component_frame(nutrient_rows + compound_rows)


def iter_foodb(root: Path, chunk_size: int = 200_000) -> tuple[pd.DataFrame, pd.DataFrame, Iterable[pd.DataFrame]]:
    """Return FooDB components and a streaming iterator over source-observation values."""
    source_key = "foodb"
    base_food = pd.read_csv(root / "foodb_2020_04_07_csv/Food.csv", low_memory=False).set_index("id")
    components = _foodb_components(root)
    content_path = root / "foodb_2020_04_07_csv/Content.csv"
    food_records: dict[str, dict[str, Any]] = {}

    # First pass constructs source-observation foods without retaining 5M values.
    usecols = ["food_id", "orig_food_id", "orig_food_common_name", "orig_food_scientific_name", "orig_food_part", "citation", "preparation_type"]
    for chunk in pd.read_csv(content_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        for row in chunk.drop_duplicates().itertuples(index=False):
            lineage_food = row.orig_food_id if pd.notna(row.orig_food_id) else row.food_id
            part = "" if pd.isna(row.orig_food_part) else str(row.orig_food_part).strip()
            prep = "" if pd.isna(row.preparation_type) else str(row.preparation_type).strip()
            citation = "FOODB" if pd.isna(row.citation) else str(row.citation).strip()
            key = f"{citation}|{lineage_food}|{part}|{prep}"
            if key in food_records:
                continue
            base = base_food.loc[row.food_id] if row.food_id in base_food.index else None
            name = row.orig_food_common_name if pd.notna(row.orig_food_common_name) else (base["name"] if base is not None else f"FooDB food {row.food_id}")
            food_records[key] = {
                "food_observation_id": stable_id("foodobs", source_key, key), "source_key": source_key,
                "source_food_id": key, "original_name": name,
                "description": base["description"] if base is not None else "",
                "scientific_name": row.orig_food_scientific_name if pd.notna(row.orig_food_scientific_name) else (base["name_scientific"] if base is not None else ""),
                "food_group": base["food_group"] if base is not None else "",
                "food_subgroup": base["food_subgroup"] if base is not None else "",
                "food_type": base["food_type"] if base is not None else "",
                "taxonomy_id": base["ncbi_taxonomy_id"] if base is not None else "", "part": part,
                "processing": prep, "recipe_status": "not_reported", "brand_status": "not_reported",
                "edible_status": "edible_food_assumed_by_source", "source_lineage_id": _foodb_food_lineage(citation, lineage_food),
                "source_record_url": "https://foodb.ca/",
            }
    foods = _food_frame(list(food_records.values()))

    def measurement_iterator() -> Iterable[pd.DataFrame]:
        for chunk in pd.read_csv(content_path, chunksize=chunk_size, low_memory=False):
            lineage_food = chunk["orig_food_id"].where(chunk["orig_food_id"].notna(), chunk["food_id"])
            part = chunk["orig_food_part"].fillna("").astype(str).str.strip()
            prep = chunk["preparation_type"].fillna("").astype(str).str.strip()
            citation = chunk["citation"].fillna("FOODB").astype(str).str.strip()
            keys = citation + "|" + lineage_food.astype(str) + "|" + part + "|" + prep
            lineage_upper = citation.str.upper()
            copied = lineage_upper.str.contains("USDA|DTU|FRIDA|PHENOL", regex=True, na=False)
            specialist = lineage_upper.str.contains("PHENOL|DUKE|ARTICLE|PUBMED", regex=True, na=False)
            source_type = chunk["source_type"].fillna("").astype(str)
            component_ids = [stable_id("compobs", source_key, typ, sid) for typ, sid in zip(source_type, chunk["source_id"], strict=True)]
            measurements = pd.DataFrame({
                "measurement_id": [stable_id("measure", source_key, x) for x in chunk["id"]], "source_key": source_key,
                "source_measurement_id": chunk["id"].astype(str),
                "food_observation_id": [stable_id("foodobs", source_key, x) for x in keys],
                "component_observation_id": component_ids, "raw_value": chunk["orig_content"],
                "raw_min": chunk["orig_min"], "raw_max": chunk["orig_max"], "raw_unit": chunk["orig_unit"],
                "raw_basis": chunk["orig_unit_expression"].where(chunk["orig_unit_expression"].notna(), chunk["orig_unit"]),
                "analytical_method": chunk["orig_method"], "method_expression": chunk["orig_unit_expression"],
                "sample_count": pd.NA, "standard_error": pd.NA,
                "quality_tier": np.where(specialist & ~copied, "C", "D"), "quality_score_available": False,
                "source_reference": citation, "lineage_source_key": citation.map(_foodb_lineage_key),
                "independent_evidence": ~copied, "data_layer": "specialist_composition",
                "exclusion_reason": np.where(copied, "copied_value_retained_for_provenance_only", "specialist_layer_not_primary_benchmark"),
            })
            yield _apply_training_gate(_measurement_frame(measurements, implicit_per_100g=False))

    return foods, components, measurement_iterator()


def _foodb_lineage_key(citation: Any) -> str:
    text = normalize_text(citation)
    if "usda" in text:
        return "usda_sr_legacy"
    if "dtu" in text or "frida" in text:
        return "frida"
    if "phenol" in text:
        return "phenol_explorer"
    if "duke" in text:
        return "duke_phytochemical_database"
    return f"foodb_lineage:{text or 'unknown'}"


def _foodb_food_lineage(citation: Any, source_food_id: Any) -> str:
    """Keep FooDB citation-local IDs out of primary-database namespaces.

    ``orig_food_id`` is only guaranteed to be meaningful inside the cited
    FooDB import. Treating it as an USDA NDB or Frida primary key produced
    false identity links when the same integer referred to different foods.
    """
    citation_text = normalize_text(citation)
    identifier = str(source_food_id).strip()
    try:
        identifier = str(int(float(identifier)))
    except ValueError:
        pass
    return f"FOODB_CITATION:{citation_text or 'unknown'}:{identifier}"


SOURCE_LOADERS = {
    "usda_sr_legacy": load_usda_sr,
    "usda_foundation": load_usda_foundation,
    "fndds": load_fndds,
    "cnf": load_cnf,
    "frida": load_frida,
    "ciqual": load_ciqual,
    "afcd": load_afcd,
    "cofid": load_cofid,
}


def gate_component_expressions(frame: pd.DataFrame, components: pd.DataFrame, definitions: dict) -> pd.DataFrame:
    """A mass-looking numerator does not establish chemical-mass identity."""
    result = frame.copy()
    lookup = components.set_index("component_observation_id")
    reasons = {}
    for identifier, row in lookup.iterrows():
        tag = str(row.get("infoods_tag", "")).strip().upper()
        official = definitions.get(tag, {}).get("definition", "")
        name = normalize_text(row.get("original_name"))
        description = normalize_text(official)
        equivalent = (
            tag in {"VITA", "VITA-", "VITARE", "VITDEQ", "NIATRP", "NIAEQ", "FOLDFE", "CARTBEQ", "VITE", "VITE-"}
            or any(x in name for x in ("equivalent", "from tryptophan", "derived from tryptophan"))
            or "equivalents" in description or "vitamin a activities" in description
        )
        if equivalent:
            reasons[identifier] = "activity_equivalent_not_chemical_mass"
        elif "per quantity of nitrogen" in description or "per gram of nitrogen" in description:
            reasons[identifier] = "nitrogen_denominator_not_food_mass"
    reasons_by_row = result["component_observation_id"].map(reasons)
    excluded = reasons_by_row.notna()
    result.loc[excluded, ["main_value_eligible", "strict_validation_eligible"]] = False
    if "validation_reference_eligible" in result:
        result.loc[excluded, "validation_reference_eligible"] = False
        changed = excluded & result["source_policy_decision"].eq("admitted_trusted_reference")
        result.loc[changed, "source_policy_decision"] = "held_record_level_check"
        result.loc[changed, "source_policy_reason"] = reasons_by_row[changed]
        result.loc[excluded, "validation_evidence_basis"] = "not_eligible"
    result.loc[excluded, "exclusion_reason"] = reasons_by_row[excluded]
    result.loc[excluded, "measurement_modality"] = "non_main_definition_expression"
    result["source_component_definition"] = result["component_observation_id"].map(lookup["source_definition"])
    return result


def load_norway(root: Path) -> SourceBundle:
    """Admit only values linked to explicit Norwegian analytical projects."""
    source = "norway"
    directory = root / "data/raw/expansion_2026_09_06/norway"
    food_data = json.loads((directory / "foods.json").read_text())["foods"]
    nutrients = json.loads((directory / "nutrients.json").read_text())["nutrients"]
    references = {x["sourceId"]: x["description"] for x in json.loads((directory / "sources.json").read_text())["sources"]}
    references["UNREPORTED"] = "Source reference absent from official constituent record; excluded from primary values"
    groups = {x["foodGroupId"]: x["name"] for x in json.loads((directory / "food-groups.json").read_text())["foodGroups"]}
    # Only explicitly reviewed identical definitions are linked to INFOODS.
    same_tags = set("WATER FAT FASAT FAMS FAPU FAN3 FAN6 CHORL STARCH SUGAR ALC RETOL CARTB VITD THIA RIBF NIA VITB6 FOL VITB12 VITC CA FE NA K MG ZN SE CU P ID".split())
    fatty_tags = {
        "F12:0": "F12D0", "F14:0": "F14D0", "F16:0": "F16D0", "F18:0": "F18D0",
        "F16:1": "F16D1", "F18:1": "F18D1", "F18:2CN6": "F18D2CN6", "F18:3N3": "F18D3N3",
        "F20:3N3": "F20D3N3", "F20:3N6": "F20D3N6", "F20:4N3": "F20D4N3", "F20:4N6": "F20D4N6",
        "F20:5N3": "F20D5N3", "F22:5N3": "F22D5N3", "F22:6N3": "F22D6N3",
    }
    component_rows = []
    for item in nutrients:
        eurofir = item["euroFirId"]
        tag = eurofir if eurofir in same_tags else fatty_tags.get(eurofir, "PROCNT" if eurofir == "PROT" else "")
        component_rows.append({
            "component_observation_id": stable_id("compobs", source, item["nutrientId"]),
            "source_key": source, "source_component_id": item["nutrientId"],
            "original_name": item["name"], "original_unit": item["unit"], "infoods_tag": tag,
            "eurofir_component_id": eurofir, "source_definition": item["euroFirName"],
            "description": item["uri"], "source_component_group": "Norwegian nutrient",
            "authority_status": "official_source_eurofir_identifier",
        })
    nutrient_ids = {x["nutrientId"] for x in nutrients}
    foods, values, unknown = [], [], set()
    for item in food_data:
        fid = stable_id("foodobs", source, item["foodId"])
        foods.append({
            "food_observation_id": fid, "source_key": source, "source_food_id": item["foodId"],
            "original_name": item["foodName"], "scientific_name": item.get("latinName", ""),
            "description": "LanguaL: " + ";".join(item.get("langualCodes", [])),
            "food_group": groups.get(item["foodGroupId"], item["foodGroupId"]),
            "recipe_status": "per_value_reference_controls_admission", "brand_status": "not_reported",
            "edible_status": "edible_portion_defined_by_source", "source_lineage_id": "NORWAY:" + item["foodId"],
            "source_record_url": item["uri"],
        })
        for constituent in item["constituents"]:
            nid, ref = constituent["nutrientId"], constituent.get("sourceId", "UNREPORTED")
            if ref not in references:
                raise ValueError(f"Unknown Norway source reference: {ref}")
            if nid not in nutrient_ids:
                unknown.add(nid)
            reviewed = ref in NORWAY_ANALYTICAL_REFERENCES and nid in nutrient_ids and nid != "SUGAN"
            values.append({
                "measurement_id": stable_id("measure", source, item["foodId"], nid),
                "source_key": source, "source_measurement_id": item["foodId"] + ":" + nid,
                "food_observation_id": fid, "component_observation_id": stable_id("compobs", source, nid),
                "raw_value": constituent.get("quantity"), "raw_unit": constituent.get("unit"),
                "raw_basis": "per 100 g edible portion, fresh weight", "raw_min": None, "raw_max": None,
                "method_expression": references[ref], "source_reference": ref + ": " + references[ref],
                "quality_tier": "B" if reviewed else "D", "quality_score_available": False,
                "independent_evidence": reviewed, "lineage_source_key": "NORWAY_PROJECT:" + ref,
                "data_layer": "primary_reference" if reviewed else "reference_review_archive",
                "value_origin": "source_linked_analytical_project" if reviewed else "unreviewed_calculated_or_borrowed",
                "quality_evidence": references[ref] if reviewed else "",
                "exclusion_reason": "" if reviewed else "not_an_admitted_analytical_reference_or_identity",
            })
    for nid in sorted(unknown):
        component_rows.append({
            "component_observation_id": stable_id("compobs", source, nid), "source_key": source,
            "source_component_id": nid, "original_name": nid, "source_definition": "Not defined in downloaded nutrient registry",
            "authority_status": "unresolved_component_identity", "source_component_group": "unresolved",
        })
    measurements = _apply_training_gate(_measurement_frame(pd.DataFrame(values), implicit_per_100g=True))
    assumed = measurements["source_reference"].str.startswith("50:")
    censored = measurements["source_reference"].str.startswith("60a:")
    measurements.loc[assumed, "value_status"] = "assumed_zero"
    measurements.loc[censored, "value_status"] = "left_censored_zero_encoding"
    measurements.loc[censored, "is_censored"] = True
    measurements.loc[assumed | censored, "is_explicit_zero"] = False
    measurements.loc[assumed | censored, ["main_value_eligible", "strict_validation_eligible"]] = False
    return SourceBundle(_food_frame(foods), _component_frame(component_rows), measurements)


SOURCE_LOADERS["norway"] = load_norway


def write_source_staging(root: Path, output_dir: Path, sources: list[str] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = parse_infoods_tagnames(root / "data/raw/reference/infoods_tagnames_2022")
    selected = sources or list(SOURCE_LOADERS) + ["foodb"]
    for source_key in selected:
        target = output_dir / source_key
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ("food_observation.csv.gz", "component_observation.csv.gz", "measurement_part_*.csv.gz"):
            for stale in target.glob(pattern):
                stale.unlink()
        if source_key == "foodb":
            foods, components, measurements = iter_foodb(root)
            write_csv(foods, target / "food_observation.csv.gz")
            write_csv(components, target / "component_observation.csv.gz")
            for index, part in enumerate(measurements):
                part = part[~part["value_status"].eq("missing")].copy()
                write_csv(part, target / f"measurement_part_{index:04d}.csv.gz")
            print(f"{source_key}: foods={len(foods):,}, components={len(components):,}, measurement parts={index + 1:,}")
            continue
        if source_key not in SOURCE_LOADERS:
            raise ValueError(f"Unknown source: {source_key}")
        bundle = SOURCE_LOADERS[source_key](root)
        bundle.measurements = gate_component_expressions(bundle.measurements, bundle.components, definitions)
        bundle.measurements = bundle.measurements[~bundle.measurements["value_status"].eq("missing")].copy()
        write_csv(bundle.foods, target / "food_observation.csv.gz")
        write_csv(bundle.components, target / "component_observation.csv.gz")
        write_csv(bundle.measurements, target / "measurement_part_0000.csv.gz")
        print(f"{source_key}: foods={len(bundle.foods):,}, components={len(bundle.components):,}, measurements={len(bundle.measurements):,}")

#!/usr/bin/env python3
"""Build the audited SR Legacy + CNF + FooDB mass-only pretraining corpus.

The build order is intentional and recorded in output artifacts:

1. concatenate source-level food rows;
2. retain only quantities convertible to g/100g and convert them;
3. deduplicate columns using official codes or reviewed FooDB mappings only;
4. deduplicate food rows using name candidates plus numerical agreement;
5. apply support and food-quality filters;
6. classify macro nutrients, micro nutrients, and compounds;
7. create leakage-safe train/validation/test splits and multi-target masks.

No missing cell is converted to zero.  Zero in a source table remains an
observed value.  The script never uses Foundation Foods or FNDDS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
SR_DIR = ROOT / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04"
CNF_DIR = ROOT / "data/raw/cnf_2026/extracted"
FOODB_OBSERVATION_DIR = ROOT / "data/processed/fooddb_observations_v1"
FOODB_COMPOUND_DIR = ROOT / "data/processed/fooddb_raw_mass_compounds_v2"
FOODB_RAW_DIR = ROOT / "foodb_2020_04_07_csv"
CROSSWALK_PATH = ROOT / "data/audits/foodb_component_crosswalk_review.csv"
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_mass_v4"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/multisource_mass_v4"
DEFAULT_AUDIT_DIR = ROOT / "data/audits/multisource_mass_v4"

SOURCE_PRIORITY = {"foodb": 3, "sr_legacy": 2, "cnf": 1}
MASS_FACTORS = {"g": 1.0, "gram": 1.0, "mg": 1e-3, "milligram": 1e-3, "ug": 1e-6, "microgram": 1e-6}
MIN_AXIS_SUPPORT = 20
MASKABLE_AXIS_SUPPORT = 50
MIN_FOOD_VALUES = 3
MIN_FOOD_NUTRIENTS = 3
NUMERIC_TOLERANCE = 0.05
MAX_COMPONENT_CONFLICT = 0.30
SPLIT_SEED = 20260806
MASK_SEED = 20260807
MASK_RATIO = 0.50
MAX_MASKED_TARGETS_PER_FOOD_AND_TASK = 5

# Dietary fibre is included with macronutrients as a dietary bulk component.
MACRO_CODES = {"203", "204", "205", "291"}
MICRO_CODES = {
    # Minerals
    "301", "303", "304", "305", "306", "307", "309", "312", "314", "315", "316", "317",
    # Vitamins and other recognised essential micronutrients in their
    # direct-mass forms. Vitamin A RAE and other activity equivalents are
    # excluded before this classification.
    "319", "323", "328", "401", "404", "405", "406", "410", "415", "416", "417", "418",
    "421", "430", "431", "432",
}
EQUIVALENT_NAME_PATTERN = re.compile(r"\b(iu|rae|dfe|ne|te|retinol activity|activity equivalent)\b", re.I)


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def from_items(cls, items: list[str]) -> "UnionFind":
        return cls(parent={item: item for item in items})

    def find(self, item: str) -> str:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def normalize_food_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def stable_integer(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def relative_difference(left: float, right: float) -> float:
    if left == right:
        return 0.0
    denominator = max(abs(left), abs(right))
    return math.inf if denominator == 0 else abs(left - right) / denominator


def canonical_axis_class(axis_id: str) -> tuple[str, str]:
    """Return target kind and nutrient subtype without adding a model token."""
    if axis_id.startswith("fdc:"):
        code = axis_id.removeprefix("fdc:")
        if code in MACRO_CODES:
            return "nutrient", "macro_nutrient"
        if code in MICRO_CODES:
            return "nutrient", "micro_nutrient"
    return "compound", "compound"


def valid_mass_axis(unit: object, axis_name: object) -> bool:
    unit_key = str(unit).strip().lower()
    name = "" if pd.isna(axis_name) else str(axis_name)
    return unit_key in MASS_FACTORS and not EQUIVALENT_NAME_PATTERN.search(name)


def read_sr_legacy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    foods = pd.read_csv(SR_DIR / "food.csv", usecols=["fdc_id", "description", "food_category_id"], low_memory=False)
    foods = foods.rename(columns={"fdc_id": "source_food_id", "description": "food_name"})
    foods["source"] = "sr_legacy"
    foods["source_food_key"] = "sr_legacy:" + foods["source_food_id"].astype(str)
    foods["food_description"] = foods["food_name"]

    nutrients = pd.read_csv(SR_DIR / "nutrient.csv", usecols=["id", "name", "unit_name", "nutrient_nbr"], low_memory=False)
    nutrients["code_numeric"] = pd.to_numeric(nutrients["nutrient_nbr"], errors="coerce")
    nutrients = nutrients[nutrients["code_numeric"].notna()].copy()
    # SR Legacy also uses valid decimal subcomponent codes (for example 338.1
    # for lutein).  Preserve them as source-standard axis identifiers instead
    # of lossy integer casting; CNF only shares the integer official codes.
    nutrients["code"] = nutrients["code_numeric"].map(lambda value: format(float(value), "g"))
    nutrients = nutrients[nutrients.apply(lambda row: valid_mass_axis(row.unit_name, row.name), axis=1)].copy()
    nutrients["axis_id"] = "fdc:" + nutrients["code"]
    nutrients["axis_name"] = nutrients["name"]
    nutrients["source_axis_key"] = "sr_nutrient:" + nutrients["id"].astype(str)

    values = pd.read_csv(SR_DIR / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"], low_memory=False)
    values["amount"] = pd.to_numeric(values["amount"], errors="coerce")
    values = values.merge(nutrients[["id", "axis_id", "axis_name", "unit_name", "source_axis_key"]], left_on="nutrient_id", right_on="id", how="inner", validate="many_to_one")
    values["value_g_per_100g"] = values["amount"] * values["unit_name"].str.lower().map(MASS_FACTORS)
    values = values.rename(columns={"fdc_id": "source_food_id"})
    values["source"] = "sr_legacy"
    values["source_food_key"] = "sr_legacy:" + values["source_food_id"].astype(str)
    return foods, values, nutrients[["axis_id", "axis_name", "source_axis_key"]].drop_duplicates()


def read_cnf(sr_axes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    foods = pd.read_csv(CNF_DIR / "Food_Name.csv", usecols=["Food_Code", "Food_Description_EN", "CNF_Food_Group_Code"], low_memory=False)
    foods = foods.rename(columns={"Food_Code": "source_food_id", "Food_Description_EN": "food_name"})
    foods["source"] = "cnf"
    foods["source_food_key"] = "cnf:" + foods["source_food_id"].astype(str)
    foods["food_description"] = foods["food_name"]

    nutrients = pd.read_csv(CNF_DIR / "Nutrient_Name.csv", usecols=["Nutrient_Code", "Nutrient_Name_EN", "Nutrient_Unit"], low_memory=False)
    nutrients = nutrients[nutrients.apply(lambda row: valid_mass_axis(row.Nutrient_Unit, row.Nutrient_Name_EN), axis=1)].copy()
    nutrients["code"] = pd.to_numeric(nutrients["Nutrient_Code"], errors="raise").astype(int).astype(str)
    nutrients["axis_id"] = np.where(
        ("fdc:" + nutrients["code"]).isin(sr_axes),
        "fdc:" + nutrients["code"],
        "cnf:" + nutrients["code"],
    )
    nutrients["axis_name"] = nutrients["Nutrient_Name_EN"]
    nutrients["source_axis_key"] = "cnf_nutrient:" + nutrients["Nutrient_Code"].astype(str)

    values = pd.read_csv(CNF_DIR / "Nutrient_Amount.csv", usecols=["Food_Code", "Nutrient_Code", "Nutrient_Amount"], low_memory=False)
    values["Nutrient_Amount"] = pd.to_numeric(values["Nutrient_Amount"], errors="coerce")
    values = values.merge(
        nutrients[["Nutrient_Code", "axis_id", "axis_name", "Nutrient_Unit", "source_axis_key"]],
        on="Nutrient_Code", how="inner", validate="many_to_one"
    )
    values["value_g_per_100g"] = values["Nutrient_Amount"] * values["Nutrient_Unit"].str.lower().map(MASS_FACTORS)
    values = values.rename(columns={"Food_Code": "source_food_id"})
    values["source"] = "cnf"
    values["source_food_key"] = "cnf:" + values["source_food_id"].astype(str)
    return foods, values, nutrients[["axis_id", "axis_name", "source_axis_key"]].drop_duplicates()


def fooDB_mapping_tables() -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    review = pd.read_csv(CROSSWALK_PATH)
    review = review[review["review_status"].eq("accepted")].copy()
    mapping = {
        (str(row.foodb_axis_kind).lower(), str(row.foodb_axis_name)): f"fdc:{int(row.canonical_usda_code)}"
        for row in review.itertuples(index=False)
        if int(row.canonical_usda_code) != 208
    }
    return mapping, review


def read_foodb() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, _ = fooDB_mapping_tables()
    foods = pd.read_csv(FOODB_OBSERVATION_DIR / "Food.csv", usecols=["id", "name", "description", "origin_group_id", "origin_entity_key"], low_memory=False)
    foods = foods.rename(columns={"id": "source_food_id", "name": "food_name", "description": "food_description"})
    foods["source"] = "foodb"
    foods["source_food_key"] = "foodb:" + foods["source_food_id"].astype(str)

    nutrient_catalog = pd.read_csv(FOODB_RAW_DIR / "Nutrient.csv", usecols=["id", "name"], low_memory=False)
    compound_catalog = pd.read_csv(FOODB_COMPOUND_DIR / "compound_axis_registry.csv", usecols=["foodb_compound_id", "axis_id", "axis_name"], low_memory=False)
    compound_catalog = compound_catalog.rename(columns={"foodb_compound_id": "source_id"})

    content = pd.read_csv(
        FOODB_OBSERVATION_DIR / "Content.csv",
        usecols=["source_content_id", "source_id", "source_type", "food_id", "standard_content", "orig_unit"],
        low_memory=False,
    )
    nutrients = content[content["source_type"].eq("Nutrient")].copy()
    nutrients["standard_content"] = pd.to_numeric(nutrients["standard_content"], errors="coerce")
    nutrients["unit"] = nutrients["orig_unit"].fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
    nutrients = nutrients[nutrients["standard_content"].notna() & nutrients["unit"].eq("mg/100g")].copy()
    nutrients = nutrients.merge(nutrient_catalog, left_on="source_id", right_on="id", how="inner", validate="many_to_one")
    nutrients["axis_id"] = nutrients.apply(
        lambda row: mapping.get(("nutrient", str(row["name"])), f"foodb_nutrient:{int(row.source_id)}"), axis=1
    )
    nutrients["axis_name"] = nutrients["name"]
    nutrients["value_g_per_100g"] = nutrients["standard_content"] / 1000.0
    nutrients["source"] = "foodb"
    nutrients["source_food_id"] = nutrients["food_id"]
    nutrients["source_food_key"] = "foodb:" + nutrients["food_id"].astype(str)
    nutrients["source_axis_key"] = "foodb_nutrient:" + nutrients["source_id"].astype(str)

    compounds = pd.read_csv(FOODB_COMPOUND_DIR / "observed_compound_values.csv", low_memory=False)
    compounds = compounds.merge(compound_catalog[["source_id", "axis_name"]], on="source_id", how="inner", validate="many_to_one")
    compounds["axis_id"] = compounds.apply(
        lambda row: mapping.get(("compound", str(row["axis_name"])), f"foodb_compound:{int(row.source_id)}"), axis=1
    )
    compounds["source"] = "foodb"
    compounds["source_food_id"] = compounds["food_id"]
    compounds["source_food_key"] = "foodb:" + compounds["food_id"].astype(str)
    compounds["source_axis_key"] = "foodb_compound:" + compounds["source_id"].astype(str)

    value_columns = ["source", "source_food_id", "source_food_key", "axis_id", "axis_name", "source_axis_key", "value_g_per_100g"]
    values = pd.concat([nutrients[value_columns], compounds[value_columns]], ignore_index=True)
    axis_rows = pd.concat(
        [
            nutrients[["axis_id", "axis_name", "source_axis_key"]].drop_duplicates(),
            compounds[["axis_id", "axis_name", "source_axis_key"]].drop_duplicates(),
        ],
        ignore_index=True,
    )
    return foods, values, axis_rows


def remove_invalid_values(values: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    values = values.copy()
    values["value_g_per_100g"] = pd.to_numeric(values["value_g_per_100g"], errors="coerce")
    invalid = values[
        values["value_g_per_100g"].isna() | ~np.isfinite(values["value_g_per_100g"]) |
        values["value_g_per_100g"].lt(0) | values["value_g_per_100g"].gt(100)
    ].copy()
    invalid["exclusion_reason"] = np.select(
        [invalid["value_g_per_100g"].isna(), invalid["value_g_per_100g"].lt(0), invalid["value_g_per_100g"].gt(100)],
        ["non_numeric_or_non_finite", "negative_mass_value", "mass_fraction_exceeds_100g_per_100g"],
        default="unknown_invalid_value",
    )
    invalid.to_csv(audit_dir / "invalid_mass_values_after_source_loading.csv", index=False)
    return values.drop(index=invalid.index).copy()


def resolve_column_collisions(values: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    """Resolve repeated source-food/semantic-axis cells after column mapping."""
    group_columns = ["source_food_key", "axis_id"]
    stats = values.groupby(group_columns, sort=False).agg(
        record_count=("value_g_per_100g", "size"),
        minimum=("value_g_per_100g", "min"),
        maximum=("value_g_per_100g", "max"),
        median=("value_g_per_100g", "median"),
    ).reset_index()
    stats["relative_range"] = np.where(
        stats["maximum"].eq(0),
        0.0,
        (stats["maximum"] - stats["minimum"]) / stats["maximum"],
    )
    stats["consistent_within_5pct"] = stats["relative_range"].le(NUMERIC_TOLERANCE)
    conflicts = stats[stats["record_count"].gt(1) & ~stats["consistent_within_5pct"]].copy()
    conflicts.to_csv(audit_dir / "column_collision_conflicts.csv", index=False)
    retained = stats[stats["consistent_within_5pct"]].copy()
    metadata = values.sort_values(group_columns + ["source_axis_key"], kind="stable").drop_duplicates(group_columns)
    retained = retained.merge(
        metadata[["source_food_key", "axis_id", "source", "source_food_id", "axis_name", "source_axis_key"]],
        on=group_columns, how="left", validate="one_to_one"
    )
    retained = retained.rename(columns={"median": "value_g_per_100g"})
    retained["column_collision_record_count"] = retained["record_count"]
    return retained[["source", "source_food_id", "source_food_key", "axis_id", "axis_name", "source_axis_key", "value_g_per_100g", "column_collision_record_count"]]


def make_food_value_maps(values: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        food_key: dict(zip(group["axis_id"], group["value_g_per_100g"]))
        for food_key, group in values.groupby("source_food_key", sort=False)
    }


def compare_food_pair(left: str, right: str, value_maps: dict[str, dict[str, float]]) -> tuple[int, int, float]:
    left_values, right_values = value_maps[left], value_maps[right]
    common_axes = left_values.keys() & right_values.keys()
    if not common_axes:
        return 0, 0, 0.0
    agreements = sum(relative_difference(left_values[axis], right_values[axis]) <= NUMERIC_TOLERANCE for axis in common_axes)
    return len(common_axes), agreements, agreements / len(common_axes)


def candidate_pairs(foods: pd.DataFrame) -> pd.DataFrame:
    """Generate exact-name candidates, then high-recall similar-name candidates."""
    rows: dict[tuple[str, str], dict[str, object]] = {}
    valid = foods[foods["normalized_food_name"].str.len().ge(3)].copy()
    for _, group in valid.groupby("normalized_food_name", sort=False):
        if len(group) < 2:
            continue
        for left, right in combinations(sorted(group["source_food_key"]), 2):
            rows[(left, right)] = {"left_food_key": left, "right_food_key": right, "name_method": "exact", "name_similarity": 1.0, "token_jaccard": 1.0}

    names = valid[["source_food_key", "normalized_food_name"]].drop_duplicates("source_food_key").reset_index(drop=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2")
    matrix = vectorizer.fit_transform(names["normalized_food_name"])
    neighbors = NearestNeighbors(n_neighbors=min(7, len(names)), metric="cosine", algorithm="brute").fit(matrix)
    distances, indices = neighbors.kneighbors(matrix, return_distance=True)
    for index, candidate_indices in enumerate(indices):
        left_key = names.at[index, "source_food_key"]
        left_name = names.at[index, "normalized_food_name"]
        for distance, candidate_index in zip(distances[index], candidate_indices):
            if candidate_index == index:
                continue
            right_key = names.at[candidate_index, "source_food_key"]
            right_name = names.at[candidate_index, "normalized_food_name"]
            similarity = 1.0 - float(distance)
            jaccard = token_jaccard(left_name, right_name)
            if similarity < 0.78 or jaccard < 0.50:
                continue
            key = tuple(sorted((left_key, right_key)))
            if key not in rows:
                rows[key] = {"left_food_key": key[0], "right_food_key": key[1], "name_method": "similar", "name_similarity": similarity, "token_jaccard": jaccard}
    return pd.DataFrame(rows.values())


def deduplicate_food_rows(foods: pd.DataFrame, values: pd.DataFrame, audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_maps = make_food_value_maps(values)
    candidates = candidate_pairs(foods)
    comparisons = []
    for candidate in candidates.itertuples(index=False):
        shared, agreements, ratio = compare_food_pair(candidate.left_food_key, candidate.right_food_key, value_maps)
        conflict_fraction = 1.0 - ratio if shared else 1.0
        accepted = shared >= 3 and conflict_fraction <= MAX_COMPONENT_CONFLICT
        comparisons.append({
            **candidate._asdict(),
            "shared_axis_count": shared,
            "within_5pct_count": agreements,
            "agreement_fraction": ratio,
            "conflict_fraction": conflict_fraction,
            "accepted_duplicate": accepted,
            "decision": "merge" if accepted else "retain_separate",
            "decision_reason": (
                "At least three shared axes and no more than 30% disagree beyond 5%."
                if accepted else "Insufficient shared evidence or more than 30% of shared axes differ beyond 5%."
            ),
        })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(audit_dir / "row_duplicate_candidates.csv", index=False)

    union_find = UnionFind.from_items(foods["source_food_key"].tolist())
    if not comparison_frame.empty:
        for row in comparison_frame[comparison_frame["accepted_duplicate"]].itertuples(index=False):
            union_find.union(row.left_food_key, row.right_food_key)
    foods = foods.copy()
    foods["duplicate_component_root"] = foods["source_food_key"].map(union_find.find)
    component_members = foods.groupby("duplicate_component_root", sort=False)["source_food_key"].apply(list).to_dict()
    ordered_roots = sorted(component_members, key=lambda root: min(component_members[root]))
    canonical_id_by_root = {root: f"food:{index:06d}" for index, root in enumerate(ordered_roots, start=1)}
    foods["canonical_food_id"] = foods["duplicate_component_root"].map(canonical_id_by_root)

    canonical_by_source = foods.set_index("source_food_key")["canonical_food_id"].to_dict()
    values = values.copy()
    values["canonical_food_id"] = values["source_food_key"].map(canonical_by_source)
    resolution_keys = ["canonical_food_id", "axis_id"]
    statistics = values.groupby(resolution_keys, sort=False).agg(
        axis_name=("axis_name", "first"),
        median_value=("value_g_per_100g", "median"),
        minimum_input_value_g_per_100g=("value_g_per_100g", "min"),
        maximum_input_value_g_per_100g=("value_g_per_100g", "max"),
        source_record_count=("source_food_key", "size"),
        source_count=("source", "nunique"),
    ).reset_index()
    # For a conflicting cell, select the highest-priority source.  Column
    # collisions were already resolved, therefore each source-food contributes
    # at most one value for this semantic axis.
    priority_values = values.assign(_source_priority=values["source"].map(SOURCE_PRIORITY)).sort_values(
        resolution_keys + ["_source_priority", "source_food_key"], ascending=[True, True, False, True], kind="stable"
    ).drop_duplicates(resolution_keys)
    priority_values = priority_values[resolution_keys + ["value_g_per_100g", "source"]].rename(
        columns={"value_g_per_100g": "priority_value", "source": "priority_source"}
    )
    resolved_values = statistics.merge(priority_values, on=resolution_keys, how="inner", validate="one_to_one")
    resolved_values["relative_range"] = np.where(
        resolved_values["maximum_input_value_g_per_100g"].eq(0),
        0.0,
        (resolved_values["maximum_input_value_g_per_100g"] - resolved_values["minimum_input_value_g_per_100g"])
        / resolved_values["maximum_input_value_g_per_100g"],
    )
    agreeing = resolved_values["relative_range"].le(NUMERIC_TOLERANCE)
    resolved_values["value_g_per_100g"] = np.where(agreeing, resolved_values["median_value"], resolved_values["priority_value"])
    resolved_values["value_resolution"] = np.where(
        agreeing, "median_agreeing_records", "source_priority:" + resolved_values["priority_source"]
    )
    resolved_values = resolved_values.drop(columns=["median_value", "priority_value", "priority_source", "relative_range"])

    members = foods.sort_values(["canonical_food_id", "source_food_key"], kind="stable").copy()
    component_sizes = members.groupby("canonical_food_id")["source_food_key"].size().rename("duplicate_component_size")
    members = members.merge(component_sizes, on="canonical_food_id", how="left")
    primary = members.sort_values(
        ["canonical_food_id", "source"],
        key=lambda column: column.map(SOURCE_PRIORITY) if column.name == "source" else column,
        ascending=[True, False],
        kind="stable",
    ).drop_duplicates("canonical_food_id")
    canonical_foods = primary[["canonical_food_id", "source", "source_food_key", "food_name", "food_description", "duplicate_component_size"]].rename(
        columns={"source": "primary_source", "source_food_key": "primary_source_food_key"}
    )
    members.to_csv(audit_dir / "row_duplicate_components.csv", index=False)
    resolved_values[resolved_values["value_resolution"].str.startswith("source_priority")].to_csv(
        audit_dir / "row_deduplication_source_priority_resolutions.csv", index=False
    )
    return canonical_foods, resolved_values, members


def stable_filters(foods: pd.DataFrame, values: pd.DataFrame, axis_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the requested support and food thresholds to convergence."""
    active_foods = foods.copy()
    active_values = values.copy()
    while True:
        before = (len(active_foods), len(active_values), active_values["axis_id"].nunique())
        support = active_values.groupby("axis_id")["canonical_food_id"].nunique()
        active_values = active_values[active_values["axis_id"].isin(support[support >= MIN_AXIS_SUPPORT].index)].copy()
        total_counts = active_values.groupby("canonical_food_id")["axis_id"].nunique()
        active_foods = active_foods[active_foods["canonical_food_id"].isin(total_counts[total_counts >= MIN_FOOD_VALUES].index)].copy()
        active_values = active_values[active_values["canonical_food_id"].isin(active_foods["canonical_food_id"])].copy()
        nutrients = set(axis_registry.loc[axis_registry["target_kind"].eq("nutrient"), "axis_id"])
        nutrient_counts = active_values[active_values["axis_id"].isin(nutrients)].groupby("canonical_food_id")["axis_id"].nunique()
        active_foods = active_foods[active_foods["canonical_food_id"].isin(nutrient_counts[nutrient_counts >= MIN_FOOD_NUTRIENTS].index)].copy()
        active_values = active_values[active_values["canonical_food_id"].isin(active_foods["canonical_food_id"])].copy()
        after = (len(active_foods), len(active_values), active_values["axis_id"].nunique())
        if after == before:
            break
    support = active_values.groupby("axis_id").agg(
        observed_foods=("canonical_food_id", "nunique"),
        positive_foods=("value_g_per_100g", lambda values: int((values > 0).sum())),
        explicit_zero_foods=("value_g_per_100g", lambda values: int((values == 0).sum())),
    ).reset_index()
    axis_registry = axis_registry.merge(support, on="axis_id", how="inner", validate="one_to_one")
    axis_registry["mask_policy"] = np.where(
        axis_registry["observed_foods"].ge(MASKABLE_AXIS_SUPPORT) & axis_registry["positive_foods"].ge(MIN_AXIS_SUPPORT),
        "maskable_target", "context_only"
    )
    return active_foods, active_values, axis_registry


def split_foods(foods: pd.DataFrame) -> dict[str, list[str]]:
    labels = foods["primary_source"].astype(str)
    if labels.value_counts().min() < 3:
        raise ValueError("At least three foods are required in every source stratum for a 70/15/15 split.")
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SPLIT_SEED)
    train_positions, temporary_positions = next(first.split(foods, labels))
    temporary = foods.iloc[temporary_positions].reset_index(drop=True)
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SPLIT_SEED + 1)
    validation_positions, test_positions = next(second.split(temporary, temporary["primary_source"]))
    return {
        "train": sorted(foods.iloc[train_positions]["canonical_food_id"].tolist()),
        "validation": sorted(temporary.iloc[validation_positions]["canonical_food_id"].tolist()),
        "test": sorted(temporary.iloc[test_positions]["canonical_food_id"].tolist()),
    }


def fit_train_normalization(values: pd.DataFrame, train_ids: set[str], axis_ids: set[str]) -> pd.DataFrame:
    rows = []
    train_values = values[values["canonical_food_id"].isin(train_ids) & values["axis_id"].isin(axis_ids)]
    for axis_id, group in train_values.groupby("axis_id", sort=True):
        positive = group.loc[group["value_g_per_100g"] > 0, "value_g_per_100g"].to_numpy(dtype=float)
        transformed = np.log1p(positive)
        median = float(np.median(transformed)) if len(transformed) else 0.0
        q1, q3 = (np.quantile(transformed, [0.25, 0.75]) if len(transformed) else (0.0, 0.0))
        scale = max(float(q3 - q1), 1e-3)
        rows.append({"axis_id": axis_id, "transform": "log1p", "median": median, "iqr_scale": scale, "positive_train_foods": len(positive)})
    return pd.DataFrame(rows)


def build_sequential_masks(values: pd.DataFrame, axis_registry: pd.DataFrame, split_ids: dict[str, list[str]]) -> pd.DataFrame:
    target_axes = axis_registry[axis_registry["mask_policy"].eq("maskable_target")][["axis_id", "target_kind"]]
    candidate = values.merge(target_axes, on="axis_id", how="inner", validate="many_to_one")
    candidate = candidate[candidate["value_g_per_100g"].gt(0)].copy()
    rows = []
    for split_name in ("validation", "test"):
        split_candidate = candidate[candidate["canonical_food_id"].isin(split_ids[split_name])]
        for (food_id, target_kind), group in split_candidate.groupby(["canonical_food_id", "target_kind"], sort=True):
            ordered = group.sort_values("axis_id", kind="stable").copy()
            count = min(MAX_MASKED_TARGETS_PER_FOOD_AND_TASK, max(1, math.ceil(len(ordered) * MASK_RATIO)))
            rng = np.random.default_rng(MASK_SEED + stable_integer(f"{split_name}|{food_id}|{target_kind}") % (2**32))
            selected = ordered.iloc[np.sort(rng.choice(len(ordered), size=count, replace=False))].sort_values("axis_id", kind="stable")
            for order, row in enumerate(selected.itertuples(index=False), start=1):
                rows.append({
                    "split": split_name,
                    "canonical_food_id": food_id,
                    "axis_id": row.axis_id,
                    "target_kind": target_kind,
                    "prediction_order": order,
                    "mask_ratio_requested": MASK_RATIO,
                    "max_targets_per_food_and_task": MAX_MASKED_TARGETS_PER_FOOD_AND_TASK,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    data_dir, split_dir, audit_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.audit_dir.resolve()
    for path in (data_dir, split_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    sr_foods, sr_values, sr_axes = read_sr_legacy()
    cnf_foods, cnf_values, cnf_axes = read_cnf(set(sr_axes["axis_id"]))
    foodb_foods, foodb_values, foodb_axes = read_foodb()
    foods = pd.concat([sr_foods, cnf_foods, foodb_foods], ignore_index=True, sort=False)
    foods["normalized_food_name"] = foods["food_name"].map(normalize_food_name)
    raw_values = pd.concat([sr_values, cnf_values, foodb_values], ignore_index=True, sort=False)
    raw_values.to_csv(audit_dir / "row_concatenated_mass_values.csv", index=False)
    foods.to_csv(audit_dir / "row_concatenated_foods.csv", index=False)

    raw_values = remove_invalid_values(raw_values, audit_dir)
    column_values = resolve_column_collisions(raw_values, audit_dir)
    foods_without_retained_values = foods[~foods["source_food_key"].isin(column_values["source_food_key"])].copy()
    foods_without_retained_values.to_csv(audit_dir / "foods_without_retained_mass_values.csv", index=False)
    foods = foods[foods["source_food_key"].isin(column_values["source_food_key"])].copy()
    axis_names = pd.concat([sr_axes, cnf_axes, foodb_axes], ignore_index=True, sort=False)
    axis_names = axis_names.sort_values(["axis_id", "axis_name", "source_axis_key"], kind="stable").drop_duplicates("axis_id")
    classifications = axis_names["axis_id"].map(canonical_axis_class)
    axis_names[["target_kind", "axis_class"]] = pd.DataFrame(classifications.tolist(), index=axis_names.index)
    axis_names["model_unit"] = "g/100g"
    axis_names.to_csv(audit_dir / "axis_registry_after_column_deduplication.csv", index=False)

    canonical_foods, canonical_values, duplicate_members = deduplicate_food_rows(foods, column_values, audit_dir)
    prefilter_axis_registry = axis_names.copy()
    final_foods, final_values, final_axis_registry = stable_filters(canonical_foods, canonical_values, prefilter_axis_registry)
    final_axis_registry = final_axis_registry.sort_values("axis_id", kind="stable").reset_index(drop=True)
    final_axis_registry["axis_index"] = np.arange(len(final_axis_registry), dtype=int)

    splits = split_foods(final_foods)
    split_rows = [{"canonical_food_id": food_id, "split": split} for split, food_ids in splits.items() for food_id in food_ids]
    split_frame = pd.DataFrame(split_rows)
    final_foods = final_foods.merge(split_frame, on="canonical_food_id", how="inner", validate="one_to_one")
    normalization = fit_train_normalization(final_values, set(splits["train"]), set(final_axis_registry["axis_id"]))
    masks = build_sequential_masks(final_values, final_axis_registry, splits)

    final_foods.to_csv(data_dir / "food_entities.csv", index=False)
    final_values.to_csv(data_dir / "observed_axis_values.csv", index=False)
    final_axis_registry.to_csv(data_dir / "axis_registry.csv", index=False)
    normalization.to_csv(data_dir / "train_only_axis_normalization.csv", index=False)
    duplicate_members[duplicate_members["canonical_food_id"].isin(final_foods["canonical_food_id"])].to_csv(data_dir / "duplicate_group_members.csv", index=False)
    masks.to_csv(split_dir / "sequential_masks.csv", index=False)
    final_axis_registry[final_axis_registry["mask_policy"].eq("maskable_target")][["axis_id", "target_kind", "axis_class"]].to_csv(split_dir / "loss_axis_ids.csv", index=False)
    with (split_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol_version": "multisource_mass_v4", "seed": SPLIT_SEED, **splits}, handle, indent=2)

    audit = {
        "protocol_version": "multisource_mass_v4",
        "sources": ["sr_legacy", "cnf", "foodb"],
        "mass_unit": "g/100g",
        "source_priority": ["foodb", "sr_legacy", "cnf"],
        "numeric_agreement_relative_tolerance": NUMERIC_TOLERANCE,
        "maximum_conflicting_shared_axis_fraction_for_row_merge": MAX_COMPONENT_CONFLICT,
        "minimum_axis_support": MIN_AXIS_SUPPORT,
        "maskable_axis_support": MASKABLE_AXIS_SUPPORT,
        "mask_ratio": MASK_RATIO,
        "max_sequential_targets_per_food_and_task": MAX_MASKED_TARGETS_PER_FOOD_AND_TASK,
        "row_concatenated_foods": int(len(foods)),
        "row_concatenated_values": int(len(raw_values)),
        "foods_after_row_deduplication": int(len(canonical_foods)),
        "final_foods": int(len(final_foods)),
        "final_axes": int(len(final_axis_registry)),
        "final_observed_values": int(len(final_values)),
        "final_axis_classes": final_axis_registry["axis_class"].value_counts().to_dict(),
        "final_mask_policies": final_axis_registry["mask_policy"].value_counts().to_dict(),
        "food_sources_after_filtering": final_foods["primary_source"].value_counts().to_dict(),
        "split_counts": split_frame["split"].value_counts().to_dict(),
        "sequential_mask_counts": {
            f"{split}:{target_kind}": int(count)
            for (split, target_kind), count in masks.groupby(["split", "target_kind"]).size().items()
        },
    }
    with (audit_dir / "final_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)
    pd.DataFrame([audit]).to_csv(audit_dir / "final_audit_summary.csv", index=False)
    print(json.dumps(audit, indent=2, default=str))
    print(f"Data: {data_dir}")
    print(f"Splits: {split_dir}")
    print(f"Audits: {audit_dir}")


if __name__ == "__main__":
    main()

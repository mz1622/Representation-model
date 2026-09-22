#!/usr/bin/env python3
"""Build FoodNutriGPT v5: a provenance-audited nutrition/form corpus.

v5 reuses the vetted raw SR Legacy, CNF 2026, and FooDB observations but does
not reuse v4's binary nutrient/compound classification.  Every retained axis
is placed in one of two trainable layers:

* core_nutrition: four macronutrients and the mass-scale core micronutrients;
* nutrient_chemical_form: fatty acids, amino acids, carbohydrate fractions,
  and nutrient forms.

Bioactive chemistry and structural/derived components are exported in an
exclusion catalogue rather than silently treated as equivalent regression
targets.  Missing observations remain absent throughout.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import build_multisource_mass_corpus_v4 as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/processed/multisource_nutrition_v5"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/multisource_nutrition_v5"
DEFAULT_AUDIT_DIR = ROOT / "data/audits/multisource_nutrition_v5"
VMH_CROSSWALK = ROOT / "data/audits/vmh_nutrition_schema_2026_08/vmh_to_local_crosswalk.csv"

CORE_MACRO_CODES = {"203", "204", "205", "291"}
CORE_MICRO_CODES = {
    "301", "302", "303", "304", "305", "306", "307", "309", "310", "312", "313", "314", "315", "316", "317",
    "319", "323", "328", "401", "404", "405", "406", "410", "415", "416", "417", "418", "421", "430",
}
CORE_SPECIAL_NAMES = {
    "core:chloride": "Chloride",
    "core:chromium": "Chromium",
    "core:iodine": "Iodine",
    "core:molybdenum": "Molybdenum",
}
CANONICAL_CORE_IDS = (
    {f"fdc:{code}" for code in CORE_MACRO_CODES}
    | {f"fdc:{code}" for code in CORE_MICRO_CODES - {"302", "310", "314", "316"}}
    | set(CORE_SPECIAL_NAMES)
)
STRUCTURAL_CODES = {"207", "221", "255"}
ACTIVITY_EQUIVALENT_CODES = {"318", "320", "324", "435"}
BIOACTIVE_FDC_CODES = {"262", "263"}

# These are exact name/chemical-entity mappings verified against the FooDB
# compound catalogue.  Form-specific entries (for example pyridoxine and
# folic acid) deliberately stay separate from aggregate nutrient targets.
FOODB_CORE_COMPOUND_MAP = {
    "3514": "fdc:301",       # Calcium
    "16258": "fdc:303",      # Iron
    "3521": "fdc:305",       # Phosphorus
    "3522": "fdc:306",       # Potassium
    "3524": "fdc:307",       # Sodium
    "3730": "fdc:309",       # Zinc
    "3583": "fdc:312",       # Copper
    "4486": "fdc:313",       # Fluoride
    "3637": "fdc:315",       # Manganese
    "13403": "fdc:317",      # Selenium
    "13831": "fdc:319",      # Retinol
    "565": "fdc:323",        # alpha-Tocopherol
    "1224": "fdc:401",       # L-Ascorbic acid
    "8425": "fdc:404",       # Thiamine
    "12163": "fdc:405",      # Riboflavin (FooDB spelling: Riboflavine)
    "8323": "fdc:410",       # Pantothenic acid
    "14513": "fdc:416",      # Biotin
    "23049": "fdc:418",      # Cobalamin
    "710": "fdc:421",        # Choline
    "12360": "fdc:430",      # Phytomenadione / vitamin K1
    "6558": "core:chloride", # Chloride
    "3517": "core:chromium", # Chromium
    "3636": "core:iodine",   # Iodine
    "3654": "core:molybdenum",  # Molybdenum
}

# CNF keeps biotin under an identifier that is absent from SR Legacy.  The
# identifier is nevertheless the same FDC nutrient identity, so leaving it as
# ``cnf:416`` would create a duplicate target alongside FooDB's ``fdc:416``.
CNF_EXACT_CORE_AXIS_MAP = {
    "416": "fdc:416",  # Biotin
}

CANONICAL_AXIS_NAMES = {
    "fdc:301": "Calcium", "fdc:303": "Iron", "fdc:305": "Phosphorus", "fdc:306": "Potassium",
    "fdc:307": "Sodium", "fdc:309": "Zinc", "fdc:312": "Copper", "fdc:313": "Fluoride",
    "fdc:315": "Manganese", "fdc:317": "Selenium", "fdc:319": "Retinol",
    "fdc:323": "Vitamin E (alpha-tocopherol)", "fdc:401": "Vitamin C, total ascorbic acid",
    "fdc:404": "Thiamin", "fdc:405": "Riboflavin", "fdc:410": "Pantothenic acid",
    "fdc:416": "Biotin", "fdc:418": "Vitamin B-12", "fdc:421": "Choline, total",
    "fdc:430": "Vitamin K (phylloquinone)", **CORE_SPECIAL_NAMES,
}
FAMILY_BY_AXIS = {
    "fdc:417": "folate", "fdc:431": "folate", "fdc:432": "folate", "fdc:435": "folate",
    "fdc:328": "vitamin_d", "fdc:325": "vitamin_d", "fdc:326": "vitamin_d",
}


def source_axis_id_count(values: pd.DataFrame) -> pd.DataFrame:
    grouped = values.groupby(["source", "axis_id"], sort=True).agg(
        source_observed_food_rows=("source_food_key", "nunique"),
        source_positive_food_rows=("value_g_per_100g", lambda data: int((data > 0).sum())),
        source_zero_food_rows=("value_g_per_100g", lambda data: int((data == 0).sum())),
        source_axis_keys=("source_axis_key", lambda data: "; ".join(sorted(set(data.astype(str))))),
    ).reset_index()
    return grouped


def remap_foodb_core_values(values: pd.DataFrame) -> pd.DataFrame:
    values = values.copy()
    compound_id = values["source_axis_key"].str.extract(r"^foodb_compound:(\d+)$", expand=False)
    replacement = compound_id.map(FOODB_CORE_COMPOUND_MAP)
    mapped = replacement.notna()
    values.loc[mapped, "axis_id"] = replacement[mapped]
    values.loc[mapped, "axis_name"] = values.loc[mapped, "axis_id"].map(CANONICAL_AXIS_NAMES)
    return values


def remap_cnf_exact_core_values(values: pd.DataFrame) -> pd.DataFrame:
    """Map CNF-only exact core identities to the canonical FDC axis ID."""
    values = values.copy()
    nutrient_code = values["source_axis_key"].str.extract(r"^cnf_nutrient:(\d+)$", expand=False)
    replacement = nutrient_code.map(CNF_EXACT_CORE_AXIS_MAP)
    mapped = replacement.notna()
    values.loc[mapped, "axis_id"] = replacement[mapped]
    values.loc[mapped, "axis_name"] = values.loc[mapped, "axis_id"].map(CANONICAL_AXIS_NAMES)
    return values


def vmh_category_by_axis() -> dict[str, str]:
    if not VMH_CROSSWALK.exists():
        raise FileNotFoundError(
            f"VMH crosswalk not found: {VMH_CROSSWALK}. Run audit_vmh_nutrition_schema.py before building v5."
        )
    crosswalk = pd.read_csv(VMH_CROSSWALK)
    return dict(zip(crosswalk["axis_id"].dropna(), crosswalk["vmh_category"].dropna()))


def classify_axis(axis_id: str, axis_name: str, vmh_categories: dict[str, str]) -> dict[str, str]:
    """Classify every source axis without relying on a model token."""
    if axis_id in CORE_SPECIAL_NAMES:
        return {
            "training_layer": "core_nutrition", "target_kind": "core_nutrition", "axis_class": "micro_nutrient",
            "classification_evidence": "Exact FooDB elemental compound mapping; core micronutrient definition.",
            "exclusion_reason": "", "mask_family": axis_id,
        }
    if axis_id.startswith("fdc:"):
        code = axis_id.removeprefix("fdc:")
        if code in CORE_MACRO_CODES:
            return {
                "training_layer": "core_nutrition", "target_kind": "core_nutrition", "axis_class": "macro_nutrient",
                "classification_evidence": "USDA nutrient identifier; core macronutrient schema.", "exclusion_reason": "", "mask_family": axis_id,
            }
        if code in CORE_MICRO_CODES:
            return {
                "training_layer": "core_nutrition", "target_kind": "core_nutrition", "axis_class": "micro_nutrient",
                "classification_evidence": "USDA nutrient identifier; core mass-scale micronutrient schema.", "exclusion_reason": "", "mask_family": FAMILY_BY_AXIS.get(axis_id, axis_id),
            }
        if code in STRUCTURAL_CODES:
            return {
                "training_layer": "excluded", "target_kind": "excluded", "axis_class": "structural_or_derived",
                "classification_evidence": "VMH/USDA structural component.",
                "exclusion_reason": "Water, ash, and alcohol are not nutrient mass targets in v5.", "mask_family": axis_id,
            }
        if code in ACTIVITY_EQUIVALENT_CODES:
            return {
                "training_layer": "excluded", "target_kind": "excluded", "axis_class": "activity_equivalent",
                "classification_evidence": "USDA vitamin activity-equivalent identifier.",
                "exclusion_reason": "RAE, DFE, and IU are not chemically interchangeable mass values.", "mask_family": FAMILY_BY_AXIS.get(axis_id, axis_id),
            }
        if code in BIOACTIVE_FDC_CODES:
            return {
                "training_layer": "excluded", "target_kind": "excluded", "axis_class": "bioactive_chemistry",
                "classification_evidence": "VMH/USDA other component.",
                "exclusion_reason": "Bioactive alkaloid; excluded until a coherent bioactive benchmark is defined.", "mask_family": axis_id,
            }
        category = vmh_categories.get(axis_id, "USDA mass composition")
        return {
            "training_layer": "nutrient_chemical_form", "target_kind": "nutrient_chemical_form", "axis_class": "nutrient_chemical_form",
            "classification_evidence": f"USDA identifier; VMH category: {category}.", "exclusion_reason": "", "mask_family": FAMILY_BY_AXIS.get(axis_id, axis_id),
        }
    if axis_id.startswith("cnf:"):
        if axis_name.strip().lower() == "aspartame":
            return {
                "training_layer": "excluded", "target_kind": "excluded", "axis_class": "food_additive",
                "classification_evidence": "CNF-specific nutrient-name record.",
                "exclusion_reason": "Food additive, not a nutrient target.", "mask_family": axis_id,
            }
        return {
            "training_layer": "nutrient_chemical_form", "target_kind": "nutrient_chemical_form", "axis_class": "nutrient_chemical_form",
            "classification_evidence": "CNF-specific mass nutrient-form record.", "exclusion_reason": "", "mask_family": axis_id,
        }
    if axis_id.startswith("foodb_nutrient:"):
        if axis_name.strip().lower() in {"energy", "ash"}:
            return {
                "training_layer": "excluded", "target_kind": "excluded", "axis_class": "structural_or_derived",
                "classification_evidence": "FooDB Nutrient table label.",
                "exclusion_reason": "Energy is non-mass; ash is a residue measurement.", "mask_family": axis_id,
            }
        return {
            "training_layer": "nutrient_chemical_form", "target_kind": "nutrient_chemical_form", "axis_class": "nutrient_chemical_form",
            "classification_evidence": "FooDB Nutrient table fatty-acid/form label without an accepted exact crosswalk.",
            "exclusion_reason": "", "mask_family": axis_id,
        }
    if axis_id.startswith("foodb_compound:"):
        return {
            "training_layer": "excluded", "target_kind": "excluded", "axis_class": "bioactive_chemistry",
            "classification_evidence": "FooDB Compound table record without an accepted nutrient-entity mapping.",
            "exclusion_reason": "Sparse/method-dependent bioactive chemistry; retained in raw provenance only.", "mask_family": axis_id,
        }
    raise ValueError(f"Unclassified axis identifier: {axis_id}")


def build_axis_catalogue(values: pd.DataFrame, vmh_categories: dict[str, str]) -> pd.DataFrame:
    source_keys = values.groupby("axis_id", sort=True).agg(
        axis_name=("axis_name", "first"),
        source_axis_keys=("source_axis_key", lambda keys: "; ".join(sorted(set(keys.astype(str))))),
    ).reset_index()
    details = source_keys.apply(lambda row: classify_axis(row.axis_id, row.axis_name, vmh_categories), axis=1)
    return pd.concat([source_keys, pd.DataFrame(details.tolist())], axis=1)


def stable_filters(foods: pd.DataFrame, values: pd.DataFrame, axis_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_foods, active_values = foods.copy(), values.copy()
    core_axes = set(axis_registry.loc[axis_registry["target_kind"].eq("core_nutrition"), "axis_id"])
    while True:
        before = (len(active_foods), len(active_values), active_values["axis_id"].nunique())
        support = active_values.groupby("axis_id")["canonical_food_id"].nunique()
        active_values = active_values[active_values["axis_id"].isin(support[support >= base.MIN_AXIS_SUPPORT].index)].copy()
        total_counts = active_values.groupby("canonical_food_id")["axis_id"].nunique()
        active_foods = active_foods[active_foods["canonical_food_id"].isin(total_counts[total_counts >= base.MIN_FOOD_VALUES].index)].copy()
        active_values = active_values[active_values["canonical_food_id"].isin(active_foods["canonical_food_id"])].copy()
        core_counts = active_values[active_values["axis_id"].isin(core_axes)].groupby("canonical_food_id")["axis_id"].nunique()
        active_foods = active_foods[active_foods["canonical_food_id"].isin(core_counts[core_counts >= base.MIN_FOOD_NUTRIENTS].index)].copy()
        active_values = active_values[active_values["canonical_food_id"].isin(active_foods["canonical_food_id"])].copy()
        after = (len(active_foods), len(active_values), active_values["axis_id"].nunique())
        if after == before:
            break
    support = active_values.groupby("axis_id").agg(
        observed_foods=("canonical_food_id", "nunique"),
        positive_foods=("value_g_per_100g", lambda data: int((data > 0).sum())),
        explicit_zero_foods=("value_g_per_100g", lambda data: int((data == 0).sum())),
    ).reset_index()
    registry = axis_registry.merge(support, on="axis_id", how="inner", validate="one_to_one")
    registry["mask_policy"] = np.where(
        registry["observed_foods"].ge(base.MASKABLE_AXIS_SUPPORT) & registry["positive_foods"].ge(base.MIN_AXIS_SUPPORT),
        "maskable_target", "context_only",
    )
    return active_foods, active_values, registry


def build_sequential_masks(values: pd.DataFrame, axis_registry: pd.DataFrame, split_ids: dict[str, list[str]]) -> pd.DataFrame:
    target_axes = axis_registry[axis_registry["mask_policy"].eq("maskable_target")][["axis_id", "target_kind", "mask_family"]]
    candidate = values.merge(target_axes, on="axis_id", how="inner", validate="many_to_one")
    candidate = candidate[candidate["value_g_per_100g"].gt(0)].copy()
    rows = []
    for split_name in ("validation", "test"):
        split_candidate = candidate[candidate["canonical_food_id"].isin(split_ids[split_name])]
        for (food_id, target_kind), group in split_candidate.groupby(["canonical_food_id", "target_kind"], sort=True):
            # One target per aggregate/component family prevents a later
            # sequential prediction from revealing an earlier target.
            representatives = group.sort_values("axis_id", kind="stable").drop_duplicates("mask_family", keep="first")
            count = min(base.MAX_MASKED_TARGETS_PER_FOOD_AND_TASK, max(1, math.ceil(len(representatives) * base.MASK_RATIO)))
            rng = np.random.default_rng(base.MASK_SEED + base.stable_integer(f"v5|{split_name}|{food_id}|{target_kind}") % (2**32))
            selected = representatives.iloc[np.sort(rng.choice(len(representatives), size=count, replace=False))].sort_values("axis_id", kind="stable")
            for order, row in enumerate(selected.itertuples(index=False), start=1):
                rows.append({
                    "split": split_name, "canonical_food_id": food_id, "axis_id": row.axis_id,
                    "target_kind": target_kind, "mask_family": row.mask_family, "prediction_order": order,
                    "mask_ratio_requested": base.MASK_RATIO, "max_targets_per_food_and_task": base.MAX_MASKED_TARGETS_PER_FOOD_AND_TASK,
                })
    return pd.DataFrame(rows)


def write_report(path: Path, all_axes: pd.DataFrame, retained: pd.DataFrame, source_coverage: pd.DataFrame, foods: pd.DataFrame) -> None:
    source_pivot = source_coverage.pivot_table(index="axis_id", columns="source", values="source_observed_food_rows", aggfunc="sum", fill_value=0)
    core = retained[retained["target_kind"].eq("core_nutrition")]
    micro = core[core["axis_class"].eq("micro_nutrient")]
    macro = core[core["axis_class"].eq("macro_nutrient")]
    layers = retained.groupby(["training_layer", "axis_class"]).size().sort_values(ascending=False)
    missing_core = sorted(CANONICAL_CORE_IDS - set(core["axis_id"]))
    lines = [
        "# FoodNutriGPT v5 Axis Provenance and Classification",
        "",
        "## Scope and Completeness",
        "",
        "This audit is complete for every mass-scale axis emitted by the current three raw sources: USDA SR Legacy, CNF 2026, and FooDB. It is not a claim that all food chemistry is completely measured. The latter is not scientifically possible with these sources, so sparse FooDB bioactives are explicitly excluded from the v5 training matrix rather than being treated as missing nutrient labels.",
        "",
        f"The source concatenation contains {len(all_axes):,} unique mass axes before semantic exclusion. After removing structural/derived measures and unsupported bioactive chemistry, v5 retains {len(retained):,} trainable or context axes across {len(foods):,} food entities.",
        "",
        "## Scientific Classification Rule",
        "",
        "1. Core nutrition starts from four macronutrient components and 29 candidate mass-scale micronutrients; only axes meeting the support threshold are retained. The schema separates chemical mass from activity equivalents: RAE, DFE, TE and IU are excluded.",
        "2. Nutrient chemical forms include fatty acids, amino acids, carbohydrate fractions and vitamin forms. They are trained in Stage 2 and linked to aggregate targets by a mask family where a direct aggregate/component relationship exists.",
        "3. Bioactive chemistry is not discarded from provenance, but is not used as a v5 prediction target because sparse analytical coverage and heterogeneous assay definitions make a single compound MSE invalid.",
        "4. Water, ash, alcohol and activity-equivalent measures are not mass-nutrient targets.",
        "",
        "The classification evidence for every individual axis is available in `all_axis_classification.csv`; VMH/USDA IDs establish canonical identities where available, while exact FooDB mappings are recorded separately from unreviewed FooDB compounds.",
        "",
        "## Retained Core Schema",
        "",
        f"Retained macronutrients: {len(macro)}. Retained micronutrients: {len(micro)}. Missing core axes after current-source filtering: {', '.join(missing_core) if missing_core else 'none'}.",
        "",
        "| Layer and class | Axis count |",
        "|---|---:|",
        *[f"| {layer} / {axis_class} | {count:,} |" for (layer, axis_class), count in layers.items()],
        "",
        "## Data Sources",
        "",
        "Each retained axis has source-level coverage in `axis_source_coverage_pre_row_dedup.csv`, before food deduplication. This is intentional: row deduplication chooses one canonical food representation but must not erase evidence that an axis was measured by more than one source. Source axes are provided in `source_axis_keys`.",
        "",
        "| Source | Role in v5 |",
        "|---|---|",
        "| USDA SR Legacy | Main U.S. food composition source; official USDA nutrient identifiers. |",
        "| CNF 2026 | Independent Canadian food-composition source; shared USDA identifiers are merged only by exact code. |",
        "| FooDB | Food-level observations; exact vitamin/mineral mappings are explicit and collision-screened. Unmapped chemical compounds are retained in the exclusion catalogue, not coerced into nutrient targets. |",
        "",
        "## Source Coverage Summary",
        "",
        f"Source-axis observations before row deduplication: SR Legacy {int(source_coverage[source_coverage.source.eq('sr_legacy')]['source_observed_food_rows'].sum()):,}; CNF {int(source_coverage[source_coverage.source.eq('cnf')]['source_observed_food_rows'].sum()):,}; FooDB {int(source_coverage[source_coverage.source.eq('foodb')]['source_observed_food_rows'].sum()):,}.",
        "",
        "## Important Limits",
        "",
        "- The 29-candidate-micronutrient schema is complete for the current mass-only core-nutrition objective; the present data retain 28 after support filtering. This is not a complete list of all dietary bioactives or metabolite forms.",
        "- Current-source coverage does not establish universal biological truth for a food: missing remains unknown and is not zero.",
        "- CoFID and AFCD are not yet admitted. They can improve chloride/chromium and other coverage only after their per-100mL rows, trace symbols and derivation-quality flags are preserved in a source-aware ingestion path.",
        "- VMH is used as an ontology and metabolic interface, not as independent training food rows, because all current VMH food records are USDA-derived.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    data_dir, split_dir, audit_dir = args.data_dir.resolve(), args.split_dir.resolve(), args.audit_dir.resolve()
    for path in (data_dir, split_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    sr_foods, sr_values, _ = base.read_sr_legacy()
    cnf_foods, cnf_values, _ = base.read_cnf(set(sr_values["axis_id"].unique()))
    cnf_values = remap_cnf_exact_core_values(cnf_values)
    foodb_foods, foodb_values, _ = base.read_foodb()
    foodb_values = remap_foodb_core_values(foodb_values)
    foods = pd.concat([sr_foods, cnf_foods, foodb_foods], ignore_index=True, sort=False)
    foods["normalized_food_name"] = foods["food_name"].map(base.normalize_food_name)
    raw_values = pd.concat([sr_values, cnf_values, foodb_values], ignore_index=True, sort=False)
    raw_values.to_csv(audit_dir / "row_concatenated_mass_values.csv", index=False)
    foods.to_csv(audit_dir / "row_concatenated_foods.csv", index=False)

    raw_values = base.remove_invalid_values(raw_values, audit_dir)
    column_values = base.resolve_column_collisions(raw_values, audit_dir)
    vmh_categories = vmh_category_by_axis()
    all_axes = build_axis_catalogue(column_values, vmh_categories)
    source_coverage = source_axis_id_count(column_values).merge(
        all_axes[["axis_id", "training_layer", "target_kind", "axis_class", "classification_evidence"]], on="axis_id", how="left", validate="many_to_one"
    )
    source_coverage.to_csv(audit_dir / "axis_source_coverage_pre_row_dedup.csv", index=False)
    all_axes.to_csv(audit_dir / "all_axis_classification.csv", index=False)

    excluded = all_axes[all_axes["training_layer"].eq("excluded")].copy()
    excluded.to_csv(audit_dir / "excluded_axis_catalog.csv", index=False)
    retained_pre = all_axes[~all_axes["training_layer"].eq("excluded")].copy()
    usable_values = column_values[column_values["axis_id"].isin(retained_pre["axis_id"])].copy()
    usable_foods = foods[foods["source_food_key"].isin(usable_values["source_food_key"])].copy()

    canonical_foods, canonical_values, duplicate_members = base.deduplicate_food_rows(usable_foods, usable_values, audit_dir)
    final_foods, final_values, final_axes = stable_filters(canonical_foods, canonical_values, retained_pre)
    final_axes = final_axes.sort_values("axis_id", kind="stable").reset_index(drop=True)
    final_axes["axis_index"] = np.arange(len(final_axes), dtype=int)
    splits = base.split_foods(final_foods)
    split_rows = [{"canonical_food_id": food_id, "split": split} for split, food_ids in splits.items() for food_id in food_ids]
    split_frame = pd.DataFrame(split_rows)
    final_foods = final_foods.merge(split_frame, on="canonical_food_id", how="inner", validate="one_to_one")
    normalization = base.fit_train_normalization(final_values, set(splits["train"]), set(final_axes["axis_id"]))
    masks = build_sequential_masks(final_values, final_axes, splits)

    final_foods.to_csv(data_dir / "food_entities.csv", index=False)
    final_values.to_csv(data_dir / "observed_axis_values.csv", index=False)
    final_axes.to_csv(data_dir / "axis_registry.csv", index=False)
    normalization.to_csv(data_dir / "train_only_axis_normalization.csv", index=False)
    duplicate_members[duplicate_members["canonical_food_id"].isin(final_foods["canonical_food_id"])].to_csv(data_dir / "duplicate_group_members.csv", index=False)
    masks.to_csv(split_dir / "sequential_masks.csv", index=False)
    final_axes[final_axes["mask_policy"].eq("maskable_target")][["axis_id", "target_kind", "axis_class", "mask_family"]].to_csv(split_dir / "loss_axis_ids.csv", index=False)
    with (split_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol_version": "multisource_nutrition_v5", "seed": base.SPLIT_SEED, **splits}, handle, indent=2)

    write_report(audit_dir / "V5_AXIS_PROVENANCE_AND_CLASSIFICATION.md", all_axes, final_axes, source_coverage, final_foods)
    audit = {
        "protocol_version": "multisource_nutrition_v5",
        "sources": ["sr_legacy", "cnf", "foodb"],
        "model_unit": "g/100g", "row_concatenated_foods": int(len(foods)), "row_concatenated_values": int(len(raw_values)),
        "excluded_axes": int(len(excluded)), "retained_axes": int(len(final_axes)), "retained_foods": int(len(final_foods)),
        "retained_observed_values": int(len(final_values)),
        "axis_classes": final_axes["axis_class"].value_counts().to_dict(),
        "target_kinds": final_axes["target_kind"].value_counts().to_dict(),
        "mask_policies": final_axes["mask_policy"].value_counts().to_dict(),
        "split_counts": split_frame["split"].value_counts().to_dict(),
        "sequential_mask_counts": {f"{split}:{kind}": int(count) for (split, kind), count in masks.groupby(["split", "target_kind"]).size().items()},
    }
    with (audit_dir / "final_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))
    print(f"Data: {data_dir}")
    print(f"Audits: {audit_dir}")


if __name__ == "__main__":
    main()

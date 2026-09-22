#!/usr/bin/env python3
"""Build the proposed post-global-audit prediction-axis collection.

This does not replace the immutable v1 panel or update model data. It records
the scientific v2 proposal for review before source-value mappings are added.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.util import stable_id, write_csv, write_json  # noqa: E402


OUTPUT_DIR = ROOT / "data/processed/proposed_prediction_axis_panel_v2"
REPORT_PATH = ROOT / "reports/proposed_prediction_axis_panel_v2/PREDICTION_AXIS_PANEL_V2.md"


# These additions are direct, fresh-weight mass expressions or explicitly
# defined nutrient expressions. They are not automatic source-value merges.
ADDITIONS = [
    ("Vitamin B6, total", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_defined_expression", "Stage 1 nutrition composition", "Vitamin B6 total is a nutritional expression distinct from each vitamer.", "USDA Foundation/SR Legacy/FNDDS; AFCD; Frida; Ciqual; WAFCT"),
    ("Pyridoxal", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_chemical_form", "Stage 1 nutrition composition", "Pyridoxal is a defined vitamin B6 chemical form.", "USDA Foundation/SR Legacy/FNDDS"),
    ("Pyridoxamine", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_chemical_form", "Stage 1 nutrition composition", "Pyridoxamine is a defined vitamin B6 chemical form.", "USDA Foundation/SR Legacy/FNDDS"),
    ("Vitamin C, total", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_defined_expression", "Stage 1 nutrition composition", "Keep the defined total vitamin-C expression separate from individual redox forms.", "USDA Foundation/SR Legacy/FNDDS; AFCD; Frida; Ciqual; WAFCT"),
    ("Dehydroascorbic acid", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_chemical_form", "Stage 1 nutrition composition", "Defined oxidised vitamin-C chemical form.", "USDA Foundation/SR Legacy/FNDDS; Frida"),
    ("Alpha-tocopherol", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_chemical_form", "Stage 1 nutrition composition", "Defined vitamin-E chemical form; it must not be substituted by alpha-TE activity equivalents.", "USDA Foundation/SR Legacy/FNDDS; AFCD; Frida; WAFCT"),
    ("5-Methyltetrahydrofolate", "Vitamins, vitamers, and provitamins", "essential_micronutrient_or_chemical_form", "Stage 1 nutrition composition", "Defined food folate form, separate from total folate and synthetic folic acid.", "USDA Foundation/SR Legacy/FNDDS"),
    ("Phytic acid", "Nutritional chemical forms and food metabolites", "food_metabolite_or_antinutrient", "Stage 2 food metabolome", "Defined myo-inositol hexakisphosphate expression; phytate-phosphorus and method variants remain distinct provenance relations.", "USDA Foundation/SR Legacy/FNDDS; PhyFoodComp; WAFCT; Bangladesh"),
    ("Raffinose", "Nutritional chemical forms and food metabolites", "defined_oligosaccharide", "Stage 2 food metabolome", "Defined trisaccharide.", "USDA Foundation/SR Legacy/FNDDS; AFCD; Frida"),
    ("Stachyose", "Nutritional chemical forms and food metabolites", "defined_oligosaccharide", "Stage 2 food metabolome", "Defined tetrasaccharide.", "USDA Foundation/SR Legacy/FNDDS; AFCD"),
    ("Resistant starch", "Nutritional chemical forms and food metabolites", "defined_nutrition_expression", "Stage 2 food metabolome", "Recognised nutritional starch fraction; analytical-method provenance is mandatory.", "USDA Foundation/SR Legacy/FNDDS; AFCD"),
    ("Beta-glucan", "Nutritional chemical forms and food metabolites", "defined_polysaccharide", "Stage 2 food metabolome", "Defined beta-glucan polysaccharide; retain method provenance.", "USDA Foundation/SR Legacy/FNDDS; AFCD"),
    ("Glycerol", "Nutritional chemical forms and food metabolites", "food_metabolite", "Stage 2 food metabolome", "Defined polyol and food constituent.", "AFCD; CoFID; Frida"),
    ("Maltitol", "Nutritional chemical forms and food metabolites", "food_metabolite", "Stage 2 food metabolome", "Defined sugar alcohol.", "AFCD; Frida"),
    ("Glutamine", "Amino acids and protein-related metabolites", "amino_acid_composition", "Stage 2 food metabolome", "Defined amino acid, distinct from glutamic acid and protein total.", "USDA Foundation/SR Legacy/FNDDS"),
    ("Asparagine", "Amino acids and protein-related metabolites", "amino_acid_composition", "Stage 2 food metabolome", "Defined amino acid, distinct from aspartic acid and protein total.", "USDA Foundation/SR Legacy/FNDDS"),
    ("Campestanol", "Sterols and sterol expressions", "defined_sterol", "Stage 2 food metabolome", "Defined saturated plant sterol, distinct from campesterol.", "USDA Foundation/SR Legacy/FNDDS"),
]


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    source = ROOT / "data/processed/frozen_prediction_axis_panel_v1/frozen_prediction_axis_registry.csv"
    panel = pd.read_csv(source, keep_default_na=False)

    # Same analyte/expression, corrected nomenclature. These are not additions.
    panel.loc[panel.canonical_name.eq("Iodide"), "canonical_name"] = "Iodine, total"
    panel.loc[panel.canonical_name.eq("Iodine, total"), "aliases_for_review"] += "; Iodide; Iodine, I"
    panel.loc[panel.canonical_name.eq("Vitamin B6 - Pyridoxin"), "canonical_name"] = "Pyridoxine (vitamin B6)"
    panel.loc[panel.canonical_name.eq("Pyridoxine (vitamin B6)"), "aliases_for_review"] += "; Vitamin B6 - Pyridoxin; pyridoxine"

    new_rows = []
    for name, family, role, stage, reason, source_basis in ADDITIONS:
        new_rows.append({
            "target_axis_id": stable_id("proposed_prediction_axis_v2", name),
            "canonical_name": name,
            "original_axis_names": name,
            "axis_family": family,
            "chemical_identity": "defined chemical entity or explicitly defined nutritional expression",
            "food_composition_role": "essential_micronutrient" if stage.startswith("Stage 1") else "food_metabolite_or_nutrition_related",
            "nutritional_role": role,
            "measurement_modality_required": "direct quantified mass expression; denominator and method retained",
            "selection_tier": "nutrition_core" if stage.startswith("Stage 1") else "food_metabolome_extension",
            "recommended_training_stage": stage,
            "mask_family": family,
            "selection_reason": reason,
            "scientific_basis": "Global post-expansion review: identity is explicit and source expression is a direct mass or defined nutritional expression.",
            "authority_urls": "FAO/INFOODS component standards; source-specific component definition retained in evidence ledger",
            "source_basis": source_basis,
            "source_match_count_exact_candidate": "",
            "source_match_count_fuzzy_candidate": "",
            "aliases_for_review": name,
            "direct_prediction_target": True,
            "numeric_pooling_permitted": False,
            "notes": "Before value use, build a versioned source-definition/unit/method mapping; do not pool method or chemical-form variants.",
        })
    result = pd.concat([panel, pd.DataFrame(new_rows)], ignore_index=True)
    if result.canonical_name.duplicated().any() or result.target_axis_id.duplicated().any():
        raise ValueError("Proposed v2 contains duplicate canonical names or IDs.")
    if len(result) != 386:
        raise AssertionError(f"Expected 386 proposed targets, found {len(result)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(result, OUTPUT_DIR / "proposed_prediction_axis_registry.csv")
    write_csv(pd.DataFrame(new_rows), OUTPUT_DIR / "new_targets_only.csv")
    write_json({
        "status": "proposed_not_frozen_not_used_for_value_mapping_or_training",
        "base_panel": str(source),
        "base_target_count": 369,
        "renamed_without_new_target": {
            "Iodide": "Iodine, total",
            "Vitamin B6 - Pyridoxin": "Pyridoxine (vitamin B6)",
        },
        "new_direct_prediction_targets": len(new_rows),
        "proposed_target_count": len(result),
        "holds": [
            "generic inositol and inositol phosphate family pending chemical-form review",
            "vitamin K2 family aggregates pending individual menaquinone mappings",
            "activity equivalents, energy, contamination/exposure analytes, ratios, and by-difference expressions",
        ],
    }, OUTPUT_DIR / "proposal_manifest.json")

    stage_counts = result.groupby("recommended_training_stage").size().to_dict()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Proposed Prediction-Axis Panel v2\n\n"
        "This proposed panel contains 386 direct prediction targets: 369 retained v1 targets, "
        "two nomenclature corrections without creating duplicate targets, and 17 additions. "
        "It is not frozen and is not yet used for numerical pooling, splitting, or training.\n\n"
        f"- Stage 1 nutrition composition: {stage_counts.get('Stage 1 nutrition composition', 0)} targets\n"
        f"- Stage 2 food metabolome: {stage_counts.get('Stage 2 food metabolome', 0)} targets\n\n"
        "The full machine-readable collection is `proposed_prediction_axis_registry.csv`; "
        "the additions are in `new_targets_only.csv`. Each addition requires a source-definition, "
        "unit, denominator, and method mapping before it can receive data.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(result)} proposed prediction targets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

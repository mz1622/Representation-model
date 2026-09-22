from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from foodcomp.dataset import masked_mse, FoodCompositionDataset
from foodcomp.harmonize import UnionFind, _formal_food_lineage, assign_partitions, _aggregate_group, _authority_key, _component_family, _role_from_component
from foodcomp.benchmark import select_hidden_families
from foodcomp.frozen_release import assign_frozen_partitions, attach_new_family_blocks
from foodcomp.schema import normalize_unit, parse_value
from foodcomp.sources import _foodb_food_lineage, gate_component_expressions, _apply_training_gate, _gate_afcd_derivation_conflicts
from foodcomp.util import normalize_text, stable_id


class ValueSemanticsTests(unittest.TestCase):
    def test_missing_is_not_zero(self) -> None:
        missing = parse_value(None)
        zero = parse_value(0)
        self.assertEqual(missing["value_status"], "missing")
        self.assertIsNone(missing["numeric_value"])
        self.assertEqual(zero["value_status"], "explicit_zero")
        self.assertEqual(zero["numeric_value"], 0.0)

    def test_censored_and_range_are_not_point_labels(self) -> None:
        censored = parse_value("< 0.2")
        interval = parse_value("0.1-0.3")
        self.assertTrue(censored["is_censored"])
        self.assertIsNone(censored["numeric_value"])
        self.assertTrue(interval["is_range_only"])
        self.assertIsNone(interval["numeric_value"])
        self.assertTrue(parse_value("not detected")["is_censored"])
        self.assertNotEqual(parse_value("not detected")["value_status"], "missing")

    def test_exact_mass_conversion(self) -> None:
        self.assertEqual(normalize_unit("mg/100 g")["conversion_factor"], 1e-3)
        self.assertEqual(normalize_unit("µg/100g")["conversion_factor"], 1e-6)
        self.assertEqual(normalize_unit("g", "per 100 g edible portion")["conversion_factor"], 1.0)

    def test_non_main_modalities_are_excluded(self) -> None:
        self.assertIsNone(normalize_unit("kcal/100g")["conversion_factor"])
        self.assertIsNone(normalize_unit("IU/100g")["conversion_factor"])
        self.assertIsNone(normalize_unit("mg/100 mL")["conversion_factor"])
        self.assertIsNone(normalize_unit("g/100g fatty acids")["conversion_factor"])

    def test_table_basis_cannot_override_column_denominator(self) -> None:
        for unit in ("mg/gN", "mg/g N", "mg/g protein", "g/kg", "mg/100mL", "mg-ATE"):
            with self.subTest(unit=unit):
                self.assertIsNone(normalize_unit(unit, "per 100 g edible portion", True)["conversion_factor"])

    def test_bare_mass_unit_does_not_override_activity_definition(self) -> None:
        components = pd.DataFrame([
            {"component_observation_id": "activity", "original_name": "Vitamin A", "infoods_tag": "VITA", "source_definition": "Vitamin A activity"},
            {"component_observation_id": "mass", "original_name": "Retinol", "infoods_tag": "RETOL", "source_definition": "Retinol mass"},
        ])
        frame = pd.DataFrame({"component_observation_id": ["activity", "mass"], "main_value_eligible": [True, True], "strict_validation_eligible": [True, True]})
        result = gate_component_expressions(frame, components, {})
        self.assertFalse(result.iloc[0].main_value_eligible)
        self.assertTrue(result.iloc[1].main_value_eligible)

    def test_microgram_scale_conflicts_are_not_hidden_by_log1p(self) -> None:
        group = pd.DataFrame({
            "quality_tier": ["B", "B"], "normalized_value_g_per_100g": [0.00018, 0.00241],
            "sample_count": [np.nan, np.nan], "standard_error": [np.nan, np.nan],
            "conversion_factor": [0.001, 0.001], "measurement_id": ["m1", "m2"],
            "source_key": ["a", "b"], "independent_evidence": [True, True],
        })
        result = _aggregate_group(group)
        self.assertNotEqual(result["aggregation_status"], "accepted")
        self.assertGreater(result["max_min_ratio"], 13)
        self.assertTrue(np.isnan(result["total_sample_count"]))

    def test_curated_reference_trains_but_cannot_validate(self) -> None:
        frame = pd.DataFrame({"main_value_eligible": [True], "independent_evidence": [False], "quality_tier": ["C"], "data_layer": ["curated_reference_training"], "quality_evidence": ["official reference"], "strict_validation_eligible": [False], "value_status": ["observed_positive"], "exclusion_reason": [""]})
        result = _apply_training_gate(frame)
        self.assertTrue(result.iloc[0].main_value_eligible)
        self.assertFalse(result.iloc[0].strict_validation_eligible)

    def test_afcd_conflicting_or_absent_metadata_is_not_an_assay(self) -> None:
        frame = pd.DataFrame({"main_value_eligible": [True] * 3, "strict_validation_eligible": [True] * 3, "independent_evidence": [True] * 3, "quality_tier": ["B"] * 3, "data_layer": ["primary_reference"] * 3, "exclusion_reason": [""] * 3})
        result = _gate_afcd_derivation_conflicts(frame, pd.Series(["Analysed"] * 3), pd.Series(["Recipe", "Analysed", None]))
        self.assertEqual(result.main_value_eligible.tolist(), [False, True, False])
        self.assertEqual(result.independent_evidence.tolist(), [False, True, False])
        self.assertIn("profile_derivation=Analysed", result.iloc[0].method_expression)
        self.assertIn("food_details_derivation=Recipe", result.iloc[0].method_expression)


class DeterminismTests(unittest.TestCase):
    def test_context_only_sibling_is_hidden_with_its_family(self) -> None:
        dataset = FoodCompositionDataset.__new__(FoodCompositionDataset)
        dataset.values = np.array([[4.0, 0.1, 12.0]])
        dataset.target_columns = np.array([0])
        dataset.family_by_column = np.array(["protein", "protein", "lipid"])
        dataset.foods = pd.DataFrame({"food_concept_id": ["food"], "canonical_name": ["Example"]})
        dataset.seed = 42
        item = dataset[0]
        np.testing.assert_array_equal(item["target_mask"], [1, 0, 0])
        np.testing.assert_array_equal(item["context_mask"], [0, 0, 1])
        np.testing.assert_array_equal(item["context_values"], [0, 0, 12])

    def test_new_bridge_does_not_collapse_frozen_families(self) -> None:
        previous = pd.DataFrame({"food_concept_id": ["a", "b"], "family_cluster_id": ["old-a", "old-b"]})
        concepts = pd.DataFrame({"food_concept_id": ["a", "b", "bridge"], "family_label": ["a", "b", "bridge"]})
        mapping = pd.DataFrame({"food_concept_id": ["a", "b", "bridge"], "normalized_lineage": ["a", "b", "c"]})
        candidates = pd.DataFrame({"left_concept_id": ["bridge", "bridge"], "right_concept_id": ["a", "b"]})
        result = attach_new_family_blocks(concepts, mapping, candidates, previous).set_index("food_concept_id")
        self.assertEqual(result.at["a", "family_cluster_id"], "old-a")
        self.assertEqual(result.at["b", "family_cluster_id"], "old-b")
        self.assertTrue(result.at["bridge", "family_expansion_conflict"])

    def test_whole_family_masks_keep_nonempty_context(self) -> None:
        families = {"large": list(range(9)), "small": [9]}
        for visibility in (0.1, 0.3, 0.5, 0.7):
            hidden = select_hidden_families(families, ["large", "small"], visibility)
            self.assertTrue(hidden)
            self.assertLess(sum(len(families[x]) for x in hidden), 10)

    def test_choline_and_amino_acid_families(self) -> None:
        self.assertEqual(_component_family("CHOLN", "choline", "essential_nutrient_choline"), "choline_family")
        for tag, name in (("LEU", "leucine"), ("ILE", "isoleucine"), ("VAL", "valine"), ("PROT", "protein")):
            role = _role_from_component(tag, name, "")[0]
            self.assertEqual(_component_family(tag, name, role), "protein_and_amino_acid_family")
        row = pd.Series({"infoods_tag": "PROT", "original_name": "Protein"})
        self.assertEqual(_authority_key(row, {"PROT", "PROCNT"})[1], "PROCNT")
        self.assertEqual(_component_family("VITB12", "cobalamin", "micronutrient_vitamin"), "vitamin:b12")

    def test_frozen_membership_not_just_count(self) -> None:
        concepts = pd.DataFrame([
            {"food_concept_id": "old", "canonical_name": "old", "family_cluster_id": "f1", "ml_exclusion_reason": ""},
            {"food_concept_id": "new_related", "canonical_name": "related", "family_cluster_id": "f1", "ml_exclusion_reason": ""},
            {"food_concept_id": "new_independent", "canonical_name": "new", "family_cluster_id": "f2", "ml_exclusion_reason": ""},
        ])
        mapping = pd.DataFrame({"food_observation_id": ["a", "b", "c"], "food_concept_id": concepts.food_concept_id, "normalized_lineage": ["a", "b", "c"]})
        observations = pd.DataFrame({"food_observation_id": ["a", "b", "c"], "source_key": ["cnf"] * 3})
        previous = pd.DataFrame({"food_concept_id": ["old"], "partition": ["validation"], "validation_panel": ["family_holdout"]})
        result, exclusions = assign_frozen_partitions(concepts, mapping, observations, {"new_related", "new_independent"}, previous)
        self.assertEqual(set(result.loc[result.partition.eq("validation"), "food_concept_id"]), {"old"})
        self.assertEqual(set(result.loc[result.partition.eq("train"), "food_concept_id"]), {"new_independent"})
        self.assertEqual(set(exclusions.food_concept_id), {"new_related"})

    def test_food_union_rejects_conflicting_facets(self) -> None:
        union = UnionFind(["a", "b", "c"], {
            "a": {"part": "leg"}, "b": {}, "c": {"part": "breast"},
        })
        self.assertTrue(union.union("a", "b"))
        self.assertFalse(union.union("b", "c"))

    def test_normalization_and_ids_are_stable(self) -> None:
        self.assertEqual(normalize_text("  Roasted—Chicken  "), "roasted chicken")
        self.assertEqual(stable_id("x", "a", 1), stable_id("x", "a", 1))

    def test_foodb_citation_ids_do_not_impersonate_primary_ids(self) -> None:
        lineage = _foodb_food_lineage("Frida 2019", 1248)
        self.assertTrue(lineage.startswith("FOODB_CITATION:"))
        self.assertNotEqual(lineage, "FRIDA:1248")
        self.assertEqual(_formal_food_lineage(lineage), "")

    def test_masked_mse_uses_only_masked_cells(self) -> None:
        prediction = np.array([10.0, 2.0])
        target = np.array([0.0, 0.0])
        mask = np.array([0.0, 1.0])
        self.assertAlmostEqual(masked_mse(prediction, target, mask), 4.0)

    def test_source_family_relatives_train_but_formal_lineage_is_held_out(self) -> None:
        concept_rows = [
            {"food_concept_id": "foundation", "family_cluster_id": "family-a", "ml_exclusion_reason": ""},
            {"food_concept_id": "relative", "family_cluster_id": "family-a", "ml_exclusion_reason": ""},
            {"food_concept_id": "lineage-copy", "family_cluster_id": "family-b", "ml_exclusion_reason": ""},
        ]
        concept_rows.extend(
            {"food_concept_id": f"other-{index}", "family_cluster_id": f"family-{index}", "ml_exclusion_reason": ""}
            for index in range(10)
        )
        concepts = pd.DataFrame(concept_rows)
        food_rows = [
            {"food_observation_id": "obs-foundation", "source_key": "usda_foundation"},
            {"food_observation_id": "obs-relative", "source_key": "usda_sr_legacy"},
            {"food_observation_id": "obs-lineage", "source_key": "cnf"},
        ]
        mapping_rows = [
            {"food_observation_id": "obs-foundation", "food_concept_id": "foundation", "normalized_lineage": "lineage-x"},
            {"food_observation_id": "obs-relative", "food_concept_id": "relative", "normalized_lineage": "lineage-y"},
            {"food_observation_id": "obs-lineage", "food_concept_id": "lineage-copy", "normalized_lineage": "lineage-x"},
        ]
        for index in range(10):
            food_rows.append({"food_observation_id": f"obs-{index}", "source_key": "cnf"})
            mapping_rows.append({
                "food_observation_id": f"obs-{index}",
                "food_concept_id": f"other-{index}",
                "normalized_lineage": f"lineage-{index}",
            })

        partitions = assign_partitions(
            concepts,
            pd.DataFrame(mapping_rows),
            pd.DataFrame(food_rows),
            set(concepts["food_concept_id"]),
        ).set_index("food_concept_id")

        self.assertEqual(partitions.at["foundation", "validation_panel"], "source_holdout")
        self.assertEqual(partitions.at["lineage-copy", "validation_panel"], "source_holdout")
        self.assertEqual(partitions.at["relative", "partition"], "train")
        self.assertTrue(partitions.at["relative", "source_holdout_family_overlap"])
        family_holdouts = set(partitions.loc[partitions["family_holdout"], "family_cluster_id"])
        train_families = set(partitions.loc[partitions["partition"].eq("train"), "family_cluster_id"])
        self.assertFalse(family_holdouts & train_families)


if __name__ == "__main__":
    unittest.main()

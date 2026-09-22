import unittest

import numpy as np
import pandas as pd

from foodcomp.target_coverage import (
    allocate_validation, apply_candidate_partition, component_identity_collisions,
    coverage_counts, eligibility_issues, make_transfer_blocks, profile_evidence,
)


class TargetCoverageTests(unittest.TestCase):
    def test_decimal_numbers_are_not_treated_as_analyte_synonyms(self):
        observations = pd.DataFrame({
            "component_observation_id": ["a", "b", "c"], "source_key": ["usda_foundation"] * 3,
            "source_definition": ["USDA nutrient number 338", "USDA nutrient number 338.1", "USDA nutrient number 338.2"],
            "original_name": ["Lutein plus zeaxanthin", "Lutein", "Zeaxanthin"],
        })
        mapping = pd.DataFrame({"component_observation_id": ["a", "b", "c"], "component_concept_id": ["incorrect"] * 3})
        rows = component_identity_collisions(observations, mapping)
        self.assertEqual(set(rows.original_nutrient_number), {"338", "338.1", "338.2"})
        self.assertEqual(set(rows.component_concept_id), {"incorrect"})

    def test_mixed_evidence_is_not_made_strict_by_changing_partition(self):
        profiles = pd.DataFrame({"food_concept_id": ["f"], "component_concept_id": ["a"],
                                 "partition": ["train"], "aggregation_status": ["accepted"],
                                 "canonical_value_g_per_100g": [0.5], "measurement_ids": ["m1;m2"]})
        measurements = pd.DataFrame({"measurement_id": ["m1", "m2"], "main_value_eligible": [True, True],
                                     "strict_validation_eligible": [True, False]})
        self.assertFalse(profile_evidence(profiles, measurements).iloc[0].strict_label_eligible)
        profiles.loc[0, "partition"] = "validation"
        with self.assertRaisesRegex(ValueError, "source policy"):
            profile_evidence(profiles, measurements)

    def test_missing_measurement_provenance_fails(self):
        profiles = pd.DataFrame({"food_concept_id": ["f"], "component_concept_id": ["a"],
                                 "partition": ["train"], "aggregation_status": ["accepted"],
                                 "canonical_value_g_per_100g": [0.5], "measurement_ids": ["missing"]})
        measurements = pd.DataFrame({"measurement_id": ["m1"], "main_value_eligible": [True],
                                     "strict_validation_eligible": [True]})
        with self.assertRaisesRegex(ValueError, "provenance"):
            profile_evidence(profiles, measurements)

    def test_family_and_borrowed_reference_make_one_transfer_block(self):
        foods = pd.DataFrame({"food_concept_id": ["a", "b", "c", "v"],
                              "partition": ["train"] * 3 + ["validation"],
                              "family_cluster_id": ["family1", "family1", "family2", "sourcefamily"]})
        mapping = pd.DataFrame({"food_observation_id": ["oa", "ob", "oc", "ov"],
                                "food_concept_id": ["a", "b", "c", "v"],
                                "normalized_lineage": ["lineage-a", "lineage-b", "lineage-c", "lineage-v"]})
        observations = pd.DataFrame({"food_observation_id": ["oa", "ob", "oc", "ov"],
                                     "source_key": ["usda_sr_legacy", "cnf", "afcd", "usda_foundation"],
                                     "source_food_id": ["123", "456", "789", "999"],
                                     "potential_reference_lineages": ["", "", "USDA:FDC:123", ""]})
        blocks = make_transfer_blocks(foods, mapping, observations)
        self.assertEqual(len(set(blocks.values())), 1)
        observations.loc[2, "potential_reference_lineages"] = "USDA:FDC:999"
        with self.assertRaisesRegex(ValueError, "validation copies"):
            make_transfer_blocks(foods, mapping, observations)
        blocks = make_transfer_blocks(foods, mapping, observations,
                                       selected_observation_ids={"oa", "ob", "ov"})
        self.assertNotEqual(blocks["a"], blocks["c"])

    @staticmethod
    def _ledger():
        return pd.DataFrame({"component_concept_id": ["a"], "nutrition_scope": ["core_nutrition_expression"],
                             "eligible_for_support_allocation": [True], "validation_deficit": [30],
                             "training_role": ["context_only"], "current_train_count": [140]})

    def test_transfer_keeps_one_hundred_training_concepts(self):
        ledger = self._ledger()
        profiles = pd.DataFrame({"food_concept_id": [f"f{i:03}" for i in range(140)],
                                 "component_concept_id": ["a"] * 140,
                                 "strict_label_eligible": [True] * 140,
                                 "transfer_block": [f"b{i // 10}" for i in range(140)]})
        assignments = dict(zip(profiles.food_concept_id, profiles.transfer_block))
        moved, result = allocate_validation(ledger, profiles, pd.DataFrame(), assignments)
        self.assertEqual(len(moved), 30)
        self.assertEqual(result["stages"][0]["optimum"], 1)
        self.assertGreaterEqual(140 - len(moved), 100)

    def test_one_large_family_cannot_be_split_to_manufacture_coverage(self):
        ledger = self._ledger()
        profiles = pd.DataFrame({"food_concept_id": [f"f{i:03}" for i in range(140)],
                                 "component_concept_id": ["a"] * 140,
                                 "strict_label_eligible": [True] * 140, "transfer_block": ["one-family"] * 140})
        assignments = dict(zip(profiles.food_concept_id, profiles.transfer_block))
        moved, result = allocate_validation(ledger, profiles, pd.DataFrame(), assignments)
        self.assertEqual(moved, set())
        self.assertEqual(result["stages"][0]["optimum"], 0)

    def test_robust_scale_zero_is_not_called_constant_when_range_is_nonzero(self):
        row = dict(component_concept_id="a", identity_status="authority_verified", expression_variant="standard_expression",
                   capability_adjusted_coverage=0.5, train_robust_scale=0, train_raw_min=0, train_raw_max=0.5,
                   nutrition_scope="core_nutrition_expression", minimum_counts_feasible_without_grouping=True)
        ledger = eligibility_issues(pd.DataFrame([row]), set())
        self.assertIn("values_not_constant", ledger.iloc[0].non_support_review_reasons)
        self.assertFalse(ledger.iloc[0].eligible_for_support_allocation)

    def test_total_109_cannot_fund_100_train_plus_30_validation(self):
        ids = [f"t{i}" for i in range(104)] + [f"v{i}" for i in range(5)]
        profiles = pd.DataFrame({"food_concept_id": ids, "component_concept_id": ["a"] * 109,
                                 "partition": ["train"] * 104 + ["validation"] * 5,
                                 "strict_label_eligible": [True] * 109})
        registry = pd.DataFrame({"component_concept_id": ["a"], "training_role": ["context_only"],
                                 "nutritional_role": ["micronutrient_vitamin"]})
        ledger, _ = coverage_counts(registry, profiles, pd.DataFrame(), {food: food for food in ids[:104]})
        self.assertEqual(ledger.iloc[0].validation_deficit, 25)
        self.assertEqual(ledger.iloc[0].train_count_after_minimum_transfer, 79)
        self.assertFalse(ledger.iloc[0].minimum_counts_feasible_without_grouping)

    def test_candidate_rechecks_evidence_and_fits_scale_on_remaining_train(self):
        foods = pd.DataFrame({"food_concept_id": ["t1", "t2", "m", "v"],
                              "partition": ["train"] * 3 + ["validation"],
                              "validation_panel": ["", "", "", "source_holdout"],
                              "cv_fold": [0, 1, 2, -1], "family_holdout": [False] * 4,
                              "source_holdout": [False] * 3 + [True],
                              "frozen_validation_identity": [False] * 3 + [True]})
        profiles = pd.DataFrame({"food_concept_id": ["t1", "t2", "m", "v"],
                                 "component_concept_id": ["a"] * 4,
                                 "partition": ["train"] * 3 + ["validation"],
                                 "canonical_value_g_per_100g": [1.0, 2.0, 99.0, 98.0],
                                 "strict_label_eligible": [True, True, False, True]})
        ledger = pd.DataFrame({"component_concept_id": ["a"], "nutrition_scope": ["core_nutrition_expression"],
                               "non_support_review_reasons": [""], "current_train_count": [3],
                               "current_validation_count": [1], "validation_deficit": [29],
                               "train_count_after_minimum_transfer": [-26],
                               "strict_labels_sufficient_for_transfer": [False], "strict_train_count": [2]})
        partition, updated, withheld = apply_candidate_partition(
            foods, profiles, ledger, {"m"}, {"t1": "b1", "t2": "b2", "m": "b3"})
        self.assertEqual(set(withheld.food_concept_id), {"m"})
        self.assertEqual(updated.iloc[0].proposed_train_count, 2)
        self.assertEqual(updated.iloc[0].proposed_validation_count, 1)
        logs = np.log1p([1.0, 2.0])
        expected_scale = max((np.quantile(logs, .75) - np.quantile(logs, .25)) / 1.349,
                             np.median(abs(logs - np.median(logs))) * 1.4826)
        self.assertAlmostEqual(updated.iloc[0].proposed_train_robust_scale, expected_scale)
        self.assertEqual(partition.set_index("food_concept_id").loc["v", "validation_panel"], "source_holdout")
        self.assertFalse(partition.benchmark_eligible.any())
        self.assertTrue(partition.candidate_only_not_for_training.all())
        with self.assertRaisesRegex(ValueError, "split a family"):
            apply_candidate_partition(foods, profiles, ledger, {"m"}, {"t1": "b1", "t2": "b3", "m": "b3"})


if __name__ == "__main__":
    unittest.main()

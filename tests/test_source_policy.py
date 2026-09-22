import unittest

import numpy as np
import pandas as pd

from foodcomp.harmonize import aggregate_profiles
from foodcomp.source_policy import (
    TRUSTED_REFERENCE_POLICY, apply_source_policy, validation_reference_mask,
)
from foodcomp.sources import gate_component_expressions
from foodcomp.target_coverage import profile_evidence


def record(**changes):
    row = {
        "measurement_id": "m", "source_key": "usda_sr_legacy",
        "food_observation_id": "food-observation", "component_observation_id": "axis-observation",
        "main_value_eligible": False, "strict_validation_eligible": False,
        "normalized_value_g_per_100g": 0.1, "value_status": "observed",
        "conversion_status": "converted_exact_mass", "measurement_modality": "mass_fraction_fresh_weight",
        "is_censored": False, "is_range_only": False, "source_reference": np.nan,
        "method_expression": np.nan, "analytical_method": np.nan, "sample_count": np.nan,
        "lineage_source_key": "usda_sr_legacy", "independent_evidence": False,
        "quality_tier": "D", "data_layer": "primary_reference",
        "exclusion_reason": "non_independent_or_derived_value",
    }
    row.update(changes)
    return row


class SourcePolicyTests(unittest.TestCase):
    def review(self, **changes):
        return apply_source_policy(pd.DataFrame([record(**changes)]), TRUSTED_REFERENCE_POLICY)

    def test_missing_methods_and_sample_counts_no_longer_block_three_sources(self):
        for source in ["usda_sr_legacy", "cnf", "foodb"]:
            with self.subTest(source=source):
                result = self.review(source_key=source, lineage_source_key=source).iloc[0]
                self.assertTrue(result.main_value_eligible)
                self.assertTrue(result.validation_reference_eligible)
                self.assertFalse(result.strict_validation_eligible)
                self.assertFalse(result.independent_evidence)
                self.assertEqual(result.quality_tier, "D")

    def test_other_sources_keep_their_existing_gates(self):
        for source in ["afcd", "cofid", "ciqual", "usda_foundation"]:
            for old in [False, True]:
                with self.subTest(source=source, old=old):
                    result = self.review(source_key=source, main_value_eligible=old,
                                         strict_validation_eligible=old).iloc[0]
                    self.assertEqual(result.main_value_eligible, old)
                    self.assertEqual(result.validation_reference_eligible, old)
                    self.assertFalse(result.source_trusted)

    def test_missing_documentation_does_not_upgrade_quality_or_independence(self):
        result = self.review(quality_tier="C", source_reference="Analytical data from the literature, partial documentation").iloc[0]
        self.assertTrue(result.main_value_eligible)
        self.assertEqual(result.quality_tier, "C")
        self.assertEqual(result.source_independence_status, "not_established")
        self.assertFalse(result.source_gate_previous_eligible)

    def test_numerical_and_status_guards_cannot_be_overridden(self):
        bad_rows = [
            {"normalized_value_g_per_100g": np.nan}, {"normalized_value_g_per_100g": -1},
            {"normalized_value_g_per_100g": 101}, {"normalized_value_g_per_100g": np.inf},
            {"value_status": "assumed_zero", "normalized_value_g_per_100g": 0},
            {"value_status": "missing"}, {"is_censored": True}, {"is_range_only": True},
            {"conversion_status": "unsupported_or_missing_unit"},
            {"measurement_modality": "biological_activity"},
            {"exclusion_reason": "activity_equivalent_not_chemical_mass"},
            {"exclusion_reason": "unresolved_identity_conflict"},
        ]
        for changes in bad_rows:
            with self.subTest(changes=changes):
                row = self.review(**changes).iloc[0]
                self.assertFalse(row.main_value_eligible)
                self.assertFalse(row.validation_reference_eligible)

    def test_explicit_zero_is_not_made_missing(self):
        row = self.review(value_status="explicit_zero", normalized_value_g_per_100g=0).iloc[0]
        self.assertTrue(row.main_value_eligible)
        self.assertEqual(row.normalized_value_g_per_100g, 0)

    def test_explicit_imputation_label_recipe_and_censoring_stay_held(self):
        for method in ["Nutrient imputed from a similar USDA food", "Label claim",
                       "Calculated using a recipe", "Analytical data; derived by linear regression",
                       "Calculated from a less than value per serving size measure"]:
            with self.subTest(method=method):
                self.assertFalse(self.review(method_expression=method).iloc[0].main_value_eligible)

    def test_routine_reference_calculation_is_not_automatically_imputation(self):
        row = self.review(source_reference="Calculated or imputed", method_expression="Calculated").iloc[0]
        self.assertTrue(row.main_value_eligible)
        self.assertEqual(row.source_value_expression, "definition_derived_reference")
        self.assertFalse(self.review(source_reference="Calculated or imputed").iloc[0].main_value_eligible)

    def test_unresolved_external_copies_never_count_as_new_independent_labels(self):
        for changes in [
            {"source_key": "foodb", "source_reference": "USDA"},
            {"source_key": "foodb", "source_reference": "DTU"},
            {"source_key": "foodb", "source_reference": "Phenol-Explorer"},
            {"source_key": "cnf", "source_reference": "No change from USDA"},
        ]:
            with self.subTest(changes=changes):
                row = self.review(**changes).iloc[0]
                self.assertFalse(row.main_value_eligible)
                self.assertEqual(row.source_independence_status, "copied_or_derived_from_external_reference")

    def test_phenolic_paper_title_is_not_a_database_copy_marker(self):
        row = self.review(source_key="foodb", source_reference="Phenolic compounds in blueberries. Journal article.",
                          exclusion_reason="copied_value_retained_for_provenance_only").iloc[0]
        self.assertTrue(row.main_value_eligible)
        self.assertFalse(row.independent_evidence)

    def test_numeric_reference_ids_remain_valid_identifiers(self):
        row = self.review(source_key="foodb", source_reference=27719959).iloc[0]
        self.assertTrue(row.main_value_eligible)
        self.assertEqual(row.source_reference, 27719959)
        row = self.review(source_key="ciqual", source_reference=359,
                          main_value_eligible=True, strict_validation_eligible=True).iloc[0]
        self.assertEqual(row.source_policy_decision, "unchanged_other_source")

    def test_expression_gate_still_overrides_trust(self):
        result = self.review()
        components = pd.DataFrame({"component_observation_id": ["axis-observation"],
                                   "original_name": ["Vitamin A equivalents"], "infoods_tag": ["VITA"],
                                   "source_definition": ["Retinol equivalents"]})
        result = gate_component_expressions(result, components, {})
        self.assertFalse(result.iloc[0].main_value_eligible)
        self.assertFalse(result.iloc[0].validation_reference_eligible)
        self.assertEqual(result.iloc[0].source_policy_decision, "held_record_level_check")

    def test_reference_eligibility_respected_by_aggregation_and_coverage(self):
        evidence = self.review()
        foods = pd.DataFrame({"food_observation_id": ["food-observation"], "food_concept_id": ["food"],
                              "exclusion_flag": [""]})
        components = pd.DataFrame({"component_observation_id": ["axis-observation"], "component_concept_id": ["axis"]})
        partition = pd.DataFrame({"food_concept_id": ["food"], "partition": ["validation"],
                                  "validation_panel": ["family_holdout"], "source_holdout": [False],
                                  "cv_fold": [-1], "family_cluster_id": ["family"]})
        profiles, _ = aggregate_profiles(evidence, foods, components, partition, {"food-observation"})
        self.assertEqual(profiles.iloc[0].canonical_value_g_per_100g, 0.1)
        self.assertTrue(profile_evidence(profiles, evidence).iloc[0].strict_label_eligible)
        self.assertFalse(evidence.iloc[0].strict_validation_eligible)
        self.assertTrue(aggregate_profiles(evidence, foods, components, partition, set())[0].empty)

    def test_old_release_flags_and_boolean_csv_strings(self):
        old = pd.DataFrame({"main_value_eligible": [True, False], "strict_validation_eligible": [True, True]})
        self.assertEqual(validation_reference_mask(old).tolist(), [True, False])
        row = self.review(is_censored="False", is_range_only="False", main_value_eligible="False",
                          strict_validation_eligible="False", independent_evidence="False").iloc[0]
        self.assertTrue(row.main_value_eligible)
        self.assertFalse(row.strict_validation_eligible)

    def test_wholly_archived_foodb_legacy_schema_has_no_previous_validation(self):
        frame = pd.DataFrame([record(source_key="foodb")]).drop(columns="strict_validation_eligible")
        result = apply_source_policy(frame, TRUSTED_REFERENCE_POLICY)
        self.assertTrue(result.iloc[0].validation_reference_eligible)
        self.assertFalse(result.iloc[0].source_gate_previous_validation_eligible)
        self.assertEqual(result.iloc[0].source_gate_previous_validation_status,
                         "legacy_foodb_archive_without_validation_field")
        self.assertTrue(apply_source_policy(frame.iloc[:0], TRUSTED_REFERENCE_POLICY).empty)
        frame.loc[0, "main_value_eligible"] = True
        with self.assertRaisesRegex(ValueError, "wholly archival"):
            apply_source_policy(frame, TRUSTED_REFERENCE_POLICY)

    def test_unknown_policy_and_reapplication_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            apply_source_policy(pd.DataFrame([record()]), "all_sources")
        with self.assertRaisesRegex(ValueError, "immutable staging"):
            apply_source_policy(self.review(), TRUSTED_REFERENCE_POLICY)

    def test_input_and_raw_provenance_unchanged(self):
        frame = pd.DataFrame([record()])
        original = frame.copy(deep=True)
        result = apply_source_policy(frame, TRUSTED_REFERENCE_POLICY)
        pd.testing.assert_frame_equal(frame, original)
        for column in ["normalized_value_g_per_100g", "source_reference", "method_expression",
                       "sample_count", "analytical_method", "quality_tier", "independent_evidence"]:
            pd.testing.assert_series_equal(result[column], frame[column])


if __name__ == "__main__":
    unittest.main()

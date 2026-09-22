"""Focused contract checks for the static Atlas builder."""

from __future__ import annotations

import unittest

import pandas as pd

from foodcomp.atlas import _annotate_training_eligibility, _build_source_table, _shard_key


class FoodCompositionAtlasTest(unittest.TestCase):
    def test_shard_key_uses_digest_prefix(self) -> None:
        self.assertEqual(_shard_key("exact_name_group:ab12ef"), "ab")
        with self.assertRaises(ValueError):
            _shard_key("not-a-stable-id")

    def test_source_registry_prefers_nonempty_registry_metadata(self) -> None:
        registry = pd.DataFrame(
            {
                "source_key": ["source_a"],
                "name": ["Source A"],
                "official_url": ["https://example.org"],
                "license_or_terms": ["Open"],
                "region": ["Europe"],
                "countries_or_coverage": ["Exampleland"],
                "access_status": ["open"],
                "source_version": ["v1"],
            }
        )
        ingestion = pd.DataFrame(
            {
                "source_key": ["source_a"],
                "source_name": ["Different display name"],
                "region": ["Other"],
                "countries_or_coverage": ["Otherland"],
                "access_status": ["restricted"],
                "ingestion_status": ["integrated_numeric"],
                "raw_food_observations_loaded": [10],
                "mapped_numeric_measurements_retained": [20],
                "source_version": ["v0"],
            }
        )
        actual = _build_source_table(registry, ingestion).iloc[0]
        self.assertEqual(actual.display_name, "Source A")
        self.assertEqual(actual.region, "Europe")
        self.assertEqual(actual.source_status, "numerically_integrated")

    def test_training_eligibility_audit_preserves_source_evidence(self) -> None:
        measurements = pd.DataFrame(
            {
                "direct_mass_label_candidate": [True, False, False, False],
                "value_status": ["observed", "observed", "observed", "reported_zero_unresolved"],
                "normalized_value_g_per_100g": [1.0, 2.0, 3.0, 0.0],
                "conversion_status": ["converted_exact_mass"] * 4,
                "source_value_origin": [
                    "source_reported",
                    "food_level_derivation_not_individual_assay_certification",
                    "recipe_or_estimation",
                    "source_reported",
                ],
                "source_note": [
                    "",
                    "profile_derivation=Analysed; food_details_derivation=Analysed",
                    "recipe",
                    "",
                ],
            }
        )

        actual = _annotate_training_eligibility(measurements)

        self.assertEqual(actual.training_review_role.tolist(), [
            "strict_model_label",
            "tier_b_curated_analytical_candidate",
            "evidence_only_derived_or_recipe",
            "evidence_only_value_semantics",
        ])


if __name__ == "__main__":
    unittest.main()

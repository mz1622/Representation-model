from pathlib import Path
import unittest

import pandas as pd

from foodcomp.composition_classification import (
    classify_components,
    classification_counts,
    classification_findings,
    load_classification,
)


ROOT = Path(__file__).resolve().parents[1]


class CompositionClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_classification(
            ROOT / "data/reference/composition_classification_v5_1.json"
        )

    def frame(self, *keys):
        return pd.DataFrame([
            {
                "component_concept_id": f"test:{i}",
                "authority_namespace": namespace,
                "authority_id": identity,
                "canonical_name": "Deliberately uninformative display name",
                "nutritional_role": "legacy_placeholder",
                "training_role": "context_only",
                "train_count": 120,
                "validation_count": 30,
                "train_raw_median": 0.0,
            }
            for i, (namespace, identity) in enumerate(keys)
        ])

    def category(self, namespace, identity):
        data = classify_components(self.frame((namespace, identity)), self.spec)
        return data.iloc[0]["classification_category"]

    def test_exact_snapshot_coverage(self):
        self.assertEqual(len(self.spec["lookup"]), 293)
        axes = self.frame(*self.spec["lookup"])
        annotated = classify_components(axes, self.spec)
        counts = classification_counts(annotated, self.spec)
        self.assertEqual(counts["axes"].sum(), 293)
        self.assertEqual(len(annotated), 293)
        self.assertTrue(annotated.classification_reason_en.str.len().gt(30).all())
        self.assertTrue(annotated.classification_reason_zh.str.len().gt(15).all())
        self.assertTrue(annotated.classification_evidence_urls.str.startswith("https://").all())

    def test_unknown_identity_is_not_fuzzy_matched(self):
        axes = self.frame(("SOURCE", "unknown:1"))
        axes["canonical_name"] = "alpha-tocopherol"
        with self.assertRaisesRegex(ValueError, "Unreviewed classification"):
            classify_components(axes, self.spec)

    def test_e_forms_do_not_all_become_core_vitamins(self):
        self.assertEqual(self.category("INFOODS", "TOCPHA"), "micro_vitamin")
        for code in ("TOCPHG", "TOCPHD", "TOCPHB", "TOCTRA"):
            self.assertEqual(self.category("INFOODS", code), "vitamin_related")

    def test_minerals_are_not_all_small_amount_elements(self):
        self.assertEqual(self.category("INFOODS", "CA"), "micro_mineral")
        self.assertEqual(self.category("INFOODS", "ID"), "micro_mineral")
        for code in ("AS", "CD", "HG", "PB", "NI"):
            self.assertEqual(self.category("INFOODS", code), "contaminant")
        self.assertEqual(self.category("INFOODS", "CR"), "element_boundary")
        self.assertEqual(self.category("INFOODS", "FD"), "element_boundary")

    def test_macronutrient_and_constituent_are_distinct(self):
        self.assertEqual(self.category("INFOODS", "FAT"), "macro")
        self.assertEqual(self.category("INFOODS", "CHO-"), "macro")
        self.assertEqual(self.category("INFOODS", "CHOCDF"), "macro")
        self.assertEqual(self.category("INFOODS", "WATER"), "macro")
        self.assertEqual(self.category("INFOODS", "F18D3CN3"), "fatty_acid")
        self.assertEqual(self.category("INFOODS", "LEU"), "amino_acid")
        self.assertEqual(self.category("INFOODS", "FIBINS"), "carbohydrate_fraction")

    def test_carotenoids_and_metabolites(self):
        self.assertEqual(self.category("INFOODS", "CARTB"), "vitamin_related")
        self.assertEqual(self.category("INFOODS", "LYCPN"), "phytochemical")
        self.assertEqual(self.category("USDA_CNF_NUTRIENT_NBR", "338"), "phytochemical")
        self.assertEqual(self.category("INFOODS", "CHOCALOH"), "vitamin_related")

    def test_choline_is_neither_cholesterol_nor_betaine(self):
        self.assertEqual(self.category("INFOODS", "CHOLN"), "choline")
        self.assertEqual(self.category("INFOODS", "CHOL-"), "sterol")
        self.assertEqual(self.category("INFOODS", "BETN"), "nitrogenous")

    def test_invalid_and_ambiguous_entries_are_not_hidden(self):
        axes = self.frame(
            ("SOURCE", "foodb:Nutrient:38"),
            ("SOURCE", "foodb:Compound:6288"),
            ("EUROFIR_EFSA", "RF-00000252-NTR"),
        )
        annotated = classify_components(axes, self.spec)
        self.assertEqual(
            annotated.classification_category.tolist(),
            ["non_mass_or_censored", "unresolved", "non_mass_or_censored"],
        )
        issues = classification_findings(pd.DataFrame(), annotated)
        self.assertEqual(len(issues), 3)

    def test_annotation_preserves_values_and_eligibility(self):
        axes = self.frame(("INFOODS", "CHOCDF"), ("INFOODS", "TOCPHG"))
        original = axes.copy(deep=True)
        annotated = classify_components(axes, self.spec)
        pd.testing.assert_frame_equal(axes, original)
        for field in ("component_concept_id", "train_raw_median", "training_role"):
            pd.testing.assert_series_equal(axes[field], annotated[field])
        self.assertIn("legacy_nutritional_role", annotated.columns)
        self.assertNotIn("nutritional_role", annotated.columns)
        self.assertFalse(annotated.classification_expert_signed_off.any())

    def test_missing_protein_is_reported_not_invented(self):
        axes = pd.DataFrame([{
            "component_concept_id": "protein",
            "canonical_name": "protein, total",
            "authority_id": "PROCNT",
            "expression_variant": "label_expression",
            "training_exclusion_reason": "non_mass_or_label_expression",
            "train_count": 11317,
            "validation_count": 743,
        }])
        retained = classify_components(self.frame(("INFOODS", "NT")), self.spec)
        findings = classification_findings(axes, retained)
        self.assertEqual(len(findings), 1)
        self.assertIn("absent", findings.iloc[0]["issue"])


if __name__ == "__main__":
    unittest.main()

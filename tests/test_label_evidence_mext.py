from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from foodcomp.label_evidence import gate_reference_evidence, inspect_reference, reference_ids
from foodcomp.mext_audit import main_unit, modality, parse_mext_value
from foodcomp.sources import _cofid_food_key, _usda_infoods_map


class MextEvidenceTests(unittest.TestCase):
    def test_zero_is_not_certified_absence(self):
        for value, status in ((0, "reported_zero_censored_or_nondetect"), ("(0)", "estimated_zero"),
                              ("Tr", "trace_interval"), ("(Tr)", "estimated_trace"), ("-", "unmeasured_dash")):
            result = parse_mext_value(value)
            self.assertEqual(result["value_status"], status)
            self.assertFalse(result["is_exact_observed_zero"])

    def test_parenthesized_positive_is_not_an_assay(self):
        self.assertEqual(parse_mext_value("(11.3)")["value_status"], "estimated_positive")
        self.assertEqual(parse_mext_value(11.3)["value_status"], "reported_positive_origin_unresolved")
        self.assertEqual(parse_mext_value("14.0\u2020")["value_status"], "unparsed_requires_review")
        self.assertEqual(parse_mext_value("14.0\u2020", prescribed_method_annotation_verified=True)["numeric_value"], 14.0)
        self.assertEqual(parse_mext_value("*")["value_status"], "footnote_only_value_unresolved")

    def test_sodium_unit_and_different_denominators(self):
        self.assertEqual(main_unit(23), "mg/100 g")
        self.assertEqual(modality("NA", "100g_edible_food"), "mass_per_100g_edible_food")
        self.assertEqual(modality("FASATF", "100g_total_fatty_acids"), "relative_denominator_excluded")
        self.assertEqual(modality("ILEN", "1g_reference_nitrogen"), "relative_denominator_excluded")

    def test_equivalents_and_ambiguous_identity_held(self):
        for tag in ("CHOAVLM", "CARTBEQ", "VITA_RAE", "NE", "NACL_EQ", "FATNLEA"):
            self.assertEqual(modality(tag, "100g_edible_food"), "equivalent_expression_excluded")
        self.assertEqual(modality("VITK", "100g_edible_food"), "mixed_mass_equivalent_definition_hold")
        self.assertEqual(modality("FAUN", "100g_edible_food"), "unidentified_substances_excluded")


class LabelEvidenceTests(unittest.TestCase):
    def test_sodium_na_tag_survives_cnf_crosswalk(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            pd.DataFrame({"Nutrient_Code": [307, 999], "Tagname": ["NA", ""]}).to_csv(base / "Nutrient_Name.csv", index=False)
            mapping = _usda_infoods_map(base)
            self.assertEqual(mapping["307"], "NA")
            self.assertNotIn("999", mapping)

    def test_reused_cofid_code_cannot_merge_unrelated_foods(self):
        ambiguous = {"13-669"}
        self.assertNotEqual(_cofid_food_key("13-669", "Aubergine, roasted", ambiguous), _cofid_food_key("13-669", "Watercress, raw", ambiguous))
        self.assertEqual(_cofid_food_key("14-362", "Apples, raw", ambiguous), "14-362")

    def setUp(self):
        self.catalogs = {
            "frida": {"1655": {"title": "Natural zero value for content. Not analyzed", "reference_type": "E"},
                      "2187": {"title": "FoodData Central", "reference_type": "WW"},
                      "2201": {"title": "Nutrient Content in Ice Cream", "reference_type": "R"}},
            "ciqual": {"444": {"title": "Valeur ajust\u00e9e/calcul\u00e9e/imput\u00e9e Ciqual", "reference_type": "official"},
                       "405": {"title": "Analyses Ciqual 2020", "reference_type": "official"}},
        }

    def test_reference_lists_not_cast_to_single_number(self):
        self.assertEqual(reference_ids("2201, 2187"), ["2201", "2187"])
        check = inspect_reference("frida", "2201, 2187", self.catalogs)
        self.assertTrue(check["strict_evidence_rejected"])
        self.assertIn("copied_database", check["evidence_issues"])

    def test_confidence_grade_cannot_override_estimation(self):
        self.assertTrue(inspect_reference("ciqual", "444", self.catalogs)["strict_evidence_rejected"])
        self.assertFalse(inspect_reference("ciqual", "405", self.catalogs)["strict_evidence_rejected"])
        self.assertTrue(inspect_reference("ciqual", "unknown", self.catalogs)["strict_evidence_rejected"])

    def test_source_evidence_rejections_preserve_values(self):
        frame = pd.DataFrame({"source_key": ["frida", "ciqual", "usda_sr_legacy"], "source_reference": ["1655", "444", "Analytical or derived from analytical"],
                              "method_expression": ["compiled", "compiled", "Recipe; known formulation"], "raw_value": [0, 4.1, 0.2],
                              "main_value_eligible": [True] * 3, "strict_validation_eligible": [True] * 3,
                              "independent_evidence": [True] * 3, "quality_tier": ["A"] * 3,
                              "quality_evidence": [""] * 3, "is_explicit_zero": [True, False, False]})
        result = gate_reference_evidence(frame, self.catalogs)
        self.assertFalse(result.main_value_eligible.any())
        self.assertFalse(result.strict_validation_eligible.any())
        self.assertFalse(result.independent_evidence.any())
        self.assertEqual(result.raw_value.tolist(), frame.raw_value.tolist())
        self.assertFalse(result.iloc[0].is_explicit_zero)
        self.assertEqual(result.iloc[0].value_status, "assumed_zero")


if __name__ == "__main__":
    unittest.main()

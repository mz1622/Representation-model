"""Unit tests for conservative semantic axis-candidate eligibility."""

from __future__ import annotations

import unittest

import pandas as pd

from foodcomp.semantic_axis_coverage import (
    _best_name_similarity,
    _identity_evidence,
)


def source(name: str, definition: str = "") -> pd.Series:
    return pd.Series({
        "original_name": name,
        "original_name_local": "",
        "source_definition": definition,
        "source_component_group": "",
        "source_chemical_class": "",
        "infoods_tag": "",
    })


def target(name: str, aliases: str = "", original: str = "") -> pd.Series:
    return pd.Series({
        "canonical_name": name,
        "aliases_for_review": aliases,
        "original_axis_names": original,
    })


class SemanticAxisCoverageTests(unittest.TestCase):
    def test_display_variants_are_exact_name_leads(self) -> None:
        candidate = source("Tocotrienol, alpha")
        expected = target("Tocotrienol, alpha", aliases="tocotrienol alpha")
        lexical, overlap, exact = _best_name_similarity(candidate, expected)
        self.assertTrue(exact)
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=exact,
            semantic_score=0.80, lexical_score=lexical, token_overlap=overlap,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "exact_name_or_registered_alias")

    def test_matching_fatty_acid_signature_is_a_manual_same_analyte_lead(self) -> None:
        candidate = source("Fatty acids, polyunsaturated, 20:5 n-3, eicosapentaenoic (EPA)")
        expected = target("Eicosapentaenoic acid (EPA; 20:5n-3)")
        lexical, overlap, exact = _best_name_similarity(candidate, expected)
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=exact,
            semantic_score=0.85, lexical_score=lexical, token_overlap=overlap,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "matching_explicit_structural_signature")

    def test_systematic_all_cis_alias_does_not_reject_common_epa_label(self) -> None:
        candidate = source("Fatty acids, polyunsaturated, 20:5 n-3, eicosapentaenoic (EPA)")
        expected = target(
            "Eicosapentaenoic acid (EPA; 20:5n-3)",
            aliases="eicosapentaenoic acid all cis 5 8 11 14 17",
        )
        lexical, overlap, exact = _best_name_similarity(candidate, expected)
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=exact,
            semantic_score=0.85, lexical_score=lexical, token_overlap=overlap,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "matching_explicit_structural_signature")

    def test_different_vitamer_number_is_not_same_analyte(self) -> None:
        candidate = source("Menaquinone 5")
        expected = target("Menaquinone-4")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.90, lexical_score=0.92, token_overlap=0.5,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_explicit_structural_or_vitamer_signature")

    def test_positional_isomer_is_not_same_analyte(self) -> None:
        candidate = source("4,5-Dicaffeoylquinic acid")
        expected = target("3,4-Dicaffeoylquinic acid")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.90, lexical_score=0.95, token_overlap=0.6,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_explicit_positional_isomer")

    def test_acylated_glycoside_is_not_the_parent_glycoside(self) -> None:
        candidate = source("Pelargonidin 3-(6''-succinyl-glucoside)")
        expected = target("Pelargonidin 3-O-glucoside")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.80, lexical_score=0.82, token_overlap=0.67,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_glycoside_or_acylated_form")

    def test_specific_fatty_acid_isomer_does_not_fill_unspecified_target(self) -> None:
        candidate = source("Fatty acids, monounsaturated, 16:1c, hexadecenoic")
        expected = target("Hexadecenoic acid (16:1; cis/trans unspecified)")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.84, lexical_score=0.35, token_overlap=0.4,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "source_is_more_specific_fatty_acid_isomer")

    def test_available_carbohydrate_is_not_total_carbohydrate(self) -> None:
        candidate = source("Available carbohydrate, labelling")
        expected = target("Carbohydrate, total")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.78, lexical_score=0.76, token_overlap=0.6,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_calculated_or_nutritional_expression")

    def test_definition_text_does_not_change_exact_name_identity(self) -> None:
        candidate = source(
            "Capsaicin",
            definition="Narrative mentions 8-methyl-N-vanillyl-6-nonenamide and 47.2 mg/kg.",
        )
        expected = target("Capsaicin")
        lexical, overlap, exact = _best_name_similarity(candidate, expected)
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=exact,
            semantic_score=0.8, lexical_score=lexical, token_overlap=overlap,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "exact_name_or_registered_alias")

    def test_different_glycoside_aglycone_is_not_same_analyte(self) -> None:
        candidate = source("Pelargonidin 3-O-glucoside")
        expected = target("Peonidin 3-O-glucoside")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.77, lexical_score=0.92, token_overlap=0.6,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_glycoside_aglycone")

    def test_different_gingerol_position_is_not_same_analyte(self) -> None:
        candidate = source("(S)-[8]-Gingerol")
        expected = target("[6]-Gingerol")
        accepted, reason = _identity_evidence(
            candidate, expected, exact_name=False,
            semantic_score=0.75, lexical_score=0.89, token_overlap=0.5,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "different_explicit_small_molecule_position")


if __name__ == "__main__":
    unittest.main()

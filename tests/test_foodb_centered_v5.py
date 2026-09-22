"""Regression checks for FooDB recovery, unique axes and source provenance."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodcomp.foodb_centered import name_key, nutrient_number, prepare_identities, foo_first_selection, retain_internal_reference_values
from foodcomp.harmonize import _authority_key


class FoodbCenteredTests(unittest.TestCase):
    def test_decimal_codes_do_not_collapse_total_and_form(self):
        for definition in ["CNF nutrient code 307.1", "USDA nutrient number 307.1"]:
            self.assertEqual(nutrient_number(definition), "307.1")
        row = pd.Series(dict(source_key="cnf", source_definition="CNF nutrient code 307.1", original_name="form"))
        self.assertEqual(_authority_key(row, set())[1], "307.1")

    def test_original_usda_code_does_not_create_foodb_duplicate(self):
        native = pd.Series(dict(source_key="usda_sr_legacy", source_definition="USDA nutrient number 578", original_name="Vitamin B-12, added"))
        imported = pd.Series(dict(source_key="foodb", source_definition="FooDB Content.orig_source_id = USDA nutrient number 578", original_name="Vitamin B-12, added"))
        self.assertEqual(_authority_key(native, set()), _authority_key(imported, set()))

    def test_gadoleic_inchikey_preserves_double_bond_position(self):
        row = dict(original_name="Gadoleic acid", infoods_tag="", source_key="foodb", source_component_id="Compound:1", inchikey="LQJBNNIYVWPHFW-QXMHVHEDSA-N")
        result = prepare_identities(pd.DataFrame([row]), {})
        self.assertEqual(result.iloc[0].harmonization_chebi, "CHEBI:32419")
        self.assertNotEqual(result.iloc[0].harmonization_chebi, "CHEBI:32425")

    def test_display_matching_preserves_stereochemistry(self):
        self.assertNotEqual(name_key("(+)-Valine"), name_key("(-)-Valine"))
        self.assertNotEqual(name_key("L-valine"), name_key("DL-valine"))

    def test_carbohydrate_method_is_not_invented(self):
        rows = [dict(original_name=name, infoods_tag="", source_key="foodb", source_component_id="Nutrient:3")
                for name in ["Carbohydrate", "Carbohydrate, by difference", "carbohydrates, total available"]]
        out = prepare_identities(pd.DataFrame(rows), {x: {} for x in ["CHO-", "CHOCDF", "CHOAVL"]})
        self.assertEqual(out.harmonization_tag.tolist(), ["CHO-", "CHOCDF", "CHOAVL"])

    def test_copies_do_not_duplicate_labels_and_foodb_is_anchor(self):
        records = [
            ("copy", "foodb", "USDA", False, "a", 2.0),
            ("origin", "usda_sr_legacy", "analytical", True, "a", 2.0),
            ("anchor", "foodb", "DUKE", True, "b", 4.0),
            ("supplement", "cnf", "CNF", True, "b", 5.0),
            ("held", "foodb", "USDA report", False, "c", 7.0),
        ]
        m = pd.DataFrame(records, columns=["measurement_id", "source_key", "source_reference", "main_value_eligible", "component_observation_id", "normalized_value_g_per_100g"])
        m["food_observation_id"] = "obs"
        fm = pd.DataFrame([dict(food_observation_id="obs", food_concept_id="food")])
        cm = pd.DataFrame([dict(component_observation_id=x, component_concept_id=x) for x in "abc"])
        split = pd.DataFrame([dict(food_concept_id="food", partition="train", source_holdout=False)])
        selected, ledger = foo_first_selection(m, fm, cm, split)
        self.assertEqual(set(selected.measurement_id), {"origin", "anchor"})
        self.assertEqual(ledger.set_index("measurement_id").loc["copy", "selection_decision"], "copied_value_resolved_to_original")

        m.loc[m.measurement_id.isin(["copy", "held"]), "main_value_eligible"] = True
        selected, ledger = foo_first_selection(m, fm, cm, split, resolve_internal_copies=False)
        self.assertEqual(set(selected.measurement_id), {"copy", "anchor", "held"})
        self.assertFalse(ledger.selection_decision.str.contains("copied_value|unresolved_copy").any())

    def test_fatty_acid_codes_are_one_identity(self):
        rows = [dict(original_name=n, infoods_tag=t, chebi_id="CHEBI:30772", source_key=s, source_component_id=i)
                for n,t,s,i in [("C4:0", "F4:0", "frida", "103"), ("4:0", "", "foodb", "Compound:12065")]]
        result = prepare_identities(pd.DataFrame(rows), {"F4D0": {}})
        self.assertEqual(result.harmonization_tag.tolist(), ["F4D0", "F4D0"])

    def test_frida_b6_is_not_a_single_pyridoxal_molecule(self):
        row = dict(original_name="Vitamin B6", infoods_tag="VITB6", chebi_id="CHEBI:18405", source_key="frida", source_component_id="40")
        result = prepare_identities(pd.DataFrame([row]), {})
        self.assertIn("pyridoxine hydrochloride", result.iloc[0].harmonization_name)
        self.assertEqual(result.iloc[0].harmonization_exclusion_reason, "documented_vitamer_equivalent_not_single_chemical_mass")

    def test_internal_citation_policy_does_not_relax_value_gates(self):
        rows = [dict(exclusion_reason=r, normalized_value_g_per_100g=x, value_status=s,
                     conversion_status="converted_exact_mass", measurement_modality="mass_fraction_fresh_weight",
                     main_value_eligible=False, independent_evidence=False)
                for r,x,s in [("external_reference_requires_origin_resolution_and_deduplication", 2.0, "observed"),
                              ("invalid_missing_censored_assumed_zero_or_non_mass_value", -1, "missing")]]
        result = retain_internal_reference_values(pd.DataFrame(rows))
        self.assertEqual(result.main_value_eligible.tolist(), [True, False])
        self.assertEqual(result.independent_evidence.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()

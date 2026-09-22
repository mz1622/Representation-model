import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from foodcomp.composition_revision import MERGES, ROLE_POLICY, revise_compositions
from foodcomp.dataset import FoodCompositionDataset
from foodcomp.harmonize import classify_training_roles


class RevisionTests(unittest.TestCase):
    @staticmethod
    def roles(n_train=50, n_validation=5, identity="unresolved_component_identity", expression="label_expression"):
        ids = [f"f{i}" for i in range(n_train + n_validation)]
        part = ["train"] * n_train + ["validation"] * n_validation
        profiles = pd.DataFrame(dict(component_concept_id=["a"] * len(ids), food_concept_id=ids,
                                     partition=part, aggregation_status=["accepted"] * len(ids),
                                     canonical_value_g_per_100g=np.arange(len(ids)) / 10))
        components = pd.DataFrame([dict(component_concept_id="a", identity_status=identity, expression_variant=expression)])
        foods = pd.DataFrame(dict(food_concept_id=ids, partition=part))
        mapping = pd.DataFrame(dict(food_concept_id=ids, food_observation_id=ids))
        observations = pd.DataFrame(dict(food_observation_id=ids, source_key=["foodb"] * len(ids)))
        return classify_training_roles(profiles, components, foods,
                                       pd.DataFrame([dict(component_observation_id="o", component_concept_id="a")]),
                                       pd.DataFrame([dict(component_observation_id="o", source_key="foodb")]),
                                       pd.DataFrame([dict(component_observation_id="o", main_value_eligible=True)]),
                                       mapping, observations, **ROLE_POLICY)[0].iloc[0]

    def test_support_boundaries(self):
        self.assertEqual(self.roles().training_role, "maskable_target")
        self.assertEqual(self.roles(49, 5).training_role, "context_only")
        self.assertEqual(self.roles(50, 4).training_role, "context_only")

    def test_identity_and_expression_flags_do_not_exclude(self):
        for expression in ("label_expression", "biological_equivalent", "standard_expression"):
            self.assertEqual(self.roles(expression=expression).training_role, "maskable_target")

    def test_fibre_aliases_merge_before_cell_aggregation(self):
        data = []
        for i, (ns, key) in enumerate([("EUROFIR_EFSA", "RF-00000284-NTR"), ("SOURCE", "foodb:Nutrient:5"), ("INFOODS", "FIBTG")]):
            data.append(dict(component_concept_id=f"a{i}", authority_namespace=ns, authority_id=key,
                             canonical_name="fibre", source_keys="foodb", component_family="carbohydrate_family"))
        concepts, mapping, ledger, _ = revise_compositions(pd.DataFrame(data), pd.DataFrame(dict(
            component_observation_id=["o0", "o1", "o2"], component_concept_id=["a0", "a1", "a2"])))
        self.assertEqual(len(concepts), 1)
        self.assertEqual(mapping.component_concept_id.nunique(), 1)
        self.assertEqual(len(ledger), 3)
        self.assertEqual(concepts.iloc[0].canonical_name, "Dietary fibre, total")
        self.assertEqual(set(concepts.iloc[0].original_component_concept_ids.split(";")), {"a0", "a1", "a2"})

    def test_groups_do_not_conflate_totals_and_isomers(self):
        all_members = [key for g in MERGES for key in g["members"]]
        self.assertEqual(len(all_members), len(set(all_members)))
        self.assertNotIn("INFOODS:F20D4", all_members)
        self.assertNotIn("INFOODS:F18D1", all_members)
        self.assertNotIn("INFOODS:VITB12", all_members)

    def test_unit_holds_are_separate_from_removed_expression_gate(self):
        axes = pd.DataFrame([dict(component_concept_id=str(i), canonical_name=key, authority_namespace=ns,
                                  authority_id=key, source_keys="foodb") for i, (ns, key) in enumerate([
                                      ("INFOODS", "CHOCDF"), ("SOURCE", "foodb:Nutrient:38"),
                                      ("USDA_CNF_NUTRIENT_NBR", "320"), ("INFOODS", "PROCNT"),
                                      ("SOURCE", "foodb:Compound:0")])])
        _, _, _, held = revise_compositions(axes, pd.DataFrame(dict(component_concept_id=axes.component_concept_id, component_observation_id=axes.component_concept_id)))
        self.assertEqual(set(held.component_concept_id), {"0", "1", "2", "4"})

    def test_stages_share_hidden_labels_but_not_loss_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "canonical_profile_matrix.npz", values=np.array([[1., 2., 3., np.nan]]),
                     component_ids=np.array(["n", "c", "context", "missing"]), food_ids=np.array(["food"]))
            pd.DataFrame([dict(food_concept_id="food", partition="train", canonical_name="food")]).to_csv(root / "ml_partition.csv", index=False)
            pd.DataFrame(dict(component_concept_id=["n", "c", "context", "missing"],
                              training_role=["maskable_target", "maskable_target", "context_only", "maskable_target"],
                              prediction_stage=[1, 2, 0, 2], component_family=["n", "c", "c", "other"])).to_csv(root / "component_concept.csv.gz", index=False)
            a = FoodCompositionDataset(root, stage=1)[0]
            b = FoodCompositionDataset(root, stage=2)[0]
            np.testing.assert_array_equal(a["context_mask"], b["context_mask"])
            np.testing.assert_array_equal(a["target_mask"], [1, 0, 0, 0])
            np.testing.assert_array_equal(b["target_mask"], [0, 1, 0, 0])
            self.assertEqual(a["context_mask"][2], 0)
            self.assertEqual(b["target_mask"][3], 0)


if __name__ == "__main__":
    unittest.main()

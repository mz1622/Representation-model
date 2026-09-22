from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from foodcomp.benchmark import load_release
from foodcomp.dataset import FoodCompositionDataset
from foodcomp.frozen_release import attach_new_family_blocks
from foodcomp.harmonize import aggregate_profiles, classify_training_roles, _authority_key
from foodcomp.util import read_component_csv, write_csv


class EvidenceRebuildTests(unittest.TestCase):
    def test_existing_ambiguous_bridge_stays_held_on_next_version(self):
        previous = pd.DataFrame({"food_concept_id": ["a", "b", "bridge"],
                                 "family_cluster_id": ["family-a", "family-b", "ambiguous"],
                                 "family_expansion_conflict": [False, False, True]})
        concepts = pd.DataFrame({"food_concept_id": ["a", "b", "bridge"], "family_label": ["a", "b", "bridge"]})
        mapping = pd.DataFrame({"food_concept_id": ["a", "b", "bridge"], "normalized_lineage": ["a", "b", "bridge"]})
        updated = attach_new_family_blocks(concepts, mapping, pd.DataFrame(), previous).set_index("food_concept_id")
        self.assertTrue(updated.loc["bridge", "family_expansion_conflict"])
        self.assertEqual(updated.loc["bridge", "family_cluster_id"], "ambiguous")

    def test_validation_only_axis_is_archival_not_model_context(self):
        profiles = pd.DataFrame({
            "food_concept_id": ["train-food", "validation-food"],
            "component_concept_id": ["train-axis", "validation-axis"],
            "partition": ["train", "validation"], "aggregation_status": ["accepted"] * 2,
            "canonical_value_g_per_100g": [0.1, 0.2],
        })
        components = pd.DataFrame({"component_concept_id": ["train-axis", "validation-axis"],
                                   "identity_status": ["authority_verified"] * 2,
                                   "expression_variant": ["standard_expression"] * 2})
        cm = pd.DataFrame({"component_concept_id": ["train-axis", "validation-axis"],
                           "component_observation_id": ["co-train", "co-val"]})
        co = pd.DataFrame({"component_observation_id": ["co-train", "co-val"], "source_key": ["a", "b"]})
        fm = pd.DataFrame({"food_concept_id": ["train-food", "validation-food"], "food_observation_id": ["fo-train", "fo-val"]})
        fo = pd.DataFrame({"food_observation_id": ["fo-train", "fo-val"], "source_key": ["a", "b"]})
        measurements = pd.DataFrame({"component_observation_id": ["co-train", "co-val"], "main_value_eligible": [True] * 2})
        registry, _ = classify_training_roles(profiles, components, profiles[["food_concept_id", "partition"]], cm, co, measurements, fm, fo)
        row = registry.set_index("component_concept_id").loc["validation-axis"]
        self.assertEqual(row.training_role, "excluded")
        self.assertEqual(row.training_exclusion_reason, "no_training_observations_archival_only")

    def test_literal_sodium_identifier_survives_registry_and_model_interface(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"component_concept_id": ["sodium"], "infoods_tag": ["NA"],
                                  "authority_id": ["NA"], "original_name": ["Sodium"],
                                  "training_role": ["maskable_target"], "component_family": ["mineral:NA"],
                                  "description": [""]})
            write_csv(frame, root / "component_concept.csv.gz")
            loaded = read_component_csv(root / "component_concept.csv.gz")
            self.assertEqual(_authority_key(loaded.iloc[0], {"NA"})[:2], ("INFOODS", "NA"))
            self.assertTrue(pd.isna(loaded.iloc[0].description))
            write_csv(pd.DataFrame({"food_concept_id": ["food"], "partition": ["train"],
                                    "canonical_name": ["Example"]}), root / "ml_partition.csv")
            np.savez_compressed(root / "canonical_profile_matrix.npz", values=np.array([[0.1]]),
                                food_ids=np.array(["food"]), component_ids=np.array(["sodium"]))
            self.assertEqual(load_release(root)[2].iloc[0].authority_id, "NA")
            self.assertEqual(FoodCompositionDataset(root).components.iloc[0].infoods_tag, "NA")

    def test_reaggregate_remaining_valid_evidence_not_old_center(self):
        measurements = pd.DataFrame({
            "measurement_id": ["good", "bad", "new-training-reference"],
            "food_observation_id": ["old-observation", "old-observation", "new-observation"],
            "component_observation_id": ["component"] * 3,
            "main_value_eligible": [True, False, True],
            "strict_validation_eligible": [True, False, False],
            "normalized_value_g_per_100g": [0.7, 3.4, 0.4],
            "sample_count": [3] * 3, "quality_tier": ["B", "B", "C"],
            "source_key": ["a", "b", "c"], "independent_evidence": [True, False, False],
        })
        foods = pd.DataFrame({"food_observation_id": ["old-observation", "new-observation"],
                              "food_concept_id": ["food"] * 2, "exclusion_flag": [""] * 2})
        components = pd.DataFrame({"component_observation_id": ["component"], "component_concept_id": ["axis"]})
        partition = pd.DataFrame({"food_concept_id": ["food"], "partition": ["validation"],
                                  "validation_panel": ["family_holdout"], "source_holdout": [False],
                                  "cv_fold": [-1], "family_cluster_id": ["family"]})
        profiles, _ = aggregate_profiles(measurements, foods, components, partition, {"old-observation"})
        self.assertEqual(profiles.iloc[0].canonical_value_g_per_100g, 0.7)
        self.assertEqual(profiles.iloc[0].measurement_ids, "good")
        measurements.loc[0, "main_value_eligible"] = False
        self.assertTrue(aggregate_profiles(measurements, foods, components, partition, {"old-observation"})[0].empty)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from foodcomp.cell_selection import SOURCE_PRIORITY, select_cell_records
from foodcomp.harmonize import aggregate_profiles


class CellSelectionTests(unittest.TestCase):
    @staticmethod
    def records():
        return pd.DataFrame([
            dict(measurement_id=f"m{i}", source_key=s, food_concept_id="food", component_concept_id="axis",
                 food_observation_id=f"f{i}", component_observation_id="a", normalized_value_g_per_100g=float(i),
                 quality_tier="B", sample_count=10, standard_error=np.nan, conversion_factor=1.0,
                 main_value_eligible=True, validation_reference_eligible=True, independent_evidence=True)
            for i, s in enumerate(SOURCE_PRIORITY)
        ])

    def test_source_priority_before_quality_and_no_averaging(self):
        records = self.records()
        records.loc[0, "quality_tier"] = "D"
        records.loc[1, "quality_tier"] = "A"
        winners, audit = select_cell_records(records)
        self.assertEqual(winners.measurement_id.tolist(), ["m0"])
        self.assertEqual(winners.normalized_value_g_per_100g.tolist(), [0.0])
        self.assertEqual(audit.candidate_source_count.tolist(), [6])
        self.assertTrue(audit.candidate_value_disagreement.all())
        for i in range(1, 6):
            winner, _ = select_cell_records(records.iloc[i:])
            self.assertEqual(winner.source_key.iloc[0], SOURCE_PRIORITY[i])

    def test_within_source_quality_sample_count_then_stable_id(self):
        records = self.records()
        records["source_key"] = "cnf"
        records["quality_tier"] = ["C", "A", "A", "B", "A", "A"]
        records["sample_count"] = [100, np.nan, 10, 100, 10, -1]
        for seed in range(3):
            winners, _ = select_cell_records(records.sample(frac=1, random_state=seed))
            self.assertEqual(winners.measurement_id.tolist(), ["m2"])

    def test_invalid_values_unknown_sources_and_duplicate_ids_raise(self):
        for field, value in [("source_key", "unknown"), ("normalized_value_g_per_100g", np.nan),
                             ("normalized_value_g_per_100g", -1), ("measurement_id", "m1")]:
            records = self.records()
            records.loc[0, field] = value
            with self.assertRaises(ValueError):
                select_cell_records(records)
        with self.assertRaises(ValueError):
            select_cell_records(self.records(), ("foodb", "foodb"))

    def aggregate(self, records, *, holdout=False, frozen=None, partition="validation"):
        mapping = records[["food_observation_id", "food_concept_id"]].assign(exclusion_flag="")
        components = pd.DataFrame([dict(component_observation_id="a", component_concept_id="axis")])
        partitions = pd.DataFrame([dict(food_concept_id="food", partition=partition,
                                       source_holdout=holdout, validation_panel="source_holdout" if holdout else "family_holdout",
                                       cv_fold=np.nan, family_cluster_id="family")])
        return aggregate_profiles(records.drop(columns=["food_concept_id", "component_concept_id"]),
                                  mapping, components, partitions, frozen, source_priority=SOURCE_PRIORITY)

    def test_validation_eligibility_precedes_precedence(self):
        records = self.records()
        records.loc[0, "validation_reference_eligible"] = False
        profile, conflicts = self.aggregate(records)
        self.assertEqual(profile.source_keys.tolist(), ["usda_foundation"])
        self.assertTrue(conflicts.empty)
        train_profile, _ = self.aggregate(records, partition="train")
        self.assertEqual(train_profile.source_keys.tolist(), ["foodb"])

    def test_frozen_observations_and_source_holdout_precede_precedence(self):
        records = self.records()
        profile, _ = self.aggregate(records, frozen={"f2", "f3"})
        self.assertEqual(profile.source_keys.tolist(), ["usda_sr_legacy"])
        profile, _ = self.aggregate(records, holdout=True)
        self.assertEqual(profile.source_keys.tolist(), ["usda_foundation"])
        profile, _ = self.aggregate(records, holdout=True, frozen={"f0", "f2"})
        self.assertTrue(profile.empty)

    def test_same_source_conflict_keeps_exactly_one_original_record(self):
        records = self.records().iloc[:2].copy()
        records["source_key"] = "cnf"
        records["normalized_value_g_per_100g"] = [0.0, 50.0]
        records["standard_error"] = 0.001
        profile, conflicts = self.aggregate(records)
        self.assertTrue(conflicts.empty)
        self.assertEqual(profile.measurement_ids.tolist(), ["m0"])
        self.assertEqual(profile.canonical_value_g_per_100g.tolist(), [0.0])
        self.assertEqual(profile.selected_measurement_count.tolist(), [1])
        self.assertEqual(profile.candidate_record_count.tolist(), [2])


if __name__ == "__main__":
    unittest.main()

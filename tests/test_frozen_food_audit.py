from pathlib import Path
import unittest

import pandas as pd

from foodcomp.frozen_food_audit import (
    build_exact_component_mapping,
    build_exact_name_food_groups,
    eligible_direct_mass_measurements,
    summarize_exact_name_axis_cells,
)


class FrozenFoodAuditTests(unittest.TestCase):
    def test_exact_name_groups_are_case_and_whitespace_normalized_only(self):
        foods = pd.DataFrame(
            {
                "food_observation_id": ["f1", "f2", "f3"],
                "source_key": ["a", "b", "c"],
                "source_food_id": ["1", "2", "3"],
                "original_name": ["Apple", "  apple ", "Apple, raw"],
                "part": ["fruit", "fruit", "fruit"],
            }
        )
        observations, groups = build_exact_name_food_groups(foods)
        self.assertEqual(observations.exact_name_group_id.nunique(), 2)
        self.assertEqual(groups.loc[groups.display_name.eq("Apple"), "food_observation_count"].item(), 2)
        self.assertNotEqual(
            observations.loc[observations.food_observation_id.eq("f1"), "exact_name_group_id"].item(),
            observations.loc[observations.food_observation_id.eq("f3"), "exact_name_group_id"].item(),
        )

    def test_mapping_is_exact_and_excludes_by_difference(self):
        panel = pd.DataFrame(
            {
                "target_axis_id": ["fat", "carb"],
                "canonical_name": ["Fat, total", "Carbohydrate, total"],
                "original_axis_names": ["Fat, total", "Carbohydrate, total"],
                "aliases_for_review": ["fat total", "carbohydrate total"],
            }
        )
        components = pd.DataFrame(
            {
                "component_observation_id": ["c1", "c2", "c3"],
                "source_key": ["foodb", "usda", "usda"],
                "source_component_id": ["Nutrient:1", "x", "y"],
                "original_name": ["Fat", "Carbohydrate, total (by difference)", "Fatty acids"],
                "infoods_tag": ["", "CHOCDF", ""],
            }
        )
        mapping = build_exact_component_mapping(panel, components).set_index("component_observation_id")
        self.assertEqual(mapping.at["c1", "target_axis_id"], "fat")
        self.assertEqual(mapping.at["c2", "mapping_status"], "not_mapped_by_exact_rule")
        self.assertEqual(mapping.at["c3", "mapping_status"], "not_mapped_by_exact_rule")

    def test_conflicts_are_retained_and_direct_mass_gate_excludes_censored_values(self):
        raw = pd.DataFrame(
            {
                "normalized_value_g_per_100g": [1.0, 1.0, 2.0, 3.0],
                "value_status": ["observed", "observed", "observed", "observed"],
                "is_censored": [False, False, False, True],
                "is_range_only": [False, False, False, False],
                "conversion_status": ["converted_exact_mass"] * 4,
                "measurement_modality": ["mass_fraction_fresh_weight"] * 4,
            }
        )
        self.assertEqual(int(eligible_direct_mass_measurements(raw).sum()), 3)
        measurements = pd.DataFrame(
            {
                "exact_name_group_id": ["food", "food"],
                "target_axis_id": ["axis", "axis"],
                "source_key": ["a", "b"],
                "normalized_value_g_per_100g": [1.0, 2.0],
                "measurement_id": ["m1", "m2"],
            }
        )
        cells = summarize_exact_name_axis_cells(measurements)
        self.assertEqual(cells.cell_status.item(), "cross_source_value_disagreement_preserved")
        self.assertEqual(cells.measurement_count.item(), 2)


if __name__ == "__main__":
    unittest.main()

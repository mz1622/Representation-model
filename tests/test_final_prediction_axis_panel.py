from pathlib import Path
import tempfile
import unittest

import pandas as pd

from foodcomp.final_prediction_axes import build_final_prediction_axis_panel
from foodcomp.prediction_axis_design import _normalise_axis_name


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data/processed/vmh_aligned_prediction_axis_design_v1/prediction_axis_registry.csv"


class FinalPredictionAxisPanelTests(unittest.TestCase):
    def test_panel_is_unique_and_keeps_semantic_safeguards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_final_prediction_axis_panel(
                draft_registry_path=DRAFT,
                output_dir=root / "data",
                report_dir=root / "report",
            )
            panel = pd.read_csv(root / "data/final_prediction_axis_registry.csv")

        self.assertGreaterEqual(manifest["direct_prediction_axes"], 200)
        self.assertEqual(manifest["direct_prediction_axes"], len(panel))
        self.assertTrue(panel.canonical_name.is_unique)
        self.assertTrue(panel.target_axis_id.is_unique)
        self.assertTrue(panel.canonical_name.map(_normalise_axis_name).is_unique)
        self.assertTrue(panel.direct_prediction_target.all())
        self.assertFalse(panel.numeric_pooling_permitted.any())
        self.assertEqual(
            set(panel.recommended_training_stage),
            {"Stage 1 nutrition composition", "Stage 2 food metabolome"},
        )
        self.assertFalse(panel.canonical_name.isin([
            "Vitamin E - Tocopherol", "Chlorogenic acid", "Rutin",
            "Glucosinalbin", "Butyric acid",
        ]).any())
        self.assertIn("5-Caffeoylquinic acid", set(panel.canonical_name))
        self.assertIn("Quercetin 3-O-rutinoside", set(panel.canonical_name))
        self.assertIn("Butyric acid (4:0)", set(panel.canonical_name))


if __name__ == "__main__":
    unittest.main()

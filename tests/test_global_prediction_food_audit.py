"""Focused invariants for global source integration helpers."""

from __future__ import annotations

import unittest

from foodcomp.global_prediction_food_audit import _parse_value, _source_value_column_codes, _unit_factor


class GlobalPredictionFoodAuditTests(unittest.TestCase):
    def test_zero_is_not_missing(self) -> None:
        self.assertEqual(_parse_value("0"), (0.0, "explicit_zero", True))

    def test_trace_is_not_zero(self) -> None:
        self.assertEqual(_parse_value("tr"), (None, "trace_or_censored", False))
        self.assertEqual(_parse_value("Spuren"), (None, "trace_or_censored", False))
        self.assertEqual(_parse_value("<0.1"), (None, "trace_or_censored", False))

    def test_bracketed_estimate_is_retained_but_not_direct(self) -> None:
        self.assertEqual(_parse_value("[0.2]"), (0.2, "bracketed_or_parenthesized_estimate", False))

    def test_mass_conversions(self) -> None:
        self.assertEqual(_unit_factor("g"), 1.0)
        self.assertEqual(_unit_factor("mg/100 g"), 1e-3)
        self.assertEqual(_unit_factor("µg / 100 g"), 1e-6)
        self.assertIsNone(_unit_factor("kcal"))

    def test_infoods_value_headers_preserve_code_and_strip_final_unit(self) -> None:
        self.assertEqual(_source_value_column_codes("WATER(G)"), ["WATER(G)", "WATER"])
        self.assertEqual(_source_value_column_codes("PHYTC- (MG)"), ["PHYTC- (MG)", "PHYTC-"])
        self.assertEqual(
            _source_value_column_codes("ENERC(KJ) (ORIGINAL)"),
            ["ENERC(KJ) (ORIGINAL)", "ENERC(KJ)(ORIGINAL)"],
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from foodcomp.distributions import distribution_statistics, summarize_compositions


class DistributionTests(unittest.TestCase):
    def fixture(self, values):
        return summarize_compositions(np.array(values, dtype=float),
            pd.DataFrame({'partition': ['train'] * (len(values) - 1) + ['validation']}),
            pd.DataFrame({'component_concept_id': ['a'], 'canonical_name': ['A']})).iloc[0]

    def test_validation_is_not_used(self):
        a = self.fixture([[1], [2], [3], [100]])
        b = self.fixture([[1], [2], [3], [np.nan]])
        pd.testing.assert_series_equal(a, b)
        self.assertEqual(a.raw_mean, 2)

    def test_missing_and_zeros_are_separate(self):
        a = self.fixture([[0], [2], [np.nan], [99]])
        self.assertEqual(a.n_observed, 2)
        self.assertEqual(a.n_missing, 1)
        self.assertEqual(a.n_zero, 1)
        self.assertEqual(a.zero_pct_observed, 50)
        self.assertEqual(a.raw_median, 1)
        self.assertEqual(a.positive_raw_median, 2)

    def test_constant_and_small_samples_are_explicitly_undefined(self):
        for v in ([0, 0, 0], [1], [1, 2], [], [.001] * 2000):
            self.assertTrue(np.isnan(distribution_statistics(v)['skewness']))
        self.assertEqual(distribution_statistics([.001] * 2000)['sd'], 0)
        a = self.fixture([[0], [0], [0], [50]])
        self.assertTrue(a.current_scale_fallback)
        self.assertEqual(a.current_effective_scale, 1)
        self.assertTrue(np.isnan(a.positive_raw_mean))

    def test_zero_heavy_but_variable_scale_collapse(self):
        a = self.fixture([[0]] * 9 + [[1], [90]])
        self.assertGreater(a.raw_sd, 0)
        self.assertEqual(a.log_iqr, 0)
        self.assertEqual(a.log_mad, 0)
        self.assertTrue(a.current_scale_fallback)

    def test_known_statistics_and_skew(self):
        s = distribution_statistics([1, 2, 3, 4, 5])
        self.assertEqual(s['mean'], 3)
        self.assertEqual(s['median'], 3)
        self.assertAlmostEqual(s['sd'], np.sqrt(2.5))
        self.assertEqual(s['skewness'], 0)
        self.assertEqual(s['iqr'], 2)
        self.assertGreater(distribution_statistics([0, 0, 0, 10])['skewness'], 1)

    def test_invalid_training_values_rejected(self):
        for invalid in (-1, 101, np.inf):
            with self.assertRaises(ValueError):
                self.fixture([[invalid], [0]])

    def test_center_change_does_not_change_same_scale_residual(self):
        y, prediction, scale = 4., 2., .5
        for center in (1., 3.):
            error = ((y - center) / scale - (prediction - center) / scale) ** 2
            self.assertEqual(error, (y - prediction) ** 2 / scale ** 2)


if __name__ == '__main__':
    unittest.main()

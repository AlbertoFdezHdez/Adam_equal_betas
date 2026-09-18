from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze_training_dynamics import diagonal_selection_rate, ema, omega1, omega2
from analysis.literal_decomposition import accumulate, corrected_ema, finalize, new_stats, sign_stable_mask


class LiteralDecompositionTests(unittest.TestCase):
    def test_bias_corrected_ema_of_constant_is_constant(self) -> None:
        values = np.ones((8, 3), dtype=np.float64)
        for beta in (0.9, 0.99, 0.999):
            np.testing.assert_allclose(corrected_ema(values, beta), values, atol=1e-14)

    def test_inclusive_sign_window_and_zero_filter(self) -> None:
        gradients = np.asarray(
            [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        )
        mask = sign_stable_mask(gradients, window=3, burn_in=0)
        np.testing.assert_array_equal(mask[:, 0], [False, False, True, False, False])
        np.testing.assert_array_equal(mask[:, 1], [False, False, False, False, True])

    def test_exact_decomposition_identity_and_budget(self) -> None:
        update = np.asarray([[1.2, -0.8], [0.9, -1.1]])
        sign = np.asarray([[1.0, -1.0], [1.0, -1.0]])
        lag = np.asarray([[0.15, 0.10], [-0.05, -0.20]])
        valid = np.ones_like(update, dtype=bool)
        stats = new_stats()
        accumulate(stats, update, sign, lag, valid)
        row = finalize(stats, window=1)
        self.assertEqual(row["n"], 4)
        self.assertEqual(row["max_identity_error"], 0.0)
        self.assertAlmostEqual(row["S_pct"] + row["L_pct"] + row["E_pct"], 100.0)


class OscillationTests(unittest.TestCase):
    def test_ema_and_differences(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0])
        np.testing.assert_allclose(ema(values, 1), values)
        self.assertAlmostEqual(omega1(values), 1.5)
        self.assertAlmostEqual(omega2(values), 1.0)

    def test_nonfinite_values_are_worst_in_selection_rate(self) -> None:
        grid = np.asarray([[[1.0], [math.nan]], [[2.0], [1.0]]])
        self.assertEqual(diagonal_selection_rate(grid), (2, 2, 1.0))


if __name__ == "__main__":
    unittest.main()

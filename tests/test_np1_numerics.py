"""Regressions for the leader's 2026-10-21 numerical discrepancy report."""
import unittest

import numpy as np

import NP1
from bcs.algorithms import solve as reference_solve
from bcs.data import generate


class NP1NumericsTests(unittest.TestCase):
    def test_reported_sensitive_trials_match_reference_trajectory(self):
        # Compare in the current runtime, not to platform-specific golden data.
        # Truth is deliberately discarded; it never enters either solver.
        for trial in (52, 135, 161):
            with self.subTest(trial=trial):
                A, y, _ = generate(100, 10, 3, 20261019, trial)
                actual = NP1.solve(A.copy(), y.copy())
                expected = reference_solve(A.copy(), y.copy(), kind="NP1")
                self.assertEqual(actual.status, expected.status)
                self.assertEqual(len(actual.history), len(expected.history))
                np.testing.assert_allclose(actual.history, expected.history,
                                           rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(actual.x, expected.x,
                                           rtol=1e-10, atol=1e-12)

    def test_zero_and_truncated_singular_values_are_safe(self):
        for rcond, rank in ((1e-12, 1), (1e-16, 2)):
            with self.subTest(rcond=rcond):
                A = np.diag([1.0, 1e-14, 0.0])
                mean, variance, actual_rank = NP1._posterior(
                    A, np.array([2.0, 3e-14, 0.0]),
                    np.ones(3), np.ones(3), rcond)
                self.assertEqual(actual_rank, rank)
                np.testing.assert_allclose(mean, [2.0, 3.0 if rank == 2 else 0.0, 0.0])
                np.testing.assert_allclose(variance, [0.0, 0.0 if rank == 2 else 1.0, 1.0])
        with np.errstate(divide="raise", invalid="raise"):
            mean, variance, rank = NP1._posterior(
                np.zeros((2, 3)), np.zeros(2), np.ones(3), np.ones(3), 1e-12)
        self.assertEqual(rank, 0)
        np.testing.assert_array_equal(mean, np.zeros(3))
        np.testing.assert_array_equal(variance, np.ones(3))


if __name__ == "__main__":
    unittest.main()

"""Independent numerical checks for the paper's noiseless NP-1 algorithm.

The reference deliberately forms full diagonal and covariance matrices, unlike
the diagonal-only production implementation. Run with unittest discovery.
"""

import unittest

import numpy as np

from NP1 import np1, solve_np1


def paper_reference(A, y, iterations, rcond=1e-14):
    """Equations (10)--(15), alpha=beta=0, with explicit matrix products."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n = A.shape[1]
    a = np.ones(n)
    b = np.ones(n)
    prior_a = np.ones(n)
    prior_b = np.ones(n)
    states = []
    for _ in range(iterations):
        # Eq. (12) and the complete covariance in Eq. (13).
        Da = np.diag(np.sqrt(prior_a))
        Ba = A @ np.diag(b)
        Pa = np.linalg.pinv(Ba @ Da, rcond=rcond)
        a = Da @ Pa @ y
        covariance_a = (np.eye(n) - Da @ Pa @ Ba) @ (Da @ Da)
        variance_a = np.diag(covariance_a).copy()

        # Eq. (10) uses this iteration's updated a, not the old a.
        Db = np.diag(np.sqrt(prior_b))
        Bb = A @ np.diag(a)
        Pb = np.linalg.pinv(Bb @ Db, rcond=rcond)
        b = Db @ Pb @ y
        covariance_b = (np.eye(n) - Db @ Pb @ Bb) @ (Db @ Db)
        variance_b = np.diag(covariance_b).copy()

        # E[a_i^2] and E[b_i^2] include posterior variance.
        prior_a = a * a + variance_a
        prior_b = b * b + variance_b
        states.append({
            "x": a * b,
            "mean_a": a.copy(),
            "mean_b": b.copy(),
            "variance_a": variance_a,
            "variance_b": variance_b,
            "precision_a": 1.0 / prior_a,
            "precision_b": 1.0 / prior_b,
        })
    return states


class NP1FormulaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(108)
        self.A = rng.normal(size=(4, 9))
        self.y = np.array([1.0, -0.4, 0.6, -1.2])

    def assert_matches_reference(self, A, y, iterations):
        expected = paper_reference(A, y, iterations)
        actual = solve_np1(
            A, y, max_iter=iterations, tol=0.0,
            rcond=1e-14, store_history=True,
        )
        self.assertEqual(actual.n_iter, iterations)
        for field, value in expected[-1].items():
            with self.subTest(iterations=iterations, field=field):
                np.testing.assert_allclose(
                    getattr(actual, field), value, rtol=2e-9, atol=2e-11,
                )
        np.testing.assert_allclose(
            actual.x_history, np.stack([state["x"] for state in expected]),
            rtol=2e-9, atol=2e-11,
        )
        return actual

    def test_one_iteration_matches_full_pseudoinverse_formulas(self):
        self.assert_matches_reference(self.A, self.y, 1)

    def test_multiple_iterations_update_second_moments_in_order(self):
        self.assert_matches_reference(self.A, self.y, 7)

    def test_rank_deficient_measurements_match_full_formulas(self):
        A = np.vstack([self.A[:3], self.A[0], self.A[1] + self.A[2]])
        y = np.concatenate([self.y[:3], [self.y[0], self.y[1] + self.y[2]]])
        actual = self.assert_matches_reference(A, y, 4)
        self.assertTrue(np.all(np.isfinite(actual.x)))
        np.testing.assert_allclose(A @ actual.x, y, rtol=1e-10, atol=1e-10)


class NP1RecoveryTests(unittest.TestCase):
    @staticmethod
    def sparse_problem():
        rng = np.random.default_rng(7)
        A = rng.normal(size=(30, 100))
        x = np.zeros(100)
        x[[9, 44, 71]] = [1.0, -1.2, 0.7]
        return A, A @ x, x

    def test_recovers_simple_sparse_signal(self):
        A, y, truth = self.sparse_problem()
        result = solve_np1(A, y)
        relative_error = np.linalg.norm(result.x - truth) / np.linalg.norm(truth)
        self.assertLess(relative_error, 1e-3)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.n_iter, 300)
        np.testing.assert_allclose(A @ result.x, y, rtol=1e-8, atol=1e-8)

    def test_zero_observation_terminates_with_finite_zero_solution(self):
        rng = np.random.default_rng(9)
        result = solve_np1(rng.normal(size=(4, 9)), np.zeros(4))
        np.testing.assert_array_equal(result.x, np.zeros(9))
        self.assertTrue(result.converged)
        self.assertLess(result.n_iter, 300)
        for field in (
            "mean_a", "mean_b", "variance_a", "variance_b",
            "precision_a", "precision_b",
        ):
            self.assertTrue(np.all(np.isfinite(getattr(result, field))), field)

    def test_determinism_wrapper_and_input_preservation(self):
        A, y, _ = self.sparse_problem()
        original_A, original_y = A.copy(), y.copy()
        first = solve_np1(A, y, store_history=True)
        second = solve_np1(A, y, store_history=True)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.x_history, second.x_history)
        np.testing.assert_array_equal(np1(A, y), first.x)
        np.testing.assert_array_equal(A, original_A)
        np.testing.assert_array_equal(y, original_y)
        self.assertEqual(first.n_iter, second.n_iter)
        self.assertEqual(first.x_history.shape, (first.n_iter, A.shape[1]))
        np.testing.assert_array_equal(first.x_history[-1], first.x)
        self.assertIsNone(solve_np1(A, y).x_history)


class NP1StoppingTests(unittest.TestCase):
    def test_iteration_cap_does_not_report_convergence(self):
        A, y, _ = NP1RecoveryTests.sparse_problem()
        result = solve_np1(A, y, max_iter=1, tol=0.0, store_history=True)
        self.assertEqual(result.n_iter, 1)
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iter")
        self.assertGreater(result.history["relative_change"][-1], 0.0)
        self.assertEqual(result.x_history.shape, (1, A.shape[1]))
        for values in result.history.values():
            self.assertEqual(values.shape, (1,))

    def test_convergence_and_histories_match_saved_estimates(self):
        A, y, _ = NP1RecoveryTests.sparse_problem()
        tolerance = 1e-3
        result = solve_np1(A, y, tol=tolerance, store_history=True)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "relative_change")
        self.assertLess(result.n_iter, 300)
        self.assertEqual(result.x_history.shape, (result.n_iter, A.shape[1]))
        for values in result.history.values():
            self.assertEqual(values.shape, (result.n_iter,))

        previous = np.vstack([np.ones(A.shape[1]), result.x_history[:-1]])
        measured_changes = np.linalg.norm(result.x_history - previous, axis=1)
        measured_changes /= np.linalg.norm(previous, axis=1)
        measured_residuals = np.linalg.norm(result.x_history @ A.T - y, axis=1)
        np.testing.assert_allclose(result.history["relative_change"], measured_changes)
        np.testing.assert_allclose(
            result.history["residual_norm"], measured_residuals, atol=1e-12,
        )
        self.assertLessEqual(result.history["relative_change"][-1], tolerance)
        self.assertTrue(np.all(result.history["relative_change"][:-1] > tolerance))
        self.assertTrue(np.all(result.history["elapsed_seconds"] >= 0.0))
        self.assertTrue(np.all(np.diff(result.history["elapsed_seconds"]) >= 0.0))
        for name in ("rank_a", "rank_b"):
            self.assertTrue(np.all(result.history[name] >= 0))
            self.assertTrue(np.all(result.history[name] <= min(A.shape)))


class NP1InputTests(unittest.TestCase):
    def test_column_measurements_are_equivalent_to_vector(self):
        A, y, _ = NP1RecoveryTests.sparse_problem()
        expected = solve_np1(A, y, max_iter=5, tol=0.0)
        actual = solve_np1(A, y[:, None], max_iter=5, tol=0.0)
        self.assertEqual(actual.x.shape, (A.shape[1],))
        np.testing.assert_array_equal(actual.x, expected.x)
        self.assertEqual(actual.n_iter, expected.n_iter)

    def test_rejects_invalid_iteration_limits(self):
        for value in (0, -1, 1.5, True, np.bool_(False), np.nan, np.inf, [2]):
            with self.subTest(max_iter=value):
                with self.assertRaises(ValueError):
                    solve_np1(np.eye(2), np.ones(2), max_iter=value)

    def test_rejects_invalid_numerical_options(self):
        invalid_options = {
            "tol": (-1.0, np.nan, np.inf, -np.inf, 1j, [1e-3]),
            "rcond": (-1e-8, 1.0, 2.0, np.nan, np.inf, 1j, [1e-8]),
            "variance_floor": (0.0, -1.0, 1e-301, 1.01, np.nan, np.inf, 1j, [1e-9]),
        }
        for name, values in invalid_options.items():
            for value in values:
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        solve_np1(np.eye(2), np.ones(2), **{name: value})

    def test_valid_numerical_boundaries(self):
        A = np.array([[1.0, 0.2, -0.3], [0.1, 1.0, 0.4]])
        y = np.array([0.7, -0.2])
        for floor in (1e-300, 1.0):
            with self.subTest(variance_floor=floor):
                result = solve_np1(
                    A, y, max_iter=np.int64(1), tol=0.0,
                    rcond=0.0, variance_floor=floor,
                )
                self.assertEqual(result.n_iter, 1)
                self.assertTrue(np.all(np.isfinite(result.x)))
                self.assertTrue(np.all(np.isfinite(result.precision_a)))
                self.assertTrue(np.all(result.precision_a <= 1.0 / floor))
                self.assertTrue(np.all(np.isfinite(result.precision_b)))
                self.assertTrue(np.all(result.precision_b <= 1.0 / floor))

    def test_rejects_incompatible_dimensions(self):
        bad_inputs = [
            (np.ones(3), np.ones(3)),
            (np.ones((2, 3)), np.ones(3)),
            (np.ones((2, 3)), np.ones((1, 2))),
            (np.ones((2, 3)), np.ones((2, 2))),
            (np.ones((0, 3)), np.ones(0)),
            (np.ones((2, 0)), np.ones(2)),
        ]
        for A, y in bad_inputs:
            with self.subTest(A_shape=A.shape, y_shape=y.shape):
                with self.assertRaises(ValueError):
                    solve_np1(A, y)

    def test_rejects_nonfinite_data(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad_value, argument="A"):
                A = np.ones((2, 3))
                A[0, 0] = bad_value
                with self.assertRaises(ValueError):
                    solve_np1(A, np.ones(2))
            with self.subTest(value=bad_value, argument="y"):
                with self.assertRaises(ValueError):
                    solve_np1(np.ones((2, 3)), np.array([1.0, bad_value]))

    def test_rejects_complex_data_without_silent_cast(self):
        with self.assertRaises((TypeError, ValueError)):
            solve_np1(np.ones((2, 3), dtype=complex) + 1j, np.ones(2))
        with self.assertRaises((TypeError, ValueError)):
            solve_np1(np.ones((2, 3)), np.ones(2, dtype=complex) + 1j)


if __name__ == "__main__":
    unittest.main()


import unittest
import numpy as np
from bcs.data import generate, fingerprint
from bcs.metrics import metrics
from bcs.algorithms import solve, posterior


class FrameworkTests(unittest.TestCase):
    def test_metrics_definition(self):
        result = metrics(np.array([2., 0.]), np.array([1., 0.]))
        self.assertEqual(result["mse"], .5)
        self.assertEqual(result["nmse"], 1)
        self.assertEqual(result["paper_mse_db"], 0)
        self.assertFalse(result["success"])
        self.assertFalse(metrics(np.array([0., .001]), np.array([0., 1.]), threshold=.999)["success"])
        self.assertEqual(metrics(np.array([1.]), np.array([1.]))["paper_mse_db"], -np.inf)
        with self.assertRaises(ValueError):
            metrics(np.array([np.nan]), np.array([1.]))
        with self.assertRaises(ValueError):
            metrics(np.array([1.]), np.array([0.]))

    def test_dataset_is_repeatable_and_noiseless(self):
        A, y, x = generate(100, 30, 3, 12, 0)
        self.assertEqual(np.count_nonzero(x), 3)
        np.testing.assert_array_equal(A @ x, y)
        self.assertEqual(fingerprint(A,y,x), fingerprint(*generate(100,30,3,12,0)))
        self.assertNotEqual(fingerprint(A,y,x), fingerprint(*generate(100,30,3,12,1)))

    def test_posterior_matches_independent_conditional_formula(self):
        rng = np.random.default_rng(8)
        A = rng.normal(size=(5, 12))
        other = rng.uniform(.5, 2, 12)
        v = rng.uniform(.5, 2, 12)
        y = rng.normal(size=5)
        H = A * other
        prior = np.diag(v)
        gain = prior @ H.T @ np.linalg.inv(H @ prior @ H.T)
        mean, variance = posterior(A, other, v, y)
        np.testing.assert_allclose(mean, gain @ y, atol=1e-12)
        np.testing.assert_allclose(variance, np.diag(prior-gain @ H @ prior), atol=1e-12)

    def test_all_solvers_recover_identity_case(self):
        A, x = np.eye(5), np.array([1.,0.,-2.,0.,0.])
        for name in ("NP0", "NP1", "SBL", "IRLS", "FOCUSS", "BP"):
            with self.subTest(algorithm=name):
                result = solve(A, x, kind=name)
                np.testing.assert_allclose(result.x, x, atol=1e-10)
                self.assertTrue(metrics(result.x,x)["success"])

    def test_bp_exact_equality_and_l1_optimum(self):
        A = np.array([[1., 2., 0.]])
        result = solve(A, np.array([2.]), kind="BP")
        np.testing.assert_allclose(result.x, [0.,1.,0.], atol=1e-9)

    def test_sbl_does_not_stop_at_initial_minimum_norm_solution(self):
        A, y, x = generate(100, 30, 3, 20261019, 0)
        result = solve(A, y, kind="SBL", convergence_tol=1e-6)
        self.assertGreater(len(result.history), 1)
        self.assertTrue(metrics(result.x, x)["success"])


if __name__ == "__main__":
    unittest.main()

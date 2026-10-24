"""Analytical baseline checks, framework integration, and display-only behavior."""
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from algorithms.baselines_member import (
    _weighted_posterior, solve, solve_sbl, solve_focuss, solve_irls, solve_bp,
)
from bcs.algorithms import Result
from bcs.data import generate
from bcs.metrics import metrics


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/"algorithms_baselines_member.json"
METHODS = {"SBL": solve_sbl, "FOCUSS": solve_focuss, "IRLS": solve_irls, "BP": solve_bp}


class BaselineFormulaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.A = rng.normal(size=(4, 9))
        self.y = rng.normal(size=4)

    def test_posterior_matches_independent_conditional_gaussian(self):
        q = np.linspace(.5, 2, 9)
        prior = np.diag(q)
        gain = prior @ self.A.T @ np.linalg.inv(self.A @ prior @ self.A.T)
        x, variance = _weighted_posterior(self.A, self.y, q, 1e-12, 1e-30, True)
        np.testing.assert_allclose(x, gain @ self.y, atol=1e-12)
        np.testing.assert_allclose(variance, np.diag(prior-gain @ self.A @ prior), atol=1e-12)

    def test_sbl_three_em_updates_match_dense_conditioning(self):
        q = np.ones(9)
        expected = []
        for _ in range(3):
            prior = np.diag(q)
            gain = prior @ self.A.T @ np.linalg.inv(self.A @ prior @ self.A.T)
            x = gain @ self.y
            variance = np.diag(prior-gain @ self.A @ prior)
            q = x*x + variance
            expected.append(x.copy())
        result = solve_sbl(self.A, self.y, max_iter=3, tol=1e-30)
        np.testing.assert_allclose(result.history, expected, rtol=1e-10, atol=1e-12)

    def test_focuss_update_is_squared_coefficient_weighting(self):
        initial = np.linalg.pinv(self.A) @ self.y
        Q = np.diag(initial**2)
        expected = Q @ self.A.T @ np.linalg.solve(self.A @ Q @ self.A.T, self.y)
        result = solve_focuss(self.A, self.y, max_iter=1)
        np.testing.assert_allclose(result.x, expected, rtol=1e-10, atol=1e-12)

    def test_irls_epsilon_is_not_squared(self):
        initial = np.linalg.pinv(self.A) @ self.y
        epsilon = .04
        Q = np.diag((initial**2+epsilon)**.75)
        expected = Q @ self.A.T @ np.linalg.solve(self.A @ Q @ self.A.T, self.y)
        result = solve_irls(self.A, self.y, max_iter=1, epsilon_start=epsilon)
        np.testing.assert_allclose(result.x, expected, rtol=1e-10, atol=1e-12)

    def test_bp_known_l1_optimum_and_signed_solution(self):
        for y in (np.array([1.]), np.array([-1.])):
            result = solve_bp(np.array([[1., 2., 0.]]), y)
            np.testing.assert_allclose(result.x, [0, y[0]/2, 0], atol=1e-10)
            self.assertEqual(result.status, "lp_optimal")
            self.assertEqual(len(result.history), 1)

    def test_rank_deficient_consistent_data(self):
        A = np.array([[1., 2., 0.], [2., 4., 0.]])
        y = np.array([1., 2.])
        for name, method in METHODS.items():
            with self.subTest(name=name):
                result = method(A, y)
                np.testing.assert_allclose(A @ result.x, y, atol=1e-9)

    def test_infeasible_noiseless_system_is_not_reported_as_success(self):
        A = np.array([[1., 2.], [2., 4.]])
        for name, method in METHODS.items():
            with self.subTest(name=name), self.assertRaises((RuntimeError, FloatingPointError)):
                method(A, np.array([1., 1.]))

    def test_bp_solver_failure_is_propagated(self):
        failed = mock.Mock(success=False, message="test failure")
        with mock.patch("algorithms.baselines_member.linprog", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                solve_bp(self.A, self.y)


class BaselineContractTests(unittest.TestCase):
    def test_result_history_timing_input_and_rng_contract(self):
        A, y, truth = generate(100, 30, 3, 20261019, 0)
        A0, y0 = A.copy(), y.copy()
        A.flags.writeable = y.flags.writeable = False
        before = np.random.get_state()
        for name, method in METHODS.items():
            with self.subTest(name=name):
                first = method(A, y, seed=1)
                second = method(A, y, seed=999)
                self.assertIsInstance(first, Result)
                self.assertTrue(metrics(first.x, truth)["success"])
                np.testing.assert_array_equal(first.x, second.x)
                np.testing.assert_array_equal(first.history, second.history)
                np.testing.assert_array_equal(first.history[-1], first.x)
                self.assertEqual(np.asarray(first.history).shape, (len(first.times), 100))
                self.assertTrue(np.all(np.diff(first.times) >= 0))
                self.assertTrue(np.all(np.asarray(first.times) >= 0))
                self.assertTrue(all(not np.shares_memory(x, first.x) for x in first.history))
                self.assertFalse(any(np.shares_memory(a, b) for a, b in
                                     zip(first.history, first.history[1:])))
        np.testing.assert_array_equal(A, A0)
        np.testing.assert_array_equal(y, y0)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_zero_measurements_and_vector_shapes(self):
        A = np.eye(4)
        for method in METHODS.values():
            for shape in ((4,), (4, 1), (1, 4)):
                result = method(A, np.zeros(shape))
                np.testing.assert_array_equal(result.x, np.zeros(4))
                self.assertTrue(result.converged)

    def test_invalid_data_and_options(self):
        cases = [(np.ones(3), np.ones(3)), (np.eye(2), np.ones(3)),
                 (np.empty((0, 2)), np.empty(0)), (np.eye(2), np.ones((2, 2))),
                 (np.array([[np.nan]]), np.ones(1)), (np.eye(2), np.array([1, np.inf]))]
        for A, y in cases:
            with self.subTest(A=A.shape, y=y.shape), self.assertRaises(ValueError):
                solve(A, y)
        for A, y in [(np.eye(2, dtype=complex), np.ones(2)),
                     (np.eye(2), np.ones(2, dtype=complex)),
                     (np.array([["bad"]]), np.ones(1))]:
            with self.assertRaises(TypeError):
                solve(A, y)
        for params in ({"tol": 0}, {"rcond": 1}, {"rcond": -1}, {"p": 1.1},
                       {"epsilon_min": 0}, {"epsilon_factor": 1}, {"max_iter": 0},
                       {"epsilon_start": 1e-9}, {"convergence_tol": np.inf},
                       {"lp_tolerance": 1e-12}, {"seed": -1}, {"kind": "NP1"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                solve(np.eye(2), np.ones(2), **params)
        for params in ({"tol": "1e-3"}, {"max_iter": True}, {"max_iter": 2.5},
                       {"fixed_iterations": 1}, {"seed": False}, {"typo": 1}):
            with self.subTest(params=params), self.assertRaises(TypeError):
                solve(np.eye(2), np.ones(2), **params)

    def test_iteration_limit_and_fixed_budget_are_truthful(self):
        A, y, _ = generate(12, 6, 2, 1234, 0)
        for method in (solve_sbl, solve_focuss, solve_irls):
            result = method(A, y, max_iter=1, tol=1e-30)
            self.assertEqual(len(result.history), 1)
            self.assertFalse(result.converged)
            self.assertEqual(result.status, "max_iter")
            fixed = method(A, y, max_iter=7, fixed_iterations=True)
            self.assertEqual(len(fixed.history), 7)

    def test_irls_does_not_stop_before_final_smoothing_level(self):
        result = solve_irls(np.eye(2), np.array([1., -2.]), max_iter=30)
        self.assertGreaterEqual(len(result.history), 9)
        self.assertTrue(result.converged)

    def test_default_registry_is_preserved_by_additional_registry(self):
        old = json.loads((ROOT/"algorithms.json").read_text(encoding="utf-8-sig"))
        new = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual({k: new[k] for k in old}, old)
        self.assertEqual(set(new)-set(old), {name+"_MEMBER" for name in METHODS})
        for path in (ROOT/"configs/baselines_member").glob("*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(set(config["algorithms"])-set(new))
            self.assertEqual(len(set(map(tuple, config["cases"]))), len(config["cases"]))


class BaselineIntegrationTests(unittest.TestCase):
    def test_original_cli_accepts_new_registry_and_replays_independent_processes(self):
        (ROOT/"tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            folder = Path(folder)
            config = dict(n=12, seed=20261019, trials=2, amplitudes="normal",
                          max_iter=30, tol=.001, success_threshold=.001,
                          cases=[[7, 1]], algorithms=[name+"_MEMBER" for name in METHODS])
            path = folder/"config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            for name in ("a", "b"):
                run = subprocess.run([sys.executable, str(ROOT/"main.py"), "run",
                                      "--config", str(path), "--registry", str(REGISTRY),
                                      "--out", str(folder/name), "--data-dir", str(folder/"data")],
                                     cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            compare = subprocess.run([sys.executable, str(ROOT/"main.py"), "compare",
                                      str(folder/"a"), str(folder/"b"), "--report", str(folder/"report.json")],
                                     cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(compare.returncode, 0, compare.stdout+compare.stderr)
            with (folder/"a/trials.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["error"] == "" for row in rows))
            self.assertEqual(len({row["data_hash"] for row in rows}), 2)
            with np.load(folder/"a/histories.npz") as histories:
                for row in rows:
                    key = f"m7_k1_t{row['trial']}_{row['algorithm']}"
                    self.assertEqual(histories[key].shape, (int(row["iterations"]), 12))
            plot = subprocess.run([sys.executable, str(ROOT/"main.py"), "plot", str(folder/"a")],
                                  cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(plot.returncode, 0, plot.stdout+plot.stderr)

    def test_display_path_creates_no_output_files(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        display = importlib.import_module("scripts.baselines_member_display")
        config = dict(n=100, seed=20261019, trials=1, amplitudes="normal",
                      max_iter=8, tol=.001, success_threshold=.001,
                      cases=[[30, 3]], algorithms=[name+"_MEMBER" for name in METHODS])
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows, traces = display.evaluate(config, registry)
        with mock.patch.object(plt.Figure, "savefig", side_effect=AssertionError("Unexpected save")):
            display.display(config, rows, traces)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(traces), 4)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()

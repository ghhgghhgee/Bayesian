"""Member NP-0 unit tests and a small registered CLI integration case."""

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

from algorithms.np0 import NP0Diagnostics, np0, solve
from bcs.algorithms import Result
from bcs.data import generate
from bcs.metrics import metrics


ROOT = Path(__file__).resolve().parents[1]


def direct_np0(A, y, *, tol, max_iter, rcond=1e-15):
    """Reference the original iteration using explicit diagonal matrices.

    This deliberately does not call the production iteration or adapter.
    It is a numerical regression reference, not a recovery guarantee.
    """
    a = np.ones(A.shape[1])
    b = np.ones(A.shape[1])
    previous = a * b
    history = []
    converged = False
    for _ in range(max_iter):
        a = np.linalg.pinv(A @ np.diag(b), rcond=rcond) @ y
        b = np.linalg.pinv(A @ np.diag(a), rcond=rcond) @ y
        estimate = a * b
        numerator = float(np.linalg.norm(estimate - previous))
        denominator = float(np.linalg.norm(previous))
        change = numerator / denominator if denominator else (
            0.0 if numerator == 0.0 else float("inf")
        )
        history.append(estimate.copy())
        previous = estimate
        if change <= tol:
            converged = True
            break
    return previous, np.asarray(history), converged, change


class NP0Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261020)
        self.A = rng.normal(size=(5, 11))
        truth = np.zeros(11)
        truth[[2, 8]] = [1.25, -0.75]
        self.y = self.A @ truth

    def assert_result_contract(self, result, n):
        self.assertIsInstance(result, Result)
        self.assertEqual(result.x.shape, (n,))
        history = np.asarray(result.history)
        times = np.asarray(result.times)
        self.assertEqual(history.ndim, 2)
        self.assertEqual(history.shape[1], n)
        self.assertGreater(len(history), 0)
        self.assertEqual(times.shape, (len(history),))
        self.assertTrue(np.all(np.isfinite(history)))
        self.assertTrue(np.all(np.isfinite(times)))
        self.assertTrue(np.all(times >= 0))
        self.assertTrue(np.all(np.diff(times) >= 0))
        np.testing.assert_array_equal(history[-1], result.x)
        self.assertIsInstance(result.converged, bool)
        self.assertEqual(
            result.status, "converged" if result.converged else "max_iter"
        )
        for estimate in result.history:
            self.assertFalse(np.shares_memory(estimate, result.x))

    def test_standalone_default_and_diagnostics(self):
        A = np.eye(5)
        truth = np.array([1.0, 0.0, -2.0, 0.0, 0.5])
        estimate = np0(A, truth)
        with_info, info = np0(A, truth, return_info=True)
        explicit = np0(A, truth, eps=1e-3, max_iter=300)
        self.assertIsInstance(estimate, np.ndarray)
        self.assertEqual(estimate.shape, (5,))
        np.testing.assert_allclose(estimate, truth, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(estimate, with_info)
        np.testing.assert_array_equal(estimate, explicit)
        self.assertIsInstance(info, NP0Diagnostics)
        self.assertTrue(info.converged)
        self.assertGreaterEqual(info.iterations, 1)
        self.assertLessEqual(info.relative_change, 1e-3)

    def test_zero_measurements_converge_without_zero_division(self):
        y = np.zeros(self.A.shape[0])
        estimate, info = np0(self.A, y, return_info=True)
        np.testing.assert_array_equal(estimate, np.zeros(self.A.shape[1]))
        self.assertTrue(info.converged)
        self.assertEqual(info.iterations, 2)
        self.assertEqual(info.relative_change, 0.0)
        result = solve(self.A, y)
        self.assert_result_contract(result, self.A.shape[1])
        self.assertEqual(result.status, "converged")

    def test_row_and_column_measurement_vectors(self):
        expected = np0(self.A, self.y, max_iter=8)
        for y in (self.y[:, None], self.y[None, :]):
            with self.subTest(shape=y.shape):
                np.testing.assert_array_equal(
                    np0(self.A, y, max_iter=8), expected
                )
        np.testing.assert_array_equal(
            solve(self.A, self.y[:, None], max_iter=8).x, expected
        )

    def test_invalid_data_is_rejected(self):
        cases = [
            (np.empty((0, 3)), np.empty(0), ValueError),
            (np.ones(3), np.ones(3), ValueError),
            (np.eye(2), np.ones(3), ValueError),
            (np.eye(2), np.ones((2, 2)), ValueError),
            (np.array([[np.nan]]), np.ones(1), ValueError),
            (np.eye(2), np.array([0.0, np.inf]), ValueError),
            (np.eye(2, dtype=complex), np.ones(2), TypeError),
            (np.eye(2), np.ones(2, dtype=complex), TypeError),
            (np.array([["invalid"]]), np.ones(1), TypeError),
        ]
        for A, y, error in cases:
            with self.subTest(A_shape=A.shape, y_shape=y.shape, error=error):
                with self.assertRaises(error):
                    np0(A, y)

    def test_invalid_stopping_parameters_are_rejected(self):
        for eps in (0, -1, np.nan, np.inf):
            with self.subTest(eps=eps):
                with self.assertRaises(ValueError):
                    np0(self.A, self.y, eps=eps)
        for eps in (None, "invalid"):
            with self.subTest(eps=eps):
                with self.assertRaises(TypeError):
                    np0(self.A, self.y, eps=eps)
        for max_iter, error in (
            (0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)
        ):
            with self.subTest(max_iter=max_iter):
                with self.assertRaises(error):
                    np0(self.A, self.y, max_iter=max_iter)
        with self.assertRaises(ValueError):
            solve(self.A, self.y, tol=0)

    def test_adapter_rejects_invalid_rcond_and_unknown_parameters(self):
        for rcond in (-1, 1, np.nan, np.inf):
            with self.subTest(rcond=rcond):
                with self.assertRaises(ValueError):
                    solve(self.A, self.y, rcond=rcond)
        with self.assertRaises(TypeError):
            solve(self.A, self.y, rcond="invalid")
        with self.assertRaises(TypeError):
            solve(self.A, self.y, misspelled_tol=1e-3)

    def test_adapter_records_real_history_and_matches_standalone(self):
        result = solve(self.A, self.y, max_iter=8, tol=1e-10)
        self.assert_result_contract(result, self.A.shape[1])
        expected, info = np0(
            self.A, self.y, max_iter=8, eps=1e-10, return_info=True
        )
        np.testing.assert_array_equal(result.x, expected)
        self.assertEqual(len(result.history), info.iterations)
        self.assertEqual(result.converged, info.converged)
        self.assertGreater(len(result.history), 1)
        self.assertFalse(np.array_equal(result.history[0], result.history[-1]))

    def test_inputs_and_global_random_state_are_unchanged(self):
        original_A, original_y = self.A.copy(), self.y.copy()
        original_state = np.random.get_state()
        first = solve(self.A, self.y, seed=1, max_iter=8)
        second = solve(self.A, self.y, seed=987654, max_iter=8)
        np0(self.A, self.y, max_iter=8)
        np.testing.assert_array_equal(self.A, original_A)
        np.testing.assert_array_equal(self.y, original_y)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.history, second.history)
        self.assertEqual(first.status, second.status)
        after_state = np.random.get_state()
        self.assertEqual(original_state[0], after_state[0])
        np.testing.assert_array_equal(original_state[1], after_state[1])
        self.assertEqual(original_state[2:], after_state[2:])

    def test_iteration_limit_does_not_claim_convergence(self):
        result = solve(np.eye(3), np.array([2.0, 0.0, -1.0]), max_iter=1)
        self.assert_result_contract(result, 3)
        self.assertEqual(len(result.history), 1)
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "max_iter")
        _, info = np0(
            np.eye(3), np.array([2.0, 0.0, -1.0]), max_iter=1, return_info=True
        )
        self.assertEqual(info.iterations, 1)
        self.assertFalse(info.converged)

    def test_matches_independent_dense_diagonal_iteration(self):
        expected, history, converged, change = direct_np0(
            self.A, self.y, tol=1e-10, max_iter=12
        )
        result = solve(self.A, self.y, tol=1e-10, max_iter=12)
        estimate, info = np0(
            self.A, self.y, eps=1e-10, max_iter=12, return_info=True
        )
        np.testing.assert_allclose(result.history, history, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(estimate, expected, rtol=1e-10, atol=1e-12)
        self.assertEqual(result.converged, converged)
        self.assertEqual(info.iterations, len(history))
        self.assertAlmostEqual(info.relative_change, change, places=11)

    def test_standalone_import_does_not_import_framework_or_scipy(self):
        code = (
            "import sys; from algorithms.np0 import np0; "
            "assert not any(n == 'bcs' or n.startswith('bcs.') or "
            "n == 'scipy' or n.startswith('scipy.') for n in sys.modules)"
        )
        run = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=60
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_plot_uses_member_history_for_both_convergence_axes(self):
        from matplotlib.axes import Axes

        main = importlib.import_module("main")
        config = dict(
            n=40, seed=20261020, trials=1, amplitudes="normal",
            cases=[[30, 3]], algorithms=["NP0_MEMBER"]
        )
        A, y, truth = generate(40, 30, 3, config["seed"], 0)
        result = solve(A, y, max_iter=4, tol=1e-30)
        expected_db = np.asarray([
            max(metrics(estimate, truth)["paper_mse_db"], -160)
            for estimate in result.history
        ])
        calls = []
        original_plot = Axes.plot

        def record_plot(axis, *args, **kwargs):
            calls.append((args, kwargs))
            return original_plot(axis, *args, **kwargs)

        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="np0_plot_", dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            (directory / "manifest.json").write_text(
                json.dumps({"config": config}), encoding="utf-8"
            )
            with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["m", "k", "algorithm", "success_rate"]
                )
                writer.writeheader()
                writer.writerow(dict(m=30, k=3, algorithm="NP0_MEMBER", success_rate=0))
            key = "m30_k3_t0_NP0_MEMBER"
            np.savez_compressed(
                directory / "histories.npz",
                **{key: np.asarray(result.history), key + "_times": np.asarray(result.times)}
            )
            with mock.patch.object(Axes, "plot", new=record_plot):
                main.plot(directory)
            member_calls = [args for args, kwargs in calls
                            if kwargs.get("label") == "NP0_MEMBER" and len(args) >= 2]
            self.assertTrue(any(
                np.array_equal(np.asarray(args[0]), np.asarray(result.times))
                and np.array_equal(np.asarray(args[1]), expected_db)
                for args in member_calls
            ), "Elapsed-time axis did not plot the member's measured history")
            self.assertTrue(any(
                np.array_equal(np.asarray(args[0]), np.arange(1, len(expected_db) + 1))
                and np.array_equal(np.asarray(args[1]), expected_db)
                for args in member_calls
            ), "Iteration axis did not plot the member's numerical history")
            self.assertTrue((directory / "success_rate.png").is_file())
            self.assertTrue((directory / "convergence.png").is_file())

    def test_registered_member_runs_through_actual_cli(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="np0_cli_", dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            config = dict(
                n=12, seed=20261020, trials=2, amplitudes="normal",
                max_iter=12, tol=1e-3, success_threshold=1e-3,
                cases=[[7, 1]], algorithms=["NP0_MEMBER"]
            )
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = directory / "run"
            run = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), "run",
                 "--config", str(config_path), "--registry", str(ROOT / "algorithms.json"),
                 "--out", str(output), "--data-dir", str(directory / "data")],
                cwd=ROOT, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["errors"], 0)
            self.assertEqual(set(manifest["algorithms"]), {"NP0_MEMBER"})
            registration = manifest["algorithms"]["NP0_MEMBER"]
            self.assertEqual(registration["entrypoint"], "algorithms.np0:solve")
            self.assertEqual(registration["params"], {"rcond": 1e-15})
            self.assertEqual(
                Path(registration["source_file"]).resolve(),
                (ROOT / "algorithms" / "np0.py").resolve()
            )
            with (output / "trials.csv").open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["algorithm"] == "NP0_MEMBER" for row in records))
            self.assertTrue(all(row["error"] == "" for row in records))
            self.assertTrue(all(row["status"] in ("converged", "max_iter") for row in records))
            with np.load(output / "estimates.npz") as estimates, \
                    np.load(output / "histories.npz") as histories, \
                    np.load(manifest["dataset_file"]) as data:
                self.assertEqual(len(estimates.files), 2)
                for trial, row in enumerate(records):
                    key = f"m7_k1_t{trial}_NP0_MEMBER"
                    history = histories[key]
                    times = histories[key + "_times"]
                    self.assertEqual(history.shape, (int(row["iterations"]), 12))
                    self.assertEqual(times.shape, (len(history),))
                    self.assertTrue(np.all(np.isfinite(history)))
                    self.assertTrue(np.all(np.isfinite(times)))
                    self.assertTrue(np.all(times >= 0))
                    self.assertTrue(np.all(np.diff(times) >= 0))
                    np.testing.assert_array_equal(history[-1], estimates[key])
                    expected = np0(
                        data["m7_k1_A"][trial], data["m7_k1_y"][trial],
                        eps=config["tol"], max_iter=config["max_iter"]
                    )
                    np.testing.assert_allclose(
                        estimates[key], expected, rtol=1e-10, atol=1e-12
                    )


if __name__ == "__main__":
    unittest.main()

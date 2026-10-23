"""NP1 member integration tests; run from the team repository root.

    python -m unittest discover -s tests -p test_np1_member.py -v

These checks use the team's real Result type, registry, and data generator.
The standalone NP-1 mathematical tests remain in the member's source project.
"""

import importlib
import json
from pathlib import Path
import unittest

import numpy as np

import NP1
from bcs.algorithms import Result
from bcs.data import generate
from bcs.metrics import metrics


ROOT = Path(__file__).resolve().parents[1]


class NP1MemberTests(unittest.TestCase):
    def setUp(self):
        self.A, self.y, self.truth = generate(100, 30, 3, 20261019, 0)

    def test_registry_imports_member_and_keeps_reference_algorithms(self):
        registry = json.loads((ROOT / "algorithms.json").read_text(encoding="utf-8-sig"))
        member = registry["NP1_MEMBER"]
        self.assertEqual(member["entrypoint"], "NP1:solve")
        module_name, function_name = member["entrypoint"].split(":")
        entrypoint = getattr(importlib.import_module(module_name), function_name)
        self.assertIs(entrypoint, NP1.solve)
        result = entrypoint(self.A, self.y, seed=17, max_iter=2, tol=0,
                            **member.get("params", {}))
        self.assertIs(type(result), Result)

        expected_params = {
            "NP0": {"kind": "NP0"},
            "NP1": {"kind": "NP1"},
            "SBL": {"kind": "SBL", "convergence_tol": 1e-6},
            "IRLS": {"kind": "IRLS", "p": 0.5},
            "FOCUSS": {"kind": "FOCUSS", "p": 0.0},
            "BP": {"kind": "BP"},
        }
        for name, params in expected_params.items():
            with self.subTest(algorithm=name):
                self.assertEqual(registry[name]["entrypoint"], "bcs.algorithms:solve")
                self.assertEqual(registry[name]["params"], params)

    def test_result_satisfies_framework_history_and_timing_contract(self):
        result = NP1.solve(self.A, self.y, seed=42)
        self.assertIs(type(result), Result)
        self.assertEqual(result.x.shape, (100,))
        self.assertTrue(np.all(np.isfinite(result.x)))
        self.assertIsInstance(result.history, list)
        self.assertIsInstance(result.times, list)
        history = np.asarray(result.history)
        times = np.asarray(result.times)
        self.assertEqual(history.ndim, 2)
        self.assertEqual(history.shape[1], 100)
        self.assertGreater(len(history), 0)
        self.assertLessEqual(len(history), 300)
        self.assertEqual(times.shape, (len(history),))
        self.assertTrue(np.all(np.isfinite(history)))
        self.assertTrue(np.all(np.isfinite(times)))
        self.assertTrue(np.all(times >= 0))
        self.assertTrue(np.all(np.diff(times) >= 0))
        np.testing.assert_array_equal(history[-1], result.x)
        self.assertIsInstance(result.converged, bool)
        self.assertEqual(result.status, "converged" if result.converged else "max_iter")

    def test_default_adapter_preserves_every_standalone_iterate(self):
        expected = NP1.solve_np1(
            self.A, self.y, max_iter=30, tol=1e-3, rcond=1e-12,
            variance_floor=1e-30, store_history=True,
        )
        actual = NP1.solve(self.A, self.y, max_iter=30, tol=1e-3)
        np.testing.assert_array_equal(actual.x, expected.x)
        np.testing.assert_array_equal(np.asarray(actual.history), expected.x_history)
        self.assertEqual(len(actual.times), expected.n_iter)
        self.assertEqual(actual.converged, expected.converged)

    def test_numerical_parameter_overrides_reach_core(self):
        options = dict(max_iter=3, tol=0, rcond=1e-8, variance_floor=1e-10)
        expected = NP1.solve_np1(self.A, self.y, store_history=True, **options)
        actual = NP1.solve(self.A, self.y, **options)
        np.testing.assert_array_equal(actual.x, expected.x)
        np.testing.assert_array_equal(np.asarray(actual.history), expected.x_history)

    def test_seed_does_not_change_estimates_or_global_random_state(self):
        before = np.random.get_state()
        try:
            first = NP1.solve(self.A, self.y, seed=1, max_iter=8)
            second = NP1.solve(self.A, self.y, seed=987654321, max_iter=8)
            after = np.random.get_state()
            self.assertEqual(after[0], before[0])
            np.testing.assert_array_equal(after[1], before[1])
            self.assertEqual(after[2:], before[2:])
            np.testing.assert_array_equal(first.x, second.x)
            np.testing.assert_array_equal(np.asarray(first.history), np.asarray(second.history))
            self.assertEqual(first.converged, second.converged)
            self.assertEqual(first.status, second.status)
        finally:
            np.random.set_state(before)

    def test_iteration_limit_is_reported_without_false_convergence(self):
        result = NP1.solve(np.eye(4), np.array([2., 0., -3., 0.]), max_iter=1, tol=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "max_iter")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(len(result.times), 1)
        np.testing.assert_array_equal(result.history[-1], result.x)

    def test_inputs_are_not_modified_and_read_only_arrays_work(self):
        matrix_before, measurement_before = self.A.copy(), self.y.copy()
        self.A.setflags(write=False)
        self.y.setflags(write=False)
        NP1.solve(self.A, self.y, max_iter=3)
        np.testing.assert_array_equal(self.A, matrix_before)
        np.testing.assert_array_equal(self.y, measurement_before)

    def test_history_snapshots_do_not_alias_final_estimate_or_each_other(self):
        result = NP1.solve(self.A, self.y, max_iter=3, tol=0)
        self.assertEqual(len(result.history), 3)
        final_before = result.x.copy()
        second_before = result.history[1].copy()
        self.assertFalse(np.shares_memory(result.history[-1], result.x))
        self.assertFalse(np.shares_memory(result.history[0], result.history[1]))
        result.history[0][:] = 12345
        np.testing.assert_array_equal(result.x, final_before)
        np.testing.assert_array_equal(result.history[1], second_before)

    def test_known_noiseless_sparse_case_recovers_using_framework_metric(self):
        result = NP1.solve(self.A, self.y, seed=20261019)
        self.assertTrue(metrics(result.x, self.truth, threshold=1e-3)["success"])
        self.assertLess(np.linalg.norm(result.x - self.truth) / np.linalg.norm(self.truth), 1e-3)
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "converged")

    def test_acceptance_config_compares_member_with_original_reference(self):
        config = json.loads((ROOT / "configs" / "np1_member.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(config["n"], 100)
        self.assertEqual(config["seed"], 20261019)
        self.assertEqual(config["trials"], 5)
        self.assertEqual(config["cases"], [[10, 3], [30, 3], [30, 15]])
        self.assertEqual(set(config["algorithms"]), {"NP1", "NP1_MEMBER"})


if __name__ == "__main__":
    unittest.main()


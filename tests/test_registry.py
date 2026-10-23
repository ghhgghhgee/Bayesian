"""Regression checks for the shared registry and every registered CLI entry."""

import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def strict_registry():
    """Reject duplicate keys and non-JSON numeric constants as well as syntax errors."""
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Duplicate registry key: {name}")
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(
        (ROOT / "algorithms.json").read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique_object,
        parse_constant=invalid_constant,
    )


class RegistryTests(unittest.TestCase):
    def test_registry_is_strict_json_with_unique_callable_entries(self):
        registry = strict_registry()
        self.assertIsInstance(registry, dict)
        expected = {"NP0", "NP0_MEMBER", "NP1", "NP1_MEMBER", "SBL", "IRLS", "FOCUSS", "BP"}
        self.assertTrue(expected.issubset(registry), "A current reference or member entry was lost")
        for name, item in registry.items():
            with self.subTest(algorithm=name):
                self.assertIsInstance(name, str)
                self.assertTrue(name)
                self.assertIsInstance(item, dict)
                self.assertIsInstance(item["owner"], str)
                self.assertTrue(item["owner"].strip())
                self.assertEqual(item["entrypoint"].count(":"), 1)
                module_name, function_name = item["entrypoint"].split(":")
                self.assertTrue(module_name)
                self.assertTrue(function_name)
                function = getattr(importlib.import_module(module_name), function_name)
                self.assertTrue(callable(function))
                params = item.get("params", {})
                self.assertIsInstance(params, dict)
                self.assertFalse({"seed", "max_iter", "tol"}.intersection(params))

    def test_checked_in_configs_only_reference_registered_algorithms(self):
        registry = strict_registry()
        paths = sorted((ROOT / "configs").glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text(encoding="utf-8-sig"))
                names = config["algorithms"]
                self.assertIsInstance(names, list)
                self.assertTrue(names)
                self.assertEqual(len(names), len(set(names)), "Duplicate algorithms in config")
                self.assertFalse(set(names).difference(registry), "Unregistered algorithm in config")

    def test_every_registered_entry_runs_through_actual_cli(self):
        registry = strict_registry()
        config = dict(
            n=12, seed=20261021, trials=1, amplitudes="normal",
            max_iter=4, tol=1e-3, success_threshold=1e-3,
            cases=[[7, 1]], algorithms=list(registry),
        )
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="registry_cli_", dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = directory / "run"
            run = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), "run",
                 "--config", str(config_path), "--registry", str(ROOT / "algorithms.json"),
                 "--out", str(output), "--data-dir", str(directory / "data")],
                cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["errors"], 0)
            self.assertEqual(set(manifest["algorithms"]), set(registry))
            with (output / "trials.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(registry))
            self.assertEqual({row["algorithm"] for row in rows}, set(registry))
            self.assertEqual(len({row["data_hash"] for row in rows}), 1)
            by_name = {row["algorithm"]: row for row in rows}
            expected_keys = {f"m7_k1_t0_{name}" for name in registry}
            with np.load(output / "estimates.npz", allow_pickle=False) as estimates, \
                    np.load(output / "histories.npz", allow_pickle=False) as histories:
                self.assertEqual(set(estimates.files), expected_keys)
                self.assertEqual(
                    set(histories.files), expected_keys | {key + "_times" for key in expected_keys}
                )
                for name, item in registry.items():
                    with self.subTest(algorithm=name):
                        provenance = manifest["algorithms"][name]
                        for field in ("entrypoint", "owner", "params"):
                            self.assertEqual(provenance[field], item[field])
                        module_name, function_name = item["entrypoint"].split(":")
                        function = getattr(importlib.import_module(module_name), function_name)
                        source = Path(inspect.getsourcefile(function)).resolve()
                        self.assertEqual(Path(provenance["source_file"]).resolve(), source)
                        self.assertEqual(
                            provenance["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
                        )
                        row = by_name[name]
                        self.assertEqual(row["error"], "")
                        self.assertIn(row["status"], ("converged", "max_iter", "lp_optimal"))
                        iterations = int(row["iterations"])
                        self.assertGreaterEqual(iterations, 1)
                        self.assertLessEqual(iterations, config["max_iter"])
                        key = f"m7_k1_t0_{name}"
                        estimate = estimates[key]
                        history = histories[key]
                        times = histories[key + "_times"]
                        self.assertEqual(estimate.shape, (config["n"],))
                        self.assertEqual(history.shape, (iterations, config["n"]))
                        self.assertEqual(times.shape, (iterations,))
                        self.assertTrue(np.all(np.isfinite(estimate)))
                        self.assertTrue(np.all(np.isfinite(history)))
                        self.assertTrue(np.all(np.isfinite(times)))
                        self.assertTrue(np.all(times >= 0))
                        self.assertTrue(np.all(np.diff(times) >= 0))
                        np.testing.assert_array_equal(history[-1], estimate)


if __name__ == "__main__":
    unittest.main()


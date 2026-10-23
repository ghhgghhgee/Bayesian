"""Compare a saved pre-fix NP1.py with the current member and reference.

Run from the repository root with --before PATH --out PATH. The old source
must come from a trusted checkout; loading it executes Python code.
Truth is used only here for scoring, never passed to a solver.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import NP1
from bcs import algorithms
from bcs.data import generate, fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("np1_before_audit", args.before)
    before = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = before
    spec.loader.exec_module(before)
    report = {
        "base_commit": "09b125cd3f9d26aeaf88e8b97211b068dacc2dfe",
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {"before": digest(args.before),
                          "after": digest(NP1.__file__),
                          "reference": digest(algorithms.__file__)},
        "parameters": {"n": 100, "seed": 20261019, "trials": 300,
                       "max_iter": 300, "tol": 1e-3, "rcond": 1e-12,
                       "variance_floor": 1e-30, "success_threshold": 1e-3},
        "cases": [], "reported_trials": [],
    }
    for m, k in ((10, 3), (30, 3), (30, 15)):
        case = {"m": m, "k": k,
                "success": {key: 0 for key in ("before", "after", "reference")},
                "before_max_abs_difference": 0.0, "after_max_abs_difference": 0.0,
                "before_label_mismatches": [], "after_label_mismatches": [],
                "after_history_mismatches": [], "after_status_mismatches": []}
        for trial in range(300):
            A, y, truth = generate(100, m, k, 20261019, trial)
            results = {"before": before.solve(A.copy(), y.copy()),
                       "after": NP1.solve(A.copy(), y.copy()),
                       "reference": algorithms.solve(A.copy(), y.copy(), kind="NP1")}
            errors = {key: float(np.linalg.norm(result.x-truth)/np.linalg.norm(truth))
                      for key, result in results.items()}
            for key, error in errors.items():
                case["success"][key] += int(error < 1e-3)
            for key in ("before", "after"):
                delta = float(np.max(np.abs(results[key].x-results["reference"].x)))
                case[key+"_max_abs_difference"] = max(case[key+"_max_abs_difference"], delta)
                if (errors[key] < 1e-3) != (errors["reference"] < 1e-3):
                    case[key+"_label_mismatches"].append(trial)
            if not np.array_equal(results["after"].history, results["reference"].history):
                case["after_history_mismatches"].append(trial)
            if results["after"].status != results["reference"].status:
                case["after_status_mismatches"].append(trial)
            if m == 10 and trial in (52, 135, 161):
                old_m, old_v, _ = before._posterior(A, y, np.ones(100), np.ones(100), 1e-12)
                ref_m, ref_v = algorithms.posterior(A, np.ones(100), np.ones(100), y)
                old_h = np.array(results["before"].history)
                ref_h = np.array(results["reference"].history)
                overlap = min(len(old_h), len(ref_h))
                gaps = np.max(np.abs(old_h[:overlap]-ref_h[:overlap]), axis=1)
                report["reported_trials"].append({
                    "trial": trial, "data_sha256": fingerprint(A, y, truth),
                    "relative_error": errors,
                    "iterations": {key: len(result.history) for key, result in results.items()},
                    "first_a_mean_max_abs_difference": float(np.max(abs(old_m-ref_m))),
                    "first_a_variance_max_abs_difference": float(np.max(abs(old_v-ref_v))),
                    "first_iteration_gap_over_1e_6": next((i+1 for i, v in enumerate(gaps) if v > 1e-6), None),
                })
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    report["passed"] = all(
        c["after_max_abs_difference"] == 0 and not c["after_label_mismatches"]
        and not c["after_history_mismatches"] and not c["after_status_mismatches"]
        for c in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

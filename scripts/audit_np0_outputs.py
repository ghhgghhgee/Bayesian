"""Independent audit of saved NP-0 framework runs; not a solver test."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcs.data import generate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def number_equal(actual, expected):
    return actual == expected or math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    config = manifest["config"]
    rows = load_rows(directory / "trials.csv")
    summaries = load_rows(directory / "summary.csv")
    errors, checked = Counter(), Counter()
    examples = []

    def check(condition, name, context=""):
        checked[name] += 1
        if not condition:
            errors[name] += 1
            if len(examples) < 30:
                examples.append(dict(check=name, context=str(context)))

    check(manifest.get("complete") is True and manifest.get("errors") == 0,
          "manifest_complete_without_errors")
    for filename, expected in manifest.get("outputs", {}).items():
        path = directory / filename
        check(path.is_file() and sha256(path) == expected, "manifest_output_integrity", filename)

    dataset_path = Path(manifest["dataset_file"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    check(dataset_path.is_file() and sha256(dataset_path) == manifest["dataset_sha256"],
          "dataset_file_integrity", dataset_path)
    expected_keys = {
        f"m{m}_k{k}_t{trial}_{name}"
        for m, k in config["cases"] for trial in range(config["trials"])
        for name in config["algorithms"]
    }
    check(len(rows) == len(expected_keys), "trial_row_count")
    by_key = {f"m{r['m']}_k{r['k']}_t{r['trial']}_{r['algorithm']}": r for r in rows}
    check(len(by_key) == len(rows), "unique_trial_records")
    check(set(by_key) == expected_keys, "complete_trial_keys")
    calculated = defaultdict(list)
    differences = defaultdict(list)
    max_measurement_residual = defaultdict(float)
    max_metric_difference = defaultdict(float)

    with np.load(directory / "estimates.npz", allow_pickle=False) as estimates, \
         np.load(directory / "histories.npz", allow_pickle=False) as histories, \
         np.load(dataset_path, allow_pickle=False) as dataset:
        check(set(estimates.files) == expected_keys, "estimate_keys")
        check(set(histories.files) == expected_keys | {key + "_times" for key in expected_keys},
              "history_keys")
        for m, k in config["cases"]:
            prefix = f"m{m}_k{k}"
            for trial in range(config["trials"]):
                A, y, truth = generate(config["n"], m, k, config["seed"], trial,
                                        config["amplitudes"])
                expected_hash = data_hash(A, y, truth)
                for label, expected in zip(("A", "y", "x"), (A, y, truth)):
                    check(np.array_equal(dataset[f"{prefix}_{label}"][trial], expected),
                          "regenerated_dataset_equal", f"{prefix} trial={trial} {label}")
                shared_hashes, pair = [], {}
                for name in config["algorithms"]:
                    key = f"{prefix}_t{trial}_{name}"
                    row = by_key[key]
                    check(row["status"] != "error" and not row["error"], "no_algorithm_error", key)
                    check(row["data_hash"] == expected_hash, "regenerated_data_hash_matches", key)
                    shared_hashes.append(row["data_hash"])
                    seed = int.from_bytes(hashlib.sha256(
                        f"{config['seed']}:{key}".encode()).digest()[:4], "little")
                    check(int(row["algorithm_seed"]) == seed, "algorithm_seed_matches", key)
                    if key not in estimates or key not in histories or key + "_times" not in histories:
                        check(False, "saved_arrays_present", key)
                        continue
                    estimate, history, times = estimates[key], histories[key], histories[key + "_times"]
                    check(estimate.shape == truth.shape and np.isfinite(estimate).all(),
                          "estimate_shape_and_finite", key)
                    shape_ok = history.ndim == 2 and history.shape[1] == config["n"] and len(history) > 0
                    check(shape_ok and np.isfinite(history).all(), "history_shape_and_finite", key)
                    check(times.shape == (len(history),) and np.isfinite(times).all()
                          and np.all(times >= 0) and np.all(np.diff(times) >= 0), "time_contract", key)
                    check(int(row["iterations"]) == len(history) and 1 <= len(history) <= config["max_iter"],
                          "iteration_count", key)
                    check(shape_ok and np.array_equal(history[-1], estimate), "final_history_equal", key)
                    seconds = float(row["seconds"])
                    check(math.isfinite(seconds) and seconds >= 0 and times[-1] <= seconds,
                          "elapsed_time_bounds", key)
                    previous = history[-2] if len(history) > 1 else np.ones(config["n"])
                    previous_norm = float(np.linalg.norm(previous))
                    change = float(np.linalg.norm(history[-1] - previous))
                    relative_change = change / previous_norm if previous_norm else (0 if not change else math.inf)
                    converged = row["converged"].lower() == "true"
                    check((converged and row["status"] == "converged" and relative_change <= config["tol"])
                          or (not converged and row["status"] == "max_iter"
                              and len(history) == config["max_iter"] and relative_change > config["tol"]),
                          "stopping_status_consistent", key)
                    error = estimate - truth
                    squared = float(np.dot(error, error))
                    nmse = squared / float(np.dot(truth, truth))
                    relative = math.sqrt(nmse)
                    independent = dict(mse=squared / truth.size, nmse=nmse,
                                       relative_error=relative,
                                       paper_mse_db=10 * math.log10(nmse) if nmse else -math.inf)
                    for metric, expected in independent.items():
                        actual = float(row[metric])
                        check(number_equal(actual, expected), "trial_metric_" + metric, key)
                        if math.isfinite(actual) and math.isfinite(expected):
                            max_metric_difference[metric] = max(max_metric_difference[metric], abs(actual - expected))
                    success = relative < config["success_threshold"]
                    check((row["success"].lower() == "true") == success, "trial_success_threshold", key)
                    group = (m, k, name)
                    calculated[group].append(dict(**independent, success=success,
                                                   iterations=len(history), seconds=seconds, converged=converged))
                    residual = float(np.linalg.norm(A @ estimate - y) / max(float(np.linalg.norm(y)), 1e-300))
                    max_measurement_residual[str(group)] = max(max_measurement_residual[str(group)], residual)
                    pair[name] = estimate
                check(len(set(shared_hashes)) == 1, "algorithms_share_data_hash", f"{prefix} trial={trial}")
                if "NP0" in pair and "NP0_MEMBER" in pair:
                    delta = pair["NP0_MEMBER"] - pair["NP0"]
                    differences[(m, k)].append(dict(
                        l2=float(np.linalg.norm(delta)),
                        max_abs=float(np.max(np.abs(delta))),
                        relative_to_truth=float(np.linalg.norm(delta) / np.linalg.norm(truth))))

    expected_groups = {(m, k, name) for m, k in config["cases"] for name in config["algorithms"]}
    summary_keys = [(int(row["m"]), int(row["k"]), row["algorithm"]) for row in summaries]
    check(set(summary_keys) == expected_groups and len(summary_keys) == len(expected_groups), "summary_groups")
    recomputed = []
    for row in summaries:
        group = (int(row["m"]), int(row["k"]), row["algorithm"])
        values = calculated[group]
        if not values:
            check(False, "summary_has_values", group)
            continue
        successes = sum(value["success"] for value in values)
        expected = dict(trials=len(values), errors=0, successes=successes,
                        success_rate=successes / len(values),
                        mean_mse=float(np.mean([v["mse"] for v in values])),
                        mean_nmse=float(np.mean([v["nmse"] for v in values])),
                        mean_iterations=float(np.mean([v["iterations"] for v in values])),
                        median_seconds=float(np.median([v["seconds"] for v in values])))
        for field, value in expected.items():
            check(number_equal(float(row[field]), value), "summary_" + field, group)
        recomputed.append(dict(m=group[0], k=group[1], algorithm=group[2], **expected,
                               converged=sum(v["converged"] for v in values),
                               max_relative_measurement_residual=max_measurement_residual[str(group)]))
    pair_stats = []
    for (m, k), values in differences.items():
        pair_stats.append(dict(m=m, k=k, trials=len(values), **{
            f"{stat}_{metric}": float(reducer([v[metric] for v in values]))
            for metric in ("l2", "max_abs", "relative_to_truth")
            for stat, reducer in (("max", np.max), ("median", np.median))}))
    return dict(directory=str(directory), passed=not errors, checks=dict(checked),
                failures=dict(errors), failure_examples=examples, config=config,
                independent_summary=recomputed, algorithm_pair_differences=pair_stats,
                max_absolute_trial_metric_difference=dict(max_metric_difference),
                manifest_sha256=sha256(directory / "manifest.json"),
                note="Metrics recalculated directly from saved estimates and regenerated truth; bcs.metrics is not used. Different rcond values do not require identical NP0 and NP0_MEMBER estimates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--out", default="results/np0_20261020_numerical_audit.json")
    args = parser.parse_args()
    runs = [audit(path) for path in args.directories]
    result = dict(passed=all(run["passed"] for run in runs), python=sys.version,
                  numpy=np.__version__, platform=platform.platform(),
                  script_sha256=sha256(__file__), runs=runs)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], output=args.out,
                         runs=[dict(directory=r["directory"], passed=r["passed"],
                                    failures=r["failures"], summary=r["independent_summary"],
                                    pair_differences=r["algorithm_pair_differences"]) for r in runs]),
                     ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

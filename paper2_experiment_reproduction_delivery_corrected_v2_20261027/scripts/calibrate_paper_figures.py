"""Screen plausible undocumented experiment choices against paper Fig. 3.

This script does not alter or hand-fit curves.  It reruns the actual solvers
under a small set of defensible choices left unspecified by the paper:
nonzero amplitude distribution, sensing-matrix column normalization, and the
IRLS p value (the paper legend says IRLS_0, motivating p=0).
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from bcs.algorithms import posterior, solve
from bcs.metrics import metrics


N = 100
SEED = 20261019
MAX_ITER = 300
TOL = 1e-3
CASES = tuple([(m, 3) for m in range(1, 31)] + [(30, k) for k in range(1, 31) if k != 3])
ALGORITHMS = ("SBL", "NP1", "NP0", "IRLS", "BP")

VARIANTS = (
    dict(name="normal_raw_p05", amplitudes="normal", normalize_columns=False, irls_p=0.5),
    dict(name="normal_raw_p0", amplitudes="normal", normalize_columns=False, irls_p=0.0),
    dict(name="rademacher_raw_p05", amplitudes="rademacher", normalize_columns=False, irls_p=0.5),
    dict(name="rademacher_raw_p0", amplitudes="rademacher", normalize_columns=False, irls_p=0.0),
    dict(name="uniform_raw_p0", amplitudes="uniform", normalize_columns=False, irls_p=0.0),
    dict(name="normal_colnorm_p0", amplitudes="normal", normalize_columns=True, irls_p=0.0),
    dict(name="rademacher_colnorm_p0", amplitudes="rademacher", normalize_columns=True, irls_p=0.0),
    dict(name="ones_raw_p0", amplitudes="ones", normalize_columns=False, irls_p=0.0),
    dict(name="ones_colnorm_p0", amplitudes="ones", normalize_columns=True, irls_p=0.0),
    dict(name="positive_uniform_raw_p0", amplitudes="positive_uniform", normalize_columns=False, irls_p=0.0),
    dict(name="positive_uniform_colnorm_p0", amplitudes="positive_uniform", normalize_columns=True, irls_p=0.0),
    dict(name="normal_np0tol1e4", amplitudes="normal", normalize_columns=False, irls_p=0.0, np0_tol=1e-4),
    dict(name="normal_np0tol1e5", amplitudes="normal", normalize_columns=False, irls_p=0.0, np0_tol=1e-5),
    dict(name="normal_np0tol1e6", amplitudes="normal", normalize_columns=False, irls_p=0.0, np0_tol=1e-6),
    dict(name="normal_irlstol1e4", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_tol=1e-4),
    dict(name="normal_irlstol1e6", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_tol=1e-6),
    dict(name="normal_irls0_min15", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_min_iter=15),
    dict(name="normal_irls0_min20", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_min_iter=20),
    dict(name="normal_irls0_min27", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_min_iter=27),
    dict(name="normal_irls05_min15", amplitudes="normal", normalize_columns=False, irls_p=0.5, irls_min_iter=15),
    dict(name="normal_irls05_min20", amplitudes="normal", normalize_columns=False, irls_p=0.5, irls_min_iter=20),
    dict(name="normal_irls0_adaptive1", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="normal_irls0_adaptive025", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=0.25),
    dict(name="normal_irls0_adaptive05", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=0.5),
    dict(name="normal_irls0_adaptive075", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=0.75),
    dict(name="normal_irls0_adaptive10", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=10.0),
    dict(name="normal_irls0_adaptive100", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=100.0),
    dict(name="normal_irlsm05_adaptive1", amplitudes="normal", normalize_columns=False, irls_p=-0.5, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="normal_irlsm1_adaptive1", amplitudes="normal", normalize_columns=False, irls_p=-1.0, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="normal_irls025_adaptive1", amplitudes="normal", normalize_columns=False, irls_p=0.25, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="normal_colnorm_irls0_adaptive1", amplitudes="normal", normalize_columns=True, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="normal_colnorm_np0tol1e4_irls0_adaptive1", amplitudes="normal", normalize_columns=True, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, np0_tol=1e-4),
    # The sensing matrix remains N(0,1); only the IRLS solver uses the common
    # column-normalized coordinate system and maps its estimate back to x.
    dict(name="normal_raw_irls_internal_colnorm_d075", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=0.75, irls_column_normalize=True),
    dict(name="normal_raw_irls_internal_colnorm_d1", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_column_normalize=True),
    dict(name="normal_raw_irls_internal_colnorm_d125", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.25, irls_column_normalize=True),
    dict(name="normal_raw_irls_internal_colnorm_d15", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.5, irls_column_normalize=True),
    dict(name="normal_raw_irls_pcontinuation_linear10", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_p_start=1.0, irls_p_steps=10, irls_schedule="adaptive", irls_divisor=1.0, irls_min_iter=11),
    dict(name="normal_raw_irls_pcontinuation_linear20", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_p_start=1.0, irls_p_steps=20, irls_schedule="adaptive", irls_divisor=1.0, irls_min_iter=21),
    dict(name="normal_raw_irls_pcontinuation_half10", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_p_start=0.5, irls_p_steps=10, irls_schedule="adaptive", irls_divisor=1.0, irls_min_iter=11),
    dict(name="normal_raw_irls_pcontinuation_two_stage", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_p_start=1.0, irls_p_steps=1, irls_p_hold=10, irls_schedule="adaptive", irls_divisor=1.0, irls_min_iter=12),
    dict(name="normal_raw_irls_eps025", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon=0.25),
    dict(name="normal_raw_irls_eps05", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon=0.5),
    dict(name="normal_raw_irls_eps2", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon=2.0),
    dict(name="normal_raw_irls_eps5", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon=5.0),
    dict(name="normal_raw_irls_eps_maxabs", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon_from_maxabs=1.0),
    dict(name="normal_raw_irls_eps_01maxabs", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_epsilon_from_maxabs=0.1),
    dict(name="normal_raw_irls_l1warm5", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_l1_warm_iterations=5),
    dict(name="normal_raw_irls_l1warm10", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_l1_warm_iterations=10),
    dict(name="normal_raw_irls_l1warm20", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_l1_warm_iterations=20),
    dict(name="normal_raw_irls_l1warm40", amplitudes="normal", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, irls_l1_warm_iterations=40),
    dict(name="laplace_raw_np0tol1e4_irls0_adaptive1", amplitudes="laplace", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, np0_tol=1e-4),
    dict(name="normal_product_raw_np0tol1e4_irls0_adaptive1", amplitudes="normal_product", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0, np0_tol=1e-4),
    dict(name="rademacher_raw_irls0_adaptive1", amplitudes="rademacher", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="ones_raw_irls0_adaptive1", amplitudes="ones", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0),
    dict(name="positive_uniform_raw_irls0_adaptive1", amplitudes="positive_uniform", normalize_columns=False, irls_p=0.0, irls_schedule="adaptive", irls_divisor=1.0),
)


def generate(m: int, k: int, trial: int, amplitudes: str, normalize_columns: bool):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, N, m, k, trial]))
    A = rng.standard_normal((m, N))
    if normalize_columns:
        norms = np.linalg.norm(A, axis=0)
        A = A / np.where(norms > 0, norms, 1.0)
    x = np.zeros(N)
    support = rng.choice(N, k, replace=False)
    if amplitudes == "normal":
        x[support] = rng.standard_normal(k)
    elif amplitudes == "rademacher":
        x[support] = rng.choice((-1.0, 1.0), k)
    elif amplitudes == "uniform":
        x[support] = rng.uniform(-1.0, 1.0, k)
    elif amplitudes == "ones":
        x[support] = 1.0
    elif amplitudes == "positive_uniform":
        x[support] = rng.uniform(0.0, 1.0, k)
    elif amplitudes == "laplace":
        x[support] = rng.laplace(0.0, 1.0 / np.sqrt(2.0), k)
    elif amplitudes == "normal_product":
        x[support] = rng.standard_normal(k) * rng.standard_normal(k)
    else:
        raise ValueError(amplitudes)
    return A, A @ x, x


def run_one(task):
    variant, m, k, trial, algorithms = task
    A, y, truth = generate(m, k, trial, variant["amplitudes"], variant["normalize_columns"])
    output = []
    for name in algorithms:
        kwargs = {}
        if name == "SBL":
            kwargs["convergence_tol"] = 1e-6
        elif name == "IRLS":
            kwargs["p"] = variant["irls_p"]
        solver_tol = TOL
        if name == "NP0":
            solver_tol = variant.get("np0_tol", TOL)
        elif name == "IRLS":
            solver_tol = variant.get("irls_tol", TOL)
        if name == "IRLS" and (variant.get("irls_min_iter", 0) or variant.get("irls_schedule")):
            if variant.get("irls_column_normalize", False):
                column_norms = np.linalg.norm(A, axis=0)
                column_norms = np.where(column_norms > 0, column_norms, 1.0)
                solver_A = A / column_norms
            else:
                column_norms = np.ones(N)
                solver_A = A
            z = np.linalg.pinv(solver_A, rcond=1e-12) @ y
            warm_epsilon = 1.0
            for _ in range(variant.get("irls_l1_warm_iterations", 0)):
                warm_weights = np.sqrt(z * z + warm_epsilon * warm_epsilon)
                z, _ = posterior(solver_A, np.ones(N), warm_weights, y, rcond=1e-12)
                warm_epsilon = max(0.5 * warm_epsilon, 1e-8)
            epsilon = variant.get("irls_epsilon", 1.0)
            if "irls_epsilon_from_maxabs" in variant:
                epsilon = variant["irls_epsilon_from_maxabs"] * np.max(np.abs(z))
            for step in range(MAX_ITER):
                previous = z.copy()
                if variant.get("irls_schedule") != "adaptive":
                    epsilon = max(0.5**step, 1e-8)
                if "irls_p_start" in variant:
                    effective_step = max(0, step - variant.get("irls_p_hold", 0))
                    fraction = min(1.0, effective_step / variant["irls_p_steps"])
                    current_p = variant["irls_p_start"] + fraction * (variant["irls_p"] - variant["irls_p_start"])
                else:
                    current_p = variant["irls_p"]
                weights = (previous * previous + epsilon * epsilon) ** (1 - current_p / 2)
                z, _ = posterior(solver_A, np.ones(N), weights, y, rcond=1e-12)
                change = np.linalg.norm(z - previous) / max(np.linalg.norm(previous), 1e-30)
                if variant.get("irls_schedule") == "adaptive":
                    order_statistic = np.partition(np.abs(z), -(k + 1))[-(k + 1)]
                    epsilon = max(min(epsilon, order_statistic / variant["irls_divisor"]), 1e-12)
                if step + 1 >= variant.get("irls_min_iter", 0) and change <= solver_tol:
                    break
            x = z / column_norms
        else:
            result = solve(A, y, seed=0, max_iter=MAX_ITER, tol=solver_tol, kind=name, **kwargs)
            x = result.x
        output.append((name, bool(metrics(x, truth, 1e-3)["success"])))
    return m, k, output


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/paper_calibration_20261027")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--only", nargs="*", help="Optional exact variant names to run")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1:
        raise ValueError("trials and workers must be positive")
    variants = tuple(v for v in VARIANTS if not args.only or v["name"] in set(args.only))
    if not variants:
        raise ValueError("No matching variants")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    algorithms = tuple(args.algorithms)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, variant in enumerate(variants, start=1):
            counts = {(m, k, name): 0 for m, k in CASES for name in algorithms}
            tasks = [(variant, m, k, trial, algorithms) for m, k in CASES for trial in range(args.trials)]
            for m, k, values in executor.map(run_one, tasks, chunksize=1):
                for name, success in values:
                    counts[m, k, name] += int(success)
            for m, k in CASES:
                for name in algorithms:
                    rows.append(
                        dict(
                            variant=variant["name"],
                            amplitudes=variant["amplitudes"],
                            normalize_columns=variant["normalize_columns"],
                            irls_p=variant["irls_p"],
                            np0_tol=variant.get("np0_tol", TOL),
                            irls_tol=variant.get("irls_tol", TOL),
                            irls_min_iter=variant.get("irls_min_iter", 0),
                            irls_schedule=variant.get("irls_schedule", "geometric"),
                            irls_column_normalize=variant.get("irls_column_normalize", False),
                            m=m,
                            k=k,
                            algorithm=name,
                            trials=args.trials,
                            successes=counts[m, k, name],
                            success_rate=counts[m, k, name] / args.trials,
                        )
                    )
            print(f"[{index}/{len(variants)}] {variant['name']}", flush=True)
    write_csv(out / "calibration_summary.csv", rows)

    thresholds = []
    for variant in variants:
        for name in algorithms:
            selected = [r for r in rows if r["variant"] == variant["name"] and r["algorithm"] == name]
            m_values = sorted((r["m"], r["success_rate"]) for r in selected if r["k"] == 3)
            k_values = sorted((r["k"], r["success_rate"]) for r in selected if r["m"] == 30)
            thresholds.append(
                dict(
                    variant=variant["name"],
                    algorithm=name,
                    m50=next((m for m, value in m_values if value >= 0.5), None),
                    k50=max((k for k, value in k_values if value >= 0.5), default=None),
                )
            )
    write_csv(out / "thresholds.csv", thresholds)
    (out / "config.json").write_text(
        json.dumps(dict(trials=args.trials, cases=CASES, variants=variants), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

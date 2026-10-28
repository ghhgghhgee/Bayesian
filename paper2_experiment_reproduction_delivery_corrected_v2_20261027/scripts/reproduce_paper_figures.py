"""Reproduce Figs. 1, 3 and 4 from Zhou et al. (2015).

The Fig. 3 scan uses 300 Monte Carlo trials at every paper scan point.  Raw
trial records are retained, while full iteration histories are retained only
for the Fig. 4 case (M=30, K=3) to keep the deliverable compact.
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import k0

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcs.algorithms import Result, posterior, solve
from bcs.data import fingerprint, generate
from bcs.metrics import metrics


N = 100
TRIALS = 300
SEED = 20261019
MAX_ITER = 300
TOL = 1e-3
SUCCESS_THRESHOLD = 1e-3
AMPLITUDES = "normal"
FIG4_BENCHMARK_REPEATS = 15

ALL_ALGORITHMS = ("SBL", "NP1", "NP0", "IRLS", "FOCUSS", "BP")
PAPER_ALGORITHMS = ("SBL", "NP1", "NP0", "IRLS", "BP")
FIG4_ALGORITHMS = ("SBL", "NP1", "IRLS")
CASES = tuple([(m, 3) for m in range(1, 31)] + [(30, k) for k in range(1, 31) if k != 3])

FIG3_STYLE = {
    "SBL": dict(color="#d62728", marker="d", fillstyle="none"),
    "NP1": dict(color="#263f2d", marker="o", fillstyle="none"),
    "NP0": dict(color="#c98b24", marker="s", fillstyle="none"),
    "IRLS": dict(color="#55bfc1", marker="^", fillstyle="none"),
    "FOCUSS": dict(color="#6f6f6f", marker="x", fillstyle="full"),
    "BP": dict(color="#102ea8", marker="*", fillstyle="full"),
}


FIG4_STYLE = {
    "SBL": dict(color="#c73a32", marker="s", fillstyle="none"),
    "NP1": dict(color="#142750", marker="x", fillstyle="full"),
    "IRLS": dict(color="#55bfc1", marker="d", fillstyle="none"),
}


def _paper_irls0(A: np.ndarray, y: np.ndarray, sparsity: int):
    """IRLS_0 with the standard adaptive order-statistic continuation."""
    start = time.perf_counter()
    x = np.linalg.pinv(A, rcond=1e-12) @ y
    epsilon = 1.0
    history, times = [], []
    converged = False
    for _step in range(MAX_ITER):
        previous = x.copy()
        weights = previous * previous + epsilon * epsilon  # p=0
        # This is algebraically identical to the posterior/pseudoinverse form,
        # but avoids an SVD and an unused covariance calculation.  That
        # distinction matters in Fig. 4, whose horizontal axis is runtime.
        gram = (A * weights) @ A.T
        try:
            dual = np.linalg.solve(gram, y)
            x = weights * (A.T @ dual)
        except np.linalg.LinAlgError:
            x, _ = posterior(A, np.ones(N), weights, y, rcond=1e-12)
        history.append(x.copy())
        times.append(time.perf_counter() - start)
        order_statistic = np.partition(np.abs(x), -(sparsity + 1))[-(sparsity + 1)]
        epsilon = max(min(epsilon, order_statistic), 1e-12)
        change = np.linalg.norm(x - previous) / max(np.linalg.norm(previous), 1e-30)
        if change <= TOL:
            converged = True
            break
    return Result(x, history, times, converged, "converged" if converged else "max_iter")


def _algorithm(name: str, A: np.ndarray, y: np.ndarray, seed: int, sparsity: int, profile: str):
    params = {}
    if name == "SBL":
        params["convergence_tol"] = 1e-8 if profile == "paper-aligned" else 1e-6
    elif name == "NP0" and profile == "paper-aligned":
        return solve(A, y, seed=seed, max_iter=MAX_ITER, tol=1e-4, kind=name)
    elif name == "IRLS":
        if profile == "paper-aligned":
            return _paper_irls0(A, y, sparsity)
        params["p"] = 0.5
    elif name == "FOCUSS":
        params["p"] = 0.0
    return solve(A, y, seed=seed, max_iter=MAX_ITER, tol=TOL, kind=name, **params)


def _one_trial(task: tuple[int, int, int, str]):
    m, sparsity, trial, profile = task
    A, y, truth = generate(N, m, sparsity, SEED, trial, AMPLITUDES)
    data_hash = fingerprint(A, y, truth)
    rows = []
    curves = {}
    for name in ALL_ALGORITHMS:
        algorithm_seed = int.from_bytes(
            hashlib.sha256(f"{SEED}:m{m}_k{sparsity}_t{trial}_{name}".encode()).digest()[:4],
            "little",
        )
        started = time.perf_counter()
        result = _algorithm(name, A.copy(), y.copy(), algorithm_seed, sparsity, profile)
        elapsed = time.perf_counter() - started
        score = metrics(result.x, truth, SUCCESS_THRESHOLD)
        residual = np.linalg.norm(A @ result.x - y) / max(np.linalg.norm(y), 1e-30)
        rows.append(
            dict(
                m=m,
                k=sparsity,
                trial=trial,
                algorithm=name,
                data_hash=data_hash,
                success=bool(score["success"]),
                mse=float(score["mse"]),
                nmse=float(score["nmse"]),
                relative_error=float(score["relative_error"]),
                paper_mse_db=float(score["paper_mse_db"]),
                iterations=len(result.history),
                converged=bool(result.converged),
                status=result.status,
                seconds=float(elapsed),
                measurement_residual=float(residual),
            )
        )
        if (m, sparsity) == (30, 3) and name in FIG4_ALGORITHMS:
            curves[name] = dict(
                history=np.asarray(result.history, dtype=float),
                times=np.asarray(result.times, dtype=float),
                truth=truth,
            )
    return trial, rows, curves


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _wilson(successes: int, n: int, z: float = 1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _summarize(rows: list[dict]):
    summaries = []
    for m, sparsity in CASES:
        for name in ALL_ALGORITHMS:
            selected = [
                row
                for row in rows
                if int(row["m"]) == m and int(row["k"]) == sparsity and row["algorithm"] == name
            ]
            if len(selected) != TRIALS:
                raise ValueError(f"Expected {TRIALS} rows for M={m}, K={sparsity}, {name}; got {len(selected)}")
            successes = sum(row["success"].lower() == "true" for row in selected)
            low, high = _wilson(successes, len(selected))
            summaries.append(
                dict(
                    m=m,
                    k=sparsity,
                    algorithm=name,
                    trials=len(selected),
                    successes=successes,
                    success_rate=successes / len(selected),
                    ci95_low=low,
                    ci95_high=high,
                    mean_mse=np.mean([float(row["mse"]) for row in selected]),
                    mean_nmse=np.mean([float(row["nmse"]) for row in selected]),
                    mean_iterations=np.mean([int(row["iterations"]) for row in selected]),
                    median_iterations=np.median([int(row["iterations"]) for row in selected]),
                    mean_seconds=np.mean([float(row["seconds"]) for row in selected]),
                    median_seconds=np.median([float(row["seconds"]) for row in selected]),
                    convergence_rate=np.mean([row["converged"].lower() == "true" for row in selected]),
                    max_measurement_residual=max(float(row["measurement_residual"]) for row in selected),
                )
            )
    return summaries


def _save_figure(fig, base: Path):
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _figure1(figures: Path, data_dir: Path):
    # The paper's peaks imply common variance 2: N(0,2), Laplace(scale=1),
    # and NP(gamma=2).  The NP density is singular at zero, so zero is omitted.
    x = np.linspace(-5, 5, 301)
    x = x[x != 0]
    normal_product = k0(np.abs(x) / np.sqrt(2)) / (np.pi * np.sqrt(2))
    laplace = 0.5 * np.exp(-np.abs(x))
    normal = np.exp(-(x * x) / 4) / np.sqrt(4 * np.pi)
    _write_csv(
        data_dir / "fig1_pdf_values.csv",
        [
            dict(x=float(a), normal_product=float(b), laplace=float(c), normal=float(d))
            for a, b, c, d in zip(x, normal_product, laplace, normal)
        ],
    )
    fig, ax = plt.subplots(figsize=(6.3, 4.3), constrained_layout=True)
    ax.plot(x, normal_product, color="black", lw=1.8, label="Normal Product")
    ax.plot(x, laplace, color="red", lw=1.6, ls="--", label="Laplace")
    ax.plot(x, normal, color="#2448d8", lw=1.6, ls="-.", label="Normal")
    ax.set(xlim=(-5, 5), ylim=(0, 0.9), xlabel="x", ylabel="P(x)")
    ax.set_xticks([-5, 0, 5])
    ax.grid(alpha=0.28, linestyle=":")
    ax.legend(frameon=True, loc="upper right")
    _save_figure(fig, figures / "fig1_pdf_comparison")


def _plot_fig3(summary_rows: list[dict], figures: Path, include_focuss: bool, suffix: str):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.35), constrained_layout=True)
    panels = ((axes[0], "m", "k", 3, "M"), (axes[1], "k", "m", 30, "K"))
    for panel_index, (ax, variable, fixed, fixed_value, xlabel) in enumerate(panels):
        algorithms = (
            ("SBL", "NP1", "NP0", "IRLS", "BP")
            if panel_index == 0
            else ("SBL", "BP", "NP1", "NP0", "IRLS")
        )
        if include_focuss:
            algorithms = (*algorithms, "FOCUSS")
        for name in algorithms:
            selected = sorted(
                [
                    row
                    for row in summary_rows
                    if int(row[fixed]) == fixed_value and row["algorithm"] == name
                ],
                key=lambda row: int(row[variable]),
            )
            style = FIG3_STYLE[name]
            ax.plot(
                [int(row[variable]) for row in selected],
                [float(row["success_rate"]) for row in selected],
                color=style["color"],
                marker=style["marker"],
                ms=5.0 if name != "BP" else 6.0,
                lw=1.15,
                markevery=1,
                fillstyle=style["fillstyle"],
                markeredgewidth=1.05,
                label="IRLS$_0$" if name == "IRLS" else name.replace("NP1", "NP-1").replace("NP0", "NP-0"),
            )
        ax.set(xlabel=xlabel, ylabel="Success Rate", xlim=(0, 30), ylim=(0, 1))
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.grid(alpha=0.55, color="#777777", linestyle=":", linewidth=0.7)
        ax.legend(fontsize=8, loc="upper right", framealpha=1.0, fancybox=False, edgecolor="black")
    axes[0].text(0.5, -0.19, "(a)", transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.19, "(b)", transform=axes[1].transAxes, ha="center", fontsize=11)
    _save_figure(fig, figures / f"fig3_success_rate{suffix}")


def _pack_fig4_curves(curves_by_trial: dict[int, dict], path: Path):
    arrays = {}
    for trial, algorithms in sorted(curves_by_trial.items()):
        for name, curve in algorithms.items():
            arrays[f"t{trial}_{name}_history"] = curve["history"]
            arrays[f"t{trial}_{name}_times"] = curve["times"]
            arrays[f"t{trial}_{name}_truth"] = curve["truth"]
    np.savez_compressed(path, **arrays)


def _benchmark_fig4(profile: str):
    """Rerun trial 0 serially and use median cumulative time per iteration.

    The Monte Carlo stage is intentionally parallel, so its per-process timing
    is not suitable for a computation-time figure.  Estimates are deterministic;
    only the elapsed-time samples are aggregated here.
    """
    A, y, truth = generate(N, 30, 3, SEED, 0, AMPLITUDES)
    curves = {}
    for name in FIG4_ALGORITHMS:
        algorithm_seed = int.from_bytes(
            hashlib.sha256(f"{SEED}:m30_k3_t0_{name}".encode()).digest()[:4], "little"
        )
        # Warm up NumPy/SciPy dispatch before collecting isolated timings.
        for _ in range(2):
            _algorithm(name, A.copy(), y.copy(), algorithm_seed, 3, profile)
        histories = []
        timing_samples = []
        for _ in range(FIG4_BENCHMARK_REPEATS):
            result = _algorithm(name, A.copy(), y.copy(), algorithm_seed, 3, profile)
            histories.append(np.asarray(result.history, dtype=float))
            timing_samples.append(np.asarray(result.times, dtype=float))
        expected_shape = histories[0].shape
        if any(value.shape != expected_shape for value in histories):
            raise RuntimeError(f"Non-deterministic Fig. 4 history length for {name}")
        if any(not np.allclose(value, histories[0], rtol=1e-9, atol=1e-11) for value in histories[1:]):
            raise RuntimeError(f"Non-deterministic Fig. 4 estimates for {name}")
        timings = np.stack(timing_samples)
        curves[name] = dict(
            history=histories[0],
            times=np.median(timings, axis=0),
            time_q25=np.quantile(timings, 0.25, axis=0),
            time_q75=np.quantile(timings, 0.75, axis=0),
            truth=truth,
        )
    return curves


def _figure4(curves: dict, figures: Path, data_dir: Path, profile: str):
    rows = []
    time_scale = 1.0
    if profile == "paper-aligned":
        sbl_end = float(curves["SBL"]["times"][-1])
        time_scale = 0.095 / sbl_end
    prepared = {}
    for name in FIG4_ALGORITHMS:
        curve = curves[name]
        history = curve["history"]
        truth = curve["truth"]
        values = np.asarray([metrics(x, truth)["paper_mse_db"] for x in history], dtype=float)
        values = np.maximum(values, -150)
        raw_times = curve["times"]
        scaled_times = raw_times * time_scale
        prepared[name] = (values, raw_times, scaled_times)
        for iteration, (raw_elapsed, q25, q75, normalized_elapsed, db) in enumerate(
            zip(raw_times, curve["time_q25"], curve["time_q75"], scaled_times, values), start=1
        ):
            rows.append(
                dict(
                    trial=0,
                    m=30,
                    k=3,
                    algorithm=name,
                    iteration=iteration,
                    benchmark_repeats=FIG4_BENCHMARK_REPEATS,
                    elapsed_seconds_raw_median=float(raw_elapsed),
                    elapsed_seconds_raw_q25=float(q25),
                    elapsed_seconds_raw_q75=float(q75),
                    elapsed_seconds_paper_normalized=float(normalized_elapsed),
                    time_scale=float(time_scale),
                    paper_mse_db=float(db),
                )
            )

    def plot_version(use_scaled_time: bool, output_name: str):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.35), constrained_layout=True)
        for name in FIG4_ALGORITHMS:
            values, raw_times, scaled_times = prepared[name]
            times = scaled_times if use_scaled_time else raw_times
            label = "IRLS$_0$" if name == "IRLS" else name.replace("NP1", "NP-1")
            style = FIG4_STYLE[name]
            plot_kwargs = dict(
                color=style["color"], marker=style["marker"], fillstyle=style["fillstyle"],
                markeredgewidth=1.0, ms=4.0, lw=1.15, label=label,
            )
            axes[0].plot(times, values, **plot_kwargs)
            axes[1].plot(np.arange(1, len(values) + 1), values, **plot_kwargs)
        if use_scaled_time:
            axes[0].set(xlabel="Paper-Hardware-Normalized Time (s)", xlim=(0, 0.1))
            axes[0].set_xticks(np.arange(0, 0.101, 0.02))
        else:
            maximum = max(float(values[1][-1]) for values in prepared.values())
            right = max(0.001, maximum * 1.06)
            axes[0].set(xlabel="Measured Computation Time (s)", xlim=(0, right))
        axes[0].set(ylabel="MSE (dB)", ylim=(-150, 20))
        axes[1].set(xlabel="The Number of Iterations", ylabel="MSE (dB)", xlim=(0, 100), ylim=(-150, 20))
        axes[1].set_xticks(np.arange(0, 101, 20))
        for ax in axes:
            ax.set_yticks(np.arange(-140, 21, 20))
            ax.grid(alpha=0.55, color="#777777", linestyle=":", linewidth=0.7)
            ax.legend(fontsize=8, loc="upper right", framealpha=1.0, fancybox=False, edgecolor="black")
        axes[0].text(0.5, -0.19, "(a)", transform=axes[0].transAxes, ha="center", fontsize=11)
        axes[1].text(0.5, -0.19, "(b)", transform=axes[1].transAxes, ha="center", fontsize=11)
        _save_figure(fig, figures / output_name)

    plot_version(False, "fig4_computational_cost_trial0")
    plot_version(True, "fig4_computational_cost_trial0_paper_scaled")
    _write_csv(data_dir / "fig4_trial0_curves.csv", rows)


def _manifest(out: Path, started: str, elapsed: float, workers: int, profile: str):
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(out)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = dict(
        paper="Zhou, Liu, and Fang, Bayesian Compressive Sensing Using Normal Product Priors, IEEE SPL 2015",
        scope="Figs. 1, 3, 4; Fig. 3 uses 300 Monte Carlo trials per point",
        config=dict(
            n=N,
            seed=SEED,
            trials=TRIALS,
            amplitudes=AMPLITUDES,
            max_iter=MAX_ITER,
            tol=TOL,
            success_threshold=SUCCESS_THRESHOLD,
            cases=CASES,
            algorithms=ALL_ALGORITHMS,
            fig4_trial=0,
            fig4_benchmark_repeats=FIG4_BENCHMARK_REPEATS,
            profile=profile,
            paper_aligned_overrides=(
                dict(sbl_tol=1e-8, np0_tol=1e-4, irls_p=0, irls_schedule="adaptive_order_statistic")
                if profile == "paper-aligned" else None
            ),
        ),
        environment=dict(
            python=sys.version,
            executable=sys.executable,
            numpy=np.__version__,
            scipy=scipy.__version__,
            matplotlib=matplotlib.__version__,
            platform=platform.platform(),
            workers=workers,
            thread_limits={name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        ),
        started_local=started,
        elapsed_seconds=elapsed,
        completed=True,
        outputs=files,
        limitations=[
            "The paper does not specify the nonzero-amplitude distribution, support sampler, or random seed.",
            "The exact IRLS_0 implementation and tuning are not provided; this repository uses its documented local continuation variant.",
            "Fig. 4 is a representative trial; absolute time is hardware-dependent.",
            "The primary Fig. 4 uses a serial median-of-15 timing benchmark, not timings recorded during the parallel Monte Carlo run.",
            "A separate paper-scaled Fig. 4 uses one common SBL-derived hardware factor; both raw and scaled times remain in the CSV.",
            "FOCUSS is retained in the result data and a supplementary Fig. 3, but is not present in the paper's Fig. 3 legend.",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/paper_reproduction_20261027")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--profile", choices=("baseline", "paper-aligned"), default="baseline")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    raw_dir = out / "raw_cases"
    data_dir = out / "data"
    figures = out / "figures"
    for directory in (raw_dir, data_dir, figures):
        directory.mkdir()

    started = time.strftime("%Y-%m-%d %H:%M:%S %z")
    wall_start = time.perf_counter()
    print(f"Running {len(CASES)} cases x {TRIALS} trials x {len(ALL_ALGORITHMS)} algorithms")
    print(f"Workers={args.workers}; output={out}", flush=True)
    all_rows = []
    curves_by_trial = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for case_index, (m, sparsity) in enumerate(CASES, start=1):
            case_start = time.perf_counter()
            tasks = [(m, sparsity, trial, args.profile) for trial in range(TRIALS)]
            case_rows = []
            for trial, rows, curves in executor.map(_one_trial, tasks, chunksize=1):
                case_rows.extend(rows)
                if curves:
                    curves_by_trial[trial] = curves
            _write_csv(raw_dir / f"m{m:02d}_k{sparsity:02d}.csv", case_rows)
            all_rows.extend(case_rows)
            print(
                f"[{case_index:02d}/{len(CASES)}] M={m:02d}, K={sparsity:02d}: "
                f"{len(case_rows)} algorithm runs in {time.perf_counter() - case_start:.1f}s",
                flush=True,
            )

    _write_csv(data_dir / "trials.csv", all_rows)
    summaries = _summarize(_read_csv(data_dir / "trials.csv"))
    _write_csv(data_dir / "summary.csv", summaries)
    _pack_fig4_curves(curves_by_trial, data_dir / "fig4_histories_300_trials.npz")
    _figure1(figures, data_dir)
    _plot_fig3(summaries, figures, include_focuss=False, suffix="")
    _plot_fig3(summaries, figures, include_focuss=True, suffix="_with_focuss")
    benchmark_curves = _benchmark_fig4(args.profile)
    _figure4(benchmark_curves, figures, data_dir, args.profile)
    elapsed = time.perf_counter() - wall_start
    _manifest(out, started, elapsed, args.workers, args.profile)
    print(f"Completed in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Noiseless NP-1 from Zhou, Liu & Fang, IEEE SPL 22(5), 2015.

Implements Algorithm 2 and equations (10)--(15), with alpha = beta = 0.
Only NumPy is required. See README.md for assumptions and integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NP1Result:
    """Estimate, final latent posterior moments, and iteration diagnostics.

    ``variance_a/b`` are posterior covariance diagonals, not prior variances.
    ``precision_a/b`` are the updated hyperparameter expectations in (14)/(15).
    History and optional x_history contain completed iterations only.
    Convergence means the relative-update criterion was met, not that the
    unknown true signal was recovered. No truth data enters the solver.
    """

    x: FloatArray
    n_iter: int
    converged: bool
    stop_reason: str
    mean_a: FloatArray
    mean_b: FloatArray
    variance_a: FloatArray
    variance_b: FloatArray
    precision_a: FloatArray
    precision_b: FloatArray
    history: dict[str, FloatArray]
    x_history: FloatArray | None


def _real_array(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex NP-1 is not implemented")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _posterior(
    A: FloatArray,
    y: FloatArray,
    other_mean: FloatArray,
    prior_variance: FloatArray,
    rcond: float,
) -> tuple[FloatArray, FloatArray, int]:
    """Evaluate (12)/(13) or (10)/(11) without n-by-n covariances.

    B = A diag(other_mean) diag(sqrt(prior_variance)),
    mean = sqrt(prior_variance) * (B^+ y),
    diag(Cov) = prior_variance * diag(I - B^+ B).
    SVD avoids squaring the condition number in B @ B.T. Numerical rank
    truncation implements the pseudoinverse; there is no support pruning.
    """
    scale = np.sqrt(prior_variance)
    B = A * (other_mean * scale)[None, :]
    if not np.all(np.isfinite(B)):
        raise FloatingPointError("NP-1 weighted matrix overflowed; rescale the input")
    left, singular, right_t = np.linalg.svd(B, full_matrices=False)
    keep = singular > rcond * singular[0]
    rank = int(np.count_nonzero(keep))
    right = right_t[keep, :]
    mean = scale * (right.T @ ((left[:, keep].T @ y) / singular[keep]))
    leverage = np.sum(right * right, axis=0)
    # An orthogonal projector has diagonal in [0,1]; clip roundoff only.
    variance = prior_variance * np.clip(1.0 - leverage, 0.0, 1.0)
    return mean, variance, rank


def solve_np1(
    A: ArrayLike,
    y: ArrayLike,
    *,
    max_iter: int = 300,
    tol: float = 1e-3,
    rcond: float | None = None,
    variance_floor: float = 1e-300,
    store_history: bool = False,
) -> NP1Result:
    """Recover a real sparse vector from noiseless measurements ``y = A @ x``.

    Parameters
    ----------
    A : array, shape (m, n)
        Dense real sensing matrix. It is not normalized or modified.
    y : array, shape (m,) or (m, 1)
        Noiseless measurement vector.
    max_iter, tol
        Paper defaults: 300 and 1e-3. Stop when
        ||x_new-x_old||_2 / ||x_old||_2 <= tol. At least one iteration runs.
        A zero denominator gives 0 for no change, otherwise infinity.
    rcond
        Relative SVD cutoff. Default: eps(float64) * max(m, n).
    variance_floor
        Lower bound on updated prior variances solely to prevent zero
        division/overflow. It is not an observation-noise regularizer.
    store_history
        Save the estimated vector after each iteration for external scoring.

    Notes
    -----
    Algorithm 2 initializes both latent means to ones. Initial precisions
    are omitted in its pseudocode; this implementation uses ones on both
    sides, consistent with the unit scales in Algorithm 1. Hyperpriors are
    fixed to the paper's alpha = beta = 0. This is not a noisy-data solver.
    No ground truth, random initialization, sparsity K, pruning, or debiasing
    is used. Each iteration costs two thin SVDs, O(m*n*min(m,n)).
    """
    A = _real_array(A, "A")
    y = _real_array(y, "y")
    if A.ndim != 2 or min(A.shape) == 0:
        raise ValueError("A must be a nonempty two-dimensional matrix")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or y.shape[0] != A.shape[0]:
        raise ValueError("y must have shape (m,) or (m, 1), matching A's rows")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isscalar(tol) or not np.isreal(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if not np.isscalar(variance_floor) or not np.isreal(variance_floor) or not np.isfinite(variance_floor) or not 1e-300 <= variance_floor <= 1:
        raise ValueError("variance_floor must be finite and in [1e-300, 1]")
    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(A.shape)
    if not np.isscalar(rcond) or not np.isreal(rcond) or not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("rcond must be finite and in [0, 1)")

    n = A.shape[1]
    a, b = np.ones(n), np.ones(n)
    prior_a, prior_b = np.ones(n), np.ones(n)
    x = a * b
    history: dict[str, list[float]] = {
        "relative_change": [],
        "residual_norm": [],
        "elapsed_seconds": [],
        "rank_a": [],
        "rank_b": [],
    }
    estimates: list[FloatArray] | None = [] if store_history else None
    converged = False
    started = perf_counter()

    for iteration in range(1, int(max_iter) + 1):
        previous_x = x
        # Algorithm 2, lines 3--6: b uses this iteration's NEW a.
        a, cov_a, rank_a = _posterior(A, y, b, prior_a, float(rcond))
        b, cov_b, rank_b = _posterior(A, y, a, prior_b, float(rcond))
        x = a * b
        # Lines 8--9, (14)/(15), alpha=beta=0:
        # E[kappa^-2] = 1 / (E[a]^2 + Var(a)), and similarly for b.
        # Work with reciprocal precisions to avoid repeated inversions.
        prior_a = np.maximum(a * a + cov_a, variance_floor)
        prior_b = np.maximum(b * b + cov_b, variance_floor)
        if not all(np.all(np.isfinite(z)) for z in (x, prior_a, prior_b)):
            raise FloatingPointError("NP-1 moments overflowed; rescale the input")

        delta = float(np.linalg.norm(x - previous_x))
        previous_norm = float(np.linalg.norm(previous_x))
        relative_change = delta / previous_norm if previous_norm else (0.0 if delta == 0 else float("inf"))
        history["relative_change"].append(relative_change)
        history["residual_norm"].append(float(np.linalg.norm(A @ x - y)))
        history["elapsed_seconds"].append(perf_counter() - started)
        history["rank_a"].append(rank_a)
        history["rank_b"].append(rank_b)
        if estimates is not None:
            estimates.append(x.copy())
        if relative_change <= tol:
            converged = True
            break

    return NP1Result(
        x=x,
        n_iter=iteration,
        converged=converged,
        stop_reason="relative_change" if converged else "max_iter",
        mean_a=a,
        mean_b=b,
        variance_a=cov_a,
        variance_b=cov_b,
        precision_a=1.0 / prior_a,
        precision_b=1.0 / prior_b,
        history={key: np.asarray(values, dtype=np.float64) for key, values in history.items()},
        x_history=np.stack(estimates) if estimates is not None else None,
    )


def np1(A: ArrayLike, y: ArrayLike, **kwargs) -> FloatArray:
    """Simple integration entry point returning only the shape-(n,) estimate."""
    return solve_np1(A, y, **kwargs).x


# Convenient spelling matching the module name in the team's allocation.
NP1 = np1

__all__ = ["NP1", "NP1Result", "np1", "solve_np1"]

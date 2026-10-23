"""Independent member baselines compatible with the repository's Result API.

Real, noiseless SBL-EM, basic FOCUSS, Chartrand-Yin smoothed IRLS, and
equality-constrained BP. These are local implementations, not author software.
See docs/baselines_member.md for formulas, defaults, and integration commands.
"""
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from bcs.algorithms import Result


def _real_parameter(name, value, *, lower=0.0, upper=None, closed=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or (value < lower if closed else value <= lower):
        raise ValueError(f"Invalid {name}: {value}")
    if upper is not None and value >= upper:
        raise ValueError(f"{name} must be less than {upper}")
    return value


def _inputs(A, y):
    A, y = np.asarray(A), np.asarray(y)
    for name, value in (("A", A), ("y", y)):
        if np.iscomplexobj(value) or value.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values")
    if A.ndim != 2 or not all(A.shape):
        raise ValueError("A must be a nonempty matrix")
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)
    if y.ndim != 1 or y.shape[0] != A.shape[0]:
        raise ValueError("y must be a vector with one entry per row of A")
    A, y = np.asarray(A, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)):
        raise ValueError("A and y must be finite")
    return A, y


def _weighted_posterior(A, y, q, rcond, variance_floor, covariance=False):
    """Q A.T (A Q A.T)^dagger y, without explicitly forming a Gram inverse.

    SVD of B=A sqrt(Q) avoids squaring its condition number. All columns are
    retained. Numerical rank truncation is controlled explicitly by rcond.
    """
    q = np.maximum(q, variance_floor)
    root = np.sqrt(q)
    B = A * root
    U, singular, Vh = np.linalg.svd(B, full_matrices=False)
    keep = singular > rcond * singular[0]
    if not np.any(keep):
        return np.zeros(A.shape[1]), q.copy() if covariance else None
    V = Vh[keep].T
    mean = root * (V @ ((U[:, keep].T @ y) / singular[keep]))
    diagonal = q * np.maximum(0.0, 1.0 - np.sum(V * V, axis=1)) if covariance else None
    return mean, diagonal


def _relative_change(current, previous):
    denominator = float(np.linalg.norm(previous))
    numerator = float(np.linalg.norm(current - previous))
    return numerator / denominator if denominator else (0.0 if numerator == 0 else np.inf)


def solve(A, y, *, seed=0, max_iter=300, tol=1e-8, kind="SBL",
          convergence_tol=None, rcond=1e-12, variance_floor=1e-30, p=None,
          epsilon_start=1.0, epsilon_min=1e-8, epsilon_factor=0.1,
          lp_tolerance=1e-9, fixed_iterations=False):
    """Return bcs.algorithms.Result; never receives ground truth or true K.

    `seed` is accepted for framework compatibility. These algorithms do not
    use random initialization or global RNG state. Every history entry is a
    completed estimate, and times are actual cumulative seconds. BP exposes
    one final estimate, not a fictitious LP iteration trajectory.
    """
    A, y = _inputs(A, y)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not isinstance(fixed_iterations, (bool, np.bool_)):
        raise TypeError("fixed_iterations must be boolean")
    tol = _real_parameter("tol", tol)
    if convergence_tol is not None:
        tol = _real_parameter("convergence_tol", convergence_tol)
    rcond = _real_parameter("rcond", rcond, upper=1.0, closed=True)
    variance_floor = _real_parameter("variance_floor", variance_floor)
    epsilon_start = _real_parameter("epsilon_start", epsilon_start)
    epsilon_min = _real_parameter("epsilon_min", epsilon_min)
    epsilon_factor = _real_parameter("epsilon_factor", epsilon_factor, upper=1.0)
    lp_tolerance = _real_parameter("lp_tolerance", lp_tolerance)
    if lp_tolerance < 1e-10:
        raise ValueError("HiGHS requires lp_tolerance >= 1e-10")
    if epsilon_start < epsilon_min:
        raise ValueError("epsilon_start must be >= epsilon_min")
    kind = str(kind).upper()
    if kind not in {"SBL", "FOCUSS", "IRLS", "BP"}:
        raise ValueError(f"Unknown member baseline: {kind}")
    if p is None:
        p = 0.0 if kind == "FOCUSS" else 0.5
    p = _real_parameter("p", p, upper=np.nextafter(1.0, np.inf), closed=True)
    n = A.shape[1]
    start = perf_counter()
    history, times = [], []
    if kind == "BP":
        # Exact equality-constrained l1 minimization; no Lasso penalty.
        opt = linprog(np.ones(2*n), A_eq=np.hstack((A, -A)), b_eq=y,
                      bounds=(0, None), method="highs", options={
                          "primal_feasibility_tolerance": lp_tolerance,
                          "dual_feasibility_tolerance": lp_tolerance,
                      })
        if not opt.success:
            raise RuntimeError(f"BP HiGHS failed: {opt.message}")
        x = opt.x[:n] - opt.x[n:]
        history.append(x.copy())
        times.append(perf_counter() - start)
        converged, status = True, "lp_optimal"
    elif not np.any(y):
        x = np.zeros(n)
        history.append(x.copy())
        times.append(perf_counter() - start)
        converged, status = True, "converged"
    else:
        gamma = np.ones(n)
        x = np.zeros(n) if kind == "SBL" else _weighted_posterior(
            A, y, np.ones(n), rcond, variance_floor)[0]
        epsilon = epsilon_start
        converged = False
        for _ in range(max_iter):
            previous = x
            if kind == "SBL":
                x, diagonal = _weighted_posterior(
                    A, y, gamma, rcond, variance_floor, covariance=True)
                gamma = np.maximum(x*x + diagonal, variance_floor)
            else:
                q = (np.abs(previous)**(2-p) if kind == "FOCUSS" else
                     (previous*previous + epsilon)**(1-p/2))
                x, _ = _weighted_posterior(A, y, q, rcond, variance_floor)
            if not np.all(np.isfinite(x)):
                raise FloatingPointError("Non-finite member estimate")
            history.append(x.copy())
            times.append(perf_counter() - start)
            change = _relative_change(x, previous)
            at_final_epsilon = kind != "IRLS" or epsilon <= epsilon_min
            converged = bool(change <= tol and at_final_epsilon)
            if converged and not fixed_iterations:
                break
            # Chartrand & Yin (2008), Section 3. epsilon is not squared here.
            if kind == "IRLS" and change < np.sqrt(epsilon)/100:
                epsilon = max(epsilon_min, epsilon * epsilon_factor)
        status = "converged" if converged else "max_iter"
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Non-finite member estimate")
    residual = np.linalg.norm(A @ x - y) / max(np.linalg.norm(y), np.finfo(float).tiny)
    if residual > 1e-7:
        raise FloatingPointError(f"Noiseless constraint not satisfied (relative residual {residual:.3g})")
    return Result(x.copy(), history, times, bool(converged), status)


def solve_sbl(A, y, *, seed=0, max_iter=300, tol=1e-8, **params):
    """SBL member entry: Gaussian posterior and EM variance update, no pruning."""
    return solve(A, y, seed=seed, max_iter=max_iter, tol=tol, kind="SBL", **params)


def solve_focuss(A, y, *, seed=0, max_iter=300, tol=1e-8, **params):
    """Basic FOCUSS member entry; p=0 unless explicitly overridden."""
    return solve(A, y, seed=seed, max_iter=max_iter, tol=tol, kind="FOCUSS", **params)


def solve_irls(A, y, *, seed=0, max_iter=300, tol=1e-8, **params):
    """Smoothed IRLS member entry with convergence-triggered epsilon reduction."""
    return solve(A, y, seed=seed, max_iter=max_iter, tol=tol, kind="IRLS", **params)


def solve_bp(A, y, *, seed=0, max_iter=300, tol=1e-8, **params):
    """BP member entry: exact equality constraints, SciPy/HiGHS LP solver."""
    return solve(A, y, seed=seed, max_iter=max_iter, tol=tol, kind="BP", **params)

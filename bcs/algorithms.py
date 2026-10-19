"""Local reference implementations, NOT the authors' original software.

NP formulas: Zhou et al. (2015), equations (10)-(15), Algorithms 1-2.
SBL: EM variance update. IRLS: smoothed l_p continuation, local baseline.
FOCUSS: p=0 reweighted minimum norm. BP: exact equality-constrained LP.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import linprog


@dataclass
class Result:
    x: np.ndarray
    history: list
    times: list
    converged: bool
    status: str


def posterior(A, other, variance, y, rcond=1e-12):
    """Mean and diagonal covariance using the paper's pseudoinverse form."""
    scale = np.sqrt(np.maximum(variance, 1e-30))
    B = A * (other * scale)
    inverse = np.linalg.pinv(B, rcond=rcond)
    mean = scale * (inverse @ y)
    diagonal = variance * np.maximum(1 - np.einsum("ij,ji->i", inverse, B), 0)
    return mean, diagonal


def solve(A, y, *, seed=0, max_iter=300, tol=1e-3, kind="NP1", rcond=1e-12,
          p=0.5, smoothing=1.0, convergence_tol=None):
    if convergence_tol is not None:
        tol = convergence_tol
    n = A.shape[1]
    start = perf_counter()
    if kind == "BP":
        opt = linprog(np.ones(2*n), A_eq=np.hstack((A, -A)), b_eq=y,
                      bounds=(0, None), method="highs")
        if not opt.success:
            raise RuntimeError("BP solver: " + opt.message)
        x = opt.x[:n] - opt.x[n:]
        return Result(x, [x.copy()], [perf_counter()-start], True, "lp_optimal")
    a, b = np.ones(n), np.ones(n)
    va, vb = np.ones(n), np.ones(n)
    x = np.ones(n) if kind in ("NP0", "NP1", "SBL") else np.linalg.pinv(A, rcond=rcond) @ y
    history, times = [], []
    converged = False
    for step in range(max_iter):
        previous = x.copy()
        if kind in ("NP0", "NP1"):
            a, ca = posterior(A, b, va, y, rcond)
            b, cb = posterior(A, a, vb, y, rcond)
            x = a * b
            if kind == "NP1":  # alpha=beta=0 => variance = second moment.
                va = np.maximum(a*a + ca, 1e-30)
                vb = np.maximum(b*b + cb, 1e-30)
        elif kind == "SBL":
            x, covariance = posterior(A, np.ones(n), va, y, rcond)
            va = np.maximum(x*x + covariance, 1e-30)
        elif kind == "IRLS":
            epsilon = max(smoothing * 0.5**step, 1e-8)
            weights = (previous*previous + epsilon*epsilon)**(1-p/2)
            x, _ = posterior(A, np.ones(n), weights, y, rcond)
        elif kind == "FOCUSS":
            weights = np.maximum(np.abs(previous), 1e-15)**(2-p)
            x, _ = posterior(A, np.ones(n), weights, y, rcond)
        else:
            raise ValueError("Unknown algorithm: " + kind)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("Non-finite estimate")
        history.append(x.copy())
        times.append(perf_counter()-start)
        change = np.linalg.norm(x-previous) / max(np.linalg.norm(previous), 1e-30)
        if change <= tol:
            converged = True
            break
    return Result(x, history, times, converged, "converged" if converged else "max_iter")

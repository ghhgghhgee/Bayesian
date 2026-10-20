"""Noise-free NP-0 solver from Zhou, Liu, and Fang (IEEE SPL, 2015).

The implementation follows the paper's alternating updates in (10) and (12)
with the NP-0 fixed scales ``kappa = gamma = 1``.  It therefore uses a
Moore-Penrose pseudoinverse for each weighted sensing matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple, Union

import numpy as np


@dataclass(frozen=True)
class NP0Diagnostics:
    """Convergence details optionally returned by :func:`np0`."""

    iterations: int
    converged: bool
    relative_change: float


def np0(
    A: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-3,
    max_iter: int = 300,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, NP0Diagnostics]]:
    """Estimate a real sparse signal with the paper's NP-0 iteration.

    Parameters
    ----------
    A:
        Real sensing matrix with shape ``(M, N)``.
    y:
        Real measurement vector with length ``M``.  A row or column vector
        is accepted.
    eps:
        Relative-change stopping tolerance (paper default: ``1e-3``).
    max_iter:
        Maximum number of alternating updates (paper default: ``300``).
    return_info:
        If true, return ``(x_hat, diagnostics)``; otherwise return ``x_hat``.

    Returns
    -------
    numpy.ndarray or tuple
        The estimated signal of length ``N``, optionally followed by an
        :class:`NP0Diagnostics` instance.

    Notes
    -----
    NP-0 fixes the scale vectors ``kappa`` and ``gamma`` to all ones and does
    not learn them.  Given current factor means, the paper's noiseless
    updates reduce to

    ``a <- pinv(A @ diag(b)) @ y``
    ``b <- pinv(A @ diag(a)) @ y``

    followed by ``x_hat = a * b``.  This is the noise-free form used in the
    paper's numerical experiments; it is not a noisy-observation solver.
    """

    x, info, _, _ = _run_np0(
        A, y, eps=eps, max_iter=max_iter, rcond=1e-15, record_history=False
    )
    return (x, info) if return_info else x


def solve(
    A: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-3,
    rcond: float = 1e-15,
):
    """Adapt NP-0 to the unified experiment framework's Result interface.

    ``seed`` is accepted for interface compatibility; NP-0 is deterministic
    and does not use random state. ``tol`` maps to the standalone ``eps``.
    ``rcond=1e-15`` preserves the original member implementation's NumPy
    pseudoinverse cutoff; the leader reference instead defaults to 1e-12.
    """

    # Import only at the framework boundary so standalone np0 stays NumPy-only.
    from bcs.algorithms import Result

    x, info, history, times = _run_np0(
        A, y, eps=tol, max_iter=max_iter, rcond=rcond, record_history=True
    )
    return Result(
        x=x,
        history=history,
        times=times,
        converged=info.converged,
        status="converged" if info.converged else "max_iter",
    )


def _run_np0(A, y, *, eps, max_iter, rcond, record_history):
    """Shared iteration for the standalone API and framework adapter."""

    A_array = np.asarray(A)
    y_array = np.asarray(y)

    if np.iscomplexobj(A_array) or np.iscomplexobj(y_array):
        raise TypeError("NP-0 in the paper is defined for real-valued A and y.")

    try:
        A_array = np.asarray(A_array, dtype=np.float64)
        y_array = np.asarray(y_array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("A and y must contain real numeric values.") from exc

    if A_array.ndim != 2 or min(A_array.shape, default=0) < 1:
        raise ValueError("A must be a non-empty two-dimensional matrix.")
    if y_array.ndim == 2 and 1 in y_array.shape:
        y_array = y_array.reshape(-1)
    if y_array.ndim != 1:
        raise ValueError("y must be a vector or a single-row/single-column array.")
    if y_array.size != A_array.shape[0]:
        raise ValueError(
            f"y has length {y_array.size}, but A has {A_array.shape[0]} rows."
        )
    if not np.all(np.isfinite(A_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("A and y must contain only finite values.")

    try:
        eps_value = float(eps)
    except (TypeError, ValueError) as exc:
        raise TypeError("eps must be a positive finite number.") from exc
    if not np.isfinite(eps_value) or eps_value <= 0:
        raise ValueError("eps must be a positive finite number.")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise TypeError("max_iter must be a positive integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    try:
        rcond_value = float(rcond)
    except (TypeError, ValueError) as exc:
        raise TypeError("rcond must be a finite number in [0, 1).") from exc
    if not np.isfinite(rcond_value) or not 0 <= rcond_value < 1:
        raise ValueError("rcond must be a finite number in [0, 1).")

    n = A_array.shape[1]
    # NP-0 initializes both latent factor means to ones and keeps the
    # corresponding scale vectors kappa and gamma fixed at ones.
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    x = a * b
    relative_change = float("inf")
    converged = False
    iterations = 0
    history, times = [], []
    start = perf_counter()

    for iterations in range(1, int(max_iter) + 1):
        # Update a using the noiseless limit of equation (12).
        weighted_A = A_array * b[np.newaxis, :]
        a = np.linalg.pinv(weighted_A, rcond=rcond_value) @ y_array

        # Update b using the noiseless limit of equation (10).
        weighted_A = A_array * a[np.newaxis, :]
        b = np.linalg.pinv(weighted_A, rcond=rcond_value) @ y_array

        x_new = a * b
        if not np.all(np.isfinite(x_new)):
            raise FloatingPointError("NP-0 produced a non-finite signal estimate.")

        change = float(np.linalg.norm(x_new - x))
        previous_norm = float(np.linalg.norm(x))
        if previous_norm == 0.0:
            relative_change = 0.0 if change == 0.0 else float("inf")
        else:
            relative_change = change / previous_norm

        x = x_new
        if record_history:
            history.append(x.copy())
            times.append(perf_counter() - start)
        if relative_change <= eps_value:
            converged = True
            break

    info = NP0Diagnostics(
        iterations=iterations,
        converged=converged,
        relative_change=relative_change,
    )
    return x, info, history, times

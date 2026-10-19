import hashlib
import numpy as np


def generate(n, m, k, seed, trial, amplitudes="normal"):
    """Independent case/trial streams; independent of algorithm/run ordering."""
    if not 0 < k <= n or not 0 < m <= n:
        raise ValueError("Require 0 < k,m <= n")
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, m, k, trial]))
    A = rng.standard_normal((m, n))  # Paper: N(0,1), no column normalization.
    x = np.zeros(n)
    support = rng.choice(n, k, replace=False)
    if amplitudes == "normal":
        x[support] = rng.standard_normal(k)
    elif amplitudes == "rademacher":
        x[support] = rng.choice([-1., 1.], k)
    else:
        raise ValueError("Unknown amplitude distribution")
    return A, A @ x, x


def fingerprint(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()

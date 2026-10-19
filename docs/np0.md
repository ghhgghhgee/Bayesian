# NP-0 implementation

`algorithms/np0.py` implements the noise-free NP-0 iteration in Zhou, Liu,
and Fang, *Bayesian Compressive Sensing Using Normal Product Priors*,
IEEE Signal Processing Letters, 2015, equations (10) and (12).

The model writes each signal coefficient as `x_i = a_i b_i`. NP-0 initializes
the factor means to ones and fixes the scale vectors `kappa` and `gamma` to
ones. It alternates the two factor updates and reports `x_hat = a * b`:

```text
a <- pinv(A @ diag(b)) @ y
b <- pinv(A @ diag(a)) @ y
x_hat <- a * b
```

The iteration stops when
`norm(x_new - x_old) / norm(x_old) <= eps`, or when `max_iter` is reached.
The defaults are `eps=1e-3` and `max_iter=300`, matching the paper. When the
previous estimate has zero norm, the implementation treats an unchanged zero
estimate as converged instead of evaluating `0 / 0`.

## Interface

```python
from algorithms.np0 import np0

x_hat = np0(A, y)
x_hat, info = np0(A, y, return_info=True)
```

`A` must be a finite real `M x N` matrix, and `y` a finite real vector of
length `M`. The function returns the length-`N` estimate. With
`return_info=True`, it also returns the iteration count, convergence flag, and
last relative change. NumPy is the only runtime dependency.

This module implements the noise-free limit used in the paper's numerical
experiments. It does not estimate observation-noise variance or implement the
NP-1 precision updates.

# Bayesian

UESTC class project: Bayesian compressed sensing with normal-product priors.

## Current implementation

- **NP-0:** noise-free implementation in [`algorithms/np0.py`](algorithms/np0.py).
- **NP-0 notes:** model, update equations, interface, and scope in
  [`docs/np0.md`](docs/np0.md).

Install the runtime dependency with:

```powershell
python -m pip install -r requirements.txt
```

The public function is `algorithms.np0.np0(A, y, eps=1e-3, max_iter=300)`.
It returns the estimated signal; pass `return_info=True` to also receive
iteration diagnostics. See the NP-0 notes for input conventions and the
noise-free scope.

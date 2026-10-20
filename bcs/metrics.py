import numpy as np


def metrics(estimate, truth, threshold=1e-3):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    if estimate.shape != truth.shape or truth.ndim != 1:
        raise ValueError("Expected equal 1-D signal shapes")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(truth)):
        raise ValueError("Non-finite signal")
    energy = float(truth @ truth)
    if energy == 0:
        raise ValueError("Relative error is undefined for zero truth")
    error = estimate - truth
    squared = float(error @ error)
    nmse = squared / energy
    relative = float(np.sqrt(nmse))
    return dict(mse=squared / truth.size, nmse=nmse, relative_error=relative,
                paper_mse_db=float(10 * np.log10(nmse)) if nmse else -np.inf,
                success=bool(relative < threshold))

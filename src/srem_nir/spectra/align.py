from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_spline_to_length(X: np.ndarray, target_length: int) -> np.ndarray:
    """Align one spectral region to `target_length` by cubic spline interpolation."""

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n_samples, old_length = X.shape
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if old_length == target_length:
        return X.copy()
    if old_length == 1:
        return np.repeat(X, target_length, axis=1).astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, old_length, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    spline = CubicSpline(x_old, X.astype(np.float64), axis=1, extrapolate=True)
    return spline(x_new).astype(np.float32, copy=False)


def align_regions_spline(
    regions: OrderedDict[str, np.ndarray],
    target_length: int | None = None,
) -> OrderedDict[str, np.ndarray]:
    """Cubic-spline align all SREM regions to a common length."""

    if target_length is None:
        target_length = max(int(X.shape[1]) for X in regions.values())
    return OrderedDict((name, cubic_spline_to_length(X, int(target_length))) for name, X in regions.items())


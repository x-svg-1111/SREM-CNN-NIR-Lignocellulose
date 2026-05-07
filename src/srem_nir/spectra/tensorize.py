from __future__ import annotations

from collections import OrderedDict

import numpy as np


def stack_regions_mnp(regions: OrderedDict[str, np.ndarray]) -> np.ndarray:
    """Stack aligned SREM regions into manuscript tensor shape M x N x P."""

    arrays = list(regions.values())
    if not arrays:
        raise ValueError("No regions to stack")
    n_samples = arrays[0].shape[0]
    n_points = arrays[0].shape[1]
    for X in arrays:
        if X.shape != (n_samples, n_points):
            raise ValueError("All aligned regions must share shape (M, N)")
    return np.stack(arrays, axis=-1).astype(np.float32, copy=False)


def to_torch_channel_first(X_mnp: np.ndarray) -> np.ndarray:
    """Convert manuscript M x N x P tensor to PyTorch Conv1d shape M x P x N."""

    X_mnp = np.asarray(X_mnp, dtype=np.float32)
    if X_mnp.ndim != 3:
        raise ValueError(f"Expected M x N x P tensor, got {X_mnp.shape}")
    return np.transpose(X_mnp, (0, 2, 1)).astype(np.float32, copy=False)


from __future__ import annotations

from collections import OrderedDict

import numpy as np


def minmax_per_sample(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each sample independently to [0, 1]."""

    X = np.asarray(X, dtype=np.float32)
    mn = np.min(X, axis=1, keepdims=True)
    mx = np.max(X, axis=1, keepdims=True)
    span = np.maximum(mx - mn, float(eps))
    return ((X - mn) / span).astype(np.float32, copy=False)


def normalize_regions_minmax(
    regions: OrderedDict[str, np.ndarray],
    eps: float = 1e-12,
) -> OrderedDict[str, np.ndarray]:
    """Apply manuscript SREM independent min-max normalization per region."""

    return OrderedDict((name, minmax_per_sample(X, eps=eps)) for name, X in regions.items())


class TargetStandardizer:
    """Standardize chemical values for neural-network training."""

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y))
        if self.std_ < self.eps:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetStandardizer is not fitted")
        return ((np.asarray(y, dtype=np.float32).reshape(-1) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetStandardizer is not fitted")
        return (np.asarray(y, dtype=np.float32).reshape(-1) * self.std_ + self.mean_).astype(np.float32)


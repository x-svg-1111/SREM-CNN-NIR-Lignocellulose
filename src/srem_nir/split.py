from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def kennard_stone(X: np.ndarray, n_select: int) -> np.ndarray:
    """Kennard-Stone sample selection by Euclidean distance in spectral space."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    n_samples = X.shape[0]
    n_select = int(n_select)
    if n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= n_samples:
        return np.arange(n_samples, dtype=int)

    D = cdist(X, X, metric="euclidean")
    i, j = np.unravel_index(np.argmax(np.triu(D, k=1)), D.shape)
    selected = [int(i), int(j)]
    remaining = np.ones(n_samples, dtype=bool)
    remaining[selected] = False

    while len(selected) < n_select:
        candidates = np.where(remaining)[0]
        min_dist = np.min(D[candidates][:, selected], axis=1)
        pick = int(candidates[np.argmax(min_dist)])
        selected.append(pick)
        remaining[pick] = False
    return np.sort(np.asarray(selected, dtype=int))


def ks_train_random_val_test_split(
    X: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 2025,
) -> SplitIndices:
    """Manuscript split: KS train selection, random val/test split from remainder."""

    X = np.asarray(X)
    n = int(X.shape[0])
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1")
    n_train = int(round(float(train_ratio) * n))
    n_val = int(round(float(val_ratio) * n))
    n_train = min(max(2, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    train = kennard_stone(X, n_train)
    rest = np.setdiff1d(np.arange(n, dtype=int), train, assume_unique=False)
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    val = np.sort(rest[:n_val])
    test = np.sort(rest[n_val:])
    return SplitIndices(train=np.sort(train), val=val, test=test)


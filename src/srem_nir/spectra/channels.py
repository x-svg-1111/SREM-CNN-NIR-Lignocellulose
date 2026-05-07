from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpectralRegion:
    """A non-overlapping NIR region in wavenumber units."""

    name: str
    low: float
    high: float
    description: str


DEFAULT_REGIONS: tuple[SpectralRegion, ...] = (
    SpectralRegion("combination", 4000.0, 5500.0, "combination band region"),
    SpectralRegion("first_overtone", 5500.0, 7500.0, "first overtone region"),
    SpectralRegion("second_overtone", 7500.0, 10000.0, "second and higher overtone region"),
)


def validate_spectrum_inputs(X: np.ndarray, wavenumbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (n_samples, n_wavenumbers), got {X.shape}")
    if X.shape[1] != wavenumbers.shape[0]:
        raise ValueError("X and wavenumbers length mismatch")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if not np.all(np.isfinite(wavenumbers)):
        raise ValueError("wavenumbers contains non-finite values")
    return X, wavenumbers


def _region_mask(wavenumbers: np.ndarray, region: SpectralRegion, *, is_last: bool) -> np.ndarray:
    """Half-open regions avoid duplicate boundary points."""

    if is_last:
        return (wavenumbers >= region.low) & (wavenumbers <= region.high)
    return (wavenumbers >= region.low) & (wavenumbers < region.high)


def partition_spectrum(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    regions: Iterable[SpectralRegion] = DEFAULT_REGIONS,
) -> OrderedDict[str, np.ndarray]:
    """Partition a full NIR spectrum into manuscript-defined SREM regions.

    The original acquisition order is preserved within each region. For a
    Bruker-style axis descending from 10000 to 4000 cm^-1, each returned region
    is also descending internally.
    """

    X, wavenumbers = validate_spectrum_inputs(X, wavenumbers)
    region_list = tuple(regions)
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    used = np.zeros(wavenumbers.shape[0], dtype=bool)
    for i, region in enumerate(region_list):
        mask = _region_mask(wavenumbers, region, is_last=i == len(region_list) - 1)
        if not np.any(mask):
            raise ValueError(f"Region {region.name!r} has no wavenumber points")
        if np.any(used & mask):
            raise ValueError(f"Region {region.name!r} overlaps a previous region")
        used |= mask
        out[region.name] = X[:, mask].astype(np.float32, copy=False)
    return out


def partition_wavenumbers(
    wavenumbers: np.ndarray,
    regions: Iterable[SpectralRegion] = DEFAULT_REGIONS,
) -> OrderedDict[str, np.ndarray]:
    """Return the wavenumber axis for each SREM region."""

    wavenumbers = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    region_list = tuple(regions)
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    for i, region in enumerate(region_list):
        mask = _region_mask(wavenumbers, region, is_last=i == len(region_list) - 1)
        if not np.any(mask):
            raise ValueError(f"Region {region.name!r} has no wavenumber points")
        out[region.name] = wavenumbers[mask]
    return out

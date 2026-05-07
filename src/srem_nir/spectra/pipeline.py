from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

import numpy as np

from srem_nir.spectra.align import align_regions_spline
from srem_nir.spectra.channels import DEFAULT_REGIONS, SpectralRegion, partition_spectrum
from srem_nir.spectra.normalize import normalize_regions_minmax
from srem_nir.spectra.tensorize import stack_regions_mnp, to_torch_channel_first


@dataclass(frozen=True)
class SREMTensorBundle:
    """Container for manuscript and PyTorch tensor views."""

    X_mnp: np.ndarray
    X_pytorch: np.ndarray
    region_names: tuple[str, ...]
    target_length: int


def build_srem_tensor(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    target_length: int | None = None,
    regions: tuple[SpectralRegion, ...] = DEFAULT_REGIONS,
) -> SREMTensorBundle:
    """Build the SREM tensor exactly as described in the manuscript.

    Steps:
    1. partition the preprocessed full spectrum into three physical regions;
    2. independently min-max normalize each region per sample;
    3. align all regions by cubic spline interpolation;
    4. stack as M x N x P and expose PyTorch M x P x N.
    """

    parts = partition_spectrum(X, wavenumbers, regions=regions)
    normed = normalize_regions_minmax(parts)
    aligned = align_regions_spline(normed, target_length=target_length)
    # Cubic interpolation can overshoot slightly near steep bands; SREM's
    # manuscript-level input contract remains [0, 1] after region normalization.
    aligned = OrderedDict((name, np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)) for name, arr in aligned.items())
    X_mnp = stack_regions_mnp(aligned)
    return SREMTensorBundle(
        X_mnp=X_mnp,
        X_pytorch=to_torch_channel_first(X_mnp),
        region_names=tuple(aligned.keys()),
        target_length=int(X_mnp.shape[1]),
    )

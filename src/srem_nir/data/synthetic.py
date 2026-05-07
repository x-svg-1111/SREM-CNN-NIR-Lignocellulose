from __future__ import annotations

from pathlib import Path

import numpy as np


def _gaussian(w: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - float(center)) / float(width)) ** 2)


def make_synthetic_srem_dataset(
    *,
    n_samples: int = 550,
    n_wavenumbers: int = 3001,
    seed: int = 2025,
) -> dict[str, np.ndarray]:
    """Create a synthetic NIR dataset with the same schema as the private data.

    The default shape mirrors the cleaned private export used during method
    development: 550 sample-level spectra and a dense 10000-to-4000 cm^-1 axis
    with 3001 points. Values are synthetic and chemically plausible only for
    smoke testing.
    """

    rng = np.random.default_rng(seed)
    wavenumbers = np.linspace(10000.0, 4000.0, n_wavenumbers, dtype=np.float32)
    base = 0.08 + 0.02 * rng.normal(size=(n_samples, 1))
    slope = rng.normal(0.0, 0.02, size=(n_samples, 1)) * np.linspace(1, -1, n_wavenumbers)
    X = base + slope

    amp_cell = rng.uniform(0.6, 1.4, size=(n_samples, 1))
    amp_hemi = rng.uniform(0.6, 1.4, size=(n_samples, 1))
    amp_lignin = rng.uniform(0.6, 1.4, size=(n_samples, 1))

    X += amp_cell * (0.09 * _gaussian(wavenumbers, 4310, 45) + 0.06 * _gaussian(wavenumbers, 6965, 55) + 0.04 * _gaussian(wavenumbers, 8950, 90))
    X += amp_hemi * (0.07 * _gaussian(wavenumbers, 4220, 55) + 0.05 * _gaussian(wavenumbers, 5840, 70) + 0.045 * _gaussian(wavenumbers, 8840, 85))
    X += amp_lignin * (0.08 * _gaussian(wavenumbers, 4550, 65) + 0.07 * _gaussian(wavenumbers, 6070, 75) + 0.05 * _gaussian(wavenumbers, 6970, 90))
    X += rng.normal(0.0, 0.006, size=X.shape)

    cellulose = 34.0 + 7.0 * amp_cell[:, 0] + 1.0 * amp_lignin[:, 0] + rng.normal(0, 0.8, n_samples)
    hemicellulose = 11.0 + 5.0 * amp_hemi[:, 0] + 0.6 * amp_cell[:, 0] + rng.normal(0, 0.6, n_samples)
    lignin = 13.0 + 7.0 * amp_lignin[:, 0] + 0.8 * amp_hemi[:, 0] + rng.normal(0, 0.7, n_samples)
    y = np.column_stack([cellulose, hemicellulose, lignin]).astype(np.float32)

    crop_names = np.array(["wheat", "maize", "rice", "other"])
    crop_labels = crop_names[rng.integers(0, crop_names.size, size=n_samples)]
    sample_ids = np.arange(1, n_samples + 1, dtype=np.int32)
    return {
        "X": X.astype(np.float32),
        "y": y,
        "wavenumbers": wavenumbers,
        "sample_ids": sample_ids,
        "crop_labels": crop_labels.astype("U16"),
    }


def save_synthetic_npz(path: str | Path, **kwargs: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_srem_dataset(**kwargs)
    np.savez_compressed(path, **data)
    return path

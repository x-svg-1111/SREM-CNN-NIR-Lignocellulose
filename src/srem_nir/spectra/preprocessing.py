from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.signal import savgol_filter

PreprocessMethod = Literal["none", "sg", "snv", "msc"]


@dataclass(frozen=True)
class SGConfig:
    window_length: int = 11
    deriv: int = 1
    polyorder: int = 2


def snv(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32, copy=False)


def savitzky_golay_derivative(X: np.ndarray, cfg: SGConfig) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    window = int(cfg.window_length)
    if window % 2 == 0:
        window += 1
    window = max(5, min(45, window))
    if window <= cfg.polyorder:
        window = cfg.polyorder + 3 + ((cfg.polyorder + 3) % 2 == 0)
    deriv = int(cfg.deriv)
    if deriv not in (1, 2):
        raise ValueError("SG derivative order must be 1 or 2")
    return savgol_filter(X, window_length=window, polyorder=int(cfg.polyorder), deriv=deriv, axis=1, mode="interp").astype(np.float32)


def fit_msc_reference(X_train: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(X_train, dtype=np.float32), axis=0)


def apply_msc(X: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    if X.shape[1] != reference.shape[0]:
        raise ValueError("MSC reference length mismatch")
    A = np.vstack([reference, np.ones_like(reference)]).T
    beta, _, _, _ = np.linalg.lstsq(A, X.T, rcond=None)
    slope = beta[0].astype(np.float32)
    intercept = beta[1].astype(np.float32)
    slope = np.where(np.abs(slope) < eps, 1.0, slope)
    return ((X - intercept[:, None]) / slope[:, None]).astype(np.float32, copy=False)


class SpectralPreprocessor:
    """Train-safe preprocessing wrapper for SG, SNV, and MSC."""

    def __init__(self, method: PreprocessMethod = "none", sg: SGConfig | None = None):
        self.method = method
        self.sg = sg or SGConfig()
        self.msc_reference_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "SpectralPreprocessor":
        if self.method == "msc":
            self.msc_reference_ = fit_msc_reference(X_train)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        method = self.method
        if method == "none":
            return np.asarray(X, dtype=np.float32)
        if method == "sg":
            return savitzky_golay_derivative(X, self.sg)
        if method == "snv":
            return snv(X)
        if method == "msc":
            if self.msc_reference_ is None:
                raise RuntimeError("MSC preprocessor must be fitted on training spectra")
            return apply_msc(X, self.msc_reference_)
        raise ValueError(f"Unsupported preprocessing method: {method!r}")

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


class PreprocessingPipeline:
    """Sequential spectral preprocessing with train-safe MSC fitting."""

    def __init__(self, steps: Iterable[SpectralPreprocessor]):
        self.steps = list(steps)

    def fit(self, X_train: np.ndarray) -> "PreprocessingPipeline":
        X_cur = np.asarray(X_train, dtype=np.float32)
        for step in self.steps:
            step.fit(X_cur)
            X_cur = step.transform(X_cur)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_cur = np.asarray(X, dtype=np.float32)
        for step in self.steps:
            X_cur = step.transform(X_cur)
        return X_cur

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)


def make_preprocessing_pipeline(spec: dict | list[dict] | str | None) -> PreprocessingPipeline:
    """Build preprocessing from compact paper-style specs.

    Examples:
    - "MSC"
    - "SNV"
    - {"method": "sg", "window": 21, "deriv": 1}
    - [{"method": "snv"}, {"method": "sg", "window": 21, "deriv": 1}]
    """

    if spec is None or spec == "none":
        return PreprocessingPipeline([SpectralPreprocessor("none")])
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "msc":
            return PreprocessingPipeline([SpectralPreprocessor("msc")])
        if s == "snv":
            return PreprocessingPipeline([SpectralPreprocessor("snv")])
        if s in {"sg", "sgdiff"}:
            return PreprocessingPipeline([SpectralPreprocessor("sg")])
        raise ValueError(f"Unsupported preprocessing string: {spec!r}")
    if isinstance(spec, dict):
        specs = [spec]
    else:
        specs = list(spec)
    steps: list[SpectralPreprocessor] = []
    for item in specs:
        method = str(item.get("method", "none")).lower()
        if method in {"sgdiff", "sg"}:
            steps.append(
                SpectralPreprocessor(
                    "sg",
                    SGConfig(
                        window_length=int(item.get("window", item.get("w", 11))),
                        deriv=int(item.get("deriv", item.get("d", item.get("ord", 1)))),
                        polyorder=int(item.get("polyorder", 2)),
                    ),
                )
            )
        elif method in {"snv", "msc", "none"}:
            steps.append(SpectralPreprocessor(method))
        else:
            raise ValueError(f"Unsupported preprocessing method: {method!r}")
    return PreprocessingPipeline(steps)

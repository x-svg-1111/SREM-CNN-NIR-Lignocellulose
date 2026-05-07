from __future__ import annotations

import numpy as np
import torch


class _ColumnOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).reshape(-1, 1)


def gradient_explainer_shap_values(
    model: torch.nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Compute Expected Gradients SHAP values via shap.GradientExplainer.

    Inputs must be in PyTorch Conv1d shape `M x P x N`.
    """

    import shap

    device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = _ColumnOutputWrapper(model.to(device).eval()).to(device).eval()
    background_t = torch.as_tensor(background, dtype=torch.float32, device=device)
    samples_t = torch.as_tensor(samples, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(model, background_t)
    values = explainer.shap_values(samples_t)
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def mean_abs_region_importance(shap_values: np.ndarray) -> np.ndarray:
    """Return mean absolute attribution per region channel."""

    values = np.asarray(shap_values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected SHAP values with shape M x P x N, got {values.shape}")
    return np.mean(np.abs(values), axis=(0, 2))


def mean_abs_wavelength_importance(shap_values: np.ndarray) -> np.ndarray:
    """Return mean absolute attribution per aligned wavelength point and channel."""

    values = np.asarray(shap_values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected SHAP values with shape M x P x N, got {values.shape}")
    return np.mean(np.abs(values), axis=0)


def normalize_importance(importance: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize mean absolute SHAP importance by its maximum value."""

    importance = np.asarray(importance, dtype=np.float32)
    denom = max(float(np.max(np.abs(importance))), float(eps))
    return (importance / denom).astype(np.float32, copy=False)


def high_contribution_mask(normalized_importance: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Flag high-contribution points after max normalization."""

    return np.asarray(normalized_importance) >= float(threshold)

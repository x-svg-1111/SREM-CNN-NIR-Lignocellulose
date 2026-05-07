from __future__ import annotations

from itertools import combinations

import numpy as np
import torch


def _predict(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = torch.as_tensor(X[i : i + 512], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def mask_regions(X: np.ndarray, keep: tuple[int, ...], *, baseline: float = 0.0) -> np.ndarray:
    """Keep selected region channels and replace other channels by baseline."""

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected M x P x N tensor, got {X.shape}")
    out = np.full_like(X, fill_value=float(baseline), dtype=np.float32)
    for c in keep:
        out[:, int(c), :] = X[:, int(c), :]
    return out


def pairwise_region_interactions(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    region_names: tuple[str, ...] = ("combination", "first_overtone", "second_overtone"),
    baseline: float = 0.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Inclusion-exclusion interaction decomposition between spectral regions.

    For regions i and j:
    I_ij(x) = f(x_{i,j}) - f(x_i) - f(x_j) + f(x_empty)
    and the reported coupling coefficient is mean(abs(I_ij)).
    """

    device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    X = np.asarray(X, dtype=np.float32)
    n_regions = X.shape[1]
    empty_pred = _predict(model, mask_regions(X, tuple(), baseline=baseline), device)
    matrix = np.zeros((n_regions, n_regions), dtype=np.float32)
    pair_values: dict[str, np.ndarray] = {}
    for i, j in combinations(range(n_regions), 2):
        pred_ij = _predict(model, mask_regions(X, (i, j), baseline=baseline), device)
        pred_i = _predict(model, mask_regions(X, (i,), baseline=baseline), device)
        pred_j = _predict(model, mask_regions(X, (j,), baseline=baseline), device)
        interaction = pred_ij - pred_i - pred_j + empty_pred
        coeff = float(np.mean(np.abs(interaction)))
        matrix[i, j] = coeff
        matrix[j, i] = coeff
        pair_values[f"{region_names[i]}__{region_names[j]}"] = interaction.astype(np.float32)
    return {
        "region_names": region_names,
        "coupling_matrix": matrix,
        "pairwise_interactions": pair_values,
    }


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from srem_nir.metrics import regression_metrics
from srem_nir.spectra.normalize import TargetStandardizer


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 0.001
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 2025
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = torch.as_tensor(X[i : i + 512], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit_regressor(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> dict[str, object]:
    """Train a PyTorch regressor with validation-loss early stopping."""

    set_seed(cfg.seed)
    device = cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    y_scaler = TargetStandardizer().fit(y_train)
    y_train_z = y_scaler.transform(y_train)
    y_val_z = y_scaler.transform(y_val)

    ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train_z, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_z, dtype=torch.float32, device=device)

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).detach().cpu())
        if best_loss - val_loss > float(cfg.min_delta):
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_train = y_scaler.inverse_transform(_predict(model, X_train, device))
    pred_val = y_scaler.inverse_transform(_predict(model, X_val, device))
    return {
        "model": model,
        "target_scaler": y_scaler,
        "best_epoch": best_epoch,
        "train_metrics": regression_metrics(y_train, pred_train),
        "val_metrics": regression_metrics(y_val, pred_val),
        "pred_train": pred_train,
        "pred_val": pred_val,
    }


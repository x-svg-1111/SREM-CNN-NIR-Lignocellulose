#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srem_nir.explain import gradient_explainer_shap_values
from srem_nir.explain.shap_values import mean_abs_region_importance, mean_abs_wavelength_importance, normalize_importance
from srem_nir.models import CNNConfig, SREMCNN
from srem_nir.models.train import TrainConfig, fit_regressor
from srem_nir.paper_config import paper_srem_params
from srem_nir.spectra import build_srem_tensor
from srem_nir.spectra.augmentation import perturb_training_spectra
from srem_nir.spectra.preprocessing import make_preprocessing_pipeline
from srem_nir.split import ks_train_random_val_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_demo" / "synthetic_srem_demo.npz"))
    ap.add_argument("--target", choices=["cellulose", "hemicellulose", "lignin"], default="cellulose")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "shap_demo"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--background-size", type=int, default=32)
    ap.add_argument("--explain-size", type=int, default=16)
    args = ap.parse_args()
    target_idx = {"cellulose": 0, "hemicellulose": 1, "lignin": 2}[args.target]
    with np.load(args.data, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.float32)[:, target_idx]
        wavenumbers = np.asarray(z["wavenumbers"], dtype=np.float32)
    split = ks_train_random_val_test_split(X, seed=2025)
    params = paper_srem_params(args.target)
    preproc = make_preprocessing_pipeline(params["preprocessing"])
    preproc.fit(X[split.train])
    X_pre = preproc.transform(X)
    X_train_aug, y_train_aug = perturb_training_spectra(X_pre[split.train], y[split.train], n_aug=1, seed=2025)
    bundle = build_srem_tensor(X_pre, wavenumbers)
    bundle_train_aug = build_srem_tensor(X_train_aug, wavenumbers, target_length=bundle.target_length)
    Xt = bundle.X_pytorch
    model = SREMCNN(
        CNNConfig(
            input_length=bundle.target_length,
            block_filters=params["block_filters"],
            block_kernels=params["block_kernels"],
            strides=params["strides"],
            se_after_conv=params["se_after_conv"],
            se_reduction=params["se_reduction"],
            dropout=params["dropout"],
            dense=params["dense"],
        )
    )
    fit = fit_regressor(
        model,
        bundle_train_aug.X_pytorch,
        y_train_aug,
        Xt[split.val],
        y[split.val],
        TrainConfig(max_epochs=args.epochs, patience=args.patience, batch_size=int(params["batch_size"]), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]), device="cpu"),
    )
    model = fit["model"]
    background = Xt[split.train[: min(int(args.background_size), len(split.train))]]
    samples = Xt[split.val[: min(int(args.explain_size), len(split.val))]]
    values = gradient_explainer_shap_values(model, background, samples, device="cpu")
    region_importance = mean_abs_region_importance(values)
    wavelength_importance = mean_abs_wavelength_importance(values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_values.npy", values)
    np.save(out_dir / "mean_abs_region_importance.npy", region_importance)
    np.save(out_dir / "mean_abs_wavelength_importance.npy", wavelength_importance)
    np.save(out_dir / "normalized_region_importance.npy", normalize_importance(region_importance))
    np.save(out_dir / "normalized_wavelength_importance.npy", normalize_importance(wavelength_importance))
    print(f"wrote SHAP arrays to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srem_nir.metrics import regression_metrics
from srem_nir.models import PlainCNN1D
from srem_nir.models.train import TrainConfig, fit_regressor
from srem_nir.paper_config import paper_baseline_params
from srem_nir.spectra.augmentation import perturb_training_spectra
from srem_nir.spectra.preprocessing import make_preprocessing_pipeline
from srem_nir.split import ks_train_random_val_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_demo" / "synthetic_srem_demo.npz"))
    ap.add_argument("--target", choices=["cellulose", "hemicellulose", "lignin"], default="cellulose")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--augment", type=int, default=1, help="number of perturbed copies for the preprocessed training set")
    args = ap.parse_args()

    target_idx = {"cellulose": 0, "hemicellulose": 1, "lignin": 2}[args.target]
    with np.load(args.data, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.float32)[:, target_idx]

    split = ks_train_random_val_test_split(X, seed=2025)
    params = paper_baseline_params(args.target, "cnn")
    preproc = make_preprocessing_pipeline(params.pop("preprocessing"))
    preproc.fit(X[split.train])
    X_pre = preproc.transform(X)
    X_train_aug, y_train_aug = perturb_training_spectra(
        X_pre[split.train],
        y[split.train],
        n_aug=int(args.augment),
        seed=2025,
    )

    model = PlainCNN1D(
        input_length=X_pre.shape[1],
        filters=params["filters"],
        kernels=params["kernels"],
        strides=params["strides"],
        se_after_conv=params["se_after_conv"],
        dense=params["dense"],
        dropout=params["dropout"],
    )
    fit = fit_regressor(
        model,
        X_train_aug[:, None, :],
        y_train_aug,
        X_pre[split.val, None, :],
        y[split.val],
        TrainConfig(
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=int(params["batch_size"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            device="cpu",
        ),
    )
    model = fit["model"]
    pred_test = fit["target_scaler"].inverse_transform(
        __import__("srem_nir.models.train", fromlist=["_predict"])._predict(model, X_pre[split.test, None, :], "cpu")
    )
    print(f"1D-CNN val RMSE={fit['val_metrics']['rmse']:.4f} test RMSE={regression_metrics(y[split.test], pred_test)['rmse']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srem_nir.baselines import fit_plsr, fit_svr, fit_xgboost
from srem_nir.paper_config import paper_baseline_params
from srem_nir.spectra.preprocessing import make_preprocessing_pipeline
from srem_nir.split import ks_train_random_val_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data_demo" / "synthetic_srem_demo.npz"))
    ap.add_argument("--target", choices=["cellulose", "hemicellulose", "lignin"], default="cellulose")
    args = ap.parse_args()
    target_idx = {"cellulose": 0, "hemicellulose": 1, "lignin": 2}[args.target]
    with np.load(args.data, allow_pickle=True) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.float32)[:, target_idx]
    split = ks_train_random_val_test_split(X, seed=2025)
    def prep_for(model_name: str):
        params = paper_baseline_params(args.target, model_name)
        preproc_spec = params.pop("preprocessing")
        pipe = make_preprocessing_pipeline(preproc_spec)
        pipe.fit(X[split.train])
        return params, pipe.transform(X[split.train]), pipe.transform(X[split.val]), pipe.transform(X[split.test])

    plsr_params, Xtr_pls, Xva_pls, Xte_pls = prep_for("plsr")
    svr_params, Xtr_svr, Xva_svr, Xte_svr = prep_for("svr")
    xgb_params, Xtr_xgb, Xva_xgb, Xte_xgb = prep_for("xgboost")
    rows = [
        fit_plsr(Xtr_pls, y[split.train], Xva_pls, y[split.val], X_test=Xte_pls, y_test=y[split.test], **plsr_params),
        fit_svr(Xtr_svr, y[split.train], Xva_svr, y[split.val], X_test=Xte_svr, y_test=y[split.test], **svr_params),
        fit_xgboost(Xtr_xgb, y[split.train], Xva_xgb, y[split.val], X_test=Xte_xgb, y_test=y[split.test], **xgb_params),
    ]
    for row in rows:
        print(f"{row.name:8s} val RMSE={row.val_metrics['rmse']:.4f} test RMSE={row.test_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()

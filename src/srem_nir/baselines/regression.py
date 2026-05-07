from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

from srem_nir.metrics import regression_metrics


@dataclass(frozen=True)
class BaselineResult:
    name: str
    model: object
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None


def _result(name: str, model: object, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray | None, y_test: np.ndarray | None) -> BaselineResult:
    pred_train = model.predict(X_train).reshape(-1)
    pred_val = model.predict(X_val).reshape(-1)
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_metrics = regression_metrics(y_test, model.predict(X_test).reshape(-1))
    return BaselineResult(
        name=name,
        model=model,
        train_metrics=regression_metrics(y_train, pred_train),
        val_metrics=regression_metrics(y_val, pred_val),
        test_metrics=test_metrics,
    )


def fit_plsr(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, n_components: int = 10, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> BaselineResult:
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("plsr", PLSRegression(n_components=int(n_components), scale=False)),
    ])
    model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    model.fit(X_train, y_train)
    return _result("PLSR", model, X_train, y_train, X_val, y_val, X_test, y_test)


def fit_svr(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, kernel: str = "linear", C: float = 10.0, gamma: str | float = "scale", epsilon: float = 0.1, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> BaselineResult:
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=float(C), gamma=gamma, epsilon=float(epsilon), kernel=str(kernel))),
    ])
    model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    model.fit(X_train, np.asarray(y_train).reshape(-1))
    return _result("SVR", model, X_train, y_train, X_val, y_val, X_test, y_test)


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, *, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05, subsample: float = 0.9, colsample_bytree: float = 0.9, min_child_weight: float = 1.0, reg_alpha: float = 0.0, reg_lambda: float = 1.0, gamma: float = 0.0, X_test: np.ndarray | None = None, y_test: np.ndarray | None = None) -> BaselineResult:
    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=float(min_child_weight),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        gamma=float(gamma),
        random_state=2025,
        n_jobs=2,
    )
    model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    model.fit(X_train, np.asarray(y_train).reshape(-1), verbose=False)
    return _result("XGBoost", model, X_train, y_train, X_val, y_val, X_test, y_test)

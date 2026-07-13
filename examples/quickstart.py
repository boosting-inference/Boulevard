"""Small release quickstart for Boulevard's sklearn-compatible estimators.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

import os
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _signal_1d(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * X[:, 0])


def _signal_additive(X: np.ndarray) -> np.ndarray:
    return (
        np.sin(2 * np.pi * X[:, 0])
        + 0.8 * (X[:, 1] - 0.5)
        + 0.4 * np.cos(4 * np.pi * X[:, 2])
    )


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(target, pred)))


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def _split(
    X: np.ndarray,
    y: np.ndarray,
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_holdout, y_train, y_holdout, _signal_train, signal_holdout = (
        train_test_split(X, y, signal, test_size=0.4, random_state=0)
    )
    X_calib, X_test, y_calib, y_test, _signal_calib, signal_test = train_test_split(
        X_holdout,
        y_holdout,
        signal_holdout,
        test_size=0.5,
        random_state=1,
    )
    return X_train, X_calib, X_test, y_train, y_calib, y_test, signal_test


def _summarize(
    name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    signal_test: np.ndarray,
) -> None:
    start = time.perf_counter()
    pred = model.predict(X_test)
    pred_seconds = time.perf_counter() - start

    ci_lower, ci_upper, _ = model.predict_intervals(
        X_test,
        level=0.95,
        mode="confidence",
    )
    pi_lower, pi_upper, _ = model.predict_intervals(
        X_test,
        level=0.95,
        mode="prediction",
    )

    print("")
    print(name)
    print(f"  prediction seconds: {pred_seconds:.4f}")
    print(f"  RMSE vs signal: {_rmse(signal_test, pred):.4f}")
    print(f"  RMSE vs noisy y: {_rmse(y_test, pred):.4f}")
    print(
        f"  95% CI coverage vs signal: "
        f"{_coverage(signal_test, ci_lower, ci_upper):.3f}"
    )
    print(f"  95% PI coverage vs noisy y: {_coverage(y_test, pi_lower, pi_upper):.3f}")
    print(f"  median CI width: {float(np.median(ci_upper - ci_lower)):.4f}")
    print(f"  sigma_hat2: {model.sigma_hat2_:.6f}")


def _fit_and_report(
    name: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    signal_test: np.ndarray,
) -> None:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model.prepare_inference(X_calib, y_calib)
    prep_seconds = time.perf_counter() - start

    print("")
    print(f"{name} fit seconds: {fit_seconds:.4f}")
    print(f"{name} prepare_inference seconds: {prep_seconds:.4f}")
    _summarize(name, model, X_test, y_test, signal_test)


def main() -> None:
    rng = np.random.default_rng(0)

    X_1d = rng.uniform(0.0, 1.0, size=(1000, 1))
    signal_1d = _signal_1d(X_1d)
    y_1d = signal_1d + rng.normal(scale=0.2, size=X_1d.shape[0])
    X_train, X_calib, X_test, y_train, y_calib, y_test, signal_test = _split(
        X_1d,
        y_1d,
        signal_1d,
    )

    print("Boulevard sklearn-compatible quickstart")
    print("")
    print("Low-dimensional signal: use DropoutBooster or ParallelBooster")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )

    _fit_and_report(
        "DropoutBooster",
        bd.DropoutBooster(
            max_iter=700,
            learning_rate=0.8,
            dropout_rate=0.1,
            subsample_rate=0.8,
            max_depth=6,
            max_leaf_nodes=64,
            min_samples_leaf=2,
            max_bins=64,
            early_stopping=False,
            random_state=0,
        ),
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        signal_test,
    )
    _fit_and_report(
        "ParallelBooster",
        bd.ParallelBooster(
            n_rounds=70,
            trees_per_round=6,
            subsample_rate=0.8,
            max_depth=8,
            max_leaf_nodes=128,
            min_samples_leaf=8,
            max_bins=64,
            drop_first_round=True,
            early_stopping=False,
            n_jobs=1,
            random_state=0,
        ),
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        signal_test,
    )

    X_add = rng.uniform(0.0, 1.0, size=(2500, 3))
    signal_add = _signal_additive(X_add)
    y_add = signal_add + rng.normal(scale=0.25, size=X_add.shape[0])
    X_train, X_calib, X_test, y_train, y_calib, y_test, signal_test = _split(
        X_add,
        y_add,
        signal_add,
    )

    print("")
    print("Higher-dimensional additive signal: start with ExplainableBooster")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )

    _fit_and_report(
        "ExplainableBooster",
        bd.ExplainableBooster(
            max_rounds=160,
            max_bins=32,
            learning_rate=0.6,
            subsample_rate=1.0,
            warmup_rounds=10,
            max_depth=4,
            min_samples_leaf=8,
            random_state=0,
        ),
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        signal_test,
    )


if __name__ == "__main__":
    main()

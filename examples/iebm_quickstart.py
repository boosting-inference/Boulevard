"""Minimal IEBM API example.

Run from the repository root with:

    python examples/iebm_quickstart.py

The current IEBM implementation is a first sklearn-style training backend:
numeric features, squared-error regression, main effects only, and experimental
bin-space intervals.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(target, pred)))


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples = 2000

    X = rng.uniform(0.0, 1.0, size=(n_samples, 3))
    signal = (
        np.sin(2 * np.pi * X[:, 0])
        + 0.75 * (X[:, 1] - 0.5)
        + 0.25 * np.cos(4 * np.pi * X[:, 2])
    )
    y = signal + rng.normal(scale=0.25, size=n_samples)

    X_train, X_holdout, y_train, y_holdout, signal_train, signal_holdout = (
        train_test_split(
            X,
            y,
            signal,
            test_size=0.35,
            random_state=0,
        )
    )
    X_calib, X_test, y_calib, y_test, _signal_calib, signal_test = train_test_split(
        X_holdout,
        y_holdout,
        signal_holdout,
        test_size=0.5,
        random_state=1,
    )

    model = bd.IEBMRegressor(
        max_rounds=80,
        max_bins=64,
        learning_rate=1.0,
        subsample_rate=0.8,
        truncation=10.0,
        max_depth=1,
        min_samples_leaf=10,
        random_state=0,
    )
    model.fit(X_train, y_train)
    model.prepare_inference(X_calib, y_calib)
    pi_scale = model.calibrate_intervals(
        X_calib,
        y_calib,
        level=0.95,
        mode="prediction",
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_bins = model.apply_bins(X_test[:5])
    ci_lower, ci_upper, ci_pred = model.predict_intervals(
        X_test,
        level=0.95,
        mode="confidence",
    )
    pi_lower, pi_upper, _ = model.predict_intervals(
        X_test,
        level=0.95,
        mode="prediction",
    )
    pi_coverage = float(np.mean((y_test >= pi_lower) & (y_test <= pi_upper)))

    print("IEBM quickstart")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print("scope: numeric squared-error main effects; intervals are experimental")
    print(f"train RMSE vs signal: {_rmse(signal_train, train_pred):.4f}")
    print(f"test RMSE vs signal: {_rmse(signal_test, test_pred):.4f}")
    print(f"test RMSE vs noisy y: {_rmse(y_test, test_pred):.4f}")
    print(f"95% CI output shape: {ci_lower.shape}")
    print(f"mean 95% CI width: {float(np.mean(ci_upper - ci_lower)):.4f}")
    print(f"CI prediction matches predict: {bool(np.allclose(ci_pred, test_pred))}")
    print(f"prediction interval calibration scale: {pi_scale:.4f}")
    print(f"calibrated 95% PI coverage vs noisy y: {pi_coverage:.3f}")
    print(f"sigma_hat2: {model.sigma_hat2_:.4f}")
    print(f"total bins: {model.fit_diagnostics_['total_bins']}")
    print(f"fit seconds: {model.fit_diagnostics_['total_seconds']:.4f}")
    print(f"first five binned test rows shape: {test_bins.shape}")


if __name__ == "__main__":
    main()

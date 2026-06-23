"""Minimal IEBM API example.

Run from the repository root with:

    python examples/iebm_quickstart.py

The current IEBM implementation is a first sklearn-style training backend:
numeric features, squared-error regression, and main effects only.  Interval
methods are intentionally not exposed yet.
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

    X_train, X_test, y_train, y_test, signal_train, signal_test = train_test_split(
        X,
        y,
        signal,
        test_size=0.35,
        random_state=0,
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

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_bins = model.apply_bins(X_test[:5])

    print("IEBM quickstart")
    print(f"train/test sizes: {X_train.shape[0]}/{X_test.shape[0]}")
    print("scope: numeric squared-error main effects; no intervals yet")
    print(f"train RMSE vs signal: {_rmse(signal_train, train_pred):.4f}")
    print(f"test RMSE vs signal: {_rmse(signal_test, test_pred):.4f}")
    print(f"test RMSE vs noisy y: {_rmse(y_test, test_pred):.4f}")
    print(f"sigma: {model.sigma_:.4f}")
    print(f"total bins: {model.fit_diagnostics_['total_bins']}")
    print(f"fit seconds: {model.fit_diagnostics_['total_seconds']:.4f}")
    print(f"first five binned test rows shape: {test_bins.shape}")


if __name__ == "__main__":
    main()

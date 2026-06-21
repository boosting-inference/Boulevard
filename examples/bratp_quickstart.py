"""Common BRAT-P histogram estimator workflow.

Run from the repository root with:

    python examples/bratp_quickstart.py

The example is synthetic so that we can report coverage against the known
regression signal. Real applications will only have ``X`` and noisy ``y``.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of targets inside the interval."""
    return float(np.mean((target >= lower) & (target <= upper)))


def main() -> None:
    rng = np.random.default_rng(0)

    # Synthetic regression problem. ``signal`` is the noiseless regression
    # function; it is used only to evaluate this example.
    n_samples = 3000
    noise_std = 0.25
    X = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    x = X[:, 0]
    signal = np.sin(2 * np.pi * x) + 0.35 * np.sin(6 * np.pi * x)
    y = signal + rng.normal(scale=noise_std, size=n_samples)

    # Use train/calibration/test splits. The calibration split estimates the
    # residual variance used by asymptotic intervals.
    X_train, X_holdout, y_train, y_holdout, _signal_train, signal_holdout = (
        train_test_split(
            X,
            y,
            signal,
            test_size=0.4,
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

    # BRAT-P has a round/slot structure. The total number of fitted trees is
    # n_rounds * trees_per_round.
    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=60,
        trees_per_round=4,
        subsample_rate=0.8,
        max_depth=10,
        max_leaf_nodes=256,
        min_samples_leaf=5,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    pred = model.predict(X_test)
    pred_seconds = time.perf_counter() - pred_start

    # Interval calls prepare the inference cache automatically. Passing
    # X_calib/y_calib makes the residual variance estimate come from held-out
    # calibration residuals instead of training residuals.
    interval_start = time.perf_counter()
    ci_lower, ci_upper = model.confidence_interval(
        X_test,
        alpha=0.05,
        X_calib=X_calib,
        y_calib=y_calib,
    )
    pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
    interval_seconds = time.perf_counter() - interval_start

    rmse_vs_y = float(np.sqrt(mean_squared_error(y_test, pred)))
    rmse_vs_signal = float(np.sqrt(mean_squared_error(signal_test, pred)))
    ci_coverage = _coverage(signal_test, ci_lower, ci_upper)
    pi_coverage = _coverage(y_test, pi_lower, pi_upper)
    ci_width = ci_upper - ci_lower
    pi_width = pi_upper - pi_lower
    abs_signal_error = np.abs(pred - signal_test)

    print("BRAT-P histogram quickstart")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print(
        "rounds/trees per round/total trees: "
        f"{model.n_rounds}/{model.trees_per_round}/{len(model._predictors)}"
    )
    print(f"RMSE vs noisy y: {rmse_vs_y:.4f}")
    print(f"RMSE vs true signal: {rmse_vs_signal:.4f}")
    print(f"median abs error vs true signal: {float(np.median(abs_signal_error)):.4f}")
    print(f"95% CI coverage vs true signal: {ci_coverage:.3f}")
    print(f"95% PI coverage vs noisy y: {pi_coverage:.3f}")
    print(f"median 95% CI half-width: {float(np.median(ci_width / 2)):.4f}")
    print(f"median 95% CI width: {float(np.median(ci_width)):.4f}")
    print(f"median 95% PI width: {float(np.median(pi_width)):.4f}")
    print(f"fit wall-clock seconds: {fit_seconds:.4f}")
    print(f"prediction wall-clock seconds: {pred_seconds:.4f}")
    print(f"interval wall-clock seconds: {interval_seconds:.4f}")
    print(f"sigma_hat2: {model.sigma_hat2_:.6f}")
    print("first five predictions and intervals:")
    for row in range(5):
        print(
            f"  x={X_test[row, 0]:.3f}, "
            f"pred={pred[row]: .4f}, "
            f"CI=({ci_lower[row]: .4f}, {ci_upper[row]: .4f}), "
            f"PI=({pi_lower[row]: .4f}, {pi_upper[row]: .4f})"
        )


if __name__ == "__main__":
    main()

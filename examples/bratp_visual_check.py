"""Visual smoke check and interval diagnostic for BRAT-P histogram intervals.

Run from the repository root with:

    python examples/bratp_visual_check.py --output /tmp/bratp_visual_check.png

If ``--output`` is omitted, the script opens an interactive matplotlib window.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(target, pred)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help=(
            "Optional path for saving the plot. If omitted, show an "
            "interactive window."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    noise_std = 1.0
    X = np.linspace(0.0, 1.0, 3000).reshape(-1, 1)
    x = X[:, 0]
    truth = np.sin(2 * np.pi * x) + 0.35 * np.sin(6 * np.pi * x)
    y = truth + rng.normal(scale=noise_std, size=X.shape[0])

    train_idx, holdout_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=0.4,
        random_state=0,
    )
    calib_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        random_state=1,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    truth_test = truth[test_idx]

    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=200,
        trees_per_round=2,
        subsample_rate=0.8,
        max_depth=10,
        max_leaf_nodes=256,
        min_samples_leaf=5,
        max_bins=63,
        early_stopping=False,
        random_state=0,
    )
    hgb_max_iter = model.n_rounds * model.trees_per_round
    vanilla_hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=hgb_max_iter,
        learning_rate=0.06,
        max_depth=model.max_depth,
        max_leaf_nodes=model.max_leaf_nodes,
        min_samples_leaf=model.min_samples_leaf,
        max_bins=255,
        l2_regularization=model.l2_regularization,
        early_stopping=False,
        random_state=0,
    )

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    hgb_fit_start = time.perf_counter()
    vanilla_hgb.fit(X_train, y_train)
    hgb_fit_seconds = time.perf_counter() - hgb_fit_start

    inference_start = time.perf_counter()
    model.prepare_inference(X_calib, y_calib)
    inference_seconds = time.perf_counter() - inference_start

    pred_start = time.perf_counter()
    pred = model.predict(X)
    test_pred = model.predict(X_test)
    pred_seconds = time.perf_counter() - pred_start

    hgb_pred_start = time.perf_counter()
    hgb_pred = vanilla_hgb.predict(X)
    hgb_test_pred = vanilla_hgb.predict(X_test)
    hgb_pred_seconds = time.perf_counter() - hgb_pred_start

    interval_start = time.perf_counter()
    ci_lower, ci_upper = model.confidence_interval(X, alpha=0.05)
    pi_lower, pi_upper = model.prediction_interval(X, alpha=0.05)
    ri_lower, ri_upper = model.reproduction_interval(X, alpha=0.05)
    interval_seconds = time.perf_counter() - interval_start

    test_interval_start = time.perf_counter()
    test_ci_lower, test_ci_upper = model.confidence_interval(X_test, alpha=0.05)
    test_pi_lower, test_pi_upper = model.prediction_interval(X_test, alpha=0.05)
    test_interval_seconds = time.perf_counter() - test_interval_start

    norm_start = time.perf_counter()
    r_norm = model.weight_norms(X)
    test_r_norm = model.weight_norms(X_test)
    norm_seconds = time.perf_counter() - norm_start

    ci_width = ci_upper - ci_lower
    pi_width = pi_upper - pi_lower
    ri_width = ri_upper - ri_lower
    test_ci_width = test_ci_upper - test_ci_lower
    abs_grid_error = np.abs(pred - truth)
    hgb_abs_grid_error = np.abs(hgb_pred - truth)
    test_abs_error = np.abs(test_pred - truth_test)
    ci_half_width = np.maximum(test_ci_width / 2, np.finfo(float).eps)
    ci_error_ratio = test_abs_error / ci_half_width

    ci_grid_coverage = _coverage(truth, ci_lower, ci_upper)
    ci_test_coverage = _coverage(truth_test, test_ci_lower, test_ci_upper)
    pi_test_coverage = _coverage(y_test, test_pi_lower, test_pi_upper)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 0.9])
    ax_ci = fig.add_subplot(grid[0, 0])
    ax_pi = fig.add_subplot(grid[0, 1])
    ax_width = fig.add_subplot(grid[1, 0])
    ax_norm = fig.add_subplot(grid[1, 1])
    ax_error = fig.add_subplot(grid[2, 0])
    ax_resid = fig.add_subplot(grid[2, 1])

    ax_ci.scatter(X_train[:, 0], y_train, s=7, alpha=0.15, label="train")
    ax_ci.plot(X[:, 0], truth, color="black", linewidth=2, label="truth")
    ax_ci.plot(X[:, 0], pred, color="#1f77b4", linewidth=2, label="BRAT-P")
    ax_ci.plot(
        X[:, 0],
        hgb_pred,
        color="0.35",
        linewidth=1.6,
        linestyle="--",
        label=f"vanilla HGBR ({hgb_max_iter} trees)",
    )
    ax_ci.fill_between(
        X[:, 0],
        ci_lower,
        ci_upper,
        color="#2ca02c",
        alpha=0.18,
        label="95% CI",
    )
    ax_ci.set_title(f"Confidence interval, grid coverage {ci_grid_coverage:.1%}")
    ax_ci.set_xlabel("x")
    ax_ci.set_ylabel("y")
    ax_ci.legend(loc="best", fontsize=8)

    ax_pi.scatter(X_test[:, 0], y_test, s=10, alpha=0.22, label="test y")
    ax_pi.plot(X[:, 0], truth, color="black", linewidth=2, label="truth")
    ax_pi.plot(X[:, 0], pred, color="#1f77b4", linewidth=2, label="BRAT-P")
    ax_pi.plot(
        X[:, 0],
        hgb_pred,
        color="0.35",
        linewidth=1.6,
        linestyle="--",
        label=f"vanilla HGBR ({hgb_max_iter} trees)",
    )
    ax_pi.fill_between(
        X[:, 0],
        pi_lower,
        pi_upper,
        color="#ff7f0e",
        alpha=0.16,
        label="95% PI",
    )
    ax_pi.set_title(f"Prediction interval, test y coverage {pi_test_coverage:.1%}")
    ax_pi.set_xlabel("x")
    ax_pi.set_ylabel("y")
    ax_pi.legend(loc="best", fontsize=8)

    ax_width.plot(X[:, 0], ci_width, label="CI width", color="#2ca02c")
    ax_width.plot(X[:, 0], pi_width, label="PI width", color="#ff7f0e")
    ax_width.plot(X[:, 0], ri_width, label="RI width", color="#9467bd")
    ax_width.set_title("Interval widths")
    ax_width.set_xlabel("x")
    ax_width.set_ylabel("width")
    ax_width.legend(loc="best", fontsize=8)

    ax_norm.plot(X[:, 0], r_norm, color="#d62728", label="grid ||r_n(x)||")
    ax_norm.scatter(
        X_test[:, 0],
        test_r_norm,
        s=8,
        alpha=0.25,
        color="#8c564b",
        label="test ||r_n(x)||",
    )
    ax_norm.set_title("BRAT-P kernel weight norm")
    ax_norm.set_xlabel("x")
    ax_norm.set_ylabel("norm")
    ax_norm.legend(loc="best", fontsize=8)

    ax_error.plot(X[:, 0], abs_grid_error, color="#1f77b4", label="|pred-truth|")
    ax_error.plot(
        X[:, 0],
        hgb_abs_grid_error,
        color="0.35",
        linestyle="--",
        label="HGBR |pred-truth|",
    )
    ax_error.plot(
        X[:, 0],
        ci_width / 2,
        color="#2ca02c",
        label="CI half-width",
    )
    ax_error.set_title("Signal error vs CI half-width")
    ax_error.set_xlabel("x")
    ax_error.set_ylabel("absolute value")
    ax_error.legend(loc="best", fontsize=8)

    calib_residuals = y_calib - model.predict(X_calib)
    ax_resid.hist(calib_residuals, bins=35, color="#4c78a8", alpha=0.82)
    ax_resid.axvline(0.0, color="black", linewidth=1)
    ax_resid.set_title("Calibration residuals")
    ax_resid.set_xlabel("residual")
    ax_resid.set_ylabel("count")

    fig.suptitle("BRAT-P histogram visual check", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output:
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()

    print(f"BRAT-P grid RMSE vs truth: {_rmse(truth, pred):.6f}")
    print(f"BRAT-P test RMSE vs noisy y: {_rmse(y_test, test_pred):.6f}")
    print(f"BRAT-P test RMSE vs truth: {_rmse(truth_test, test_pred):.6f}")
    print(f"vanilla HGBR max_iter/tree budget: {hgb_max_iter}")
    print(f"vanilla HGBR grid RMSE vs truth: {_rmse(truth, hgb_pred):.6f}")
    print(f"vanilla HGBR test RMSE vs noisy y: {_rmse(y_test, hgb_test_pred):.6f}")
    print(f"vanilla HGBR test RMSE vs truth: {_rmse(truth_test, hgb_test_pred):.6f}")
    print(f"BRAT-P training wall-clock seconds: {fit_seconds:.6f}")
    print(f"vanilla HGBR training wall-clock seconds: {hgb_fit_seconds:.6f}")
    print(f"BRAT-P inference prep wall-clock seconds: {inference_seconds:.6f}")
    print(f"BRAT-P prediction wall-clock seconds: {pred_seconds:.6f}")
    print(f"vanilla HGBR prediction wall-clock seconds: {hgb_pred_seconds:.6f}")
    print(f"BRAT-P grid interval wall-clock seconds: {interval_seconds:.6f}")
    print(f"BRAT-P test interval wall-clock seconds: {test_interval_seconds:.6f}")
    print(f"BRAT-P weight norm wall-clock seconds: {norm_seconds:.6f}")
    print(f"BRAT-P sigma_hat2: {model.sigma_hat2_:.6f}")
    print(f"BRAT-P CI truth coverage on grid: {ci_grid_coverage:.6f}")
    print(f"BRAT-P CI truth coverage on test split: {ci_test_coverage:.6f}")
    print(f"BRAT-P PI y coverage on test split: {pi_test_coverage:.6f}")
    print(
        "BRAT-P abs signal error min/median/max on test split: "
        f"{np.min(test_abs_error):.6f}/"
        f"{np.median(test_abs_error):.6f}/"
        f"{np.max(test_abs_error):.6f}"
    )
    print(
        "BRAT-P CI half-width min/median/max on test split: "
        f"{np.min(ci_half_width):.6f}/"
        f"{np.median(ci_half_width):.6f}/"
        f"{np.max(ci_half_width):.6f}"
    )
    print(
        "BRAT-P |pred-truth|/CI half-width quantiles q50/q90/q95/max: "
        f"{np.quantile(ci_error_ratio, [0.5, 0.9, 0.95, 1.0])}"
    )
    print(
        "BRAT-P weight norm min/median/max on test split: "
        f"{np.min(test_r_norm):.6f}/"
        f"{np.median(test_r_norm):.6f}/"
        f"{np.max(test_r_norm):.6f}"
    )


if __name__ == "__main__":
    main()

"""API-ready BRAT-D and BRAT-P histogram estimator demo.

Run from the repository root with:

    python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png

If ``--output`` is omitted, the script opens an interactive matplotlib window.
The data are synthetic so this example can report coverage against the known
regression signal. Real applications only have ``X`` and noisy ``y``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd


@dataclass
class IntervalResult:
    name: str
    color: str
    model: object
    fit_seconds: float
    inference_seconds: float
    grid_prediction_seconds: float
    test_prediction_seconds: float
    grid_interval_seconds: float
    test_interval_seconds: float
    grid_weight_norm_seconds: float
    test_weight_norm_seconds: float
    grid_pred: np.ndarray
    test_pred: np.ndarray
    grid_ci_lower: np.ndarray
    grid_ci_upper: np.ndarray
    grid_pi_lower: np.ndarray
    grid_pi_upper: np.ndarray
    test_ci_lower: np.ndarray
    test_ci_upper: np.ndarray
    test_pi_lower: np.ndarray
    test_pi_upper: np.ndarray
    grid_weight_norms: np.ndarray
    test_weight_norms: np.ndarray


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(target, pred)))


def _format_quantiles(values: np.ndarray) -> str:
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return (
        f"q05={q05:.4f}, q25={q25:.4f}, q50={q50:.4f}, "
        f"q75={q75:.4f}, q95={q95:.4f}"
    )


def _fit_interval_model(
    *,
    name: str,
    color: str,
    model: object,
    X_grid: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
) -> IntervalResult:
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    model.prepare_inference(X_calib, y_calib)
    inference_seconds = time.perf_counter() - inference_start

    prediction_start = time.perf_counter()
    grid_pred = model.predict(X_grid)
    grid_prediction_seconds = time.perf_counter() - prediction_start

    prediction_start = time.perf_counter()
    test_pred = model.predict(X_test)
    test_prediction_seconds = time.perf_counter() - prediction_start

    interval_start = time.perf_counter()
    grid_ci_lower, grid_ci_upper = model.confidence_interval(X_grid, alpha=0.05)
    grid_pi_lower, grid_pi_upper = model.prediction_interval(X_grid, alpha=0.05)
    grid_interval_seconds = time.perf_counter() - interval_start

    interval_start = time.perf_counter()
    test_ci_lower, test_ci_upper = model.confidence_interval(X_test, alpha=0.05)
    test_pi_lower, test_pi_upper = model.prediction_interval(X_test, alpha=0.05)
    test_interval_seconds = time.perf_counter() - interval_start

    norm_start = time.perf_counter()
    grid_weight_norms = model.weight_norms(X_grid)
    grid_weight_norm_seconds = time.perf_counter() - norm_start

    norm_start = time.perf_counter()
    test_weight_norms = model.weight_norms(X_test)
    test_weight_norm_seconds = time.perf_counter() - norm_start

    return IntervalResult(
        name=name,
        color=color,
        model=model,
        fit_seconds=fit_seconds,
        inference_seconds=inference_seconds,
        grid_prediction_seconds=grid_prediction_seconds,
        test_prediction_seconds=test_prediction_seconds,
        grid_interval_seconds=grid_interval_seconds,
        test_interval_seconds=test_interval_seconds,
        grid_weight_norm_seconds=grid_weight_norm_seconds,
        test_weight_norm_seconds=test_weight_norm_seconds,
        grid_pred=grid_pred,
        test_pred=test_pred,
        grid_ci_lower=grid_ci_lower,
        grid_ci_upper=grid_ci_upper,
        grid_pi_lower=grid_pi_lower,
        grid_pi_upper=grid_pi_upper,
        test_ci_lower=test_ci_lower,
        test_ci_upper=test_ci_upper,
        test_pi_lower=test_pi_lower,
        test_pi_upper=test_pi_upper,
        grid_weight_norms=grid_weight_norms,
        test_weight_norms=test_weight_norms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help=(
            "Optional path for saving the plot. If omitted, show an "
            "interactive window."
        ),
    )
    parser.add_argument(
        "--bratp-n-jobs",
        type=int,
        default=1,
        help=(
            "Number of joblib threads for fitting BRAT-P slot trees within "
            "each round. Use values greater than 1 to benchmark the optional "
            "parallel fit path."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    n_samples = 3000
    noise_std = 1
    X = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    x = X[:, 0]
    signal = np.sin(2 * np.pi * x) + 0.35 * np.sin(6 * np.pi * x)
    y = signal + rng.normal(scale=noise_std, size=n_samples)

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

    total_tree_budget = 1000
    brat_d = bd.BRATDHistGradientBoostingRegressor(
        max_iter=total_tree_budget,
        learning_rate=0.45,
        dropout_rate=0.3,
        subsample_rate=0.8,
        nystrom_subsample_rate=0.8,
        max_depth=10,
        max_leaf_nodes=512,
        min_samples_leaf=5,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )
    brat_p = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=100,
        trees_per_round=total_tree_budget // 100,
        subsample_rate=0.8,
        nystrom_subsample_rate=0.8,
        max_depth=10,
        max_leaf_nodes=256,
        min_samples_leaf=5,
        max_bins=255,
        early_stopping=False,
        random_state=0,
        n_jobs=args.bratp_n_jobs,
    )
    vanilla_hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=total_tree_budget,
        learning_rate=0.06,
        max_depth=10,
        max_leaf_nodes=256,
        min_samples_leaf=5,
        max_bins=255,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=0,
    )

    results = [
        _fit_interval_model(
            name="BRAT-D",
            color="#1f77b4",
            model=brat_d,
            X_grid=X,
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
        ),
        _fit_interval_model(
            name="BRAT-P",
            color="#d62728",
            model=brat_p,
            X_grid=X,
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
        ),
    ]

    hgb_fit_start = time.perf_counter()
    vanilla_hgb.fit(X_train, y_train)
    hgb_fit_seconds = time.perf_counter() - hgb_fit_start

    hgb_pred_start = time.perf_counter()
    hgb_grid_pred = vanilla_hgb.predict(X)
    hgb_grid_pred_seconds = time.perf_counter() - hgb_pred_start

    hgb_pred_start = time.perf_counter()
    hgb_test_pred = vanilla_hgb.predict(X_test)
    hgb_test_pred_seconds = time.perf_counter() - hgb_pred_start

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.9, 0.9])
    ax_fit = fig.add_subplot(grid[0, 0])
    ax_pi = fig.add_subplot(grid[0, 1])
    ax_ci_width = fig.add_subplot(grid[1, 0])
    ax_norm = fig.add_subplot(grid[1, 1])
    ax_error = fig.add_subplot(grid[2, 0])
    ax_resid = fig.add_subplot(grid[2, 1])

    ax_fit.scatter(X_train[:, 0], y_train, s=7, alpha=0.12, label="train")
    ax_fit.plot(X[:, 0], signal, color="black", linewidth=2, label="signal")
    ax_fit.plot(
        X[:, 0],
        hgb_grid_pred,
        color="0.35",
        linestyle="--",
        linewidth=1.6,
        label=f"HGBR ({total_tree_budget} trees)",
    )
    for result in results:
        ax_fit.plot(
            X[:, 0],
            result.grid_pred,
            color=result.color,
            linewidth=1.8,
            label=result.name,
        )
        ax_fit.fill_between(
            X[:, 0],
            result.grid_ci_lower,
            result.grid_ci_upper,
            color=result.color,
            alpha=0.12,
        )
    ax_fit.set_title("Fitted signal and 95% confidence bands")
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.legend(loc="best", fontsize=8)

    ax_pi.scatter(X_test[:, 0], y_test, s=9, alpha=0.18, label="test y")
    ax_pi.plot(X[:, 0], signal, color="black", linewidth=2, label="signal")
    for result in results:
        ax_pi.plot(X[:, 0], result.grid_pred, color=result.color, linewidth=1.8)
        ax_pi.fill_between(
            X[:, 0],
            result.grid_pi_lower,
            result.grid_pi_upper,
            color=result.color,
            alpha=0.10,
            label=f"{result.name} 95% PI",
        )
    ax_pi.set_title("Prediction intervals")
    ax_pi.set_xlabel("x")
    ax_pi.set_ylabel("y")
    ax_pi.legend(loc="best", fontsize=8)

    for result in results:
        ci_width = result.grid_ci_upper - result.grid_ci_lower
        pi_width = result.grid_pi_upper - result.grid_pi_lower
        ax_ci_width.plot(
            X[:, 0],
            ci_width,
            color=result.color,
            linewidth=1.8,
            label=f"{result.name} CI width",
        )
        ax_ci_width.plot(
            X[:, 0],
            pi_width,
            color=result.color,
            linestyle=":",
            linewidth=1.5,
            label=f"{result.name} PI width",
        )
    ax_ci_width.set_title("Interval widths")
    ax_ci_width.set_xlabel("x")
    ax_ci_width.set_ylabel("width")
    ax_ci_width.legend(loc="best", fontsize=8)

    for result in results:
        ax_norm.plot(
            X[:, 0],
            result.grid_weight_norms,
            color=result.color,
            linewidth=1.8,
            label=result.name,
        )
    ax_norm.set_title("Kernel weight norms")
    ax_norm.set_xlabel("x")
    ax_norm.set_ylabel(r"$||r_n(x)||$")
    ax_norm.legend(loc="best", fontsize=8)

    ax_error.plot(
        X[:, 0],
        np.abs(hgb_grid_pred - signal),
        color="0.35",
        linestyle="--",
        label="HGBR |error|",
    )
    for result in results:
        ax_error.plot(
            X[:, 0],
            np.abs(result.grid_pred - signal),
            color=result.color,
            linewidth=1.6,
            label=f"{result.name} |error|",
        )
        ax_error.plot(
            X[:, 0],
            (result.grid_ci_upper - result.grid_ci_lower) / 2,
            color=result.color,
            linestyle=":",
            linewidth=1.4,
            label=f"{result.name} CI half-width",
        )
    ax_error.set_title("Signal error vs CI half-width")
    ax_error.set_xlabel("x")
    ax_error.set_ylabel("absolute value")
    ax_error.legend(loc="best", fontsize=8)

    for result in results:
        residuals = y_calib - result.model.predict(X_calib)
        ax_resid.hist(
            residuals,
            bins=35,
            alpha=0.52,
            color=result.color,
            label=f"{result.name} calibration residuals",
        )
    ax_resid.axvline(0.0, color="black", linewidth=1)
    ax_resid.set_title("Calibration residuals")
    ax_resid.set_xlabel("residual")
    ax_resid.set_ylabel("count")
    ax_resid.legend(loc="best", fontsize=8)

    fig.suptitle("Boulevard histogram estimators: BRAT-D and BRAT-P", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    print("Boulevard BRAT histogram API demo")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print(f"total tree budget per method: {total_tree_budget}")
    print(f"Nyström inference subsample rate: {brat_d.nystrom_subsample_rate}")
    print("")
    print("vanilla HGBR")
    print(f"vanilla HGBR fit seconds: {hgb_fit_seconds:.4f}")
    print(f"  grid prediction seconds: {hgb_grid_pred_seconds:.4f}")
    print(f"  test prediction seconds: {hgb_test_pred_seconds:.4f}")
    print(
        "  test prediction seconds per point: "
        f"{hgb_test_pred_seconds / X_test.shape[0]:.8f}"
    )
    print(f"  grid RMSE vs signal: {_rmse(signal, hgb_grid_pred):.4f}")
    print(f"  test RMSE vs signal: {_rmse(signal_test, hgb_test_pred):.4f}")
    print(f"  test RMSE vs y: {_rmse(y_test, hgb_test_pred):.4f}")
    print(
        "  test abs signal error quantiles: "
        f"{_format_quantiles(np.abs(hgb_test_pred - signal_test))}"
    )

    for result in results:
        ci_width = result.test_ci_upper - result.test_ci_lower
        pi_width = result.test_pi_upper - result.test_pi_lower
        grid_ci_coverage = _coverage(
            signal,
            result.grid_ci_lower,
            result.grid_ci_upper,
        )
        test_ci_coverage = _coverage(
            signal_test,
            result.test_ci_lower,
            result.test_ci_upper,
        )
        test_pi_coverage = _coverage(y_test, result.test_pi_lower, result.test_pi_upper)
        test_abs_error = np.abs(result.test_pred - signal_test)
        compression_ratio = result.model.observed_cells_.shape[0] / X_train.shape[0]
        print("")
        print(result.name)
        print(f"  fit seconds: {result.fit_seconds:.4f}")
        print(f"  inference prep seconds: {result.inference_seconds:.4f}")
        print(f"  grid prediction seconds: {result.grid_prediction_seconds:.4f}")
        print(f"  test prediction seconds: {result.test_prediction_seconds:.4f}")
        print(
            "  test prediction seconds per point: "
            f"{result.test_prediction_seconds / X_test.shape[0]:.8f}"
        )
        print(f"  grid interval seconds: {result.grid_interval_seconds:.4f}")
        print(f"  test interval seconds: {result.test_interval_seconds:.4f}")
        print(
            "  test interval seconds per point: "
            f"{result.test_interval_seconds / X_test.shape[0]:.8f}"
        )
        print(f"  grid weight-norm seconds: {result.grid_weight_norm_seconds:.4f}")
        print(f"  test weight-norm seconds: {result.test_weight_norm_seconds:.4f}")
        print(
            "  test weight-norm seconds per point: "
            f"{result.test_weight_norm_seconds / X_test.shape[0]:.8f}"
        )
        print(f"  grid RMSE vs signal: {_rmse(signal, result.grid_pred):.4f}")
        print(f"  RMSE vs signal: {_rmse(signal_test, result.test_pred):.4f}")
        print(f"  RMSE vs y: {_rmse(y_test, result.test_pred):.4f}")
        print(f"  grid 95% CI coverage vs signal: {grid_ci_coverage:.3f}")
        print(f"  test 95% CI coverage vs signal: {test_ci_coverage:.3f}")
        print(f"  test 95% PI coverage vs y: {test_pi_coverage:.3f}")
        print(f"  sigma_hat2: {result.model.sigma_hat2_:.6f}")
        print(f"  inference method: {result.model.inference_method_}")
        if hasattr(result.model, "nystrom_landmark_count_"):
            print(
                "  Nyström landmarks / observed cells: "
                f"{result.model.nystrom_landmark_count_}/"
                f"{result.model.observed_cells_.shape[0]}"
            )
        if "parallel_rounds" in result.model.fit_diagnostics_:
            print(
                "  parallel/serial rounds: "
                f"{result.model.fit_diagnostics_['parallel_rounds']}/"
                f"{result.model.fit_diagnostics_['serial_rounds']}"
            )
            print(
                "  effective n_jobs: "
                f"{result.model.fit_diagnostics_['effective_n_jobs']}"
            )
            print(
                "  vectorized residual rounds: "
                f"{result.model.fit_diagnostics_['vectorized_residual_rounds']}"
            )
        print(
            "  observed cells / train rows: "
            f"{result.model.observed_cells_.shape[0]}/{X_train.shape[0]}"
        )
        print(f"  observed-cell compression ratio: {compression_ratio:.4f}")
        print(f"  CI width quantiles: {_format_quantiles(ci_width)}")
        print(f"  PI width quantiles: {_format_quantiles(pi_width)}")
        print(f"  abs signal error quantiles: {_format_quantiles(test_abs_error)}")
        print(
            "  weight norm quantiles: "
            f"{_format_quantiles(result.test_weight_norms)}"
        )

    if args.output:
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()


if __name__ == "__main__":
    main()

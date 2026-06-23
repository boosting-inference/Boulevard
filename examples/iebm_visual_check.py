"""Visual diagnostic for the current IEBM prototype.

Run from the repository root with:

    python examples/iebm_visual_check.py --output /tmp/iebm_visual_check.png

This example uses a synthetic additive three-feature regression function.  The
top row shows IEBM partial dependence curves with experimental confidence bands.
Coverage is available here only because the synthetic data-generating function
is known.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def _format_quantiles(values: np.ndarray) -> str:
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return (
        f"q05={q05:.4f}, q25={q25:.4f}, q50={q50:.4f}, "
        f"q75={q75:.4f}, q95={q95:.4f}"
    )


def _rmse(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(target, pred)))


def _residual_summary(name: str, residuals: np.ndarray) -> str:
    return (
        f"{name} residual mean/variance: "
        f"{np.mean(residuals):.6f}/{np.var(residuals, ddof=1):.6f}"
    )


def _term_range_summary(term_scores: Sequence[np.ndarray]) -> str:
    ranges = [
        f"x{idx}: min={np.min(scores):.4f}, max={np.max(scores):.4f}, "
        f"range={np.ptp(scores):.4f}"
        for idx, scores in enumerate(term_scores)
    ]
    return "; ".join(ranges)


def _component_values(X: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.sin(2 * np.pi * X[:, 0]) + 0.3 * np.cos(8 * np.pi * X[:, 0]),
            0.8 * (X[:, 1] - 0.5),
            0.45 * np.sin(4 * np.pi * X[:, 2]),
        ]
    )


def _signal(X: np.ndarray) -> np.ndarray:
    return np.sum(_component_values(X), axis=1)


def _component_grid(feature_idx: int, values: np.ndarray) -> np.ndarray:
    if feature_idx == 0:
        return np.sin(2 * np.pi * values) + 0.3 * np.cos(8 * np.pi * values)
    if feature_idx == 1:
        return 0.8 * (values - 0.5)
    if feature_idx == 2:
        return 0.45 * np.sin(4 * np.pi * values)
    raise ValueError("feature_idx is out of range.")


def _run_iebm_sweep(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    signal_test: np.ndarray,
    noise_variance: float,
) -> None:
    configs = [
        {"max_rounds": 80, "max_bins": 64, "max_depth": 1},
        {"max_rounds": 160, "max_bins": 128, "max_depth": 1},
        {"max_rounds": 160, "max_bins": 128, "max_depth": 2},
        {"max_rounds": 240, "max_bins": 128, "max_depth": 2},
    ]

    print("")
    print("IEBM hyperparameter sweep")
    print(
        "  columns: rounds, bins, leaves, fit seconds, prep seconds, "
        "RMSE(signal), sigma_hat2/noise_var, norm q50/q95, "
        "CI width q50/q95, CI coverage"
    )

    for config in configs:
        model = bd.IEBMRegressor(
            learning_rate=1.0,
            subsample_rate=0.8,
            warmup_rounds=20,
            truncation=10.0,
            min_samples_leaf=10,
            random_state=0,
            **config,
        )

        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        prep_start = time.perf_counter()
        model.prepare_inference(X_calib, y_calib)
        prep_seconds = time.perf_counter() - prep_start

        lower, upper, pred = model.predict_intervals(
            X_test,
            level=0.95,
            mode="confidence",
        )
        norms = model.weight_norms(X_test)
        coverage = _coverage(signal_test, lower, upper)
        norm_q50, norm_q95 = np.quantile(norms, [0.5, 0.95])
        width_q50, width_q95 = np.quantile(upper - lower, [0.5, 0.95])

        print(
            "  "
            f"{config['max_rounds']:>4}, "
            f"{config['max_bins']:>4}, "
            f"{model.max_leaves_:>2}, "
            f"{fit_seconds:.4f}, "
            f"{prep_seconds:.4f}, "
            f"{_rmse(signal_test, pred):.4f}, "
            f"{model.sigma_hat2_:.6f}/{noise_variance:.6f}, "
            f"{norm_q50:.5f}/{norm_q95:.5f}, "
            f"{width_q50:.5f}/{width_q95:.5f}, "
            f"{coverage:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Optional path for saving the plot. If omitted, show a window.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the small IEBM hyperparameter diagnostic sweep.",
    )
    args = parser.parse_args()
    if args.output:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

    rng = np.random.default_rng(0)
    n_samples = 3600
    noise_std = 0.25
    X = rng.uniform(0.0, 1.0, size=(n_samples, 3))
    signal = _signal(X)
    components = _component_values(X)
    y = signal + rng.normal(scale=noise_std, size=n_samples)

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
        signal_train,
        signal_holdout,
        components_train,
        components_holdout,
    ) = train_test_split(
        X,
        y,
        signal,
        components,
        test_size=0.4,
        random_state=0,
    )
    (
        X_calib,
        X_test,
        y_calib,
        y_test,
        signal_calib,
        signal_test,
        _components_calib,
        _components_test,
    ) = train_test_split(
        X_holdout,
        y_holdout,
        signal_holdout,
        components_holdout,
        test_size=0.5,
        random_state=1,
    )

    iebm = bd.IEBMRegressor(
        max_rounds=160,
        max_bins=128,
        learning_rate=1.0,
        subsample_rate=0.8,
        warmup_rounds=20,
        truncation=10.0,
        max_depth=7,
        min_samples_leaf=10,
        random_state=0,
    )
    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        max_bins=255,
        early_stopping=False,
        random_state=0,
    )

    fit_start = time.perf_counter()
    iebm.fit(X_train, y_train)
    iebm_fit_seconds = time.perf_counter() - fit_start

    fit_start = time.perf_counter()
    hgb.fit(X_train, y_train)
    hgb_fit_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    iebm.prepare_inference(X_calib, y_calib)
    inference_seconds = time.perf_counter() - inference_start

    pred_start = time.perf_counter()
    ci_lower, ci_upper, iebm_test_pred = iebm.predict_intervals(
        X_test,
        level=0.95,
        mode="confidence",
    )
    iebm_test_pred_seconds = time.perf_counter() - pred_start
    iebm_train_pred = iebm.predict(X_train)
    iebm_calib_pred = iebm.predict(X_calib)

    pred_start = time.perf_counter()
    pi_lower, pi_upper, _ = iebm.predict_intervals(
        X_test,
        level=0.95,
        mode="prediction",
    )
    iebm_interval_seconds = time.perf_counter() - pred_start

    pred_start = time.perf_counter()
    hgb_test_pred = hgb.predict(X_test)
    hgb_test_pred_seconds = time.perf_counter() - pred_start

    norm_start = time.perf_counter()
    test_norms = iebm.weight_norms(X_test)
    test_norm_seconds = time.perf_counter() - norm_start

    train_residuals = y_train - iebm_train_pred
    calib_residuals = y_calib - iebm_calib_pred
    test_residuals = y_test - iebm_test_pred
    test_abs_signal_error = np.abs(iebm_test_pred - signal_test)
    hgb_test_abs_signal_error = np.abs(hgb_test_pred - signal_test)
    ci_width = ci_upper - ci_lower
    bin_counts = np.concatenate(iebm.bin_counts_)
    occupied_bins = int(np.count_nonzero(bin_counts))
    component_means = np.mean(components_train, axis=0)

    partial_grid = np.linspace(0.0, 1.0, 300)
    partial_results = []
    for feature_idx in range(X.shape[1]):
        lower, upper, pred = iebm.predict_feature_intervals(
            feature_idx,
            partial_grid,
            level=0.95,
            mode="confidence",
            include_intercept=False,
        )
        truth = (
            _component_grid(feature_idx, partial_grid)
            - component_means[feature_idx]
        )
        coverage = _coverage(truth, lower, upper)
        partial_results.append((lower, upper, pred, truth, coverage))

    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.9])
    partial_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    ax_pred = fig.add_subplot(grid[1, 0])
    ax_width = fig.add_subplot(grid[1, 1])
    ax_resid = fig.add_subplot(grid[1, 2])

    for feature_idx, ax in enumerate(partial_axes):
        lower, upper, pred, truth, coverage = partial_results[feature_idx]
        ax.fill_between(
            partial_grid,
            lower,
            upper,
            color="#1f77b4",
            alpha=0.16,
            label="95% confidence band",
        )
        ax.plot(partial_grid, truth, color="black", linewidth=2.0, label="true partial")
        ax.plot(partial_grid, pred, color="#1f77b4", linewidth=1.8, label="IEBM")
        ax.set_title(f"Partial dependence: x{feature_idx}")
        ax.set_xlabel(f"x{feature_idx}")
        ax.set_ylabel("centered contribution")
        ax.text(
            0.03,
            0.95,
            f"coverage={coverage:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        if feature_idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    ax_pred.scatter(signal_test, iebm_test_pred, s=12, alpha=0.35, label="IEBM")
    ax_pred.scatter(signal_test, hgb_test_pred, s=12, alpha=0.25, label="HGBR")
    lims = [
        min(np.min(signal_test), np.min(iebm_test_pred), np.min(hgb_test_pred)),
        max(np.max(signal_test), np.max(iebm_test_pred), np.max(hgb_test_pred)),
    ]
    ax_pred.plot(lims, lims, color="black", linewidth=1)
    ax_pred.set_title("Predicted vs true signal")
    ax_pred.set_xlabel("true signal")
    ax_pred.set_ylabel("prediction")
    ax_pred.legend(loc="best", fontsize=8)

    ax_width.scatter(
        test_abs_signal_error,
        ci_width / 2,
        s=12,
        alpha=0.35,
        color="#1f77b4",
    )
    ax_width.set_title("CI half-width vs signal error")
    ax_width.set_xlabel("|prediction - signal|")
    ax_width.set_ylabel("CI half-width")
    ax_width.text(
        0.03,
        0.95,
        f"full CI coverage={_coverage(signal_test, ci_lower, ci_upper):.3f}\n"
        f"PI y coverage={_coverage(y_test, pi_lower, pi_upper):.3f}",
        transform=ax_width.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    ax_resid.hist(calib_residuals, bins=35, color="#1f77b4", alpha=0.65)
    ax_resid.axvline(0.0, color="black", linewidth=1)
    ax_resid.set_title("Calibration residuals")
    ax_resid.set_xlabel("residual")
    ax_resid.set_ylabel("count")

    fig.suptitle("IEBM multivariate additive diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    partial_coverages = [result[4] for result in partial_results]
    print("IEBM multivariate visual diagnostic")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print("top-row bands use predict_feature_intervals(..., mode='confidence')")
    print(f"IEBM fit seconds: {iebm_fit_seconds:.4f}")
    print(f"IEBM inference prep seconds: {inference_seconds:.4f}")
    print(f"IEBM test prediction seconds: {iebm_test_pred_seconds:.4f}")
    print(f"IEBM interval seconds: {iebm_interval_seconds:.4f}")
    print(f"IEBM test weight-norm seconds: {test_norm_seconds:.4f}")
    print(f"HGBR fit seconds: {hgb_fit_seconds:.4f}")
    print(f"HGBR test prediction seconds: {hgb_test_pred_seconds:.4f}")
    print(f"IEBM test RMSE vs signal: {_rmse(signal_test, iebm_test_pred):.4f}")
    print(f"IEBM test RMSE vs y: {_rmse(y_test, iebm_test_pred):.4f}")
    print(f"HGBR test RMSE vs signal: {_rmse(signal_test, hgb_test_pred):.4f}")
    print(f"HGBR test RMSE vs y: {_rmse(y_test, hgb_test_pred):.4f}")
    print(f"true noise variance: {noise_std**2:.6f}")
    print(f"sigma_hat2: {iebm.sigma_hat2_:.6f}")
    print(f"fit sigma: {iebm.sigma_:.6f}")
    print(_residual_summary("train", train_residuals))
    print(_residual_summary("calib", calib_residuals))
    print(_residual_summary("test", test_residuals))
    print(f"total bins: {iebm.fit_diagnostics_['total_bins']}")
    print(
        "occupied bins / total bins: "
        f"{occupied_bins}/{bin_counts.shape[0]}"
    )
    print(
        "bin count min/median/mean/max: "
        f"{np.min(bin_counts):.1f}/"
        f"{np.median(bin_counts):.1f}/"
        f"{np.mean(bin_counts):.1f}/"
        f"{np.max(bin_counts):.1f}"
    )
    print(f"term score ranges: {_term_range_summary(iebm.term_scores_)}")
    print(f"structure updates: {iebm.fit_diagnostics_['structure_updates']}")
    print(
        "empty training bins: "
        f"{iebm.inference_diagnostics_['empty_training_bins']}"
    )
    print(f"weight norm quantiles: {_format_quantiles(test_norms)}")
    print(
        "IEBM abs signal error quantiles: "
        f"{_format_quantiles(test_abs_signal_error)}"
    )
    print(
        "HGBR abs signal error quantiles: "
        f"{_format_quantiles(hgb_test_abs_signal_error)}"
    )
    print(f"full CI width quantiles: {_format_quantiles(ci_width)}")
    print(
        "partial confidence coverage x0/x1/x2: "
        f"{partial_coverages[0]:.3f}/"
        f"{partial_coverages[1]:.3f}/"
        f"{partial_coverages[2]:.3f}"
    )
    print(
        "full confidence coverage on test signal: "
        f"{_coverage(signal_test, ci_lower, ci_upper):.3f}"
    )
    print(
        "prediction interval coverage on noisy y: "
        f"{_coverage(y_test, pi_lower, pi_upper):.3f}"
    )

    if not args.skip_sweep:
        _run_iebm_sweep(
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            signal_test=signal_test,
            noise_variance=noise_std**2,
        )

    if args.output:
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()


if __name__ == "__main__":
    main()

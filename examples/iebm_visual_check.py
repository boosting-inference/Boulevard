"""Visual diagnostic for the current IEBM prototype.

Run from the repository root with:

    python examples/iebm_visual_check.py --output /tmp/iebm_visual_check.png

This example uses a synthetic additive three-feature regression function.  The
top row shows IEBM partial dependence curves with experimental confidence bands.
Coverage is available here only because the synthetic data-generating function
is known. If InterpretML is installed, the script also reports a vanilla EBM
main-effect baseline for fit-time and RMSE comparison.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd

try:
    from interpret.glassbox import (
        ExplainableBoostingRegressor as InterpretEBMRegressor,
    )
except ImportError:  # pragma: no cover - optional example dependency
    InterpretEBMRegressor = None


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


def _make_interpret_ebm(*, max_leaves: int) -> object | None:
    if InterpretEBMRegressor is None:
        return None

    params = {
        "interactions": 0,
        "max_bins": 128,
        "outer_bags": 1,
        "validation_size": 0,
        "learning_rate": 0.1,
        "max_rounds": 160,
        "max_leaves": max_leaves,
        "min_samples_leaf": 10,
        "n_jobs": 1,
        "random_state": 0,
    }
    try:
        return InterpretEBMRegressor(**params)
    except TypeError:
        fallback = {
            "interactions": 0,
            "max_bins": 128,
            "learning_rate": 0.1,
            "max_rounds": 160,
            "max_leaves": max_leaves,
            "n_jobs": 1,
            "random_state": 0,
        }
        return InterpretEBMRegressor(**fallback)


def _fit_iebm_for_benchmark(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int,
) -> tuple[bd.IEBMRegressor, float]:
    model = bd.IEBMRegressor(
        max_rounds=160,
        max_bins=128,
        learning_rate=1.0,
        subsample_rate=0.8,
        warmup_rounds=20,
        truncation=10.0,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=0,
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - fit_start


def _run_tree_complexity_benchmark(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    signal_test: np.ndarray,
    deep_iebm: bd.IEBMRegressor,
    deep_iebm_fit_seconds: float,
    deep_iebm_test_pred: np.ndarray,
) -> None:
    configs = [
        ("shallow", 1, 2),
        ("deep", 7, 128),
    ]

    print("")
    print("Matched tree-complexity benchmark")
    print(
        "  columns: setting, leaves, IEBM fit seconds, IEBM tree-update share, "
        "IEBM RMSE(signal), EBM fit seconds, EBM RMSE(signal)"
    )

    for name, max_depth, max_leaves in configs:
        if max_leaves == deep_iebm.max_leaves_:
            iebm_model = deep_iebm
            iebm_fit_seconds = deep_iebm_fit_seconds
            iebm_pred = deep_iebm_test_pred
        else:
            iebm_model, iebm_fit_seconds = _fit_iebm_for_benchmark(
                X_train=X_train,
                y_train=y_train,
                max_depth=max_depth,
            )
            iebm_pred = iebm_model.predict(X_test)

        fit_diag = iebm_model.fit_diagnostics_
        tree_update_share = (
            fit_diag["tree_update_seconds"] / fit_diag["total_seconds"]
            if fit_diag["total_seconds"] > 0
            else 0.0
        )

        ebm_fit_text = "skipped"
        ebm_rmse_text = "skipped"
        interpret_ebm = _make_interpret_ebm(max_leaves=max_leaves)
        if interpret_ebm is not None:
            fit_start = time.perf_counter()
            interpret_ebm.fit(X_train, y_train)
            ebm_fit_seconds = time.perf_counter() - fit_start
            ebm_pred = interpret_ebm.predict(X_test)
            ebm_fit_text = f"{ebm_fit_seconds:.4f}"
            ebm_rmse_text = f"{_rmse(signal_test, ebm_pred):.4f}"

        print(
            "  "
            f"{name:<7}, "
            f"{max_leaves:>3}, "
            f"{iebm_fit_seconds:.4f}, "
            f"{100.0 * tree_update_share:.1f}%, "
            f"{_rmse(signal_test, iebm_pred):.4f}, "
            f"{ebm_fit_text}, "
            f"{ebm_rmse_text}"
        )


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

    main_max_depth = 7
    main_max_leaves = 2**main_max_depth
    iebm = bd.IEBMRegressor(
        max_rounds=160,
        max_bins=128,
        learning_rate=1.0,
        subsample_rate=0.8,
        warmup_rounds=20,
        truncation=10.0,
        max_depth=main_max_depth,
        min_samples_leaf=10,
        random_state=0,
    )
    interpret_ebm = _make_interpret_ebm(max_leaves=main_max_leaves)

    fit_start = time.perf_counter()
    iebm.fit(X_train, y_train)
    iebm_fit_seconds = time.perf_counter() - fit_start

    if interpret_ebm is not None:
        fit_start = time.perf_counter()
        interpret_ebm.fit(X_train, y_train)
        interpret_ebm_fit_seconds = time.perf_counter() - fit_start
    else:
        interpret_ebm_fit_seconds = None

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

    if interpret_ebm is not None:
        pred_start = time.perf_counter()
        interpret_ebm_test_pred = interpret_ebm.predict(X_test)
        interpret_ebm_test_pred_seconds = time.perf_counter() - pred_start
    else:
        interpret_ebm_test_pred = None
        interpret_ebm_test_pred_seconds = None

    norm_start = time.perf_counter()
    test_norms = iebm.weight_norms(X_test)
    test_norm_seconds = time.perf_counter() - norm_start

    train_residuals = y_train - iebm_train_pred
    calib_residuals = y_calib - iebm_calib_pred
    test_residuals = y_test - iebm_test_pred
    test_abs_signal_error = np.abs(iebm_test_pred - signal_test)
    if interpret_ebm_test_pred is not None:
        ebm_test_abs_signal_error = np.abs(interpret_ebm_test_pred - signal_test)
    else:
        ebm_test_abs_signal_error = None
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
    if interpret_ebm_test_pred is not None:
        ax_pred.scatter(
            signal_test,
            interpret_ebm_test_pred,
            s=12,
            alpha=0.25,
            label="InterpretML EBM",
        )
    prediction_arrays = [signal_test, iebm_test_pred]
    if interpret_ebm_test_pred is not None:
        prediction_arrays.append(interpret_ebm_test_pred)
    lims = [
        min(float(np.min(values)) for values in prediction_arrays),
        max(float(np.max(values)) for values in prediction_arrays),
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
    fit_diag = iebm.fit_diagnostics_
    fit_total = float(fit_diag["total_seconds"])

    def _fit_timing_line(label: str, key: str) -> str:
        seconds = float(fit_diag[key])
        share = seconds / fit_total if fit_total > 0 else 0.0
        return f"  {label}: {seconds:.4f}s ({100.0 * share:.1f}%)"

    print("IEBM multivariate visual diagnostic")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print("top-row bands use predict_feature_intervals(..., mode='confidence')")
    print(f"IEBM fit seconds: {iebm_fit_seconds:.4f}")
    print("IEBM fit timing breakdown:")
    print(_fit_timing_line("binning", "binning_seconds"))
    print(_fit_timing_line("contribution", "contribution_seconds"))
    print(_fit_timing_line("residual", "residual_seconds"))
    print(_fit_timing_line("sampling", "sampling_seconds"))
    print(_fit_timing_line("tree update", "tree_update_seconds"))
    print(_fit_timing_line("structure", "structure_seconds"))
    print(_fit_timing_line("score update", "score_update_seconds"))
    print(_fit_timing_line("intercept", "intercept_seconds"))
    print(_fit_timing_line("finalization", "finalization_seconds"))
    print(f"IEBM inference prep seconds: {inference_seconds:.4f}")
    print(f"IEBM test prediction seconds: {iebm_test_pred_seconds:.4f}")
    print(f"IEBM interval seconds: {iebm_interval_seconds:.4f}")
    print(f"IEBM test weight-norm seconds: {test_norm_seconds:.4f}")
    if interpret_ebm is not None:
        print(f"InterpretML EBM fit seconds: {interpret_ebm_fit_seconds:.4f}")
        print(
            "InterpretML EBM test prediction seconds: "
            f"{interpret_ebm_test_pred_seconds:.4f}"
        )
    else:
        print(
            "InterpretML EBM baseline skipped: install the 'interpretml' extra "
            "to enable it."
        )
    print(f"IEBM test RMSE vs signal: {_rmse(signal_test, iebm_test_pred):.4f}")
    print(f"IEBM test RMSE vs y: {_rmse(y_test, iebm_test_pred):.4f}")
    if interpret_ebm_test_pred is not None:
        print(
            "InterpretML EBM test RMSE vs signal: "
            f"{_rmse(signal_test, interpret_ebm_test_pred):.4f}"
        )
        print(
            "InterpretML EBM test RMSE vs y: "
            f"{_rmse(y_test, interpret_ebm_test_pred):.4f}"
        )
    print(f"true noise variance: {noise_std**2:.6f}")
    print(f"sigma_hat2: {iebm.sigma_hat2_:.6f}")
    print(f"fit sigma: {iebm.sigma_:.6f}")
    print(_residual_summary("train", train_residuals))
    print(_residual_summary("calib", calib_residuals))
    print(_residual_summary("test", test_residuals))
    print(f"total bins: {fit_diag['total_bins']}")
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
    print(f"structure updates: {fit_diag['structure_updates']}")
    print(
        "empty training bins: "
        f"{iebm.inference_diagnostics_['empty_training_bins']}"
    )
    print(f"weight norm quantiles: {_format_quantiles(test_norms)}")
    print(
        "IEBM abs signal error quantiles: "
        f"{_format_quantiles(test_abs_signal_error)}"
    )
    if ebm_test_abs_signal_error is not None:
        print(
            "InterpretML EBM abs signal error quantiles: "
            f"{_format_quantiles(ebm_test_abs_signal_error)}"
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
    _run_tree_complexity_benchmark(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        signal_test=signal_test,
        deep_iebm=iebm,
        deep_iebm_fit_seconds=iebm_fit_seconds,
        deep_iebm_test_pred=iebm_test_pred,
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

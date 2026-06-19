"""Visual smoke check for the experimental histogram BRAT-D backend."""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from boulevard.estimators.brat import BRATDRegressor
from boulevard.estimators.brat_hist import BRATDHistGradientBoostingRegressor


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _coverage(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((target >= lower) & (target <= upper)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Optional path for saving the plot. If omitted, show an interactive window.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(3)
    noise_std = 0.12
    X = np.linspace(0, 1, 1000).reshape(-1, 1)
    x = X[:, 0]
    truth = (
        np.sin(2 * np.pi * x)
        + 0.35 * np.sin(6 * np.pi * x)
        + 0.6 * (x - 0.5) ** 2
    )
    y = truth + rng.normal(scale=noise_std, size=X.shape[0])

    train_idx, test_idx = train_test_split(
        np.arange(X.shape[0]),
        test_size=0.35,
        random_state=1,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    truth_test = truth[test_idx]

    brat_hist = BRATDHistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.45,
        dropout_rate=0.3,
        max_depth=10,
        max_leaf_nodes=1024,
        min_samples_leaf=5,
        max_bins=128,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=0,
        verbose=0,
    )
    brat_train_start = time.perf_counter()
    brat_hist.fit(X_train, y_train)
    brat_train_seconds = time.perf_counter() - brat_train_start

    brat_inference_start = time.perf_counter()
    brat_hist.prepare_inference(X_test, y_test)
    brat_inference_seconds = time.perf_counter() - brat_inference_start

    brat_exact = BRATDRegressor(
        n_estimators=brat_hist.max_iter,
        learning_rate=brat_hist.learning_rate,
        max_depth=brat_hist.max_depth,
        min_samples_split=5,
        subsample_rate=1.0,
        dropout_rate=brat_hist.dropout_rate,
        random_state=0,
        verbose=0,
    )
    exact_train_start = time.perf_counter()
    brat_exact.fit(X_train, y_train)
    exact_train_seconds = time.perf_counter() - exact_train_start

    exact_inference_start = time.perf_counter()
    brat_exact.prepare_inference(X_test, y_test)
    exact_inference_seconds = time.perf_counter() - exact_inference_start

    vanilla_hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=8,
        min_samples_leaf=8,
        max_bins=96,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=0,
    )
    vanilla_train_start = time.perf_counter()
    vanilla_hgb.fit(X_train, y_train)
    vanilla_train_seconds = time.perf_counter() - vanilla_train_start

    brat_predict_start = time.perf_counter()
    brat_pred = brat_hist.predict(X)
    brat_test_pred = brat_hist.predict(X_test)
    brat_predict_seconds = time.perf_counter() - brat_predict_start

    X_binned = brat_hist._bin_data(X)
    raw_tree_sum = brat_hist._predict_tree_sum_binned(
        X_binned,
        selected=None,
        n_threads=brat_hist._effective_n_threads(),
    )
    q = 1 - brat_hist.dropout_rate
    signal_scale = (1 + brat_hist.learning_rate * q) / brat_hist.learning_rate
    raw_boulevard_pred = (
        brat_hist.learning_rate / len(brat_hist._predictors)
    ) * raw_tree_sum
    signal_correction_max_error = float(
        np.max(np.abs(brat_pred - signal_scale * raw_boulevard_pred))
    )

    exact_predict_start = time.perf_counter()
    exact_pred = brat_exact.predict(X)
    exact_test_pred = brat_exact.predict(X_test)
    exact_predict_seconds = time.perf_counter() - exact_predict_start

    vanilla_predict_start = time.perf_counter()
    vanilla_pred = vanilla_hgb.predict(X)
    vanilla_test_pred = vanilla_hgb.predict(X_test)
    vanilla_predict_seconds = time.perf_counter() - vanilla_predict_start

    brat_interval_start = time.perf_counter()
    brat_ci_test_lower, brat_ci_test_upper = brat_hist.confidence_interval(X_test)
    brat_hist_ci_seconds = time.perf_counter() - brat_interval_start

    brat_interval_start = time.perf_counter()
    brat_pi_test_lower, brat_pi_test_upper = brat_hist.prediction_interval(X_test)
    brat_hist_pi_seconds = time.perf_counter() - brat_interval_start

    brat_interval_start = time.perf_counter()
    brat_ri_test_lower, brat_ri_test_upper = brat_hist.reproduction_interval(X_test)
    brat_hist_ri_seconds = time.perf_counter() - brat_interval_start

    exact_interval_start = time.perf_counter()
    exact_ci_test_lower, exact_ci_test_upper = brat_exact.confidence_interval(X_test)
    exact_ci_seconds = time.perf_counter() - exact_interval_start

    exact_interval_start = time.perf_counter()
    exact_pi_test_lower, exact_pi_test_upper = brat_exact.prediction_interval(X_test)
    exact_pi_seconds = time.perf_counter() - exact_interval_start

    brat_grid_weight_start = time.perf_counter()
    brat_weight_norms = brat_hist.weight_norms(X)
    brat_grid_weight_seconds = time.perf_counter() - brat_grid_weight_start

    brat_test_weight_start = time.perf_counter()
    brat_test_weight_norms = brat_hist.weight_norms(X_test)
    brat_test_weight_seconds = time.perf_counter() - brat_test_weight_start

    slow_norm_start = time.perf_counter()
    slow_test_binned = brat_hist._bin_data(X_test)
    slow_test_leaves = brat_hist._apply_leaf_indices_binned(slow_test_binned)
    slow_test_kernel_vectors = brat_hist._cell_kernel_vector(slow_test_leaves)
    slow_test_weights = brat_hist._solve_cell_brat_d_weights(slow_test_kernel_vectors)
    slow_test_weight_norms = np.sqrt(
        np.maximum((slow_test_weights**2) @ brat_hist.cell_counts_, 0.0)
    )
    slow_test_weight_seconds = time.perf_counter() - slow_norm_start
    cache_slow_norm_max_error = float(
        np.max(np.abs(brat_test_weight_norms - slow_test_weight_norms))
    )

    z_975 = 1.959963984540054
    brat_ci_half_width = (
        z_975
        * signal_scale
        * np.sqrt(brat_hist.sigma_hat2_)
        * brat_weight_norms
    )
    brat_ci_lower = brat_pred - brat_ci_half_width
    brat_ci_upper = brat_pred + brat_ci_half_width
    grid_cell_indices = brat_hist.apply_cell_indices(X)
    test_cell_indices = brat_hist.apply_cell_indices(X_test)
    train_cell_indices = brat_hist.apply_cell_indices(X_train)
    grid_observed = grid_cell_indices >= 0
    test_observed = test_cell_indices >= 0
    if np.any(grid_observed):
        cached_grid_weight_norms = brat_hist.cell_weight_norms_[
            grid_cell_indices[grid_observed]
        ]
        cache_grid_norm_max_error = float(
            np.max(np.abs(brat_weight_norms[grid_observed] - cached_grid_weight_norms))
        )
    else:
        cache_grid_norm_max_error = float("nan")

    brat_grid_rmse = _rmse(truth, brat_pred)
    exact_grid_rmse = _rmse(truth, exact_pred)
    vanilla_grid_rmse = _rmse(truth, vanilla_pred)
    brat_test_rmse = _rmse(y_test, brat_test_pred)
    exact_test_rmse = _rmse(y_test, exact_test_pred)
    vanilla_test_rmse = _rmse(y_test, vanilla_test_pred)
    brat_test_truth_rmse = _rmse(truth_test, brat_test_pred)
    exact_test_truth_rmse = _rmse(truth_test, exact_test_pred)
    vanilla_test_truth_rmse = _rmse(truth_test, vanilla_test_pred)
    brat_ci_truth_coverage = _coverage(truth_test, brat_ci_test_lower, brat_ci_test_upper)
    brat_pi_y_coverage = _coverage(y_test, brat_pi_test_lower, brat_pi_test_upper)
    exact_ci_truth_coverage = _coverage(
        truth_test,
        exact_ci_test_lower,
        exact_ci_test_upper,
    )
    exact_pi_y_coverage = _coverage(y_test, exact_pi_test_lower, exact_pi_test_upper)
    observed_cell_count = brat_hist.observed_cells_.shape[0]
    cell_count_sum = float(np.sum(brat_hist.cell_counts_))
    cell_kernel_symmetry_error = float(
        np.max(np.abs(brat_hist.cell_kernel_matrix_ - brat_hist.cell_kernel_matrix_.T))
    )
    grid_unseen_cell_rate = float(np.mean(grid_cell_indices < 0))
    test_unseen_cell_rate = float(np.mean(test_cell_indices < 0))
    train_unseen_cell_rate = float(np.mean(train_cell_indices < 0))
    grid_cache_hit_rate = float(np.mean(grid_observed))
    test_cache_hit_rate = float(np.mean(test_observed))
    train_residuals = y_train - brat_hist.predict(X_train)
    test_residuals = y_test - brat_test_pred
    centered_sigma_error = float(
        abs(brat_hist.sigma_hat2_ - np.var(test_residuals, ddof=1))
    )
    ci_test_width = brat_ci_test_upper - brat_ci_test_lower
    pi_test_width = brat_pi_test_upper - brat_pi_test_lower
    ri_test_width = brat_ri_test_upper - brat_ri_test_lower
    interval_pi_minus_ci_min = float(np.min(pi_test_width - ci_test_width))
    interval_ri_sqrt2_ci_max_error = float(
        np.max(np.abs(ri_test_width - np.sqrt(2) * ci_test_width))
    )
    cell_compression_ratio = observed_cell_count / X_train.shape[0]
    brat_abs_error = np.abs(brat_pred - truth)
    exact_abs_error = np.abs(exact_pred - truth)
    vanilla_abs_error = np.abs(vanilla_pred - truth)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 14))
    grid = fig.add_gridspec(3, 2)

    fit_ax = fig.add_subplot(grid[0, 0])
    fit_ax.scatter(X_train[:, 0], y_train, s=10, alpha=0.25, label="train")
    fit_ax.scatter(X_test[:, 0], y_test, s=10, alpha=0.18, label="test")
    fit_ax.plot(x, truth, color="black", linewidth=2, label="truth")
    fit_ax.fill_between(
        x,
        brat_ci_lower,
        brat_ci_upper,
        color="#1f77b4",
        alpha=0.14,
        label="BRAT-D hist 95% CI",
    )
    fit_ax.plot(x, brat_pred, color="#1f77b4", linewidth=2, label="BRAT-D hist")
    fit_ax.plot(
        x,
        exact_pred,
        color="#2ca02c",
        linewidth=1.5,
        linestyle=":",
        label="BRAT-D exact",
    )
    fit_ax.plot(
        x,
        vanilla_pred,
        color="#ff7f0e",
        linewidth=1.6,
        linestyle="--",
        label="vanilla HGBR",
    )
    fit_ax.set_title("Synthetic regression fit")
    fit_ax.set_xlabel("x")
    fit_ax.set_ylabel("y")
    fit_ax.legend(loc="best", fontsize=8)

    error_ax = fig.add_subplot(grid[0, 1])
    error_ax.plot(
        x,
        brat_abs_error,
        color="#1f77b4",
        linewidth=2,
        label="BRAT-D hist",
    )
    error_ax.plot(
        x,
        exact_abs_error,
        color="#2ca02c",
        linewidth=1.4,
        linestyle=":",
        label="BRAT-D exact",
    )
    error_ax.plot(
        x,
        vanilla_abs_error,
        color="#ff7f0e",
        linewidth=1.6,
        linestyle="--",
        label="vanilla HGBR",
    )
    error_ax.set_title("Absolute error against known truth")
    error_ax.set_xlabel("x")
    error_ax.set_ylabel("|prediction - truth|")
    error_ax.legend(loc="best", fontsize=8)

    timing_ax = fig.add_subplot(grid[1, 0])
    timing_labels = ["train", "prepare", "predict", "CI test", "PI test"]
    positions = np.arange(len(timing_labels))
    bar_width = 0.25
    hist_times = np.array(
        [
            brat_train_seconds,
            brat_inference_seconds,
            brat_predict_seconds,
            brat_hist_ci_seconds,
            brat_hist_pi_seconds,
        ]
    )
    exact_times = np.array(
        [
            exact_train_seconds,
            exact_inference_seconds,
            exact_predict_seconds,
            exact_ci_seconds,
            exact_pi_seconds,
        ]
    )
    vanilla_times = np.array(
        [
            vanilla_train_seconds,
            np.nan,
            vanilla_predict_seconds,
            np.nan,
            np.nan,
        ]
    )
    timing_ax.bar(positions - bar_width, hist_times, width=bar_width, label="BRAT-D hist")
    timing_ax.bar(positions, exact_times, width=bar_width, label="BRAT-D exact")
    timing_ax.bar(positions + bar_width, vanilla_times, width=bar_width, label="vanilla HGBR")
    timing_ax.set_title("Wall-clock timing comparison")
    timing_ax.set_xticks(positions)
    timing_ax.set_xticklabels(timing_labels, rotation=20)
    timing_ax.set_ylabel("seconds")
    timing_ax.set_yscale("log")
    timing_ax.legend(loc="best", fontsize=8)

    width_ax = fig.add_subplot(grid[1, 1])
    width_ax.plot(
        x,
        brat_weight_norms,
        color="#9467bd",
        linewidth=1.6,
        label="cell weight norm",
    )
    width_ax.set_title("Inference weight and width diagnostics")
    width_ax.set_xlabel("x")
    width_ax.set_ylabel("cell weight norm")
    width_ax.legend(loc="upper left", fontsize=8)

    half_width_ax = width_ax.twinx()
    half_width_ax.plot(
        x,
        brat_ci_half_width,
        color="#d62728",
        linewidth=1.2,
        linestyle="--",
        label="CI half-width",
    )
    half_width_ax.set_ylabel("CI half-width")
    half_width_ax.legend(loc="upper right", fontsize=8)

    cell_ax = fig.add_subplot(grid[2, 0])
    if brat_hist.observed_cells_.shape[1] == 1:
        cell_x = brat_hist.observed_cells_[:, 0]
        cell_xlabel = "observed histogram bin id"
    else:
        cell_x = np.arange(observed_cell_count)
        cell_xlabel = "observed multidimensional cell id"
    cell_order = np.argsort(cell_x)
    cell_ax.bar(
        cell_x[cell_order],
        brat_hist.cell_counts_[cell_order],
        color="#4c78a8",
        alpha=0.75,
    )
    cell_ax.set_title("Observed cell compression")
    cell_ax.set_xlabel(cell_xlabel)
    cell_ax.set_ylabel("training rows per cell")
    cell_ax.text(
        0.02,
        0.95,
        f"grid cache hit: {grid_cache_hit_rate:.1%}\n"
        f"test cache hit: {test_cache_hit_rate:.1%}\n"
        f"cells/train: {cell_compression_ratio:.1%}",
        transform=cell_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    resid_ax = fig.add_subplot(grid[2, 1])
    bins = np.linspace(
        min(train_residuals.min(), test_residuals.min()),
        max(train_residuals.max(), test_residuals.max()),
        28,
    )
    resid_ax.hist(
        train_residuals,
        bins=bins,
        alpha=0.45,
        color="#4c78a8",
        label="train residuals",
    )
    resid_ax.hist(
        test_residuals,
        bins=bins,
        alpha=0.45,
        color="#f58518",
        label="test residuals",
    )
    resid_ax.axvline(0, color="black", linewidth=1)
    resid_ax.set_title("BRAT-D hist residuals")
    resid_ax.set_xlabel("y - prediction")
    resid_ax.set_ylabel("count")
    resid_ax.legend(loc="best", fontsize=8)

    print(f"BRAT-D hist grid RMSE vs truth: {brat_grid_rmse:.6f}")
    print(f"BRAT-D exact grid RMSE vs truth: {exact_grid_rmse:.6f}")
    print(f"vanilla HGBR grid RMSE vs truth: {vanilla_grid_rmse:.6f}")
    print(f"BRAT-D hist test RMSE vs noisy y: {brat_test_rmse:.6f}")
    print(f"BRAT-D exact test RMSE vs noisy y: {exact_test_rmse:.6f}")
    print(f"vanilla HGBR test RMSE vs noisy y: {vanilla_test_rmse:.6f}")
    print(f"BRAT-D hist test RMSE vs truth: {brat_test_truth_rmse:.6f}")
    print(f"BRAT-D exact test RMSE vs truth: {exact_test_truth_rmse:.6f}")
    print(f"vanilla HGBR test RMSE vs truth: {vanilla_test_truth_rmse:.6f}")
    print(f"BRAT-D hist training wall-clock seconds: {brat_train_seconds:.6f}")
    print(f"BRAT-D exact training wall-clock seconds: {exact_train_seconds:.6f}")
    print(f"vanilla HGBR training wall-clock seconds: {vanilla_train_seconds:.6f}")
    print(
        "BRAT-D hist inference prep wall-clock seconds: "
        f"{brat_inference_seconds:.6f}"
    )
    print(
        "BRAT-D exact inference prep wall-clock seconds: "
        f"{exact_inference_seconds:.6f}"
    )
    print(f"BRAT-D hist prediction wall-clock seconds: {brat_predict_seconds:.6f}")
    print(f"BRAT-D exact prediction wall-clock seconds: {exact_predict_seconds:.6f}")
    print(f"vanilla HGBR prediction wall-clock seconds: {vanilla_predict_seconds:.6f}")
    print(
        "BRAT-D hist signal-correction max abs error: "
        f"{signal_correction_max_error:.6e}"
    )
    print(
        "BRAT-D hist grid weight-norm diagnostic wall-clock seconds: "
        f"{brat_grid_weight_seconds:.6f}"
    )
    print(
        "BRAT-D hist cached test weight-norm wall-clock seconds: "
        f"{brat_test_weight_seconds:.6f}"
    )
    print(
        "BRAT-D hist slow test weight-norm wall-clock seconds: "
        f"{slow_test_weight_seconds:.6f}"
    )
    print(
        "BRAT-D hist cached-vs-slow test norm max abs error: "
        f"{cache_slow_norm_max_error:.6e}"
    )
    print(
        "BRAT-D hist cached grid norm max abs error: "
        f"{cache_grid_norm_max_error:.6e}"
    )
    print(f"BRAT-D hist CI test wall-clock seconds: {brat_hist_ci_seconds:.6f}")
    print(f"BRAT-D hist PI test wall-clock seconds: {brat_hist_pi_seconds:.6f}")
    print(f"BRAT-D hist RI test wall-clock seconds: {brat_hist_ri_seconds:.6f}")
    print(f"BRAT-D exact CI test wall-clock seconds: {exact_ci_seconds:.6f}")
    print(f"BRAT-D exact PI test wall-clock seconds: {exact_pi_seconds:.6f}")
    print(f"BRAT-D hist observed cells / train rows: {observed_cell_count}/{X_train.shape[0]}")
    print(f"BRAT-D hist observed-cell compression ratio: {cell_compression_ratio:.6f}")
    print(f"BRAT-D hist cell count sum: {cell_count_sum:.0f}")
    print(f"BRAT-D hist train unseen-cell rate: {train_unseen_cell_rate:.6f}")
    print(f"BRAT-D hist test unseen-cell rate: {test_unseen_cell_rate:.6f}")
    print(f"BRAT-D hist grid unseen-cell rate: {grid_unseen_cell_rate:.6f}")
    print(f"BRAT-D hist test cache hit rate: {test_cache_hit_rate:.6f}")
    print(f"BRAT-D hist grid cache hit rate: {grid_cache_hit_rate:.6f}")
    print(f"BRAT-D hist cell kernel shape: {brat_hist.cell_kernel_matrix_.shape}")
    print(f"BRAT-D hist cell kernel symmetry error: {cell_kernel_symmetry_error:.6e}")
    print(f"BRAT-D hist sigma_hat2: {brat_hist.sigma_hat2_:.6f}")
    print(
        "BRAT-D hist centered residual variance check abs error: "
        f"{centered_sigma_error:.6e}"
    )
    print(
        "BRAT-D hist cell weight norm min/median/max: "
        f"{np.min(brat_weight_norms):.6f}/"
        f"{np.median(brat_weight_norms):.6f}/"
        f"{np.max(brat_weight_norms):.6f}"
    )
    print(
        "BRAT-D hist CI half-width min/median/max: "
        f"{np.min(brat_ci_half_width):.6f}/"
        f"{np.median(brat_ci_half_width):.6f}/"
        f"{np.max(brat_ci_half_width):.6f}"
    )
    print(f"BRAT-D hist CI truth coverage on test split: {brat_ci_truth_coverage:.6f}")
    print(f"BRAT-D hist PI y coverage on test split: {brat_pi_y_coverage:.6f}")
    print(f"BRAT-D exact CI truth coverage on test split: {exact_ci_truth_coverage:.6f}")
    print(f"BRAT-D exact PI y coverage on test split: {exact_pi_y_coverage:.6f}")
    print(
        "BRAT-D hist min PI-width minus CI-width on test split: "
        f"{interval_pi_minus_ci_min:.6e}"
    )
    print(
        "BRAT-D hist RI-width vs sqrt(2)*CI-width max abs error: "
        f"{interval_ri_sqrt2_ci_max_error:.6e}"
    )
    print(f"BRAT-D hist train residual mean: {np.mean(train_residuals):.6f}")
    print(f"BRAT-D hist test residual mean: {np.mean(test_residuals):.6f}")
    print(f"BRAT-D hist train residual variance: {np.var(train_residuals, ddof=1):.6f}")
    print(f"BRAT-D hist test residual variance: {np.var(test_residuals, ddof=1):.6f}")

    fig.suptitle("Experimental BRAT-D histogram backend visual check")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=160)
        print(f"Saved BRAT-D histogram visual check to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

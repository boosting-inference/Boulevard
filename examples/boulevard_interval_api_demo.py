"""Unified interval API demo for Boulevard's current Python estimators.

Run from the repository root with:

    python examples/boulevard_interval_api_demo.py --output /tmp/boulevard_intervals.png

The script fits BRAT-D, BRAT-P, and IEBM on the same synthetic additive
regression problem.  It demonstrates the shared interval API:

    lower, upper, pred = model.predict_intervals(X, mode="confidence")
    lower, upper, pred = model.predict_intervals(X, mode="prediction")
    lower, upper, pred = model.predict_intervals(X, mode="reproduction")

For IEBM, feature-level intervals use the EBM-style partial-function API:

    lower, upper, partial = model.predict_feature_intervals(feature_idx, grid)
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass

# Keep this example's terminal output focused on Boulevard diagnostics. Some
# macOS environments make joblib print a physical-core warning otherwise.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import boulevard as bd

MODES = ("confidence", "prediction", "reproduction")
MODE_LABELS = {
    "confidence": "CI",
    "prediction": "PI",
    "reproduction": "RI",
}
MODE_COLORS = {
    "confidence": "#1f77b4",
    "prediction": "#d62728",
    "reproduction": "#2ca02c",
}


@dataclass
class ModelSummary:
    name: str
    model: object
    fit_seconds: float
    prep_seconds: float
    pred_seconds: float
    interval_seconds: float
    norm_seconds: float
    test_pred: np.ndarray
    test_norms: np.ndarray
    intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    path_intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _components(X: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.sin(2 * np.pi * X[:, 0]) + 0.3 * np.cos(8 * np.pi * X[:, 0]),
            0.8 * (X[:, 1] - 0.5),
            0.45 * np.sin(4 * np.pi * X[:, 2]),
        ]
    )


def _signal(X: np.ndarray) -> np.ndarray:
    return np.sum(_components(X), axis=1)


def _component_grid(feature_idx: int, values: np.ndarray) -> np.ndarray:
    if feature_idx == 0:
        return np.sin(2 * np.pi * values) + 0.3 * np.cos(8 * np.pi * values)
    if feature_idx == 1:
        return 0.8 * (values - 0.5)
    if feature_idx == 2:
        return 0.45 * np.sin(4 * np.pi * values)
    raise ValueError("feature_idx is out of range.")


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


def _coverage_target(
    mode: str,
    *,
    signal_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    if mode == "prediction":
        return y_test
    return signal_test


def _fit_interval_model(
    *,
    name: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    X_path: np.ndarray,
) -> ModelSummary:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    model.prepare_inference(X_calib, y_calib)
    prep_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_pred = model.predict(X_test)
    pred_seconds = time.perf_counter() - start

    intervals = {}
    path_intervals = {}
    start = time.perf_counter()
    for mode in MODES:
        intervals[mode] = model.predict_intervals(X_test, level=0.95, mode=mode)
        path_intervals[mode] = model.predict_intervals(X_path, level=0.95, mode=mode)
    interval_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_norms = model.weight_norms(X_test)
    norm_seconds = time.perf_counter() - start

    return ModelSummary(
        name=name,
        model=model,
        fit_seconds=fit_seconds,
        prep_seconds=prep_seconds,
        pred_seconds=pred_seconds,
        interval_seconds=interval_seconds,
        norm_seconds=norm_seconds,
        test_pred=test_pred,
        test_norms=test_norms,
        intervals=intervals,
        path_intervals=path_intervals,
    )


def _make_brat_d(random_state: int) -> bd.BRATDHistGradientBoostingRegressor:
    return bd.BRATDHistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.45,
        dropout_rate=0.3,
        subsample_rate=0.8,
        nystrom_subsample_rate=1.0,
        max_depth=5,
        max_leaf_nodes=32,
        min_samples_leaf=8,
        max_bins=32,
        early_stopping=False,
        random_state=random_state,
    )


def _make_brat_p(
    random_state: int,
    n_jobs: int,
) -> bd.BRATPHistGradientBoostingRegressor:
    return bd.BRATPHistGradientBoostingRegressor(
        n_rounds=20,
        trees_per_round=4,
        subsample_rate=0.8,
        nystrom_subsample_rate=1.0,
        max_depth=5,
        max_leaf_nodes=32,
        min_samples_leaf=8,
        max_bins=32,
        early_stopping=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _make_iebm(random_state: int) -> bd.IEBMRegressor:
    return bd.IEBMRegressor(
        max_rounds=80,
        max_bins=64,
        learning_rate=1.0,
        subsample_rate=0.8,
        warmup_rounds=10,
        truncation=10.0,
        max_depth=5,
        min_samples_leaf=8,
        random_state=random_state,
    )


def _print_model_summary(
    result: ModelSummary,
    *,
    y_test: np.ndarray,
    signal_test: np.ndarray,
) -> None:
    print("")
    print(result.name)
    print(f"  fit seconds: {result.fit_seconds:.4f}")
    print(f"  prepare_inference seconds: {result.prep_seconds:.4f}")
    print(f"  test predict seconds: {result.pred_seconds:.4f}")
    print(f"  all test/path intervals seconds: {result.interval_seconds:.4f}")
    print(f"  test weight_norms seconds: {result.norm_seconds:.4f}")
    print(f"  RMSE vs signal: {_rmse(signal_test, result.test_pred):.4f}")
    print(f"  RMSE vs y: {_rmse(y_test, result.test_pred):.4f}")
    print(f"  sigma_hat2: {result.model.sigma_hat2_:.6f}")
    if hasattr(result.model, "inference_method_"):
        print(f"  inference method: {result.model.inference_method_}")
    if hasattr(result.model, "observed_cells_"):
        print(f"  observed cells: {result.model.observed_cells_.shape[0]}")
    for mode in MODES:
        lower, upper, _ = result.intervals[mode]
        target = _coverage_target(mode, signal_test=signal_test, y_test=y_test)
        print(
            f"  {MODE_LABELS[mode]} coverage: "
            f"{_coverage(target, lower, upper):.3f}"
        )
        print(
            f"  {MODE_LABELS[mode]} width quantiles: "
            f"{_format_quantiles(upper - lower)}"
        )
    print(f"  weight norm quantiles: {_format_quantiles(result.test_norms)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Optional path for saving the plot. If omitted, show a window.",
    )
    parser.add_argument(
        "--bratp-n-jobs",
        type=int,
        default=1,
        help="Optional joblib worker count for BRAT-P slot fitting.",
    )
    args = parser.parse_args()
    if args.output:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

    rng = np.random.default_rng(0)
    n_samples = 900
    noise_std = 0.35
    X = rng.uniform(0.0, 1.0, size=(n_samples, 3))
    signal = _signal(X)
    components = _components(X)
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
        _signal_calib,
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

    path_grid = np.linspace(0.0, 1.0, 120)
    X_path = np.column_stack(
        [
            path_grid,
            np.full_like(path_grid, 0.5),
            np.full_like(path_grid, 0.5),
        ]
    )
    path_signal = _signal(X_path)

    results = [
        _fit_interval_model(
            name="BRAT-D",
            model=_make_brat_d(random_state=0),
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            X_path=X_path,
        ),
        _fit_interval_model(
            name="BRAT-P",
            model=_make_brat_p(random_state=0, n_jobs=args.bratp_n_jobs),
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test,
            X_path=X_path,
        ),
    ]

    iebm_result = _fit_interval_model(
        name="IEBM",
        model=_make_iebm(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_calib=X_calib,
        y_calib=y_calib,
        X_test=X_test,
        X_path=X_path,
    )

    component_means = np.mean(components_train, axis=0)
    feature_intervals = {}
    feature_seconds = {}
    for feature_idx in range(X.shape[1]):
        feature_intervals[feature_idx] = {}
        for mode in MODES:
            start = time.perf_counter()
            feature_intervals[feature_idx][mode] = (
                iebm_result.model.predict_feature_intervals(
                    feature_idx,
                    path_grid,
                    level=0.95,
                    mode=mode,
                    include_intercept=False,
                )
            )
            feature_seconds[(feature_idx, mode)] = time.perf_counter() - start

    import matplotlib

    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 3, figsize=(16, 17), sharex=False)

    for row_idx, result in enumerate(results):
        for col_idx, mode in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            lower, upper, pred = result.path_intervals[mode]
            ax.fill_between(
                path_grid,
                lower,
                upper,
                color=MODE_COLORS[mode],
                alpha=0.18,
                label=f"95% {MODE_LABELS[mode]}",
            )
            ax.plot(
                path_grid,
                path_signal,
                color="black",
                linewidth=1.8,
                label="signal",
            )
            ax.plot(
                path_grid,
                pred,
                color=MODE_COLORS[mode],
                linewidth=1.7,
                label="fit",
            )
            test_lower, test_upper, _ = result.intervals[mode]
            target = _coverage_target(mode, signal_test=signal_test, y_test=y_test)
            coverage = _coverage(target, test_lower, test_upper)
            ax.set_title(f"{result.name} {MODE_LABELS[mode]} on x0 slice")
            ax.set_xlabel("x0 with x1=x2=0.5")
            ax.set_ylabel("response")
            ax.text(
                0.03,
                0.95,
                f"test coverage={coverage:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower left", fontsize=8)

    for feature_idx in range(X.shape[1]):
        row_idx = 2 + feature_idx
        truth = _component_grid(feature_idx, path_grid) - component_means[feature_idx]
        for col_idx, mode in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            lower, upper, pred = feature_intervals[feature_idx][mode]
            ax.fill_between(
                path_grid,
                lower,
                upper,
                color=MODE_COLORS[mode],
                alpha=0.18,
                label=f"95% {MODE_LABELS[mode]}",
            )
            ax.plot(
                path_grid,
                truth,
                color="black",
                linewidth=1.8,
                label="true partial",
            )
            ax.plot(
                path_grid,
                pred,
                color=MODE_COLORS[mode],
                linewidth=1.7,
                label="IEBM partial",
            )
            coverage = _coverage(truth, lower, upper)
            ax.set_title(f"IEBM x{feature_idx} partial {MODE_LABELS[mode]}")
            ax.set_xlabel(f"x{feature_idx}")
            ax.set_ylabel("centered contribution")
            ax.text(
                0.03,
                0.95,
                f"partial coverage={coverage:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
            if feature_idx == 0 and col_idx == 0:
                ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Boulevard interval API: BRAT-D, BRAT-P, and IEBM",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    print("Boulevard interval API demo")
    print(
        "train/calib/test sizes: "
        f"{X_train.shape[0]}/{X_calib.shape[0]}/{X_test.shape[0]}"
    )
    print(f"noise variance: {noise_std**2:.6f}")
    print(
        "interval API: "
        "predict_intervals(..., mode='confidence|prediction|reproduction')"
    )

    for result in [*results, iebm_result]:
        _print_model_summary(result, y_test=y_test, signal_test=signal_test)

    print("")
    print("IEBM feature interval diagnostics")
    for feature_idx in range(X.shape[1]):
        truth = _component_grid(feature_idx, path_grid) - component_means[feature_idx]
        print(f"  feature x{feature_idx}")
        for mode in MODES:
            lower, upper, _ = feature_intervals[feature_idx][mode]
            print(
                f"    {MODE_LABELS[mode]} coverage/seconds/width q50: "
                f"{_coverage(truth, lower, upper):.3f}/"
                f"{feature_seconds[(feature_idx, mode)]:.5f}/"
                f"{np.median(upper - lower):.4f}"
            )

    if args.output:
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()


if __name__ == "__main__":
    main()

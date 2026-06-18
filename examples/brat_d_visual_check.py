"""Visual smoke check and variance diagnostic for BRAT-D intervals."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import boulevard as bd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Optional path for saving the plot. If omitted, show an interactive window.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    noise_std = 0.15
    X = np.linspace(0, 1, 1000).reshape(-1, 1)
    truth = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 0] ** 2
    y = truth + rng.normal(scale=noise_std, size=X.shape[0])

    indices = np.arange(X.shape[0])
    train_idx, calib_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=0,
    )
    X_train, X_calib = X[train_idx], X[calib_idx]
    y_train, y_calib = y[train_idx], y[calib_idx]

    def make_model(random_state: int) -> bd.BRATDRegressor:
        return bd.BRATDRegressor(
            n_estimators=500,
            learning_rate=0.3,
            max_depth=8,
            min_samples_split=2,
            subsample_rate=0.6,
            dropout_rate=0.8,
            random_state=random_state,
        )

    model = make_model(random_state=0)
    model.fit(X_train, y_train)
    model.prepare_inference(X_calib, y_calib)

    train_residuals = y_train - model.predict(X_train)
    calib_residuals = y_calib - model.predict(X_calib)
    calib_noise = y_calib - truth[calib_idx]
    true_noise_var = noise_std**2
    oracle_calib_noise_var = float(np.var(calib_noise, ddof=1))
    train_centered_resid_var = float(np.var(train_residuals, ddof=1))
    calib_centered_resid_var = float(np.var(calib_residuals, ddof=1))
    calib_uncentered_resid_mse = float(np.mean(calib_residuals**2))

    pred = model.predict(X)
    confidence_lower, confidence_upper = model.confidence_interval(X, alpha=0.1)
    asymptotic_pred_lower, asymptotic_pred_upper = model.prediction_interval(X, alpha=0.1)
    calibrated_pred_lower, calibrated_pred_upper = model.prediction_interval(
        X,
        alpha=0.1,
        calibrated=True,
    )
    reproduction_lower, reproduction_upper = model.reproduction_interval(X, alpha=0.1)
    r_norm = model.weight_norms(X)
    confidence_half_width = np.maximum(
        (confidence_upper - confidence_lower) / 2,
        np.finfo(float).eps,
    )
    ci_error_ratio = np.abs(pred - truth) / confidence_half_width
    ci_miss_mask = ci_error_ratio > 1

    n_future_responses = 200
    future_y = truth[None, :] + rng.normal(
        scale=noise_std,
        size=(n_future_responses, X.shape[0]),
    )
    n_reproductions = 12
    reproduction_predictions = []
    for rep_idx in range(n_reproductions):
        rep_y = truth + rng.normal(scale=noise_std, size=X.shape[0])
        rep_model = make_model(random_state=1000 + rep_idx)
        rep_model.fit(X[train_idx], rep_y[train_idx])
        reproduction_predictions.append(rep_model.predict(X))
    reproduction_predictions = np.asarray(reproduction_predictions)

    ci_coverage = np.mean((confidence_lower <= truth) & (truth <= confidence_upper))
    raw_pi_coverage = np.mean(
        (asymptotic_pred_lower[None, :] <= future_y)
        & (future_y <= asymptotic_pred_upper[None, :])
    )
    calibrated_pi_coverage = np.mean(
        (calibrated_pred_lower[None, :] <= future_y)
        & (future_y <= calibrated_pred_upper[None, :])
    )
    ri_coverage = np.mean(
        (reproduction_lower[None, :] <= reproduction_predictions)
        & (reproduction_predictions <= reproduction_upper[None, :])
    )

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.75, 0.65])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
    ]

    def draw_panel(ax, lower, upper, title, band_color, extra_lines=None):
        ax.scatter(X_train[:, 0], y_train, s=10, alpha=0.28, label="train")
        ax.plot(X[:, 0], truth, color="black", linewidth=2, label="truth")
        ax.plot(X[:, 0], pred, color="#1f77b4", linewidth=2, label="BRAT-D")
        if extra_lines is not None:
            for line in extra_lines:
                ax.plot(X[:, 0], line, color="0.45", linewidth=0.8, alpha=0.18)
        ax.fill_between(
            X[:, 0],
            lower,
            upper,
            color=band_color,
            alpha=0.18,
            label="90% interval",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=8)

    draw_panel(
        axes[0],
        confidence_lower,
        confidence_upper,
        f"Asymptotic confidence interval\ncoverage vs truth: {ci_coverage:.1%}",
        "#2ca02c",
    )
    draw_panel(
        axes[1],
        asymptotic_pred_lower,
        asymptotic_pred_upper,
        f"Asymptotic prediction interval\ncoverage vs noisy y: {raw_pi_coverage:.1%}",
        "#ff7f0e",
    )
    draw_panel(
        axes[2],
        reproduction_lower,
        reproduction_upper,
        f"Asymptotic reproduction interval\ncoverage vs refit preds: {ri_coverage:.1%}",
        "#9467bd",
        extra_lines=reproduction_predictions[:8],
    )

    variance_ax = fig.add_subplot(grid[1, 1])
    variance_labels = [
        "true\nnoise",
        "oracle\ncalib",
        "calib\ncentered",
        "calib\nMSE",
        "train\ncentered",
    ]
    variance_values = [
        true_noise_var,
        oracle_calib_noise_var,
        calib_centered_resid_var,
        calib_uncentered_resid_mse,
        train_centered_resid_var,
    ]
    variance_ax.bar(
        np.arange(len(variance_values)),
        variance_values,
        color=["#4c78a8", "#72b7b2", "#59a14f", "#f28e2b", "#bab0ac"],
    )
    variance_ax.axhline(true_noise_var, color="black", linewidth=1, linestyle="--")
    variance_ax.set_title("Residual variance diagnostic")
    variance_ax.set_xticks(np.arange(len(variance_labels)), variance_labels)
    variance_ax.set_ylabel("variance")

    width_ax = fig.add_subplot(grid[2, 0])
    width_ax.plot(
        X[:, 0],
        confidence_upper - confidence_lower,
        label="asymptotic CI width",
        color="#2ca02c",
    )
    width_ax.plot(
        X[:, 0],
        asymptotic_pred_upper - asymptotic_pred_lower,
        label="asymptotic PI width",
        color="#ff7f0e",
    )
    width_ax.plot(
        X[:, 0],
        calibrated_pred_upper - calibrated_pred_lower,
        label="calibration-adjusted asymptotic PI width",
        color="#ffbb78",
        linestyle="--",
    )
    width_ax.plot(
        X[:, 0],
        reproduction_upper - reproduction_lower,
        label="reproduction width",
        color="#9467bd",
    )
    width_ax.set_title("Interval widths")
    width_ax.set_xlabel("x")
    width_ax.set_ylabel("width")
    width_ax.legend(loc="best", fontsize=8)

    weight_ax = fig.add_subplot(grid[2, 1])
    weight_ax.plot(X[:, 0], r_norm, color="#d62728", label="||r_n(x)||")
    weight_ax.set_title("BRAT-D kernel weight norm")
    weight_ax.set_xlabel("x")
    weight_ax.set_ylabel("norm")
    weight_ax.legend(loc="best", fontsize=8)

    ratio_ax = fig.add_subplot(grid[3, :])
    ratio_ax.plot(
        X[:, 0],
        ci_error_ratio,
        color="#6b4c9a",
        label="|prediction - truth| / CI half-width",
    )
    ratio_ax.scatter(
        X[ci_miss_mask, 0],
        ci_error_ratio[ci_miss_mask],
        color="#d62728",
        s=12,
        label="CI misses truth",
        zorder=3,
    )
    ratio_ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ratio_ax.set_title("Confidence interval miss diagnostic")
    ratio_ax.set_xlabel("x")
    ratio_ax.set_ylabel("error / half-width")
    ratio_ax.legend(loc="best", fontsize=8)

    print(f"sigma_hat2: {model.sigma_hat2_:.6f}")
    print(f"true noise variance: {true_noise_var:.6f}")
    print(f"oracle calibration noise variance: {oracle_calib_noise_var:.6f}")
    print(f"calibration residual mean: {np.mean(calib_residuals):.6f}")
    print(f"calibration centered residual variance: {calib_centered_resid_var:.6f}")
    print(f"calibration uncentered residual MSE: {calib_uncentered_resid_mse:.6f}")
    print(f"training centered residual variance: {train_centered_resid_var:.6f}")
    print(
        "sigma_hat2 equals calibration centered residual variance: "
        f"{np.allclose(model.sigma_hat2_, calib_centered_resid_var)}"
    )
    print(f"r_norm range: [{r_norm.min():.6f}, {r_norm.max():.6f}]")
    print(
        "CI error/half-width ratio range: "
        f"[{ci_error_ratio.min():.6f}, {ci_error_ratio.max():.6f}]"
    )
    print(f"CI miss grid points: {ci_miss_mask.sum()} / {ci_miss_mask.size}")
    print(f"CI coverage vs truth E[y|x]: {ci_coverage:.3f}")
    print(f"raw asymptotic PI coverage vs fresh noisy y: {raw_pi_coverage:.3f}")
    print(
        "calibration-adjusted asymptotic PI coverage vs fresh noisy y: "
        f"{calibrated_pi_coverage:.3f}"
    )
    print(f"RI coverage vs independently refit BRAT-D predictions: {ri_coverage:.3f}")
    print(
        "asymptotic CI width range: "
        f"[{(confidence_upper - confidence_lower).min():.6f}, "
        f"{(confidence_upper - confidence_lower).max():.6f}]"
    )
    print(
        "asymptotic PI width range: "
        f"[{(asymptotic_pred_upper - asymptotic_pred_lower).min():.6f}, "
        f"{(asymptotic_pred_upper - asymptotic_pred_lower).max():.6f}]"
    )
    print(
        "raw asymptotic PI width range: "
        f"[{(asymptotic_pred_upper - asymptotic_pred_lower).min():.6f}, "
        f"{(asymptotic_pred_upper - asymptotic_pred_lower).max():.6f}]"
    )
    print(
        "calibration-adjusted asymptotic PI width range: "
        f"[{(calibrated_pred_upper - calibrated_pred_lower).min():.6f}, "
        f"{(calibrated_pred_upper - calibrated_pred_lower).max():.6f}]"
    )

    fig.suptitle("BRAT-D visual check")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=160)
        print(f"Saved BRAT-D visual check to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

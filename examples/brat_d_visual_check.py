"""Visual smoke check for BRAT-D predictions and intervals."""

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
    X = np.linspace(0, 1, 240).reshape(-1, 1)
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
            n_estimators=240,
            learning_rate=0.8,
            max_depth=6,
            min_samples_split=2,
            subsample_rate=0.9,
            dropout_rate=0.2,
            random_state=random_state,
        )

    model = make_model(random_state=0)
    model.fit(X_train, y_train)
    model.prepare_inference(X_calib, y_calib)

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

    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75])
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

    print(f"sigma_hat2: {model.sigma_hat2_:.6f}")
    print(f"r_norm range: [{r_norm.min():.6f}, {r_norm.max():.6f}]")
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

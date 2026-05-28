"""Visual smoke check for BRAT-D predictions."""

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
    X = np.linspace(0, 1, 240).reshape(-1, 1)
    truth = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 0] ** 2
    y = truth + rng.normal(scale=0.15, size=X.shape[0])

    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )

    model = bd.BRATDRegressor(
        n_estimators=120,
        learning_rate=0.6,
        max_depth=3,
        subsample_rate=0.7,
        dropout_rate=0.4,
        random_state=0,
    )
    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib, alpha=0.1)

    pred = model.predict(X)
    lower, upper = model.predict_interval(X)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(X_train[:, 0], y_train, s=12, alpha=0.35, label="train")
    ax.plot(X[:, 0], truth, color="black", linewidth=2, label="truth")
    ax.plot(X[:, 0], pred, color="#1f77b4", linewidth=2, label="BRAT-D")
    ax.fill_between(
        X[:, 0],
        lower,
        upper,
        color="#1f77b4",
        alpha=0.18,
        label="90% conformal interval",
    )
    ax.set_title("BRAT-D prediction curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=160)
        print(f"Saved BRAT-D visual check to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

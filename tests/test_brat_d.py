import os

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, train_test_split

import boulevard as bd


def test_brat_d_fit_predict_metadata():
    X, y = make_regression(
        n_samples=120,
        n_features=5,
        noise=1.0,
        random_state=0,
    )

    model = bd.BRATDRegressor(
        n_estimators=8,
        learning_rate=0.5,
        max_depth=3,
        subsample_rate=0.7,
        dropout_rate=0.4,
        random_state=0,
    )

    model.fit(X, y)
    pred = model.predict(X)
    leaves = model.apply_leaf_indices(X)

    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    assert leaves.shape == (X.shape[0], model.n_estimators)
    assert model.in_bag_matrix_.shape == (X.shape[0], model.n_estimators)
    assert model.leaf_assignments_.shape == (X.shape[0], model.n_estimators)
    assert len(model.estimators_) == model.n_estimators


def test_brat_d_random_state_is_deterministic():
    X, y = make_regression(
        n_samples=100,
        n_features=4,
        noise=1.0,
        random_state=0,
    )

    params = dict(
        n_estimators=6,
        learning_rate=0.5,
        max_depth=2,
        subsample_rate=0.8,
        dropout_rate=0.5,
        random_state=12,
    )
    first = bd.BRATDRegressor(**params).fit(X, y).predict(X)
    second = bd.BRATDRegressor(**params).fit(X, y).predict(X)

    np.testing.assert_allclose(first, second)


def test_brat_d_sklearn_clone_and_cross_val_score():
    X, y = make_regression(
        n_samples=90,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    model = bd.BRATDRegressor(
        n_estimators=4,
        max_depth=2,
        random_state=0,
    )

    cloned = clone(model)
    assert cloned.get_params()["n_estimators"] == 4

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="neg_mean_squared_error",
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_brat_d_conformal_interval():
    X, y = make_regression(
        n_samples=180,
        n_features=4,
        noise=5.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    model = bd.BRATDRegressor(
        n_estimators=8,
        max_depth=3,
        random_state=0,
    )
    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib, alpha=0.1)

    lower, upper = model.predict_interval(X_calib[:10])

    assert lower.shape == upper.shape == (10,)
    assert np.all(lower <= upper)


def test_brat_d_visual_prediction_curve(tmp_path):
    if os.environ.get("BOULEVARD_RUN_VISUAL_TESTS") != "1":
        pytest.skip("Set BOULEVARD_RUN_VISUAL_TESTS=1 to generate visual artifacts.")

    plt = pytest.importorskip("matplotlib.pyplot")

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

    output_path = tmp_path / "brat_d_prediction_curve.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"BRAT-D visual artifact: {output_path}")
    assert output_path.exists()

import os
from statistics import NormalDist

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


def test_brat_d_residual_dropout_uses_independent_bernoulli_sampling():
    class ConstantTree:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(X.shape[0], self.value, dtype=float)

    X = np.zeros((3, 1))
    y = np.full(3, 1000.0)
    model = bd.BRATDRegressor(learning_rate=2.0, dropout_rate=0.5)
    model.estimators_ = [
        ConstantTree(1.0),
        ConstantTree(10.0),
        ConstantTree(100.0),
        ConstantTree(1000.0),
    ]

    residuals = model._residuals_for_next_tree(X, y, np.random.default_rng(0))

    # Seed 0 draws [0.637, 0.270, 0.041, 0.017], so Bernoulli(q=0.5)
    # keeps the last three trees. The residual denominator is all previous trees.
    expected = y - (2.0 / 4.0) * (10.0 + 100.0 + 1000.0)
    np.testing.assert_allclose(residuals, np.full(3, expected))


def test_brat_d_rejects_full_dropout_random_forest_limit():
    X, y = make_regression(
        n_samples=40,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    model = bd.BRATDRegressor(dropout_rate=1.0)

    with pytest.raises(ValueError, match="random-forest limit"):
        model.fit(X, y)


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

    lower, upper = model.predict_interval(X_calib[:10], method="conformal")

    assert lower.shape == upper.shape == (10,)
    assert np.all(lower <= upper)


def test_brat_d_asymptotic_intervals():
    X, y = make_regression(
        n_samples=140,
        n_features=4,
        noise=2.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    model = bd.BRATDRegressor(
        n_estimators=12,
        learning_rate=0.6,
        max_depth=3,
        subsample_rate=0.8,
        dropout_rate=0.4,
        random_state=0,
    )
    model.fit(X_train, y_train)
    model.prepare_inference(X_calib, y_calib)

    assert model.kernel_matrix_.shape == (X_train.shape[0], X_train.shape[0])
    assert model.sigma_hat2_ >= 0

    ci_lower, ci_upper = model.confidence_interval(X_calib[:8])
    pi_lower, pi_upper = model.prediction_interval(X_calib[:8])
    raw_pi_lower, raw_pi_upper = model.prediction_interval(
        X_calib[:8],
        calibrated=False,
    )
    ri_lower, ri_upper = model.reproduction_interval(X_calib[:8])
    pi2_lower, pi2_upper = model.predict_interval(X_calib[:8], method="asymptotic")

    for lower, upper in [
        (ci_lower, ci_upper),
        (pi_lower, pi_upper),
        (raw_pi_lower, raw_pi_upper),
        (ri_lower, ri_upper),
        (pi2_lower, pi2_upper),
    ]:
        assert lower.shape == upper.shape == (8,)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower <= upper)


def test_brat_d_asymptotic_intervals_use_signal_corrected_prediction_scale():
    X = np.zeros((2, 1))
    model = bd.BRATDRegressor(learning_rate=2.0, dropout_rate=0.25)
    model.sigma_hat2_ = 9.0
    model.predict = lambda X: np.full(X.shape[0], 10.0)
    model._weight_norms = lambda X: np.full(X.shape[0], 4.0)

    alpha = 0.2
    z = NormalDist().inv_cdf(1 - alpha / 2)
    prediction_scale = (1 + 2.0 * (1 - 0.25)) / 2.0

    ci_lower, ci_upper = model.confidence_interval(X, alpha=alpha)
    ci_half_width = z * prediction_scale * 3.0 * 4.0
    np.testing.assert_allclose(ci_lower, np.full(2, 10.0 - ci_half_width))
    np.testing.assert_allclose(ci_upper, np.full(2, 10.0 + ci_half_width))

    pi_lower, pi_upper = model.prediction_interval(X, alpha=alpha)
    pi_half_width = z * np.sqrt(9.0 * (1 + (prediction_scale * 4.0) ** 2))
    np.testing.assert_allclose(pi_lower, np.full(2, 10.0 - pi_half_width))
    np.testing.assert_allclose(pi_upper, np.full(2, 10.0 + pi_half_width))

    ri_lower, ri_upper = model.reproduction_interval(X, alpha=alpha)
    ri_half_width = z * np.sqrt(2) * prediction_scale * 3.0 * 4.0
    np.testing.assert_allclose(ri_lower, np.full(2, 10.0 - ri_half_width))
    np.testing.assert_allclose(ri_upper, np.full(2, 10.0 + ri_half_width))


def test_brat_d_asymptotic_intervals_require_prepare_inference():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    model = bd.BRATDRegressor(
        n_estimators=5,
        random_state=0,
    ).fit(X, y)

    with pytest.raises(RuntimeError, match="prepare_inference"):
        model.confidence_interval(X[:3])


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
    lower, upper = model.predict_interval(X, method="conformal")

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

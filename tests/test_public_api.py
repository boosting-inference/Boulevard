import numpy as np
from sklearn.base import clone

import boulevard as bd


def _make_regression_data(n_samples=90, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    signal = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] - 0.5)
    y = signal + rng.normal(scale=0.1, size=n_samples)
    return X, y


def test_top_level_public_api_names():
    assert bd.__all__ == [
        "BRATDHistGradientBoostingRegressor",
        "BRATPHistGradientBoostingRegressor",
        "IEBMRegressor",
        "__version__",
    ]
    assert not hasattr(bd, "XGBRegressor")


def test_top_level_brat_d_is_sklearn_compatible():
    model = bd.BRATDHistGradientBoostingRegressor(
        max_iter=3,
        learning_rate=0.5,
        dropout_rate=0.2,
        max_leaf_nodes=4,
        min_samples_leaf=4,
        max_bins=16,
        early_stopping=False,
        random_state=0,
    )

    cloned = clone(model)
    assert cloned.get_params()["max_iter"] == 3
    assert cloned.get_params()["dropout_rate"] == 0.2

    model.set_params(max_iter=4)
    assert model.max_iter == 4


def test_top_level_brat_p_is_sklearn_compatible():
    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=2,
        max_leaf_nodes=4,
        min_samples_leaf=4,
        max_bins=16,
        early_stopping=False,
        random_state=0,
    )

    cloned = clone(model)
    assert cloned.get_params()["n_rounds"] == 3
    assert cloned.get_params()["trees_per_round"] == 2

    model.set_params(n_rounds=4)
    assert model.n_rounds == 4


def test_top_level_iebm_is_sklearn_compatible():
    model = bd.IEBMRegressor(
        max_rounds=5,
        max_bins=8,
        max_depth=2,
        min_samples_leaf=3,
        random_state=0,
    )

    cloned = clone(model)
    assert cloned.get_params()["max_rounds"] == 5
    assert cloned.get_params()["max_depth"] == 2

    model.set_params(max_rounds=6)
    assert model.max_rounds == 6


def test_top_level_brat_d_user_workflow():
    X, y = _make_regression_data(n_samples=100, random_state=1)
    model = bd.BRATDHistGradientBoostingRegressor(
        max_iter=4,
        learning_rate=0.6,
        dropout_rate=0.2,
        max_leaf_nodes=4,
        min_samples_leaf=4,
        max_bins=16,
        early_stopping=False,
        random_state=0,
    )

    model.fit(X[:70], y[:70])
    model.prepare_inference(X[70:], y[70:])

    pred = model.predict(X[70:75])
    ci_lower, ci_upper = model.confidence_interval(X[70:75])
    pi_lower, pi_upper = model.prediction_interval(X[70:75])

    assert pred.shape == (5,)
    assert ci_lower.shape == ci_upper.shape == pred.shape
    assert pi_lower.shape == pi_upper.shape == pred.shape
    assert np.all(np.isfinite(pred))
    assert np.all(ci_lower <= ci_upper)
    assert np.all(pi_lower <= pi_upper)


def test_top_level_brat_p_user_workflow():
    X, y = _make_regression_data(n_samples=100, random_state=2)
    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=2,
        trees_per_round=2,
        max_leaf_nodes=4,
        min_samples_leaf=4,
        max_bins=16,
        early_stopping=False,
        random_state=0,
    )

    model.fit(X[:70], y[:70])
    model.prepare_inference(X[70:], y[70:])

    pred = model.predict(X[70:75])
    ci_lower, ci_upper = model.confidence_interval(X[70:75])
    pi_lower, pi_upper = model.prediction_interval(X[70:75])

    assert pred.shape == (5,)
    assert ci_lower.shape == ci_upper.shape == pred.shape
    assert pi_lower.shape == pi_upper.shape == pred.shape
    assert np.all(np.isfinite(pred))
    assert np.all(ci_lower <= ci_upper)
    assert np.all(pi_lower <= pi_upper)


def test_top_level_iebm_user_workflow():
    X, y = _make_regression_data(n_samples=100, random_state=3)
    model = bd.IEBMRegressor(
        max_rounds=8,
        max_bins=8,
        max_depth=2,
        min_samples_leaf=4,
        random_state=0,
    )

    model.fit(X[:70], y[:70])
    model.prepare_inference(X[70:], y[70:])

    pred = model.predict(X[70:75])
    lower, upper, interval_pred = model.predict_intervals(
        X[70:75],
        level=0.95,
        mode="confidence",
    )

    assert pred.shape == (5,)
    assert lower.shape == upper.shape == interval_pred.shape == pred.shape
    assert np.allclose(pred, interval_pred)
    assert np.all(np.isfinite(pred))
    assert np.all(lower <= interval_pred)
    assert np.all(interval_pred <= upper)

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import boulevard as bd


def test_xgb_regressor_fit_predict():
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=1.0,
        random_state=0,
    )

    model = bd.XGBRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_xgb_regressor_leaf_indices():
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        noise=1.0,
        random_state=0,
    )

    model = bd.XGBRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )

    model.fit(X, y)
    leaf_indices = model.apply_leaf_indices(X)

    assert leaf_indices.shape[0] == X.shape[0]
    assert leaf_indices.ndim == 2


def test_xgb_regressor_conformal_interval():
    X, y = make_regression(
        n_samples=300,
        n_features=5,
        noise=5.0,
        random_state=0,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=0,
    )

    X_calib, X_test, y_calib, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=0,
    )

    model = bd.XGBRegressor(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib, alpha=0.1)

    lower, upper = model.predict_interval(X_test)

    assert lower.shape == upper.shape
    assert lower.shape[0] == X_test.shape[0]
    assert np.all(lower <= upper)
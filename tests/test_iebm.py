import numpy as np
import pytest
from sklearn.base import clone

from boulevard.estimators.interpretml import IEBMRegressor


def _make_additive_data(n_samples=160, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    signal = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] - 0.5)
    y = signal + rng.normal(scale=0.05, size=n_samples)
    return X, y, signal


def test_iebm_regressor_is_sklearn_cloneable():
    model = IEBMRegressor(
        max_rounds=5,
        max_bins=12,
        learning_rate=0.8,
        subsample_rate=0.7,
        truncation=3.0,
        max_leaves=3,
        min_samples_leaf=4,
        random_state=0,
    )

    cloned = clone(model)

    assert cloned.max_rounds == 5
    assert cloned.max_bins == 12
    assert cloned.learning_rate == 0.8
    assert cloned.subsample_rate == 0.7
    assert cloned.truncation == 3.0
    assert cloned.max_leaves == 3
    assert cloned.min_samples_leaf == 4
    assert cloned.random_state == 0


def test_iebm_fit_predict_smoke():
    X, y, signal = _make_additive_data()
    model = IEBMRegressor(
        max_rounds=20,
        max_bins=16,
        max_leaves=3,
        min_samples_leaf=5,
        random_state=0,
    )

    model.fit(X, y)
    pred = model.predict(X[:10])

    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
    assert model.apply_bins(X[:10]).shape == (10, X.shape[1])
    assert len(model.term_scores_) == X.shape[1]
    assert model.fit_diagnostics_["max_rounds"] == 20
    assert model.fit_diagnostics_["n_features"] == X.shape[1]

    baseline_rmse = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    model_rmse = float(np.sqrt(np.mean((model.predict(X) - signal) ** 2)))
    assert model_rmse < baseline_rmse


def test_iebm_fit_is_deterministic():
    X, y, _ = _make_additive_data()
    params = dict(
        max_rounds=12,
        max_bins=12,
        subsample_rate=0.8,
        max_leaves=2,
        min_samples_leaf=5,
        random_state=0,
    )

    first = IEBMRegressor(**params).fit(X, y).predict(X)
    second = IEBMRegressor(**params).fit(X, y).predict(X)

    np.testing.assert_allclose(first, second)


def test_iebm_terms_are_centered_on_training_bins():
    X, y, _ = _make_additive_data()
    model = IEBMRegressor(
        max_rounds=8,
        max_bins=10,
        max_leaves=2,
        min_samples_leaf=4,
        random_state=0,
    ).fit(X, y)

    for feature_idx, scores in enumerate(model.term_scores_):
        weighted_mean = np.dot(model.bin_counts_[feature_idx], scores) / X.shape[0]
        assert weighted_mean == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"max_bins": 1}, "max_bins"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"subsample_rate": 0.0}, "subsample_rate"),
        ({"subsample_rate": 1.5}, "subsample_rate"),
        ({"truncation": 0.0}, "truncation"),
        ({"max_leaves": 0}, "max_leaves"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
    ],
)
def test_iebm_rejects_invalid_parameters(params, message):
    X, y, _ = _make_additive_data(n_samples=20)
    model = IEBMRegressor(**params)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_iebm_rejects_nonfinite_input():
    X, y, _ = _make_additive_data(n_samples=20)
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN|finite numeric X"):
        IEBMRegressor(max_rounds=2).fit(X, y)


def test_iebm_sample_weight_is_not_supported_yet():
    X, y, _ = _make_additive_data(n_samples=20)

    with pytest.raises(NotImplementedError, match="sample_weight"):
        IEBMRegressor(max_rounds=2).fit(X, y, sample_weight=np.ones_like(y))

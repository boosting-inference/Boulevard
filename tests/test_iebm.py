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
        warmup_rounds=3,
        truncation=3.0,
        max_leaves=3,
        max_depth=None,
        min_samples_leaf=4,
        leave_one_out=True,
        random_state=0,
    )

    cloned = clone(model)

    assert cloned.max_rounds == 5
    assert cloned.max_bins == 12
    assert cloned.learning_rate == 0.8
    assert cloned.subsample_rate == 0.7
    assert cloned.warmup_rounds == 3
    assert cloned.truncation == 3.0
    assert cloned.max_leaves == 3
    assert cloned.max_depth is None
    assert cloned.min_samples_leaf == 4
    assert cloned.leave_one_out is True
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
    assert model.term_features_ == [(0,), (1,)]
    assert model.term_names_ == ["x0", "x1"]
    assert len(model.bin_weights_) == X.shape[1]
    assert model.fit_diagnostics_["max_rounds"] == 20
    assert model.fit_diagnostics_["warmup_rounds"] == 20
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


def test_iebm_default_full_residual_improves_one_feature_fit():
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 180).reshape(-1, 1)
    signal = np.sin(2 * np.pi * X[:, 0]) + 0.3 * np.cos(8 * np.pi * X[:, 0])
    y = signal + rng.normal(scale=0.03, size=X.shape[0])
    params = dict(
        max_rounds=40,
        max_bins=32,
        max_leaves=2,
        min_samples_leaf=4,
        random_state=0,
    )

    default_pred = IEBMRegressor(**params).fit(X, y).predict(X)
    leave_one_out_pred = (
        IEBMRegressor(**params, leave_one_out=True).fit(X, y).predict(X)
    )

    default_rmse = float(np.sqrt(np.mean((default_pred - signal) ** 2)))
    leave_one_out_rmse = float(np.sqrt(np.mean((leave_one_out_pred - signal) ** 2)))
    assert default_rmse < 0.7 * leave_one_out_rmse


def test_iebm_max_depth_alias_controls_effective_leaf_count():
    X, y, _ = _make_additive_data(n_samples=80)
    model = IEBMRegressor(
        max_rounds=4,
        max_bins=8,
        max_depth=2,
        min_samples_leaf=3,
        random_state=0,
    ).fit(X, y)

    assert model.max_depth == 2
    assert model.max_leaves_ == 4
    assert model.fit_diagnostics_["max_leaves"] == 4


def test_iebm_prepare_inference_and_weight_norms():
    X, y, _ = _make_additive_data(n_samples=120)
    X_train, y_train = X[:90], y[:90]
    X_calib, y_calib = X[90:], y[90:]
    model = IEBMRegressor(
        max_rounds=10,
        max_bins=12,
        max_leaves=3,
        min_samples_leaf=4,
        random_state=0,
    ).fit(X_train, y_train)

    returned = model.prepare_inference(X_calib, y_calib)
    norms = model.weight_norms(X_calib[:8])

    assert returned is model
    assert model.sigma_hat2_ > 0
    assert model.inference_diagnostics_["n_calibration_rows"] == X_calib.shape[0]
    assert model.inference_diagnostics_["structure_updates"] == 20
    assert norms.shape == (8,)
    assert np.all(np.isfinite(norms))
    assert np.all(norms >= 0)


def test_iebm_bin_space_weight_norm_matches_hand_calculation():
    model = IEBMRegressor()
    model.n_features_in_ = 1
    model.X_train_ = np.array([[0.1], [0.2], [0.9]])
    model.y_train_ = np.zeros(3)
    model.bin_edges_ = [np.array([0.5])]
    model.n_bins_ = np.array([2])
    model.bin_counts_ = [np.array([2.0, 1.0])]
    model.structure_sums_ = [np.eye(2)]
    model.structure_update_counts_ = np.array([1])
    model.term_scores_ = [np.zeros(2)]
    model.intercept_ = 0.0
    model.sigma_hat2_ = 1.0

    model._prepare_bin_inference_cache()

    expected = np.sqrt(6.0) / 7.0
    np.testing.assert_allclose(model._feature_bin_norm_sq_[0], [6.0 / 49.0, 6.0 / 49.0])
    np.testing.assert_allclose(
        model.weight_norms(np.array([[0.1], [0.9]])),
        [expected, expected],
    )


def test_iebm_predict_intervals_follow_ebm_api_shape():
    X, y, _ = _make_additive_data(n_samples=120)
    model = IEBMRegressor(
        max_rounds=10,
        max_bins=12,
        max_depth=2,
        min_samples_leaf=4,
        random_state=0,
    ).fit(X[:90], y[:90])
    model.prepare_inference(X[90:], y[90:])

    lower, upper, pred = model.predict_intervals(
        X[:7],
        level=0.95,
        mode="confidence",
    )
    feat_lower, feat_upper, feat_pred = model.predict_feature_intervals(
        0,
        X[:7, 0],
        level=0.95,
        mode="confidence",
    )

    assert lower.shape == upper.shape == pred.shape == (7,)
    assert feat_lower.shape == feat_upper.shape == feat_pred.shape == (7,)
    assert np.all(lower <= pred)
    assert np.all(pred <= upper)
    assert np.all(feat_lower <= feat_pred)
    assert np.all(feat_pred <= feat_upper)


def test_iebm_calibrate_intervals_reaches_calibration_coverage():
    X, y, _ = _make_additive_data(n_samples=180)
    X_train, y_train = X[:120], y[:120]
    X_calib, y_calib = X[120:], y[120:]
    level = 0.9
    model = IEBMRegressor(
        max_rounds=12,
        max_bins=12,
        max_depth=2,
        min_samples_leaf=4,
        random_state=0,
    ).fit(X_train, y_train)

    scale = model.calibrate_intervals(
        X_calib,
        y_calib,
        level=level,
        mode="prediction",
        propagate_to_ci_ri=True,
    )
    lower, upper, _ = model.predict_intervals(
        X_calib,
        level=level,
        mode="prediction",
    )
    coverage = np.mean((y_calib >= lower) & (y_calib <= upper))

    assert scale > 0
    assert coverage >= level
    assert model.interval_calibrations_[("prediction", level)] == pytest.approx(scale)
    assert model.interval_calibrations_[("confidence", level)] == pytest.approx(scale)
    assert model.interval_calibrations_[("reproduction", level)] == pytest.approx(scale)


def test_iebm_predict_intervals_reject_invalid_mode():
    X, y, _ = _make_additive_data(n_samples=30)
    model = IEBMRegressor(max_rounds=2, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="mode"):
        model.predict_intervals(X[:3], mode="bad")

    with pytest.raises(ValueError, match="feature_idx"):
        model.predict_feature_intervals(99, X[:3, 0])


def test_iebm_weight_norms_auto_prepare_inference():
    X, y, _ = _make_additive_data(n_samples=80)
    model = IEBMRegressor(
        max_rounds=6,
        max_bins=8,
        max_leaves=2,
        min_samples_leaf=3,
        random_state=0,
    ).fit(X, y)

    norms = model.weight_norms(X[:5])

    assert hasattr(model, "sigma_hat2_")
    assert model.inference_diagnostics_["n_calibration_rows"] == X.shape[0]
    assert norms.shape == (5,)


def test_iebm_prepare_inference_requires_paired_calibration_data():
    X, y, _ = _make_additive_data(n_samples=30)
    model = IEBMRegressor(max_rounds=2, random_state=0).fit(X, y)

    with pytest.raises(ValueError, match="provided together"):
        model.prepare_inference(X_calib=X)

    with pytest.raises(ValueError, match="same number of rows"):
        model.prepare_inference(X_calib=X[:5], y_calib=y[:4])

    with pytest.raises(ValueError, match="At least two residuals"):
        model.prepare_inference(X_calib=X[:1], y_calib=y[:1])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"max_bins": 1}, "max_bins"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"subsample_rate": 0.0}, "subsample_rate"),
        ({"subsample_rate": 1.5}, "subsample_rate"),
        ({"warmup_rounds": -1}, "warmup_rounds"),
        ({"truncation": 0.0}, "truncation"),
        ({"max_leaves": 0}, "max_leaves"),
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": 2, "max_leaves": 3}, "Specify only one"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"leave_one_out": 1}, "leave_one_out"),
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

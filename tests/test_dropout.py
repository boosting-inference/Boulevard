import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from boulevard.estimators.sklearn.dropout import DropoutBooster


def test_dropout_booster_skeleton_is_sklearn_cloneable():
    model = DropoutBooster(
        learning_rate=0.8,
        dropout_rate=0.2,
        subsample_rate=0.7,
        max_iter=3,
        early_stopping=False,
        random_state=0,
    )

    cloned = clone(model)

    assert cloned.learning_rate == 0.8
    assert cloned.dropout_rate == 0.2
    assert cloned.subsample_rate == 0.7
    assert cloned.max_iter == 3


def test_dropout_booster_fit_predict_smoke():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=5,
        learning_rate=0.8,
        dropout_rate=0.2,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=32,
        random_state=0,
    )

    model.fit(X, y)
    pred = model.predict(X[:7])

    assert len(model._predictors) == 5
    assert model.n_iter_ == 5
    assert pred.shape == (7,)
    assert np.all(np.isfinite(pred))

    diagnostics = model.fit_diagnostics_
    expected_keys = {
        "total_seconds",
        "binning_seconds",
        "cell_metadata_seconds",
        "residual_seconds",
        "gradient_seconds",
        "grower_setup_seconds",
        "grower_grow_seconds",
        "predictor_seconds",
        "training_prediction_cache_seconds",
        "score_seconds",
        "residual_tree_prediction_calls",
        "residual_tree_traversal_calls",
        "residual_cache_hit_rounds",
        "residual_zero_keep_rounds",
        "sampled_training_rows",
    }
    assert set(diagnostics) == expected_keys
    assert diagnostics["total_seconds"] > 0
    assert diagnostics["residual_tree_prediction_calls"] >= 0
    assert diagnostics["residual_tree_traversal_calls"] == 0
    assert diagnostics["residual_zero_keep_rounds"] >= 0
    assert diagnostics["sampled_training_rows"] == X.shape[0] * model.max_iter
    assert model._train_tree_predictions_.shape == (X.shape[0], model.max_iter)
    assert model.in_bag_matrix_.shape == (X.shape[0], model.max_iter)
    assert np.all(model.in_bag_matrix_)


def test_dropout_booster_caches_training_tree_predictions():
    X, y = make_regression(
        n_samples=50,
        n_features=2,
        noise=0.5,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=4,
        learning_rate=0.7,
        dropout_rate=0.0,
        max_leaf_nodes=4,
        min_samples_leaf=3,
        max_bins=16,
        random_state=0,
    ).fit(X, y)

    direct_columns = []
    for predictor_group in model._predictors:
        direct_columns.append(
            predictor_group[0].predict_binned(
                model.X_binned_train_,
                model._bin_mapper.missing_values_bin_idx_,
                model._effective_n_threads(),
            )
        )

    np.testing.assert_allclose(
        model._train_tree_predictions_,
        np.column_stack(direct_columns),
    )
    assert model.fit_diagnostics_["residual_tree_prediction_calls"] == 6
    assert model.fit_diagnostics_["residual_tree_traversal_calls"] == 0
    assert model.fit_diagnostics_["residual_cache_hit_rounds"] == 3


def test_dropout_booster_is_deterministic():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    params = dict(
        max_iter=5,
        learning_rate=0.8,
        dropout_rate=0.2,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=32,
        random_state=0,
    )

    first = DropoutBooster(**params).fit(X, y).predict(X)
    second = DropoutBooster(**params).fit(X, y).predict(X)

    np.testing.assert_allclose(first, second)


def test_dropout_booster_rejects_unsupported_sklearn_modes():
    model = DropoutBooster(early_stopping=True)

    with pytest.raises(ValueError, match="early_stopping"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"loss": "absolute_error"}, "squared_error"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"dropout_rate": 1.0}, "dropout_rate"),
        ({"subsample_rate": 0.0}, "subsample_rate"),
        ({"warm_start": True}, "warm_start"),
        ({"categorical_features": [0]}, "categorical_features"),
        ({"monotonic_cst": [1]}, "monotonic_cst"),
        ({"interaction_cst": [{0}]}, "interaction_cst"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_bins": 1}, "max_bins"),
        ({"max_bins": 256}, "max_bins"),
    ],
)
def test_dropout_booster_rejects_unsupported_parameters(params, message):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = DropoutBooster(
        early_stopping=False,
        **params,
    )

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_dropout_booster_rejects_invalid_sample_weight():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = DropoutBooster(
        max_iter=2,
        min_samples_leaf=1,
        early_stopping=False,
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(X, y, sample_weight=[[1.0], [1.0], [1.0]])

    with pytest.raises(ValueError, match="negative"):
        model.fit(X, y, sample_weight=[1.0, -1.0, 1.0])


def test_dropout_booster_predict_applies_signal_correction():
    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=0.5,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=6,
        learning_rate=0.7,
        dropout_rate=0.2,
        max_leaf_nodes=5,
        min_samples_leaf=4,
        max_bins=16,
        random_state=0,
    ).fit(X, y)

    X_eval = X[:8]
    X_binned = model._bin_data(X_eval)
    tree_sum = model._predict_tree_sum_binned(
        X_binned,
        selected=None,
        n_threads=model._effective_n_threads(),
    )
    q = 1 - model.dropout_rate
    raw_boulevard = (model.learning_rate / len(model._predictors)) * tree_sum
    expected = ((1 + model.learning_rate * q) / model.learning_rate) * raw_boulevard

    np.testing.assert_allclose(model.predict(X_eval), expected)


def test_dropout_booster_row_subsampling_is_deterministic():
    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=0.5,
        random_state=0,
    )
    params = dict(
        max_iter=5,
        learning_rate=0.7,
        dropout_rate=0.2,
        subsample_rate=0.6,
        max_leaf_nodes=5,
        min_samples_leaf=4,
        max_bins=16,
        random_state=0,
    )

    first = DropoutBooster(**params).fit(X, y)
    second = DropoutBooster(**params).fit(X, y)

    expected_rows_per_tree = int(np.ceil(params["subsample_rate"] * X.shape[0]))
    np.testing.assert_array_equal(first.in_bag_matrix_, second.in_bag_matrix_)
    assert np.all(first.in_bag_matrix_.sum(axis=0) == expected_rows_per_tree)
    assert first.fit_diagnostics_["sampled_training_rows"] == (
        expected_rows_per_tree * params["max_iter"]
    )


def test_dropout_booster_cell_metadata_compresses_duplicate_bins():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    model = DropoutBooster(
        max_iter=3,
        learning_rate=0.5,
        dropout_rate=0.0,
        max_leaf_nodes=3,
        min_samples_leaf=1,
        max_bins=8,
        random_state=0,
    ).fit(X, y)

    assert model.observed_cells_.shape[0] == 3
    np.testing.assert_array_equal(model.cell_counts_, np.array([2.0, 2.0, 2.0]))

    cell_indices = model.apply_cell_indices(X)
    assert cell_indices[0] == cell_indices[1]
    assert cell_indices[2] == cell_indices[3]
    assert cell_indices[4] == cell_indices[5]


def test_dropout_booster_prepare_inference_uses_centered_residual_variance():
    X, y = make_regression(
        n_samples=80,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=5,
        learning_rate=0.6,
        dropout_rate=0.25,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)

    with pytest.raises(ValueError, match="provided together"):
        model.prepare_inference(X_calib=X_calib)

    model.prepare_inference(X_calib, y_calib)

    residuals = y_calib - model.predict(X_calib)
    assert model.sigma_hat2_ == pytest.approx(float(np.var(residuals, ddof=1)))
    assert model.inference_method_ == "histogram_cell"


def test_dropout_booster_intervals_prepare_from_training_data_by_default():
    X, y = make_regression(
        n_samples=80,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=5,
        learning_rate=0.6,
        dropout_rate=0.25,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)

    ci_lower, ci_upper = model.confidence_interval(X_test[:4])

    residuals = y_train - model.predict(X_train)
    assert model.sigma_hat2_ == pytest.approx(float(np.var(residuals, ddof=1)))
    assert model.inference_method_ == "histogram_cell"
    assert ci_lower.shape == ci_upper.shape == (4,)
    assert np.all(ci_lower <= ci_upper)


def test_dropout_booster_interval_call_can_use_calibration_data():
    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=5,
        learning_rate=0.6,
        dropout_rate=0.25,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)

    pi_lower, pi_upper = model.prediction_interval(
        X_calib[:4],
        X_calib=X_calib,
        y_calib=y_calib,
    )

    residuals = y_calib - model.predict(X_calib)
    assert model.sigma_hat2_ == pytest.approx(float(np.var(residuals, ddof=1)))
    assert pi_lower.shape == pi_upper.shape == (4,)
    assert np.all(pi_lower <= pi_upper)

    with pytest.raises(ValueError, match="provided together"):
        model.confidence_interval(X_calib[:4], X_calib=X_calib)


def test_dropout_booster_cached_weight_norms_match_direct_solve_for_cells():
    levels = np.linspace(0.0, 1.0, 12)
    X = np.column_stack([levels, levels])
    y = np.sin(2 * np.pi * levels) + 0.2 * levels
    model = DropoutBooster(
        max_iter=6,
        learning_rate=0.5,
        dropout_rate=0.25,
        max_leaf_nodes=4,
        min_samples_leaf=1,
        max_bins=16,
        random_state=0,
    ).fit(X, y)
    model.prepare_inference()

    X_query = np.array(
        [
            [levels[0], levels[0]],
            [levels[3], levels[3]],
            [levels[0], levels[-1]],
            [levels[-1], levels[0]],
        ]
    )
    cell_indices = model.apply_cell_indices(X_query)
    assert np.any(cell_indices >= 0)
    assert np.any(cell_indices < 0)

    cached_norms = model.weight_norms(X_query)
    X_binned = model._bin_data(X_query)
    leaf_indices = model._apply_leaf_indices_binned(X_binned)
    kernel_vectors = model._cell_kernel_vector(leaf_indices)
    weights = model._solve_cell_brat_d_weights(kernel_vectors)
    direct_norms = np.sqrt(np.maximum((weights**2) @ model.cell_counts_, 0.0))

    np.testing.assert_allclose(cached_norms, direct_norms)


def test_dropout_booster_interval_widths_are_ordered():
    X, y = make_regression(
        n_samples=120,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=8,
        learning_rate=0.6,
        dropout_rate=0.25,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)
    model.prepare_inference(X_calib, y_calib)

    ci_lower, ci_upper = model.confidence_interval(X_calib[:8])
    pi_lower, pi_upper = model.prediction_interval(X_calib[:8])
    ri_lower, ri_upper = model.reproduction_interval(X_calib[:8])
    ci_lower_new, ci_upper_new, pred_new = model.predict_intervals(
        X_calib[:8],
        level=0.95,
        mode="confidence",
    )
    pi_lower_new, pi_upper_new, _ = model.predict_intervals(
        X_calib[:8],
        level=0.95,
        mode="prediction",
    )
    ri_lower_new, ri_upper_new, _ = model.predict_intervals(
        X_calib[:8],
        level=0.95,
        mode="reproduction",
    )

    ci_width = ci_upper - ci_lower
    pi_width = pi_upper - pi_lower
    ri_width = ri_upper - ri_lower

    np.testing.assert_allclose(ci_lower_new, ci_lower)
    np.testing.assert_allclose(ci_upper_new, ci_upper)
    np.testing.assert_allclose(pi_lower_new, pi_lower)
    np.testing.assert_allclose(pi_upper_new, pi_upper)
    np.testing.assert_allclose(ri_lower_new, ri_lower)
    np.testing.assert_allclose(ri_upper_new, ri_upper)
    np.testing.assert_allclose(pred_new, model.predict(X_calib[:8]))
    assert np.all(ci_width > 0)
    assert np.all(pi_width >= ci_width)
    np.testing.assert_allclose(ri_width, np.sqrt(2) * ci_width)

    with pytest.raises(ValueError, match="mode must be"):
        model.predict_intervals(X_calib[:2], mode="bad")
    with pytest.raises(ValueError, match="calibrated=True"):
        model.predict_intervals(X_calib[:2], mode="confidence", calibrated=True)


def test_dropout_booster_observed_cell_inference_smoke():
    X, y = make_regression(
        n_samples=120,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
    )
    model = DropoutBooster(
        max_iter=8,
        learning_rate=0.6,
        dropout_rate=0.25,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)

    model.prepare_inference(X_calib, y_calib)

    assert model.observed_cells_.ndim == 2
    assert model.cell_counts_.sum() == X_train.shape[0]
    assert model.cell_kernel_matrix_.shape == (
        model.observed_cells_.shape[0],
        model.observed_cells_.shape[0],
    )
    train_cell_indices = model.apply_cell_indices(X_train[:5])
    assert np.all(train_cell_indices >= 0)

    ci_lower, ci_upper = model.confidence_interval(X_calib[:6])
    pi_lower, pi_upper = model.prediction_interval(X_calib[:6])
    ri_lower, ri_upper = model.reproduction_interval(X_calib[:6])
    norms = model.weight_norms(X_calib[:6])
    train_norms = model.weight_norms(X_train[:5])

    for lower, upper in [
        (ci_lower, ci_upper),
        (pi_lower, pi_upper),
        (ri_lower, ri_upper),
    ]:
        assert lower.shape == upper.shape == (6,)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower <= upper)
    assert norms.shape == (6,)
    assert np.all(np.isfinite(norms))
    assert np.all(norms >= 0)
    np.testing.assert_allclose(
        train_norms,
        model.cell_weight_norms_[train_cell_indices],
    )

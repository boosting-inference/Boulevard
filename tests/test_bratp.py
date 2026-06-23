import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from boulevard.estimators.bratp import BRATPHistGradientBoostingRegressor


def test_brat_p_hist_skeleton_is_sklearn_cloneable():
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=4,
        subsample_rate=0.7,
        max_leaf_nodes=5,
        early_stopping=False,
        random_state=0,
        n_jobs=2,
    )

    cloned = clone(model)

    assert cloned.n_rounds == 3
    assert cloned.trees_per_round == 4
    assert cloned.subsample_rate == 0.7
    assert cloned.max_leaf_nodes == 5
    assert cloned.n_jobs == 2


def test_brat_p_hist_fit_predict_smoke():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=2,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=32,
        random_state=0,
    )

    model.fit(X, y)
    pred = model.predict(X[:7])

    total_trees = model.n_rounds * model.trees_per_round
    assert len(model._predictors) == total_trees
    assert model.n_iter_ == total_trees
    assert pred.shape == (7,)
    assert np.all(np.isfinite(pred))
    assert model._train_tree_predictions_.shape == (X.shape[0], total_trees)
    assert model._train_prediction_table_.shape == (
        model.n_rounds,
        model.trees_per_round,
        X.shape[0],
    )
    assert model.in_bag_matrix_.shape == (X.shape[0], total_trees)
    assert np.all(model.in_bag_matrix_)

    diagnostics = model.fit_diagnostics_
    assert diagnostics["n_rounds"] == model.n_rounds
    assert diagnostics["trees_per_round"] == model.trees_per_round
    assert diagnostics["total_trees"] == total_trees
    assert diagnostics["sampled_training_rows"] == X.shape[0] * total_trees


def test_brat_p_hist_residual_formula():
    y = np.array([10.0, 20.0])
    slot_prediction_sums = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ]
    )
    current_round_prediction_sum = np.array([1.0, 3.0])

    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=3,
    )
    residual = model._residuals_for_next_tree_binned(
        y,
        round_idx=2,
        slot_idx=1,
        previous_slot_prediction_sums=slot_prediction_sums,
        current_round_prediction_sum=current_round_prediction_sum,
    )
    expected_previous = (slot_prediction_sums[0] + slot_prediction_sums[2]) / 2

    np.testing.assert_allclose(residual, y - expected_previous)

    first_round_residual = model._residuals_for_next_tree_binned(
        y,
        round_idx=0,
        slot_idx=1,
        previous_slot_prediction_sums=slot_prediction_sums,
        current_round_prediction_sum=current_round_prediction_sum,
    )
    np.testing.assert_allclose(first_round_residual, y - current_round_prediction_sum)

    model.drop_first_round = True
    dropped_first_round_residual = model._residuals_for_next_tree_binned(
        y,
        round_idx=0,
        slot_idx=1,
        previous_slot_prediction_sums=slot_prediction_sums,
        current_round_prediction_sum=current_round_prediction_sum,
    )
    np.testing.assert_allclose(dropped_first_round_residual, y)

    round_residuals = model._residuals_for_round_binned(
        y,
        round_idx=2,
        previous_slot_prediction_sums=slot_prediction_sums,
    )
    expected_round_residuals = np.vstack(
        [
            y - (slot_prediction_sums[1] + slot_prediction_sums[2]) / 2,
            y - (slot_prediction_sums[0] + slot_prediction_sums[2]) / 2,
            y - (slot_prediction_sums[0] + slot_prediction_sums[1]) / 2,
        ]
    )
    np.testing.assert_allclose(round_residuals, expected_round_residuals)


def test_brat_p_hist_fit_uses_frozen_previous_round_sums():
    class RecordingBRATP(BRATPHistGradientBoostingRegressor):
        def _residuals_for_round_binned(
            self,
            y,
            *,
            round_idx,
            previous_slot_prediction_sums,
        ):
            self.recorded_previous_sums_.append(
                (
                    round_idx,
                    previous_slot_prediction_sums.copy(),
                )
            )
            return super()._residuals_for_round_binned(
                y,
                round_idx=round_idx,
                previous_slot_prediction_sums=previous_slot_prediction_sums,
            )

    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=0.5,
        random_state=0,
    )
    model = RecordingBRATP(
        n_rounds=2,
        trees_per_round=3,
        max_leaf_nodes=5,
        min_samples_leaf=4,
        max_bins=16,
        random_state=0,
    )
    model.recorded_previous_sums_ = []

    model.fit(X, y)

    second_round_snapshots = [
        previous_sums
        for round_idx, previous_sums in model.recorded_previous_sums_
        if round_idx == 1
    ]
    assert len(second_round_snapshots) == 1
    assert np.any(second_round_snapshots[0])


def test_brat_p_hist_prediction_uses_round_slot_aggregation():
    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=0.5,
        random_state=0,
    )
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=3,
        max_leaf_nodes=5,
        min_samples_leaf=4,
        max_bins=16,
        random_state=0,
    ).fit(X, y)

    X_eval = X[:8]
    X_binned = model._bin_data(X_eval)
    expected = model._predict_brat_p_binned(
        X_binned,
        n_threads=model._effective_n_threads(),
    )

    np.testing.assert_allclose(model.predict(X_eval), expected)


def test_brat_p_hist_is_deterministic():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    params = dict(
        n_rounds=3,
        trees_per_round=2,
        max_leaf_nodes=4,
        min_samples_leaf=5,
        max_bins=32,
        random_state=0,
    )

    first = BRATPHistGradientBoostingRegressor(**params).fit(X, y).predict(X)
    second = BRATPHistGradientBoostingRegressor(**params).fit(X, y).predict(X)

    np.testing.assert_allclose(first, second)


def test_brat_p_hist_parallel_fit_matches_serial_fit():
    X, y = make_regression(
        n_samples=90,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    params = dict(
        n_rounds=4,
        trees_per_round=3,
        subsample_rate=0.8,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
        drop_first_round=True,
    )

    serial = BRATPHistGradientBoostingRegressor(**params, n_jobs=1).fit(X, y)
    parallel = BRATPHistGradientBoostingRegressor(**params, n_jobs=2).fit(X, y)

    np.testing.assert_allclose(serial.predict(X), parallel.predict(X))
    np.testing.assert_array_equal(serial.in_bag_matrix_, parallel.in_bag_matrix_)
    assert serial.fit_diagnostics_["parallel_rounds"] == 0
    assert parallel.fit_diagnostics_["parallel_rounds"] == parallel.n_rounds
    assert parallel.fit_diagnostics_["effective_n_jobs"] == 2
    assert serial.fit_diagnostics_["vectorized_residual_rounds"] == serial.n_rounds
    assert parallel.fit_diagnostics_["vectorized_residual_rounds"] == parallel.n_rounds


def test_brat_p_hist_default_first_round_stays_serial_with_parallel_jobs():
    X, y = make_regression(
        n_samples=90,
        n_features=3,
        noise=1.0,
        random_state=0,
    )
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=4,
        trees_per_round=3,
        subsample_rate=0.8,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
        n_jobs=2,
    ).fit(X, y)

    assert model.fit_diagnostics_["serial_rounds"] == 1
    assert model.fit_diagnostics_["parallel_rounds"] == model.n_rounds - 1
    assert model.fit_diagnostics_["vectorized_residual_rounds"] == model.n_rounds - 1


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"loss": "absolute_error"}, "squared_error"),
        ({"n_rounds": 0}, "n_rounds"),
        ({"trees_per_round": 1}, "trees_per_round"),
        ({"subsample_rate": 0.0}, "subsample_rate"),
        ({"warm_start": True}, "warm_start"),
        ({"categorical_features": [0]}, "categorical_features"),
        ({"monotonic_cst": [1]}, "monotonic_cst"),
        ({"interaction_cst": [{0}]}, "interaction_cst"),
        ({"max_bins": 1}, "max_bins"),
        ({"max_bins": 256}, "max_bins"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": 1.5}, "n_jobs"),
        ({"n_jobs": True}, "n_jobs"),
    ],
)
def test_brat_p_hist_rejects_unsupported_parameters(params, message):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = BRATPHistGradientBoostingRegressor(
        early_stopping=False,
        **params,
    )

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_brat_p_hist_intervals_smoke():
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
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=2,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X_train, y_train)

    model.prepare_inference(X_calib, y_calib)

    ci_lower, ci_upper = model.confidence_interval(X_calib[:6])
    pi_lower, pi_upper = model.prediction_interval(X_calib[:6])
    ri_lower, ri_upper = model.reproduction_interval(X_calib[:6])
    norms = model.weight_norms(X_calib[:6])

    assert model.inference_method_ == "histogram_cell"
    assert model.sigma_hat2_ == pytest.approx(
        float(np.var(y_calib - model.predict(X_calib), ddof=1))
    )
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


def test_brat_p_hist_cell_system_uses_parallel_scaling():
    X, y = make_regression(
        n_samples=90,
        n_features=2,
        noise=1.0,
        random_state=0,
    )
    model = BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=4,
        max_leaf_nodes=5,
        min_samples_leaf=5,
        max_bins=16,
        random_state=0,
    ).fit(X, y)
    model.prepare_inference()

    X_query = X[:8]
    X_binned = model._bin_data(X_query)
    leaf_indices = model._apply_leaf_indices_binned(X_binned)
    kernel_vectors = model._cell_kernel_vector(leaf_indices)
    weighted_kernel = model.cell_counts_[:, None] * model.cell_kernel_matrix_
    scaled_system = (
        (1 / model.trees_per_round) * np.eye(model.cell_kernel_matrix_.shape[0])
        + ((model.trees_per_round - 1) / model.trees_per_round) * weighted_kernel
    )
    unscaled_system = (
        np.eye(model.cell_kernel_matrix_.shape[0])
        + (model.trees_per_round - 1) * weighted_kernel
    )

    scaled_weights = np.linalg.solve(scaled_system.T, kernel_vectors.T).T
    unscaled_weights = np.linalg.solve(unscaled_system.T, kernel_vectors.T).T
    scaled_norms = np.sqrt(
        np.maximum((scaled_weights**2) @ model.cell_counts_, 0.0)
    )
    unscaled_norms = np.sqrt(
        np.maximum((unscaled_weights**2) @ model.cell_counts_, 0.0)
    )

    np.testing.assert_allclose(model.cell_system_matrix_t_, scaled_system.T)
    np.testing.assert_allclose(
        model._solve_cell_brat_p_weights(kernel_vectors),
        scaled_weights,
    )
    np.testing.assert_allclose(scaled_norms, model.trees_per_round * unscaled_norms)

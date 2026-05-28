"""Native BRAT estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boulevard.algorithms.brat_d import brat_d_scale
from boulevard.backends.sklearn_tree import SubsampledDecisionTreeRegressor
from boulevard.estimators._base import ConformalIntervalMixin
from boulevard.intervals.asymptotic import normal_interval, normal_quantile
from boulevard.intervals.conformal import SplitConformalInterval
from boulevard.kernels.leaf import leaf_kernel_matrix, leaf_kernel_vector
from boulevard.kernels.weights import solve_brat_d_weights


class BRATDRegressor(ConformalIntervalMixin, RegressorMixin, BaseEstimator):
    """Boulevard Regularized Additive Regression Trees with dropout.

    This is a scikit-learn-compatible port of the scratch BRAT-D algorithm.
    It uses sklearn decision trees as the native tree backend and stores the
    in-bag and leaf metadata needed for future asymptotic interval methods.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        learning_rate: float = 1.0,
        max_depth: int | None = 4,
        min_samples_split: int = 2,
        subsample_rate: float = 0.8,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample_rate = subsample_rate
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> "BRATDRegressor":
        """Fit the BRAT-D ensemble."""
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self._validate_params()

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.estimators_: list[SubsampledDecisionTreeRegressor] = []
        self.in_bag_matrix_ = np.zeros((n_samples, self.n_estimators), dtype=bool)
        self.leaf_assignments_ = np.zeros((n_samples, self.n_estimators), dtype=int)
        self.train_score_: list[float] = []
        self.conformal_interval_: SplitConformalInterval | None = None

        for tree_idx in range(self.n_estimators):
            residuals = self._residuals_for_next_tree(X, y, rng)
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = SubsampledDecisionTreeRegressor(
                subsample_rate=self.subsample_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X, residuals, sample_weight=sample_weight)

            self.estimators_.append(tree)
            self.in_bag_matrix_[:, tree_idx] = tree.in_bag_
            self.leaf_assignments_[:, tree_idx] = tree.leaf_assignments_

            if self.verbose:
                mse = float(np.mean((y - self.predict(X)) ** 2))
                self.train_score_.append(mse)

        if not self.verbose:
            self.train_score_ = []

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression targets."""
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False)

        if not self.estimators_:
            return np.zeros(X.shape[0], dtype=float)

        pred = np.zeros(X.shape[0], dtype=float)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)

        pred /= len(self.estimators_)
        pred *= brat_d_scale(self.learning_rate, self.dropout_rate)
        return pred

    def apply_leaf_indices(self, X: Any) -> np.ndarray:
        """Return leaf indices for each sample and tree."""
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False)
        return np.column_stack([tree.leaf_indices(X) for tree in self.estimators_])

    def in_bag_matrix(self) -> np.ndarray:
        """Return the training in-bag matrix."""
        check_is_fitted(self, "in_bag_matrix_")
        return self.in_bag_matrix_

    def prepare_inference(
        self,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> "BRATDRegressor":
        """Prepare exact BRAT-D asymptotic interval inference.

        This computes the empirical tree kernel from training leaf assignments
        and estimates the noise variance from calibration residuals. If no
        calibration data is provided, the training residuals are used.
        """
        check_is_fitted(self, "estimators_")

        if (X_calib is None) != (y_calib is None):
            raise ValueError("X_calib and y_calib must be provided together.")

        if X_calib is None:
            X_var = self.X_train_
            y_var = self.y_train_
        else:
            X_var = check_array(X_calib, accept_sparse=False)
            y_var = np.asarray(y_calib, dtype=float)
            if X_var.shape[0] != y_var.shape[0]:
                raise ValueError("X_calib and y_calib must contain the same number of rows.")

        residuals = y_var - self.predict(X_var)
        if residuals.shape[0] < 2:
            raise ValueError("At least two residuals are required to estimate variance.")

        self.X_inference_calib_ = X_var.copy()
        self.y_inference_calib_ = y_var.copy()
        self.sigma_hat2_ = float(np.var(residuals, ddof=1))
        self.kernel_matrix_ = leaf_kernel_matrix(
            self.leaf_assignments_,
            self.in_bag_matrix_,
        )
        self.inference_method_ = "exact"
        return self

    def predict_interval(
        self,
        X: Any,
        alpha: float | None = None,
        method: str = "asymptotic",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals.

        ``method='asymptotic'`` uses the raw BRAT-D CLT interval after
        ``prepare_inference`` has been called. ``method='conformal'`` uses split
        conformal calibration.
        """
        if method == "conformal":
            return super().predict_interval(X, alpha=alpha, method=method)
        if method == "asymptotic":
            return self.prediction_interval(
                X,
                alpha=0.05 if alpha is None else alpha,
            )
        raise ValueError("method must be 'conformal' or 'asymptotic'.")

    def confidence_interval(
        self,
        X: Any,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return asymptotic BRAT-D confidence intervals for ``f(x)``."""
        center = self.predict(X)
        r_norm = self._weight_norms(X)
        scale = brat_d_scale(self.learning_rate, self.dropout_rate)
        se = scale * np.sqrt(self.sigma_hat2_) * r_norm
        interval = normal_interval(center, se, alpha=alpha)
        return interval.lower, interval.upper

    def prediction_interval(
        self,
        X: Any,
        alpha: float = 0.05,
        calibrated: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return asymptotic BRAT-D prediction intervals for ``y | x``."""
        center = self.predict(X)
        half_width = self._prediction_half_width(X, alpha=alpha)
        if calibrated:
            half_width *= self._prediction_calibration_scale(alpha=alpha)
        return center - half_width, center + half_width

    def reproduction_interval(
        self,
        X: Any,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return asymptotic BRAT-D reproduction intervals."""
        center = self.predict(X)
        r_norm = self._weight_norms(X)
        scale = brat_d_scale(self.learning_rate, self.dropout_rate)
        se = np.sqrt(2) * scale * np.sqrt(self.sigma_hat2_) * r_norm
        interval = normal_interval(center, se, alpha=alpha)
        return interval.lower, interval.upper

    def weight_norms(self, X: Any) -> np.ndarray:
        """Return BRAT-D kernel weight norms used by asymptotic intervals."""
        return self._weight_norms(X)

    def get_backend_model(self) -> list[SubsampledDecisionTreeRegressor]:
        """Return the fitted native tree ensemble."""
        check_is_fitted(self, "estimators_")
        return self.estimators_

    def _residuals_for_next_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not self.estimators_:
            return y

        q = 1 - self.dropout_rate
        keep_count = int(np.round(q * len(self.estimators_)))
        if keep_count == 0:
            return y

        selected = rng.choice(len(self.estimators_), size=keep_count, replace=False)
        pred = np.zeros(X.shape[0], dtype=float)
        for idx in selected:
            pred += self.estimators_[int(idx)].predict(X)

        pred *= self.learning_rate / len(self.estimators_)
        return y - pred

    def _validate_params(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be in (0, 1].")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

    def _weight_norms(self, X: Any) -> np.ndarray:
        self._check_inference_prepared()
        X = check_array(X, accept_sparse=False)
        test_leaf_indices = self.apply_leaf_indices(X)
        kernel_vectors = leaf_kernel_vector(
            self.leaf_assignments_,
            test_leaf_indices,
            self.in_bag_matrix_,
        )
        weights = solve_brat_d_weights(
            kernel_vectors,
            self.kernel_matrix_,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
        )
        return np.linalg.norm(weights, axis=1)

    def _prediction_half_width(self, X: Any, alpha: float) -> np.ndarray:
        r_norm = self._weight_norms(X)
        scale = brat_d_scale(self.learning_rate, self.dropout_rate)
        se = scale * np.sqrt(self.sigma_hat2_) * np.sqrt(1 + r_norm**2)
        return normal_quantile(alpha) * se

    def _prediction_calibration_scale(self, alpha: float) -> float:
        self._check_inference_prepared()
        center = self.predict(self.X_inference_calib_)
        half_width = self._prediction_half_width(self.X_inference_calib_, alpha=alpha)
        half_width = np.maximum(half_width, np.finfo(float).eps)
        ratios = np.abs(self.y_inference_calib_ - center) / half_width
        n_calib = ratios.shape[0]
        quantile_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        quantile_level = min(quantile_level, 1.0)
        return float(np.quantile(ratios, quantile_level, method="higher"))

    def _check_inference_prepared(self) -> None:
        check_is_fitted(self, "estimators_")
        if not hasattr(self, "kernel_matrix_") or not hasattr(self, "sigma_hat2_"):
            raise RuntimeError(
                "BRAT-D asymptotic intervals require prepare_inference(...) "
                "before calling interval methods."
            )


class BRATPRegressor:
    """Reserved estimator for parallel Boulevard Regularized Additive Trees."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("BRATPRegressor has not been ported yet.")

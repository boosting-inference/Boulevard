"""Scikit-learn tree backend used by native Boulevard estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class SubsampledDecisionTreeRegressor:
    """Decision tree regressor fitted on a row subsample.

    This is the package-quality version of the tree primitive from the scratch
    BRAT implementation. It keeps explicit in-bag and leaf-assignment metadata
    so BRAT estimators can build leaf kernels for inference.
    """

    def __init__(
        self,
        *,
        subsample_rate: float = 0.8,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.subsample_rate = subsample_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.tree_: DecisionTreeRegressor | None = None
        self.sample_indices_: np.ndarray | None = None
        self.in_bag_: np.ndarray | None = None
        self.leaf_assignments_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SubsampledDecisionTreeRegressor":
        """Fit a decision tree on a random row subsample."""
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be in (0, 1].")

        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = y.shape[0]
        subsample_size = max(1, int(np.round(n_samples * self.subsample_rate)))

        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        self.sample_indices_ = rng.choice(n_samples, size=subsample_size, replace=False)

        self.in_bag_ = np.zeros(n_samples, dtype=bool)
        self.in_bag_[self.sample_indices_] = True

        tree_random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=tree_random_state,
        )

        fit_weight = None
        if sample_weight is not None:
            fit_weight = np.asarray(sample_weight)[self.sample_indices_]

        self.tree_.fit(
            X[self.sample_indices_],
            y[self.sample_indices_],
            sample_weight=fit_weight,
        )
        self.leaf_assignments_ = self.tree_.apply(X)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the fitted tree."""
        self._check_is_fitted()
        return np.asarray(self.tree_.predict(X))

    def leaf_indices(self, X: Any) -> np.ndarray:
        """Return leaf indices for ``X``."""
        self._check_is_fitted()
        return np.asarray(self.tree_.apply(X))

    def get_native_model(self) -> DecisionTreeRegressor:
        """Return the fitted sklearn tree."""
        self._check_is_fitted()
        return self.tree_

    def _check_is_fitted(self) -> None:
        if self.tree_ is None:
            raise RuntimeError(
                "This SubsampledDecisionTreeRegressor instance is not fitted yet."
            )

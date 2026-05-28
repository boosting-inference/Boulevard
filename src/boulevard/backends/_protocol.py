"""Backend protocol for tree ensemble libraries."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TreeEnsembleBackend(Protocol):
    """Minimal interface required by Boulevard estimators.

    Backends adapt concrete libraries such as XGBoost, LightGBM, CatBoost,
    InterpretML, or the native BRAT tree builder to a common surface.
    """

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "TreeEnsembleBackend":
        """Fit the backend model."""

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        """Predict responses for ``X``."""

    def leaf_indices(self, X: Any) -> np.ndarray:
        """Return leaf indices with shape ``(n_samples, n_trees)``."""

    def get_native_model(self) -> Any:
        """Return the wrapped library model."""


class SupportsPerTreePrediction(TreeEnsembleBackend, Protocol):
    """Optional backend capability for algorithms needing tree-level outputs."""

    def per_tree_predict(self, X: Any) -> np.ndarray:
        """Return one prediction column per tree."""


class SupportsInBagMatrix(TreeEnsembleBackend, Protocol):
    """Optional backend capability for CLT-style kernel inference."""

    def in_bag_matrix(self) -> np.ndarray:
        """Return a boolean matrix of training samples used by each tree."""

"""XGBoost backend adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from xgboost import XGBRegressor as _XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "XGBoost support requires the optional dependency 'xgboost'. "
        'Install it with: pip install "boulevard-boosting[xgboost]"'
    ) from exc


class XGBoostBackend:
    """Adapter around ``xgboost.XGBRegressor``."""

    def __init__(self, **xgb_params: Any) -> None:
        self.xgb_params = dict(xgb_params)
        self.model_: _XGBRegressor | None = None

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> XGBoostBackend:
        """Fit the wrapped XGBoost regressor."""
        self.model_ = _XGBRegressor(**self.xgb_params)
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        """Predict using the wrapped model."""
        self._check_is_fitted()
        return np.asarray(self.model_.predict(X, **predict_kwargs))

    def leaf_indices(self, X: Any) -> np.ndarray:
        """Return XGBoost leaf indices."""
        self._check_is_fitted()
        return np.asarray(self.model_.apply(X))

    def get_native_model(self) -> _XGBRegressor:
        """Return the fitted XGBoost model."""
        self._check_is_fitted()
        return self.model_

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("This XGBoostBackend instance is not fitted yet.")


# Backward-compatible import path. Prefer ``boulevard.estimators.XGBRegressor``.
def __getattr__(name: str) -> Any:
    if name == "XGBRegressor":
        from boulevard.estimators.xgboost import XGBRegressor

        return XGBRegressor
    raise AttributeError(name)

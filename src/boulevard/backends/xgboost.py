"""XGBoost backend for Boulevard."""

from __future__ import annotations

from typing import Any

import numpy as np

from boulevard.intervals.conformal import SplitConformalInterval

try:
    from xgboost import XGBRegressor as _XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "XGBoost support requires the optional dependency 'xgboost'. "
        'Install it with: pip install "boulevard-boosting[xgboost]"'
    ) from exc


class XGBRegressor:
    """Boulevard-style wrapper around ``xgboost.XGBRegressor``.

    This class is currently a lightweight wrapper. It exposes a user-facing
    Boulevard API while delegating training and prediction to XGBoost.

    Parameters
    ----------
    boulevard:
        Whether to enable Boulevard-style behavior. In the current prototype,
        this flag is reserved for future Boulevard aggregation logic.
    **xgb_params:
        Parameters passed directly to ``xgboost.XGBRegressor``.
    """

    def __init__(self, boulevard: bool = True, **xgb_params: Any) -> None:
        self.boulevard = boulevard
        self.xgb_params = dict(xgb_params)

        self.model_: _XGBRegressor | None = None
        self.conformal_interval_: SplitConformalInterval | None = None

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> "XGBRegressor":
        """Fit the underlying XGBoost regressor."""
        self.model_ = _XGBRegressor(**self.xgb_params)
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        """Predict using the fitted model."""
        self._check_is_fitted()
        return np.asarray(self.model_.predict(X, **predict_kwargs))

    def apply_leaf_indices(self, X: Any) -> np.ndarray:
        """Return leaf indices for each sample and tree.

        The returned array has shape roughly ``(n_samples, n_trees)``.
        """
        self._check_is_fitted()
        return np.asarray(self.model_.apply(X))

    def calibrate(
        self,
        X_calib: Any,
        y_calib: Any,
        alpha: float = 0.1,
    ) -> "XGBRegressor":
        """Calibrate a split conformal prediction interval."""
        y_pred = self.predict(X_calib)

        self.conformal_interval_ = SplitConformalInterval().fit(
            y_true=y_calib,
            y_pred=y_pred,
            alpha=alpha,
        )

        return self

    def predict_interval(
        self,
        X: Any,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal prediction intervals.

        Parameters
        ----------
        X:
            Test covariates.
        alpha:
            Optional significance level. For now, this must match the level
            used during calibration. Recalibration is required to change alpha.
        """
        self._check_is_fitted()

        if self.conformal_interval_ is None:
            raise RuntimeError(
                "The model has not been calibrated yet. "
                "Call model.calibrate(X_calib, y_calib, alpha=...) first."
            )

        if alpha is not None and alpha != self.conformal_interval_.alpha_:
            raise ValueError(
                "This model was calibrated with alpha="
                f"{self.conformal_interval_.alpha_}. "
                "Changing alpha requires recalibration."
            )

        y_pred = self.predict(X)
        return self.conformal_interval_.predict(y_pred)

    def get_backend_model(self) -> _XGBRegressor:
        """Return the fitted underlying XGBoost model."""
        self._check_is_fitted()
        return self.model_

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters for basic sklearn compatibility."""
        params = {"boulevard": self.boulevard}
        params.update(self.xgb_params)
        return params

    def set_params(self, **params: Any) -> "XGBRegressor":
        """Set parameters for basic sklearn compatibility."""
        for key, value in params.items():
            if key == "boulevard":
                self.boulevard = value
            else:
                self.xgb_params[key] = value
        return self

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("This XGBRegressor instance is not fitted yet.")
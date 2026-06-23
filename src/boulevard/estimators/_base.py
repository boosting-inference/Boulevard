"""Shared estimator utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from boulevard.intervals.conformal import SplitConformalInterval


class ConformalIntervalMixin:
    """Mixin for estimators with split conformal prediction intervals."""

    conformal_interval_: SplitConformalInterval | None

    def calibrate(
        self,
        X_calib: Any,
        y_calib: Any,
        alpha: float = 0.1,
    ) -> Any:
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
        method: str = "conformal",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals.

        Only split conformal intervals are implemented for backend wrappers
        today. The ``method`` parameter is part of the stable public API so
        asymptotic intervals can be added without changing call sites.
        """
        if method != "conformal":
            raise ValueError("Only method='conformal' is currently implemented.")

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

        return self.conformal_interval_.predict(self.predict(X))


class BackendRegressorMixin(ConformalIntervalMixin):
    """Shared wrapper behavior for library-backed regressors."""

    backend_: Any
    backend_params: dict[str, Any]

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> Any:
        """Fit the underlying backend model."""
        self.backend_ = self._make_backend()
        self.backend_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: Any, **predict_kwargs: Any) -> np.ndarray:
        """Predict using the fitted backend."""
        self._check_is_fitted()
        return self.backend_.predict(X, **predict_kwargs)

    def apply_leaf_indices(self, X: Any) -> np.ndarray:
        """Return backend leaf indices."""
        self._check_is_fitted()
        return self.backend_.leaf_indices(X)

    def get_backend_model(self) -> Any:
        """Return the fitted native backend model."""
        self._check_is_fitted()
        return self.backend_.get_native_model()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-style estimator parameters."""
        params = {"boulevard": self.boulevard}
        params.update(self.backend_params)
        return params

    def set_params(self, **params: Any) -> Any:
        """Set sklearn-style estimator parameters."""
        for key, value in params.items():
            if key == "boulevard":
                self.boulevard = value
            else:
                self.backend_params[key] = value
        return self

    def _check_is_fitted(self) -> None:
        if self.backend_ is None:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet."
            )

    def _make_backend(self) -> Any:
        raise NotImplementedError

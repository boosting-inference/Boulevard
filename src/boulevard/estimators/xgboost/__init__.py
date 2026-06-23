"""Public XGBoost estimator wrapper."""

from __future__ import annotations

from typing import Any

from boulevard.backends.xgboost import XGBoostBackend
from boulevard.estimators._base import BackendRegressorMixin
from boulevard.intervals.conformal import SplitConformalInterval


class XGBRegressor(BackendRegressorMixin):
    """Boulevard-style wrapper around ``xgboost.XGBRegressor``.

    The current implementation delegates point prediction to XGBoost and
    provides a stable Boulevard-facing API for calibration, intervals, and
    leaf extraction. Paper-faithful BRAT training lives in native Boulevard
    estimators because it changes the boosting residual construction.
    """

    def __init__(self, boulevard: bool = True, **xgb_params: Any) -> None:
        self.boulevard = boulevard
        self.backend_params = dict(xgb_params)
        self.backend_: XGBoostBackend | None = None
        self.conformal_interval_: SplitConformalInterval | None = None

    def _make_backend(self) -> XGBoostBackend:
        return XGBoostBackend(**self.backend_params)

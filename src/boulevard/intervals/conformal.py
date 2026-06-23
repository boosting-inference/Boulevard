"""Split conformal prediction intervals."""

from __future__ import annotations

import numpy as np


class SplitConformalInterval:
    """A simple split conformal interval for regression.

    This calibrator uses absolute residuals from a held-out calibration set.
    """

    def __init__(self) -> None:
        self.residual_quantile_: float | None = None
        self.alpha_: float | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.1,
    ) -> SplitConformalInterval:
        """Fit the conformal interval using calibration residuals."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape. "
                f"Got {y_true.shape} and {y_pred.shape}."
            )

        residuals = np.abs(y_true - y_pred)
        n = residuals.shape[0]

        if n == 0:
            raise ValueError("Calibration set must not be empty.")

        # Finite-sample split conformal quantile.
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self.residual_quantile_ = float(
            np.quantile(residuals, quantile_level, method="higher")
        )
        self.alpha_ = alpha

        return self

    def predict(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper conformal prediction bounds."""
        if self.residual_quantile_ is None:
            raise RuntimeError("The conformal interval has not been fitted yet.")

        y_pred = np.asarray(y_pred, dtype=float)

        lower = y_pred - self.residual_quantile_
        upper = y_pred + self.residual_quantile_

        return lower, upper

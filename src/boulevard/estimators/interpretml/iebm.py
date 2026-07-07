"""Inferable EBM-style additive regressor.

This module intentionally rebuilds the current IEBM prototype inside boulevard
instead of importing or patching InterpretML internals.  The first version keeps
the scope narrow: squared-error regression, numeric features, main effects, and
one-dimensional binned tree updates.
"""

from __future__ import annotations

import time
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class IEBMRegressor(BaseEstimator, RegressorMixin):
    """Minimal sklearn-style Inferable EBM regressor.

    The estimator follows the paper's main-effect training shape: for each round
    and each feature, fit a one-dimensional binned tree to a frozen full-model
    residual, center and truncate the update, then average it into the feature
    function.

    This initial implementation supports numeric regression only.  It exposes
    bin-space interval diagnostics, but those intervals remain experimental
    until the coverage behavior is audited more broadly.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 100,
        max_bins: int = 64,
        learning_rate: float = 1.0,
        subsample_rate: float = 1.0,
        warmup_rounds: int = 20,
        truncation: float = 100.0,
        max_leaves: int = 2,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        leave_one_out: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.subsample_rate = subsample_rate
        self.warmup_rounds = warmup_rounds
        self.truncation = truncation
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.leave_one_out = leave_one_out
        self.random_state = random_state

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> IEBMRegressor:
        """Fit the numeric main-effect IEBM model."""
        fit_start = time.perf_counter()
        self._validate_iebm_params()
        self.max_leaves_ = self._resolve_max_leaves()

        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported yet.")

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        if not np.all(np.isfinite(X)):
            raise ValueError("IEBMRegressor currently requires finite numeric X.")
        if not np.all(np.isfinite(y)):
            raise ValueError("IEBMRegressor currently requires finite numeric y.")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_in_ = np.array([f"x{idx}" for idx in range(n_features)])
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.interval_calibrations_ = {}

        self.bin_edges_ = self._fit_bin_edges(X)
        train_bins = self._bin_data(X)
        self.train_bins_ = train_bins
        self.bin_counts_ = [
            np.bincount(train_bins[:, feature_idx], minlength=self.n_bins_[feature_idx])
            .astype(float)
            for feature_idx in range(n_features)
        ]
        self.bin_weights_ = self.bin_counts_
        self.term_features_ = [(feature_idx,) for feature_idx in range(n_features)]
        self.term_names_ = list(self.feature_names_in_)
        self.term_scores_ = [
            np.zeros(self.n_bins_[feature_idx], dtype=float)
            for feature_idx in range(n_features)
        ]
        warm_scores = [
            np.zeros(self.n_bins_[feature_idx], dtype=float)
            for feature_idx in range(n_features)
        ]
        avg_scores = [
            np.zeros(self.n_bins_[feature_idx], dtype=float)
            for feature_idx in range(n_features)
        ]
        self.structure_sums_ = [
            np.zeros(
                (self.n_bins_[feature_idx], self.n_bins_[feature_idx]),
                dtype=float,
            )
            for feature_idx in range(n_features)
        ]
        self.structure_update_counts_ = np.zeros(n_features, dtype=int)
        self.intercept_ = float(np.mean(y))

        sampled_rows = 0
        for round_idx in range(1, self.max_rounds + 1):
            old_scores = [scores.copy() for scores in self.term_scores_]
            old_contributions = self._contributions_from_bins(train_bins, old_scores)
            new_scores = []
            new_warm_scores = []
            new_avg_scores = []
            coeff = (round_idx - 1.0) / round_idx

            for feature_idx in range(n_features):
                feature_bins = train_bins[:, feature_idx]
                feature_contribution = old_scores[feature_idx][feature_bins]
                prediction = self.intercept_ + old_contributions
                if self.leave_one_out:
                    prediction = prediction - feature_contribution
                residual = y - prediction

                if self.subsample_rate < 1.0:
                    in_bag = rng.random(n_samples) < self.subsample_rate
                    if not np.any(in_bag):
                        in_bag = np.ones(n_samples, dtype=bool)
                else:
                    in_bag = np.ones(n_samples, dtype=bool)
                sampled_rows += int(np.sum(in_bag))

                update, structure_delta = self._fit_binned_tree_update(
                    feature_bins=feature_bins,
                    residual=residual,
                    in_bag=in_bag,
                    n_bins=self.n_bins_[feature_idx],
                    full_bin_counts=self.bin_counts_[feature_idx],
                )
                self.structure_sums_[feature_idx] += structure_delta
                self.structure_update_counts_[feature_idx] += 1
                mean_update = float(
                    np.dot(self.bin_counts_[feature_idx], update) / n_samples
                )
                centered = np.clip(
                    update - mean_update,
                    -self.truncation,
                    self.truncation,
                )
                if round_idx <= self.warmup_rounds:
                    local_learning_rate = (
                        self.learning_rate
                        if self.learning_rate < 0.8
                        else self.learning_rate / 2
                    )
                    new_warm = warm_scores[feature_idx] + local_learning_rate * centered
                    new_avg = avg_scores[feature_idx]
                else:
                    new_warm = warm_scores[feature_idx]
                    new_avg = (
                        coeff * avg_scores[feature_idx]
                        + (self.learning_rate / round_idx) * centered
                    )

                new_warm_scores.append(new_warm)
                new_avg_scores.append(new_avg)
                new_scores.append(new_warm + new_avg)

            warm_scores = new_warm_scores
            avg_scores = new_avg_scores
            self.term_scores_ = new_scores
            contributions = self._contributions_from_bins(train_bins, self.term_scores_)
            self.intercept_ = float(np.mean(y - contributions))

        if self.leave_one_out:
            self.boulevard_scale_ = 1.0
        else:
            self.boulevard_scale_ = (1.0 + self.learning_rate) / self.learning_rate
        self.warm_scores_ = warm_scores
        self.avg_scores_ = avg_scores
        self.term_scores_ = [
            warm_scores[feature_idx] + self.boulevard_scale_ * avg_scores[feature_idx]
            for feature_idx in range(n_features)
        ]
        contributions = self._contributions_from_bins(train_bins, self.term_scores_)
        self.intercept_ = float(np.mean(y - contributions))

        residuals = y - self.predict(X)
        self.sigma_ = (
            float(np.std(residuals, ddof=1)) if residuals.shape[0] > 1 else 0.0
        )
        self.fit_diagnostics_ = {
            "total_seconds": time.perf_counter() - fit_start,
            "n_features": n_features,
            "total_bins": int(np.sum(self.n_bins_)),
            "max_rounds": self.max_rounds,
            "max_leaves": self.max_leaves_,
            "max_depth": self.max_depth,
            "sampled_rows": sampled_rows,
            "structure_updates": int(np.sum(self.structure_update_counts_)),
            "leave_one_out": self.leave_one_out,
            "warmup_rounds": self.warmup_rounds,
            "boulevard_scale": self.boulevard_scale_,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict from the fitted additive model."""
        check_is_fitted(self, "term_scores_")
        X = check_array(X, accept_sparse=False)
        if not np.all(np.isfinite(X)):
            raise ValueError("IEBMRegressor currently requires finite numeric X.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fit with "
                f"{self.n_features_in_} features."
            )

        bins = self._bin_data(X)
        return self.intercept_ + self._contributions_from_bins(bins, self.term_scores_)

    def apply_bins(self, X: Any) -> np.ndarray:
        """Return trained per-feature bin indices for numeric input data."""
        check_is_fitted(self, "term_scores_")
        X = self._validate_X_for_prediction(X)
        return self._bin_data(X)

    def prepare_inference(
        self,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> IEBMRegressor:
        """Prepare bin-space diagnostic inference quantities.

        This estimates residual variance from held-out calibration data, or
        from the training data if no calibration data are supplied, and builds
        the bin-level influence-norm cache used by :meth:`weight_norms`,
        :meth:`predict_intervals`, and :meth:`predict_feature_intervals`.
        """
        check_is_fitted(self, "term_scores_")

        if (X_calib is None) != (y_calib is None):
            raise ValueError("X_calib and y_calib must be provided together.")

        if X_calib is None:
            X_var = self.X_train_
            y_var = self.y_train_
        else:
            X_var = self._validate_X_for_prediction(X_calib)
            y_var = np.asarray(y_calib, dtype=float)
            if y_var.ndim != 1:
                raise ValueError("y_calib must be one-dimensional.")
            if X_var.shape[0] != y_var.shape[0]:
                raise ValueError(
                    "X_calib and y_calib must contain the same number of rows."
                )
            if not np.all(np.isfinite(y_var)):
                raise ValueError("IEBMRegressor currently requires finite numeric y.")

        residuals = y_var - self.predict(X_var)
        if residuals.shape[0] < 2:
            raise ValueError(
                "At least two residuals are required to estimate variance."
            )

        self.X_inference_calib_ = X_var.copy()
        self.y_inference_calib_ = y_var.copy()
        self.sigma_hat2_ = float(np.var(residuals, ddof=1))
        self.interval_calibrations_ = {}
        self._prepare_bin_inference_cache()
        self.inference_diagnostics_ = {
            "n_calibration_rows": int(X_var.shape[0]),
            "sigma_hat2": self.sigma_hat2_,
            "total_bins": int(np.sum(self.n_bins_)),
            "empty_training_bins": int(
                sum(np.count_nonzero(counts == 0) for counts in self.bin_counts_)
            ),
            "structure_updates": int(np.sum(self.structure_update_counts_)),
        }
        return self

    def weight_norms(self, X: Any) -> np.ndarray:
        """Return bin-space influence-norm diagnostics for query rows."""
        self._ensure_inference_prepared()
        X = self._validate_X_for_prediction(X)
        bins = self._bin_data(X)
        norm_sq = np.zeros(X.shape[0], dtype=float)
        for feature_idx, cached_norm_sq in enumerate(self._feature_bin_norm_sq_):
            norm_sq += cached_norm_sq[bins[:, feature_idx]]
        return np.sqrt(np.maximum(norm_sq, 0.0))

    def predict_intervals(
        self,
        X: Any,
        level: float = 0.95,
        mode: str = "prediction",
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return EBM-style intervals for full predictions.

        This mirrors the patched IEBM API shape: return lower bounds, upper
        bounds, and point predictions. The interval calculation is still
        experimental for this rebuilt estimator.
        """
        self._ensure_inference_prepared()
        X = self._validate_X_for_prediction(X)
        preds = self.predict(X)
        norms = self.weight_norms(X)
        half_width = self._interval_half_width(
            norms,
            level=level,
            mode=mode,
            sigma=sigma,
            scale=self.boulevard_scale_,
        )
        half_width = half_width * self._interval_calibration_scale(mode, level)
        return preds - half_width, preds + half_width, preds

    def calibrate_intervals(
        self,
        X_calib: Any,
        y_calib: Any,
        level: float = 0.95,
        mode: str = "prediction",
        sigma: float | None = None,
        propagate_to_ci_ri: bool = False,
    ) -> float:
        """Calibrate interval widths on held-out response data.

        The calibration factor is the smallest multiplier that makes
        ``predict(X_calib) +/- factor * raw_width`` cover at least ``level`` of
        the observed calibration responses.  This is most natural for
        ``mode="prediction"`` because the calibration targets noisy responses.
        """
        check_is_fitted(self, "term_scores_")
        self.prepare_inference(X_calib, y_calib)
        X_calib = self._validate_X_for_prediction(X_calib)
        y_calib = np.asarray(y_calib, dtype=float)
        if y_calib.ndim != 1:
            raise ValueError("y_calib must be one-dimensional.")
        if X_calib.shape[0] != y_calib.shape[0]:
            raise ValueError(
                "X_calib and y_calib must contain the same number of rows."
            )

        preds = self.predict(X_calib)
        norms = self.weight_norms(X_calib)
        raw_width = self._interval_half_width(
            norms,
            level=level,
            mode=mode,
            sigma=sigma,
            scale=self.boulevard_scale_,
        )
        calibration = self._find_min_interval_scale(
            y_true=y_calib,
            y_pred=preds,
            half_width=raw_width,
            target=level,
        )
        key = (mode, float(level))
        self.interval_calibrations_[key] = calibration
        if propagate_to_ci_ri and mode == "prediction":
            self.interval_calibrations_[("confidence", float(level))] = calibration
            self.interval_calibrations_[("reproduction", float(level))] = calibration
        return calibration

    def predict_feature_intervals(
        self,
        feature_idx: int,
        x_k: Any,
        level: float = 0.95,
        mode: str = "confidence",
        sigma: float | None = None,
        include_intercept: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return EBM-style intervals for one main-effect partial function."""
        self._ensure_inference_prepared()
        if not 0 <= feature_idx < self.n_features_in_:
            raise ValueError("feature_idx is out of range.")

        values = np.asarray(x_k)
        if values.ndim != 1:
            raise ValueError("x_k must be one-dimensional.")
        if not np.all(np.isfinite(values.astype(float, copy=False))):
            raise ValueError("x_k must contain finite numeric values.")

        bins = np.digitize(
            values.astype(float, copy=False),
            self.bin_edges_[feature_idx],
            right=False,
        )
        bins = np.clip(bins, 0, self.n_bins_[feature_idx] - 1)
        preds = self.term_scores_[feature_idx][bins]
        if include_intercept:
            preds = preds + self.intercept_

        norms = np.sqrt(np.maximum(self._feature_bin_norm_sq_[feature_idx][bins], 0.0))
        half_width = self._interval_half_width(
            norms,
            level=level,
            mode=mode,
            sigma=sigma,
            scale=2.0,
        )
        half_width = half_width * self._interval_calibration_scale(mode, level)
        return preds - half_width, preds + half_width, preds

    def _interval_calibration_scale(self, mode: str, level: float) -> float:
        if not hasattr(self, "interval_calibrations_"):
            self.interval_calibrations_ = {}
        return float(self.interval_calibrations_.get((mode, float(level)), 1.0))

    def _interval_half_width(
        self,
        norms: np.ndarray,
        *,
        level: float,
        mode: str,
        sigma: float | None,
        scale: float,
    ) -> np.ndarray:
        sigma = self._resolve_interval_sigma(sigma)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        if mode == "confidence":
            base = sigma * norms
        elif mode == "prediction":
            base = sigma * np.hypot(1.0, norms)
        elif mode == "reproduction":
            base = np.sqrt(2.0) * sigma * norms
        else:
            raise ValueError(
                "mode must be 'confidence', 'prediction', or 'reproduction'."
            )
        return scale * z * base

    def _resolve_interval_sigma(self, sigma: float | None) -> float:
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            return float(sigma)
        if hasattr(self, "sigma_hat2_") and self.sigma_hat2_ > 0:
            return float(np.sqrt(self.sigma_hat2_))
        if hasattr(self, "sigma_") and self.sigma_ > 0:
            return float(self.sigma_)
        return 1e-8

    @staticmethod
    def _interval_coverage(
        scale: float,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        half_width: np.ndarray,
    ) -> float:
        lower = y_pred - scale * half_width
        upper = y_pred + scale * half_width
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    @classmethod
    def _find_min_interval_scale(
        cls,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        half_width: np.ndarray,
        target: float,
        tol: float = 1e-3,
        c_max: float = 50.0,
        max_iters: int = 50,
    ) -> float:
        if not 0 < target < 1:
            raise ValueError("level must be in (0, 1).")
        if y_true.shape != y_pred.shape or y_true.shape != half_width.shape:
            raise ValueError("Calibration arrays must have the same shape.")
        if not np.all(np.isfinite(y_true)):
            raise ValueError("y_calib must contain finite numeric values.")
        if not np.all(np.isfinite(half_width)) or np.any(half_width < 0):
            raise ValueError("Interval half-widths must be finite and nonnegative.")

        lo = 0.0
        hi = float(c_max)
        if cls._interval_coverage(hi, y_true, y_pred, half_width) < target:
            raise ValueError("c_max is too small to reach the requested coverage.")

        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            if cls._interval_coverage(mid, y_true, y_pred, half_width) >= target:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi

    def _validate_iebm_params(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be in (0, 1].")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative.")
        if self.truncation <= 0:
            raise ValueError("truncation must be positive.")
        if self.max_leaves < 1:
            raise ValueError("max_leaves must be at least 1.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if self.max_depth is not None and self.max_leaves != 2:
            raise ValueError(
                "Specify only one of max_depth or a non-default max_leaves."
            )
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if not isinstance(self.leave_one_out, bool):
            raise ValueError("leave_one_out must be a bool.")

    def _resolve_max_leaves(self) -> int:
        if self.max_depth is None:
            return int(self.max_leaves)
        return int(2**self.max_depth)

    def _fit_bin_edges(self, X: np.ndarray) -> list[np.ndarray]:
        edges: list[np.ndarray] = []
        n_bins: list[int] = []
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]

        for feature_idx in range(X.shape[1]):
            column = X[:, feature_idx].astype(float, copy=False)
            raw_edges = np.quantile(column, quantiles)
            feature_edges = np.unique(raw_edges[np.isfinite(raw_edges)])
            edges.append(feature_edges)
            n_bins.append(int(feature_edges.shape[0] + 1))

        self.n_bins_ = np.asarray(n_bins, dtype=int)
        return edges

    def _bin_data(self, X: np.ndarray) -> np.ndarray:
        bins = np.empty((X.shape[0], len(self.bin_edges_)), dtype=int)
        for feature_idx, edges in enumerate(self.bin_edges_):
            bins[:, feature_idx] = np.digitize(
                X[:, feature_idx].astype(float, copy=False),
                edges,
                right=False,
            )
        return bins

    def _fit_binned_tree_update(
        self,
        *,
        feature_bins: np.ndarray,
        residual: np.ndarray,
        in_bag: np.ndarray,
        n_bins: int,
        full_bin_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bins = feature_bins[in_bag]
        values = residual[in_bag]
        counts = np.bincount(bins, minlength=n_bins).astype(float)
        sums = np.bincount(bins, weights=values, minlength=n_bins).astype(float)
        sums_sq = np.bincount(bins, weights=values**2, minlength=n_bins).astype(float)

        segments = self._greedy_segments(counts, sums, sums_sq)
        update = np.zeros(n_bins, dtype=float)
        structure_delta = np.zeros((n_bins, n_bins), dtype=float)
        for start, stop in segments:
            count = float(np.sum(counts[start:stop]))
            if count > 0:
                update[start:stop] = float(np.sum(sums[start:stop]) / count)

            full_count = float(np.sum(full_bin_counts[start:stop]))
            if full_count > 0:
                segment_bins = np.arange(start, stop, dtype=int)
                structure_delta[np.ix_(segment_bins, segment_bins)] += 1.0 / full_count

        return update, structure_delta

    def _validate_X_for_prediction(self, X: Any) -> np.ndarray:
        X = check_array(X, accept_sparse=False)
        if not np.all(np.isfinite(X)):
            raise ValueError("IEBMRegressor currently requires finite numeric X.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fit with "
                f"{self.n_features_in_} features."
            )
        return X

    def _ensure_inference_prepared(self) -> None:
        if not hasattr(self, "sigma_hat2_") or not hasattr(
            self,
            "_bin_cache_ready_",
        ):
            self.prepare_inference()

    def _prepare_bin_inference_cache(self) -> None:
        n_samples = float(self.X_train_.shape[0])

        self._bin_H_expectations_ = []
        self._bin_M_cholesky_ = []
        self._bin_M_pinv_ = []

        for feature_idx, counts in enumerate(self.bin_counts_):
            counts = np.asarray(counts, dtype=float)
            update_count = int(self.structure_update_counts_[feature_idx])
            if update_count > 0:
                H_avg = self.structure_sums_[feature_idx] / update_count
            else:
                H_avg = np.zeros_like(self.structure_sums_[feature_idx])
            H_avg = 0.5 * (H_avg + H_avg.T)
            self._bin_H_expectations_.append(H_avg)

            if H_avg.size:
                H_inv = self._pinv_psd(H_avg)
            else:
                H_inv = np.zeros_like(H_avg)

            centered_counts = np.diag(counts) - np.outer(counts, counts) / n_samples
            M = H_inv + centered_counts
            M = 0.5 * (M + M.T)

            try:
                self._bin_M_cholesky_.append(np.linalg.cholesky(M))
                self._bin_M_pinv_.append(None)
            except np.linalg.LinAlgError:
                self._bin_M_cholesky_.append(None)
                self._bin_M_pinv_.append(self._pinv_psd(M))

        self._feature_bin_norm_sq_ = [
            np.array(
                [
                    self._feature_influence_norm_sq(feature_idx, bin_idx)
                    for bin_idx in range(counts.shape[0])
                ],
                dtype=float,
            )
            for feature_idx, counts in enumerate(self.bin_counts_)
        ]
        self._bin_cache_ready_ = True

    def _feature_influence_norm_sq(self, feature_idx: int, bin_idx: int) -> float:
        counts = self.bin_counts_[feature_idx]
        if counts.size == 0:
            return 0.0

        bin_idx = int(np.clip(bin_idx, 0, counts.size - 1))
        H_mat = self._bin_H_expectations_[feature_idx]
        if H_mat.size == 0:
            return 0.0

        h_vec = H_mat[:, bin_idx]
        if not np.any(h_vec):
            return 0.0

        n_samples = float(self.X_train_.shape[0])
        c_dot_h = float(np.dot(counts, h_vec))
        centered_rhs = counts * h_vec - (c_dot_h / n_samples) * counts
        correction = self._solve_bin_system(feature_idx, centered_rhs)
        q = h_vec - correction

        norm_sq = float(np.dot(q * counts, q) - (np.dot(counts, q) ** 2) / n_samples)
        if not np.isfinite(norm_sq):
            return 0.0
        return max(norm_sq, 0.0)

    def _solve_bin_system(self, feature_idx: int, rhs: np.ndarray) -> np.ndarray:
        chol = self._bin_M_cholesky_[feature_idx]
        if chol is not None and chol.size:
            y = np.linalg.solve(chol, rhs)
            return np.linalg.solve(chol.T, y)

        pinv = self._bin_M_pinv_[feature_idx]
        if pinv is None or pinv.size == 0:
            return np.zeros_like(rhs)
        return pinv @ rhs

    @staticmethod
    def _pinv_psd(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros_like(matrix)
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        keep = values > tol * max(1.0, float(np.max(values)))
        if not np.any(keep):
            return np.zeros_like(matrix)
        return (vectors[:, keep] / values[keep]) @ vectors[:, keep].T

    def _greedy_segments(
        self,
        counts: np.ndarray,
        sums: np.ndarray,
        sums_sq: np.ndarray,
    ) -> list[tuple[int, int]]:
        n_bins = counts.shape[0]
        segments = [(0, n_bins)]
        prefix_counts = np.r_[0.0, np.cumsum(counts)]
        prefix_sums = np.r_[0.0, np.cumsum(sums)]
        prefix_sums_sq = np.r_[0.0, np.cumsum(sums_sq)]

        def stats(start: int, stop: int) -> tuple[float, float, float]:
            return (
                float(prefix_counts[stop] - prefix_counts[start]),
                float(prefix_sums[stop] - prefix_sums[start]),
                float(prefix_sums_sq[stop] - prefix_sums_sq[start]),
            )

        def sse(count: float, total: float, total_sq: float) -> float:
            if count <= 0:
                return 0.0
            return max(total_sq - (total * total) / count, 0.0)

        while len(segments) < self.max_leaves_:
            best: tuple[float, int, int] | None = None
            for segment_idx, (start, stop) in enumerate(segments):
                parent_count, parent_sum, parent_sum_sq = stats(start, stop)
                if parent_count < 2 * self.min_samples_leaf:
                    continue
                parent_sse = sse(parent_count, parent_sum, parent_sum_sq)
                for split in range(start + 1, stop):
                    left_count, left_sum, left_sum_sq = stats(start, split)
                    right_count, right_sum, right_sum_sq = stats(split, stop)
                    if (
                        left_count < self.min_samples_leaf
                        or right_count < self.min_samples_leaf
                    ):
                        continue
                    gain = parent_sse - sse(left_count, left_sum, left_sum_sq) - sse(
                        right_count,
                        right_sum,
                        right_sum_sq,
                    )
                    if best is None or gain > best[0]:
                        best = (gain, segment_idx, split)

            if best is None or best[0] <= 1e-12:
                break

            _, segment_idx, split = best
            start, stop = segments[segment_idx]
            segments[segment_idx : segment_idx + 1] = [(start, split), (split, stop)]

        return segments

    @staticmethod
    def _contributions_from_bins(
        bins: np.ndarray,
        term_scores: list[np.ndarray],
    ) -> np.ndarray:
        contributions = np.zeros(bins.shape[0], dtype=float)
        for feature_idx, scores in enumerate(term_scores):
            contributions += scores[bins[:, feature_idx]]
        return contributions


__all__ = ["IEBMRegressor"]

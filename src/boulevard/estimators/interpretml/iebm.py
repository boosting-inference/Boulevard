"""Inferable EBM-style additive regressor.

This module intentionally rebuilds the current IEBM prototype inside boulevard
instead of importing or patching InterpretML internals.  The first version keeps
the scope narrow: squared-error regression, numeric features, main effects, and
one-dimensional binned tree updates.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class IEBMRegressor(BaseEstimator, RegressorMixin):
    """Minimal sklearn-style Inferable EBM regressor.

    The estimator follows the paper's main-effect training shape: for each round
    and each feature, fit a one-dimensional binned tree to the residual obtained
    by leaving that feature out, center and truncate the update, then average it
    into the feature function.

    This initial implementation supports numeric regression only.  Interval
    methods will be added after the training path is audited.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 100,
        max_bins: int = 64,
        learning_rate: float = 1.0,
        subsample_rate: float = 1.0,
        truncation: float = 100.0,
        max_leaves: int = 2,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.subsample_rate = subsample_rate
        self.truncation = truncation
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
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

        self.bin_edges_ = self._fit_bin_edges(X)
        train_bins = self._bin_data(X)
        self.train_bins_ = train_bins
        self.bin_counts_ = [
            np.bincount(train_bins[:, feature_idx], minlength=self.n_bins_[feature_idx])
            .astype(float)
            for feature_idx in range(n_features)
        ]
        self.term_scores_ = [
            np.zeros(self.n_bins_[feature_idx], dtype=float)
            for feature_idx in range(n_features)
        ]
        self.intercept_ = float(np.mean(y))

        sampled_rows = 0
        for round_idx in range(1, self.max_rounds + 1):
            old_scores = [scores.copy() for scores in self.term_scores_]
            old_contributions = self._contributions_from_bins(train_bins, old_scores)
            new_scores = []
            coeff = (round_idx - 1.0) / round_idx

            for feature_idx in range(n_features):
                feature_bins = train_bins[:, feature_idx]
                feature_contribution = old_scores[feature_idx][feature_bins]
                prediction_without_feature = (
                    self.intercept_ + old_contributions - feature_contribution
                )
                residual = y - prediction_without_feature

                if self.subsample_rate < 1.0:
                    in_bag = rng.random(n_samples) < self.subsample_rate
                    if not np.any(in_bag):
                        in_bag = np.ones(n_samples, dtype=bool)
                else:
                    in_bag = np.ones(n_samples, dtype=bool)
                sampled_rows += int(np.sum(in_bag))

                update = self._fit_binned_tree_update(
                    feature_bins=feature_bins,
                    residual=residual,
                    in_bag=in_bag,
                    n_bins=self.n_bins_[feature_idx],
                )
                mean_update = float(
                    np.dot(self.bin_counts_[feature_idx], update) / n_samples
                )
                centered = np.clip(
                    update - mean_update,
                    -self.truncation,
                    self.truncation,
                )
                new_scores.append(
                    coeff * old_scores[feature_idx]
                    + (self.learning_rate / round_idx) * centered
                )

            self.term_scores_ = new_scores
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
            "sampled_rows": sampled_rows,
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
        X = check_array(X, accept_sparse=False)
        return self._bin_data(X)

    def _validate_iebm_params(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be in (0, 1].")
        if self.truncation <= 0:
            raise ValueError("truncation must be positive.")
        if self.max_leaves < 1:
            raise ValueError("max_leaves must be at least 1.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

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
    ) -> np.ndarray:
        bins = feature_bins[in_bag]
        values = residual[in_bag]
        counts = np.bincount(bins, minlength=n_bins).astype(float)
        sums = np.bincount(bins, weights=values, minlength=n_bins).astype(float)
        sums_sq = np.bincount(bins, weights=values**2, minlength=n_bins).astype(float)

        segments = self._greedy_segments(counts, sums, sums_sq)
        update = np.zeros(n_bins, dtype=float)
        for start, stop in segments:
            count = float(np.sum(counts[start:stop]))
            if count > 0:
                update[start:stop] = float(np.sum(sums[start:stop]) / count)
        return update

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

        while len(segments) < self.max_leaves:
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

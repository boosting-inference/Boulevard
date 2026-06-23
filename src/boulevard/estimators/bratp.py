"""Experimental histogram-tree BRAT-P estimator."""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boulevard.intervals.asymptotic import normal_interval, normal_quantile

try:
    from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
    from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
    from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
except ImportError as exc:  # pragma: no cover - import-time compatibility guard
    _HISTOGRAM_IMPORT_ERROR = exc
    _BinMapper = None
    G_H_DTYPE = np.float32
    TreeGrower = None
    TreePredictor = None
else:
    _HISTOGRAM_IMPORT_ERROR = None


class BRATPHistGradientBoostingRegressor(HistGradientBoostingRegressor):
    """Experimental BRAT-P backend built on sklearn histogram-tree internals.

    This class intentionally inherits sklearn's public histogram gradient
    boosting estimator for API compatibility, but replaces sklearn's boosting
    loop with the BRAT-P group/slot residual construction.
    """

    def __init__(
        self,
        loss: str = "squared_error",
        *,
        quantile: float | None = None,
        n_rounds: int = 100,
        trees_per_round: int = 10,
        subsample_rate: float = 1.0,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        categorical_features: Any | None = None,
        monotonic_cst: Any | None = None,
        interaction_cst: Any | None = None,
        warm_start: bool = False,
        early_stopping: bool | str = False,
        scoring: str | None = "loss",
        validation_fraction: float | None = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        verbose: int = 0,
        random_state: int | None = None,
        drop_first_round: bool = False,
    ) -> None:
        super().__init__(
            loss=loss,
            quantile=quantile,
            learning_rate=1.0,
            max_iter=n_rounds * trees_per_round,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.n_rounds = n_rounds
        self.trees_per_round = trees_per_round
        self.subsample_rate = subsample_rate
        self.drop_first_round = drop_first_round

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> BRATPHistGradientBoostingRegressor:
        """Fit the experimental histogram BRAT-P estimator."""
        fit_start = time.perf_counter()
        binning_seconds = 0.0
        cell_metadata_seconds = 0.0
        residual_seconds = 0.0
        gradient_seconds = 0.0
        grower_setup_seconds = 0.0
        grower_grow_seconds = 0.0
        predictor_seconds = 0.0
        training_prediction_cache_seconds = 0.0
        score_seconds = 0.0
        sampled_training_rows = 0

        self._check_histogram_backend()
        self._validate_brat_p_params()

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1 or sample_weight.shape[0] != y.shape[0]:
                raise ValueError(
                    "sample_weight must be one-dimensional with one entry per row."
                )
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values.")

        n_threads = self._effective_n_threads()
        rng = np.random.default_rng(self.random_state)
        n_bins = self.max_bins + 1

        step_start = time.perf_counter()
        self._bin_mapper = _BinMapper(
            n_bins=n_bins,
            is_categorical=None,
            known_categories=None,
            random_state=self.random_state,
            n_threads=n_threads,
        )
        X_binned = self._bin_mapper.fit_transform(X)
        has_missing_values = (
            (X_binned == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )
        binning_seconds += time.perf_counter() - step_start

        self.n_features_in_ = X.shape[1]
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.X_binned_train_ = X_binned.copy()

        step_start = time.perf_counter()
        self._init_cell_metadata(X_binned)
        cell_metadata_seconds += time.perf_counter() - step_start

        self._baseline_prediction = np.zeros((1, 1), dtype=float)
        self.n_trees_per_iteration_ = 1
        total_trees = self.n_rounds * self.trees_per_round
        self.max_iter = total_trees
        self._predictors: list[list[TreePredictor]] = []
        self._predictor_table: list[list[TreePredictor | None]] = [
            [None for _ in range(self.trees_per_round)]
            for _ in range(self.n_rounds)
        ]
        self._train_tree_predictions_ = np.empty((X_binned.shape[0], total_trees))
        self._train_prediction_table_ = np.zeros(
            (self.n_rounds, self.trees_per_round, X_binned.shape[0]),
            dtype=float,
        )
        self.in_bag_matrix_ = np.zeros((X_binned.shape[0], total_trees), dtype=bool)
        self.train_score_: list[float] = []
        self.conformal_interval_ = None
        slot_prediction_sums = np.zeros(
            (self.trees_per_round, X_binned.shape[0]),
            dtype=float,
        )
        current_round_prediction_sum = np.zeros(X_binned.shape[0], dtype=float)

        for round_idx in range(self.n_rounds):
            current_round_prediction_sum.fill(0.0)
            previous_slot_prediction_sums = slot_prediction_sums.copy()
            for slot_idx in range(self.trees_per_round):
                tree_idx = round_idx * self.trees_per_round + slot_idx

                step_start = time.perf_counter()
                residuals = self._residuals_for_next_tree_binned(
                    y,
                    round_idx=round_idx,
                    slot_idx=slot_idx,
                    previous_slot_prediction_sums=previous_slot_prediction_sums,
                    current_round_prediction_sum=current_round_prediction_sum,
                )
                residual_seconds += time.perf_counter() - step_start

                in_bag = self._sample_in_bag_indices(X_binned.shape[0], rng)
                self.in_bag_matrix_[in_bag, tree_idx] = True
                sampled_training_rows += int(in_bag.size)

                step_start = time.perf_counter()
                gradients = np.ascontiguousarray(
                    -residuals[in_bag],
                    dtype=G_H_DTYPE,
                )
                if sample_weight is None:
                    hessians = np.ones(1, dtype=G_H_DTYPE)
                else:
                    gradients = np.ascontiguousarray(
                        gradients * sample_weight[in_bag],
                        dtype=G_H_DTYPE,
                    )
                    hessians = np.ascontiguousarray(
                        sample_weight[in_bag],
                        dtype=G_H_DTYPE,
                    )
                gradient_seconds += time.perf_counter() - step_start

                step_start = time.perf_counter()
                grower = self._make_tree_grower(
                    X_binned=np.asfortranarray(X_binned[in_bag]),
                    gradients=gradients,
                    hessians=hessians,
                    n_bins=n_bins,
                    has_missing_values=has_missing_values,
                    rng=rng,
                    n_threads=n_threads,
                )
                grower_setup_seconds += time.perf_counter() - step_start

                step_start = time.perf_counter()
                grower.grow()
                grower_grow_seconds += time.perf_counter() - step_start

                step_start = time.perf_counter()
                predictor = grower.make_predictor(
                    binning_thresholds=self._bin_mapper.bin_thresholds_
                )
                predictor_seconds += time.perf_counter() - step_start
                self._predictors.append([predictor])
                self._predictor_table[round_idx][slot_idx] = predictor

                step_start = time.perf_counter()
                train_prediction = predictor.predict_binned(
                    X_binned,
                    self._bin_mapper.missing_values_bin_idx_,
                    n_threads,
                )
                self._train_tree_predictions_[:, tree_idx] = train_prediction
                self._train_prediction_table_[round_idx, slot_idx] = train_prediction
                slot_prediction_sums[slot_idx] += train_prediction
                current_round_prediction_sum += train_prediction
                training_prediction_cache_seconds += time.perf_counter() - step_start

                if self.verbose:
                    step_start = time.perf_counter()
                    mse = float(np.mean((y - self.predict(X)) ** 2))
                    self.train_score_.append(mse)
                    score_seconds += time.perf_counter() - step_start

        if not self.verbose:
            self.train_score_ = []
        total_seconds = time.perf_counter() - fit_start
        self.fit_diagnostics_ = {
            "total_seconds": total_seconds,
            "binning_seconds": binning_seconds,
            "cell_metadata_seconds": cell_metadata_seconds,
            "residual_seconds": residual_seconds,
            "gradient_seconds": gradient_seconds,
            "grower_setup_seconds": grower_setup_seconds,
            "grower_grow_seconds": grower_grow_seconds,
            "predictor_seconds": predictor_seconds,
            "training_prediction_cache_seconds": training_prediction_cache_seconds,
            "score_seconds": score_seconds,
            "n_rounds": self.n_rounds,
            "trees_per_round": self.trees_per_round,
            "total_trees": total_trees,
            "drop_first_round": self.drop_first_round,
            "sampled_training_rows": sampled_training_rows,
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression targets with BRAT-P round/slot aggregation."""
        check_is_fitted(self, "_predictors")
        X = check_array(X, accept_sparse=False)
        X_binned = self._bin_data(X)
        if not self._predictors:
            return np.zeros(X.shape[0], dtype=float)

        return self._predict_brat_p_binned(
            X_binned,
            n_threads=self._effective_n_threads(),
        )

    def apply_bin_indices(self, X: Any) -> np.ndarray:
        """Return sklearn histogram-bin indices for each sample and feature."""
        check_is_fitted(self, "_bin_mapper")
        X = check_array(X, accept_sparse=False)
        return self._bin_data(X).copy()

    def apply_cell_indices(self, X: Any) -> np.ndarray:
        """Return observed training-cell indices, or ``-1`` for unseen cells."""
        check_is_fitted(self, "observed_cells_")
        X_binned = self.apply_bin_indices(X)
        return np.array(
            [self._cell_to_index_.get(tuple(row.tolist()), -1) for row in X_binned],
            dtype=int,
        )

    def prepare_inference(
        self,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> BRATPHistGradientBoostingRegressor:
        """Prepare observed-cell BRAT-P asymptotic interval inference.

        Histogram bins compress duplicate training rows into multidimensional
        observed cells. The BRAT-P leaf kernel is then built over these cells
        with cell counts carrying the multiplicity of the original rows.
        """
        check_is_fitted(self, "_predictors")

        if (X_calib is None) != (y_calib is None):
            raise ValueError("X_calib and y_calib must be provided together.")

        if X_calib is None:
            X_var = self.X_train_
            y_var = self.y_train_
        else:
            X_var = check_array(X_calib, accept_sparse=False)
            y_var = np.asarray(y_calib, dtype=float)
            if X_var.shape[0] != y_var.shape[0]:
                raise ValueError(
                    "X_calib and y_calib must contain the same number of rows."
                )

        residuals = y_var - self.predict(X_var)
        if residuals.shape[0] < 2:
            raise ValueError(
                "At least two residuals are required to estimate variance."
            )

        self.X_inference_calib_ = X_var.copy()
        self.y_inference_calib_ = y_var.copy()
        self.sigma_hat2_ = float(np.var(residuals, ddof=1))
        self._prepare_cell_kernel()
        self.inference_method_ = "histogram_cell"
        return self

    def predict_interval(
        self,
        X: Any,
        alpha: float | None = None,
        method: str = "asymptotic",
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return asymptotic prediction intervals."""
        if method != "asymptotic":
            raise ValueError("Only method='asymptotic' is implemented for BRAT-P hist.")
        return self.prediction_interval(
            X,
            alpha=0.05 if alpha is None else alpha,
            X_calib=X_calib,
            y_calib=y_calib,
        )

    def confidence_interval(
        self,
        X: Any,
        alpha: float = 0.05,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return observed-cell asymptotic confidence intervals for ``f(x)``."""
        self._ensure_inference_prepared(X_calib=X_calib, y_calib=y_calib)
        center = self.predict(X)
        r_norm = self._weight_norms(X)
        se = np.sqrt(self.sigma_hat2_) * r_norm
        interval = normal_interval(center, se, alpha=alpha)
        return interval.lower, interval.upper

    def prediction_interval(
        self,
        X: Any,
        alpha: float = 0.05,
        calibrated: bool = False,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return observed-cell asymptotic prediction intervals for ``y | x``."""
        self._ensure_inference_prepared(X_calib=X_calib, y_calib=y_calib)
        center = self.predict(X)
        half_width = self._prediction_half_width(X, alpha=alpha)
        if calibrated:
            half_width *= self._prediction_calibration_scale(alpha=alpha)
        return center - half_width, center + half_width

    def reproduction_interval(
        self,
        X: Any,
        alpha: float = 0.05,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return observed-cell asymptotic reproduction intervals."""
        self._ensure_inference_prepared(X_calib=X_calib, y_calib=y_calib)
        center = self.predict(X)
        r_norm = self._weight_norms(X)
        se = np.sqrt(2) * np.sqrt(self.sigma_hat2_) * r_norm
        interval = normal_interval(center, se, alpha=alpha)
        return interval.lower, interval.upper

    def weight_norms(self, X: Any) -> np.ndarray:
        """Return observed-cell BRAT-P weight norms used by intervals."""
        self._ensure_inference_prepared()
        return self._weight_norms(X)

    @staticmethod
    def _check_histogram_backend() -> None:
        if _HISTOGRAM_IMPORT_ERROR is not None:
            raise ImportError(
                "BRATPHistGradientBoostingRegressor requires sklearn histogram "
                "gradient boosting private internals."
            ) from _HISTOGRAM_IMPORT_ERROR

    def _validate_brat_p_params(self) -> None:
        if self.loss != "squared_error":
            raise ValueError(
                "BRATPHistGradientBoostingRegressor currently supports only "
                "loss='squared_error'."
            )
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be at least 1.")
        if self.trees_per_round <= 1:
            raise ValueError("trees_per_round must be greater than 1.")
        if not 0 < self.subsample_rate <= 1:
            raise ValueError("subsample_rate must be in (0, 1].")
        if self.warm_start:
            raise ValueError(
                "warm_start is not supported for BRAT-P histogram fitting."
            )
        if self.early_stopping not in (False, None):
            raise ValueError(
                "early_stopping must be False or None for BRAT-P histogram fitting."
            )
        if self.categorical_features is not None:
            raise ValueError(
                "categorical_features is not supported yet for BRAT-P histogram "
                "fitting."
            )
        if self.monotonic_cst is not None:
            raise ValueError(
                "monotonic_cst is not supported yet for BRAT-P histogram fitting."
            )
        if self.interaction_cst is not None:
            raise ValueError(
                "interaction_cst is not supported yet for BRAT-P histogram fitting."
            )
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2.")
        if self.max_bins > 255:
            raise ValueError("max_bins cannot exceed 255 for sklearn histogram trees.")

    def _effective_n_threads(self) -> int:
        return 1

    def _sample_in_bag_indices(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.subsample_rate == 1.0:
            return np.arange(n_samples)

        n_in_bag = max(1, int(np.ceil(self.subsample_rate * n_samples)))
        return np.sort(rng.choice(n_samples, size=n_in_bag, replace=False))

    def _bin_data(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self._bin_mapper.transform(X))

    def _init_cell_metadata(self, X_binned: np.ndarray) -> None:
        observed_cells, inverse, counts = np.unique(
            X_binned,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        self.observed_cells_ = np.ascontiguousarray(observed_cells)
        self.train_cell_indices_ = inverse.astype(int, copy=False)
        self.cell_counts_ = counts.astype(float, copy=False)
        self._cell_to_index_ = {
            tuple(row.tolist()): idx for idx, row in enumerate(self.observed_cells_)
        }

    def _make_tree_grower(
        self,
        *,
        X_binned: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        n_bins: int,
        has_missing_values: np.ndarray,
        rng: np.random.Generator,
        n_threads: int,
    ) -> TreeGrower:
        kwargs: dict[str, Any] = {
            "X_binned": X_binned,
            "gradients": gradients,
            "hessians": hessians,
            "n_bins": n_bins,
            "n_bins_non_missing": self._bin_mapper.n_bins_non_missing_,
            "has_missing_values": has_missing_values,
            "is_categorical": None,
            "monotonic_cst": None,
            "interaction_cst": None,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "shrinkage": 1.0,
            "n_threads": n_threads,
        }
        signature = inspect.signature(TreeGrower.__init__)
        if "feature_fraction_per_split" in signature.parameters:
            kwargs["feature_fraction_per_split"] = 1.0
        if "rng" in signature.parameters:
            kwargs["rng"] = rng
        return TreeGrower(**kwargs)

    def _residuals_for_next_tree_binned(
        self,
        y: np.ndarray,
        *,
        round_idx: int,
        slot_idx: int,
        previous_slot_prediction_sums: np.ndarray,
        current_round_prediction_sum: np.ndarray,
    ) -> np.ndarray:
        if round_idx == 0 and self.drop_first_round:
            return y

        if round_idx == 0:
            return y - current_round_prediction_sum

        previous_slot_averages = previous_slot_prediction_sums / round_idx
        dropped_prediction = (
            previous_slot_averages.sum(axis=0) - previous_slot_averages[slot_idx]
        )
        return y - dropped_prediction

    def _predict_tree_sum_binned(
        self,
        X_binned: np.ndarray,
        *,
        selected: np.ndarray | None,
        n_threads: int,
    ) -> np.ndarray:
        out = np.zeros(X_binned.shape[0], dtype=float)
        if selected is None:
            selected_iter = range(len(self._predictors))
        else:
            selected_iter = selected

        for idx in selected_iter:
            predictor = self._predictors[int(idx)][0]
            out += predictor.predict_binned(
                X_binned,
                self._bin_mapper.missing_values_bin_idx_,
                n_threads,
            )
        return out

    def _predict_brat_p_binned(
        self,
        X_binned: np.ndarray,
        *,
        n_threads: int,
    ) -> np.ndarray:
        slot_sums = np.zeros((self.trees_per_round, X_binned.shape[0]), dtype=float)
        for round_idx in range(self.n_rounds):
            for slot_idx in range(self.trees_per_round):
                predictor = self._predictor_table[round_idx][slot_idx]
                if predictor is None:
                    continue
                slot_sums[slot_idx] += predictor.predict_binned(
                    X_binned,
                    self._bin_mapper.missing_values_bin_idx_,
                    n_threads,
                )
        return slot_sums.sum(axis=0) / self.n_rounds

    def _apply_leaf_indices_binned(self, X_binned: np.ndarray) -> np.ndarray:
        leaves = np.empty((X_binned.shape[0], len(self._predictors)), dtype=int)
        for tree_idx, predictor_group in enumerate(self._predictors):
            leaves[:, tree_idx] = self._apply_single_tree_leaves_binned(
                predictor_group[0],
                X_binned,
            )
        return leaves

    def _apply_single_tree_leaves_binned(
        self,
        predictor: TreePredictor,
        X_binned: np.ndarray,
    ) -> np.ndarray:
        nodes = predictor.nodes
        out = np.empty(X_binned.shape[0], dtype=int)
        missing_bin = self._bin_mapper.missing_values_bin_idx_

        for row_idx, row in enumerate(X_binned):
            node_idx = 0
            while not nodes[node_idx]["is_leaf"]:
                node = nodes[node_idx]
                feature_idx = int(node["feature_idx"])
                value = row[feature_idx]
                if value == missing_bin:
                    go_left = bool(node["missing_go_to_left"])
                else:
                    go_left = value <= node["bin_threshold"]
                node_idx = int(node["left"] if go_left else node["right"])
            out[row_idx] = node_idx

        return out

    def _prepare_cell_kernel(self) -> None:
        self.cell_leaf_assignments_ = self._apply_leaf_indices_binned(
            self.observed_cells_
        )
        self.cell_kernel_matrix_ = self._cell_kernel_matrix(
            self.cell_leaf_assignments_,
            self.cell_counts_,
        )
        self._prepare_observed_cell_norm_cache()

    def _prepare_observed_cell_norm_cache(self) -> None:
        weighted_kernel = self.cell_counts_[:, None] * self.cell_kernel_matrix_
        # BRAT-P uses k(x)' [K^{-1} I + (K - 1) K^{-1} K_n]^{-1}.
        matrix = (1 / self.trees_per_round) * np.eye(
            self.cell_kernel_matrix_.shape[0]
        )
        self.cell_system_matrix_t_ = (
            matrix
            + ((self.trees_per_round - 1) / self.trees_per_round) * weighted_kernel
        ).T

        weights = np.linalg.solve(
            self.cell_system_matrix_t_, self.cell_kernel_matrix_.T
        ).T
        self.cell_weight_norms_ = np.sqrt(
            np.maximum((weights**2) @ self.cell_counts_, 0.0)
        )

    @staticmethod
    def _cell_kernel_matrix(
        cell_leaf_indices: np.ndarray,
        cell_counts: np.ndarray,
    ) -> np.ndarray:
        leaves = np.asarray(cell_leaf_indices)
        counts = np.asarray(cell_counts, dtype=float)
        if leaves.ndim != 2:
            raise ValueError("cell_leaf_indices must have shape (n_cells, n_trees).")
        if leaves.shape[0] != counts.shape[0]:
            raise ValueError("cell_counts must have one entry per cell.")

        n_cells, n_trees = leaves.shape
        if n_trees == 0:
            raise ValueError("cell_leaf_indices must contain at least one tree.")

        kernel = np.zeros((n_cells, n_cells), dtype=float)
        for tree_idx in range(n_trees):
            tree_leaves = leaves[:, tree_idx]
            for leaf in np.unique(tree_leaves):
                members = np.flatnonzero(tree_leaves == leaf)
                denom = float(np.sum(counts[members]))
                if denom <= 0:
                    continue
                kernel[np.ix_(members, members)] += 1.0 / denom
        return kernel / n_trees

    def _cell_kernel_vector(self, test_leaf_indices: np.ndarray) -> np.ndarray:
        train_leaves = self.cell_leaf_assignments_
        test_leaves = np.asarray(test_leaf_indices)
        counts = self.cell_counts_

        if test_leaves.ndim != 2:
            raise ValueError("test leaf indices must have shape (n_samples, n_trees).")
        if train_leaves.shape[1] != test_leaves.shape[1]:
            raise ValueError(
                "train and test leaf indices must have the same number of trees."
            )

        n_test, n_trees = test_leaves.shape
        out = np.zeros((n_test, train_leaves.shape[0]), dtype=float)
        for tree_idx in range(n_trees):
            train_tree_leaves = train_leaves[:, tree_idx]
            for row_idx in range(n_test):
                members = train_tree_leaves == test_leaves[row_idx, tree_idx]
                denom = float(np.sum(counts[members]))
                if denom > 0:
                    out[row_idx, members] += 1.0 / denom
        return out / n_trees

    def _solve_cell_brat_p_weights(self, kernel_vectors: np.ndarray) -> np.ndarray:
        matrix_t = getattr(self, "cell_system_matrix_t_", None)
        if matrix_t is None:
            weighted_kernel = self.cell_counts_[:, None] * self.cell_kernel_matrix_
            # BRAT-P uses k(x)' [K^{-1} I + (K - 1) K^{-1} K_n]^{-1}.
            matrix = (1 / self.trees_per_round) * np.eye(
                self.cell_kernel_matrix_.shape[0]
            )
            matrix_t = (
                matrix
                + ((self.trees_per_round - 1) / self.trees_per_round)
                * weighted_kernel
            ).T
        return np.linalg.solve(matrix_t, kernel_vectors.T).T

    def _weight_norms(self, X: Any) -> np.ndarray:
        self._check_inference_prepared()
        X = check_array(X, accept_sparse=False)
        X_binned = self._bin_data(X)
        cell_indices = np.array(
            [self._cell_to_index_.get(tuple(row.tolist()), -1) for row in X_binned],
            dtype=int,
        )

        norms = np.empty(X_binned.shape[0], dtype=float)
        observed = cell_indices >= 0
        if np.any(observed):
            norms[observed] = self.cell_weight_norms_[cell_indices[observed]]

        if np.any(~observed):
            unseen_binned = X_binned[~observed]
            test_leaf_indices = self._apply_leaf_indices_binned(unseen_binned)
            kernel_vectors = self._cell_kernel_vector(test_leaf_indices)
            weights = self._solve_cell_brat_p_weights(kernel_vectors)
            norms[~observed] = np.sqrt(
                np.maximum((weights**2) @ self.cell_counts_, 0.0)
            )

        return norms

    def _prediction_half_width(self, X: Any, alpha: float) -> np.ndarray:
        r_norm = self._weight_norms(X)
        se = np.sqrt(self.sigma_hat2_ * (1 + r_norm**2))
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

    def _ensure_inference_prepared(
        self,
        *,
        X_calib: Any | None = None,
        y_calib: Any | None = None,
    ) -> None:
        if (X_calib is None) != (y_calib is None):
            raise ValueError("X_calib and y_calib must be provided together.")
        if X_calib is not None:
            self.prepare_inference(X_calib, y_calib)
            return
        if not self._is_inference_prepared():
            self.prepare_inference()

    def _is_inference_prepared(self) -> bool:
        return (
            hasattr(self, "cell_kernel_matrix_")
            and hasattr(self, "cell_weight_norms_")
            and hasattr(self, "sigma_hat2_")
        )

    def _check_inference_prepared(self) -> None:
        check_is_fitted(self, "_predictors")
        if not self._is_inference_prepared():
            raise RuntimeError(
                "BRAT-P histogram asymptotic intervals require "
                "prepare_inference(...) before calling interval methods."
            )


__all__ = ["BRATPHistGradientBoostingRegressor"]

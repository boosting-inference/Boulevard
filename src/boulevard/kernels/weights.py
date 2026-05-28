"""Kernel-ridge-style weight helpers for asymptotic intervals."""

from __future__ import annotations

import numpy as np


def solve_brat_d_weights(
    kernel_vector: np.ndarray,
    kernel_matrix: np.ndarray,
    *,
    learning_rate: float,
    dropout_rate: float,
) -> np.ndarray:
    """Solve BRAT-D kernel weights ``k(x)^T (lambda^-1 I + qK)^-1``."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    q = 1 - dropout_rate
    matrix = (1 / learning_rate) * np.eye(kernel_matrix.shape[0]) + q * kernel_matrix
    kernel_vector = np.asarray(kernel_vector, dtype=float)
    if kernel_vector.ndim == 1:
        return np.linalg.solve(matrix.T, kernel_vector)
    if kernel_vector.ndim == 2:
        return np.linalg.solve(matrix.T, kernel_vector.T).T
    raise ValueError("kernel_vector must be one- or two-dimensional.")

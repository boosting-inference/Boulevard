"""Nystrom sketching helpers for histogram-cell inference."""

from __future__ import annotations

import numpy as np


def sample_nystrom_landmarks(
    *,
    cell_counts: np.ndarray,
    subsample_rate: float,
    random_state: int | None,
) -> np.ndarray:
    """Sample observed cells as Nystrom landmarks.

    The paper's sketching matrix samples training rows. In the histogram-cell
    implementation, repeated training rows have already been compressed into
    observed cells, so cells are sampled with probability proportional to their
    training-row multiplicity.
    """
    counts = np.asarray(cell_counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("cell_counts must be one-dimensional.")
    if counts.size == 0:
        raise ValueError("cell_counts must contain at least one cell.")
    if np.any(counts <= 0):
        raise ValueError("cell_counts must be positive.")
    if not 0 < subsample_rate <= 1:
        raise ValueError("nystrom_subsample_rate must be in (0, 1].")

    n_train = int(np.sum(counts))
    n_landmarks = min(counts.size, max(1, int(np.ceil(subsample_rate * n_train))))
    if n_landmarks >= counts.size:
        return np.arange(counts.size, dtype=int)

    rng = np.random.default_rng(random_state)
    probabilities = counts / np.sum(counts)
    return np.sort(
        rng.choice(
            counts.size,
            size=n_landmarks,
            replace=False,
            p=probabilities,
        )
    )


def nystrom_weight_norm_cache(
    *,
    kernel_matrix: np.ndarray,
    cell_counts: np.ndarray,
    landmark_indices: np.ndarray,
    identity_scale: float,
    kernel_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached observed-cell norms and the sketched covariance matrix.

    This implements the matrix-sketching formula from the paper for systems of
    the form ``identity_scale * I + kernel_scale * K``. With ``S`` the
    subsampling matrix and ``K_S = K S``, the inverse is approximated by

    ``a^-1 I - a^-2 K_S (b^-1 S' K S + a^-1 K_S' K_S)^-1 S' K``,

    where ``a=identity_scale`` and ``b=kernel_scale``. The returned covariance
    matrix is the compressed ``Sigma`` used to estimate ``||r_n(x)||`` from the
    landmark kernel vector ``S' k_n(x)``.
    """
    kernel = np.asarray(kernel_matrix, dtype=float)
    counts = np.asarray(cell_counts, dtype=float)
    landmarks = np.asarray(landmark_indices, dtype=int)

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel_matrix must be square.")
    if counts.ndim != 1 or counts.shape[0] != kernel.shape[0]:
        raise ValueError("cell_counts must match kernel_matrix.")
    if landmarks.ndim != 1 or landmarks.size == 0:
        raise ValueError("landmark_indices must be a non-empty vector.")
    if np.any(landmarks < 0) or np.any(landmarks >= kernel.shape[0]):
        raise ValueError("landmark_indices contain out-of-range cells.")
    if identity_scale <= 0:
        raise ValueError("identity_scale must be positive.")
    if kernel_scale <= 0:
        raise ValueError("kernel_scale must be positive.")

    landmark_kernel = kernel[:, landmarks]
    landmark_gram = kernel[np.ix_(landmarks, landmarks)]
    weighted_landmark_kernel = counts[:, None] * landmark_kernel
    gram = landmark_kernel.T @ weighted_landmark_kernel

    identity_inverse = 1.0 / identity_scale
    middle = (1.0 / kernel_scale) * landmark_gram + identity_inverse * gram
    middle_pinv = np.linalg.pinv(middle)
    landmark_gram_pinv = np.linalg.pinv(landmark_gram)

    sketched_inverse_basis = (
        identity_inverse * landmark_kernel
        - (identity_inverse**2) * landmark_kernel @ middle_pinv @ gram
    ) @ landmark_gram_pinv
    sigma = sketched_inverse_basis.T @ (counts[:, None] * sketched_inverse_basis)
    norms = nystrom_weight_norms(landmark_kernel, sigma)
    return norms, sigma


def nystrom_weight_norms(
    landmark_kernel_vectors: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Estimate ``||r_n(x)||`` from landmark kernel vectors."""
    vectors = np.asarray(landmark_kernel_vectors, dtype=float)
    covariance = np.asarray(sigma, dtype=float)

    if vectors.ndim != 2:
        raise ValueError("landmark_kernel_vectors must be two-dimensional.")
    if covariance.shape != (vectors.shape[1], vectors.shape[1]):
        raise ValueError("sigma has incompatible shape.")

    squared = np.einsum("ij,jk,ik->i", vectors, covariance, vectors)
    return np.sqrt(np.maximum(squared, 0.0))

"""Leaf-indicator kernel utilities."""

from __future__ import annotations

import numpy as np


def leaf_kernel_matrix(
    leaf_indices: np.ndarray,
    in_bag: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the empirical tree leaf kernel for training samples.

    Parameters
    ----------
    leaf_indices:
        Integer array of shape ``(n_samples, n_trees)``.
    in_bag:
        Optional boolean array of the same shape indicating whether a sample
        was used to fit each tree.
    """
    leaves = np.asarray(leaf_indices)
    if leaves.ndim != 2:
        raise ValueError("leaf_indices must have shape (n_samples, n_trees).")

    n_samples, n_trees = leaves.shape
    if n_trees == 0:
        raise ValueError("leaf_indices must contain at least one tree.")

    if in_bag is None:
        mask = np.ones_like(leaves, dtype=bool)
    else:
        mask = np.asarray(in_bag, dtype=bool)
        if mask.shape != leaves.shape:
            raise ValueError("in_bag must have the same shape as leaf_indices.")

    kernel = np.zeros((n_samples, n_samples), dtype=float)
    for tree_idx in range(n_trees):
        same_leaf = leaves[:, tree_idx, None] == leaves[None, :, tree_idx]
        valid = same_leaf & mask[None, :, tree_idx]
        counts = valid.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1
        kernel += valid / counts

    return kernel / n_trees


def leaf_kernel_vector(
    train_leaf_indices: np.ndarray,
    test_leaf_indices: np.ndarray,
    in_bag: np.ndarray | None = None,
) -> np.ndarray:
    """Compute empirical leaf-kernel vectors for test samples."""
    train_leaves = np.asarray(train_leaf_indices)
    test_leaves = np.asarray(test_leaf_indices)

    if train_leaves.ndim != 2 or test_leaves.ndim != 2:
        raise ValueError("leaf indices must be two-dimensional.")
    if train_leaves.shape[1] != test_leaves.shape[1]:
        raise ValueError(
            "train and test leaf indices must have the same number of trees."
        )

    n_train, n_trees = train_leaves.shape
    n_test = test_leaves.shape[0]
    if in_bag is None:
        mask = np.ones_like(train_leaves, dtype=bool)
    else:
        mask = np.asarray(in_bag, dtype=bool)
        if mask.shape != train_leaves.shape:
            raise ValueError("in_bag must have the same shape as train_leaf_indices.")

    out = np.zeros((n_test, n_train), dtype=float)
    for tree_idx in range(n_trees):
        same_leaf = test_leaves[:, tree_idx, None] == train_leaves[None, :, tree_idx]
        valid = same_leaf & mask[None, :, tree_idx]
        counts = valid.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1
        out += valid / counts

    return out / n_trees

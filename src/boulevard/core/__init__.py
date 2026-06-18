"""Core compatibility namespace."""

from boulevard.algorithms import boulevard_scale, select_tail
from boulevard.kernels import leaf_kernel_matrix, leaf_kernel_vector

__all__ = [
    "boulevard_scale",
    "leaf_kernel_matrix",
    "leaf_kernel_vector",
    "select_tail",
]

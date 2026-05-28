"""Compatibility exports for backend interfaces."""

from boulevard.backends._protocol import (
    SupportsInBagMatrix,
    SupportsPerTreePrediction,
    TreeEnsembleBackend,
)

__all__ = [
    "SupportsInBagMatrix",
    "SupportsPerTreePrediction",
    "TreeEnsembleBackend",
]

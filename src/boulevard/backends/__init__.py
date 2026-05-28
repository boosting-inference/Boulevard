"""Backend adapters for supported tree ensemble libraries."""

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

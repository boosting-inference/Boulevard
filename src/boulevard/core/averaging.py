"""Compatibility shim for Boulevard aggregation helpers."""

from boulevard.algorithms.boulevard import boulevard_scale
from boulevard.algorithms.brat_d import brat_d_scale
from boulevard.algorithms.selection import select_tail

__all__ = ["boulevard_scale", "brat_d_scale", "select_tail"]

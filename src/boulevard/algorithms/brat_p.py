"""BRAT-P formulas and placeholders."""

from __future__ import annotations


def validate_trees_per_group(n_trees_per_group: int) -> None:
    """Validate the BRAT-P parallel group size."""
    if n_trees_per_group < 1:
        raise ValueError("n_trees_per_group must be at least 1.")

"""Vanilla Boulevard aggregation formulas."""

from __future__ import annotations


def boulevard_scale(learning_rate: float) -> float:
    """Return the final prediction scale for vanilla Boulevard."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    return (1 + learning_rate) / learning_rate

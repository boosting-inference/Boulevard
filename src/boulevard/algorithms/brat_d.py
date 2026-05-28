"""BRAT-D formulas."""

from __future__ import annotations


def brat_d_scale(learning_rate: float, dropout_rate: float) -> float:
    """Return the final prediction scale for BRAT-D."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not 0 <= dropout_rate <= 1:
        raise ValueError("dropout_rate must be between 0 and 1.")
    q = 1 - dropout_rate
    return (1 + learning_rate * q) / learning_rate

"""Validation helpers."""

from __future__ import annotations


def check_alpha(alpha: float) -> None:
    """Validate a conformal or interval significance level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

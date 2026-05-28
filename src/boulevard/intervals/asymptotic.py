"""Asymptotic interval formulas."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class NormalInterval:
    """Container for lower and upper normal-approximation bounds."""

    lower: np.ndarray
    upper: np.ndarray


def normal_interval(
    center: np.ndarray,
    standard_error: np.ndarray,
    alpha: float = 0.05,
) -> NormalInterval:
    """Return normal-approximation interval bounds."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    center = np.asarray(center, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    return NormalInterval(
        lower=center - z * standard_error,
        upper=center + z * standard_error,
    )

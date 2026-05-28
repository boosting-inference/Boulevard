"""Nyström approximation helpers."""

from __future__ import annotations

import numpy as np


def uniform_landmarks(
    n_samples: int,
    n_landmarks: int | float,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample landmark row indices without replacement."""
    if isinstance(n_landmarks, float):
        if not 0 < n_landmarks <= 1:
            raise ValueError("float n_landmarks must be in (0, 1].")
        size = int(n_samples * n_landmarks)
    else:
        size = int(n_landmarks)

    if not 1 <= size <= n_samples:
        raise ValueError("n_landmarks must select between 1 and n_samples rows.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    return rng.choice(n_samples, size=size, replace=False)

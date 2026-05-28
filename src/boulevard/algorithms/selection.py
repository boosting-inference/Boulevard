"""Tree selection utilities for Boulevard-style aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def select_tail(
    items: Sequence[T],
    *,
    burn_in: int = 0,
    buffer_size: int | float | str | None = None,
    block_size: int | None = None,
    dropout_rate: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> list[T]:
    """Select ensemble members for Boulevard-style prediction.

    This mirrors the scratch implementation's prediction-time controls while
    keeping the policy independent from any particular backend.
    """
    selected = list(items)

    if burn_in:
        selected = selected[min(burn_in, max(0, len(selected) - burn_in)) :]

    if buffer_size is not None:
        if buffer_size == "sqrt":
            keep = 5 * int(np.sqrt(len(selected)))
            selected = selected[-keep:] if keep else []
        elif isinstance(buffer_size, float) and buffer_size < 1:
            keep = int(buffer_size * len(selected))
            selected = selected[-keep:] if keep else []
        else:
            selected = selected[-int(buffer_size) :]

    if block_size is not None:
        step = max(1, block_size + len(items) % block_size)
        selected = selected[::step]

    if dropout_rate is not None:
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1.")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        keep_mask = rng.binomial(1, 1 - dropout_rate, len(selected)).astype(bool)
        selected = [item for item, keep in zip(selected, keep_mask) if keep]

    return selected

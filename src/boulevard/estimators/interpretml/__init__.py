"""InterpretML-family estimators."""

from __future__ import annotations

from boulevard.estimators.interpretml.iebm import IEBMRegressor


class EBMRegressor:
    """Reserved public wrapper for future InterpretML support."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("InterpretML support has not been implemented yet.")


__all__ = ["EBMRegressor", "IEBMRegressor"]

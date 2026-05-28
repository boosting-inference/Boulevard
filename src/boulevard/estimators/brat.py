"""Native BRAT estimator placeholders.

BRAT-D and BRAT-P will be ported here from the scratch ``BRATs`` package.
They require true custom residual construction, so they belong in native
Boulevard estimators instead of backend wrappers around existing libraries.
"""

from __future__ import annotations


class BRATDRegressor:
    """Reserved estimator for Boulevard Regularized Additive Trees with dropout."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("BRATDRegressor has not been ported yet.")


class BRATPRegressor:
    """Reserved estimator for parallel Boulevard Regularized Additive Trees."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("BRATPRegressor has not been ported yet.")

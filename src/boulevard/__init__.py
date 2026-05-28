"""Boulevard: regularized boosting wrappers with uncertainty intervals."""

from boulevard._version import __version__
from boulevard.estimators.brat import BRATDRegressor
from boulevard.estimators.xgboost import XGBRegressor

__all__ = [
    "BRATDRegressor",
    "__version__",
    "XGBRegressor",
]

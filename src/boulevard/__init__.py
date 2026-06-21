"""Boulevard: regularized boosting wrappers with uncertainty intervals."""

from boulevard._version import __version__
from boulevard.estimators.brat import BRATDRegressor
from boulevard.estimators.bratd import BRATDHistGradientBoostingRegressor
from boulevard.estimators.bratp import BRATPHistGradientBoostingRegressor
from boulevard.estimators.xgboost import XGBRegressor

__all__ = [
    "BRATDHistGradientBoostingRegressor",
    "BRATDRegressor",
    "BRATPHistGradientBoostingRegressor",
    "__version__",
    "XGBRegressor",
]

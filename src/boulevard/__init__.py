"""Boulevard: regularized boosting wrappers with uncertainty intervals."""

from boulevard._version import __version__
from boulevard.estimators.brat import BRATDRegressor
from boulevard.estimators.brat_hist import BRATDHistGradientBoostingRegressor
from boulevard.estimators.xgboost import XGBRegressor

__all__ = [
    "BRATDHistGradientBoostingRegressor",
    "BRATDRegressor",
    "__version__",
    "XGBRegressor",
]

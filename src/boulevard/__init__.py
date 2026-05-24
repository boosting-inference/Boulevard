"""Boulevard: regularized boosting wrappers with uncertainty intervals."""

from boulevard._version import __version__
from boulevard.backends.xgboost import XGBRegressor

__all__ = [
    "__version__",
    "XGBRegressor",
]
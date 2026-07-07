import boulevard as bd
from boulevard.estimators import (
    BRATDHistGradientBoostingRegressor,
    BRATPHistGradientBoostingRegressor,
    IEBMRegressor,
)
from boulevard.estimators import sklearn as bd_sklearn
from boulevard.estimators.catboost import CatBoostRegressor
from boulevard.estimators.interpretml import EBMRegressor
from boulevard.estimators.interpretml import IEBMRegressor as NamespacedIEBMRegressor
from boulevard.estimators.lightgbm import LGBMRegressor


def test_import_boulevard():
    assert bd.__version__ == "0.1.0a1"


def test_brat_d_hist_is_public_api():
    assert bd.BRATDHistGradientBoostingRegressor is BRATDHistGradientBoostingRegressor
    assert (
        bd_sklearn.BRATDHistGradientBoostingRegressor
        is BRATDHistGradientBoostingRegressor
    )

    model = bd.BRATDHistGradientBoostingRegressor(
        max_iter=3,
        learning_rate=0.4,
        dropout_rate=0.2,
        early_stopping=False,
    )

    assert model.get_params()["max_iter"] == 3
    assert model.get_params()["dropout_rate"] == 0.2
    model.set_params(max_iter=4)
    assert model.max_iter == 4


def test_brat_p_hist_is_public_api():
    assert bd.BRATPHistGradientBoostingRegressor is BRATPHistGradientBoostingRegressor
    assert (
        bd_sklearn.BRATPHistGradientBoostingRegressor
        is BRATPHistGradientBoostingRegressor
    )

    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=2,
        early_stopping=False,
    )

    assert model.get_params()["n_rounds"] == 3
    assert model.get_params()["trees_per_round"] == 2
    model.set_params(n_rounds=4)
    assert model.n_rounds == 4


def test_backend_estimator_namespaces_import():
    assert not hasattr(bd, "XGBRegressor")
    assert bd.IEBMRegressor is IEBMRegressor
    assert NamespacedIEBMRegressor is IEBMRegressor
    assert LGBMRegressor.__name__ == "LGBMRegressor"
    assert CatBoostRegressor.__name__ == "CatBoostRegressor"
    assert EBMRegressor.__name__ == "EBMRegressor"

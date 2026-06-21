import boulevard as bd
from boulevard.estimators import (
    BRATDHistGradientBoostingRegressor,
    BRATPHistGradientBoostingRegressor,
)


def test_import_boulevard():
    assert bd.__version__ == "0.0.1"


def test_brat_d_hist_is_public_api():
    assert bd.BRATDHistGradientBoostingRegressor is BRATDHistGradientBoostingRegressor

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

    model = bd.BRATPHistGradientBoostingRegressor(
        n_rounds=3,
        trees_per_round=2,
        early_stopping=False,
    )

    assert model.get_params()["n_rounds"] == 3
    assert model.get_params()["trees_per_round"] == 2
    model.set_params(n_rounds=4)
    assert model.n_rounds == 4

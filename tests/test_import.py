import boulevard as bd
from boulevard.estimators import (
    DropoutBooster,
    ExplainableBooster,
    ParallelBooster,
)
from boulevard.estimators import interpretml as bd_interpretml
from boulevard.estimators import sklearn as bd_sklearn
from boulevard.estimators.interpretml import (
    ExplainableBooster as NamespacedExplainableBooster,
)


def test_import_boulevard():
    assert bd.__version__ == "0.1.0a1"


def test_dropout_booster_is_public_api():
    assert bd.DropoutBooster is DropoutBooster
    assert (
        bd_sklearn.DropoutBooster
        is DropoutBooster
    )

    model = bd.DropoutBooster(
        max_iter=3,
        learning_rate=0.4,
        dropout_rate=0.2,
        early_stopping=False,
    )

    assert model.get_params()["max_iter"] == 3
    assert model.get_params()["dropout_rate"] == 0.2
    model.set_params(max_iter=4)
    assert model.max_iter == 4


def test_parallel_booster_is_public_api():
    assert bd.ParallelBooster is ParallelBooster
    assert (
        bd_sklearn.ParallelBooster
        is ParallelBooster
    )

    model = bd.ParallelBooster(
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
    assert bd.ExplainableBooster is ExplainableBooster
    assert NamespacedExplainableBooster is ExplainableBooster
    assert bd_interpretml.ExplainableBooster is ExplainableBooster
    assert bd_interpretml.__all__ == ["ExplainableBooster"]

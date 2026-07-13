# Boulevard Documentation

Boulevard provides sklearn-compatible research estimators for Boulevard /
BRAT-style boosting with uncertainty intervals.

The current public API is intentionally small:

```python
import boulevard as bd

bd.DropoutBooster
bd.ParallelBooster
bd.ExplainableBooster
```

The package name is `boulevard-boosting`; the import name is `boulevard`.

## Estimators

- `DropoutBooster`: histogram-tree BRAT-D / Boulevard regularized dropout
  boosting.
- `ParallelBooster`: histogram-tree BRAT-P round/slot training.
- `ExplainableBooster`: main-effect additive EBM-family estimator.

The first release does not expose XGBoost, LightGBM, or CatBoost backends.

## Where To Start

- [Quickstart](quickstart.md): short examples for all three estimators.
- [API Reference](api.md): parameters, interval methods, and limitations.
- [Hyperparameter Recommendations](hyperparameters.md): empirical starting
  points for accuracy and interval coverage.
- [Development Notes](development-notes.md): current engineering status and
  release decisions.

## Inference Workflow

For intervals, fit the model, prepare inference, then call `predict_intervals`:

```python
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
```

`prepare_inference` estimates residual variance and caches the linear algebra
used by `weight_norms` and interval methods. If it is not called explicitly, the
estimators use the training data on first interval use.

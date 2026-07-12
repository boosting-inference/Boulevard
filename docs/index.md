# Boulevard Documentation

Boulevard is a research-oriented Python package for Boulevard / BRAT-style
boosting estimators with uncertainty intervals.

The current publishable target is the scikit-learn-compatible API:

```python
import boulevard as bd

bd.BRATDHistGradientBoostingRegressor
bd.BRATPHistGradientBoostingRegressor
bd.IEBMRegressor
```

The package name is `boulevard-boosting`; the import name is `boulevard`.

```bash
pip install boulevard-boosting
```

## Current Stable Surface

- `BRATDHistGradientBoostingRegressor`: histogram-tree BRAT-D / Boulevard
  regularized dropout boosting.
- `BRATPHistGradientBoostingRegressor`: histogram-tree BRAT-P parallel
  round/slot Boulevard training.
- `IEBMRegressor`: experimental Inferable EBM-style additive regressor.

XGBoost, LightGBM, and CatBoost namespaces are kept as development placeholders.
They are not part of the stable top-level public API yet.

## Where To Start

- [Quickstart](quickstart.md): short runnable examples for BRAT-D, BRAT-P, and
  IEBM.
- [API Reference](api.md): estimator parameters, interval methods, and current
  limitations.
- [Development Notes](development-notes.md): engineering history and known
  research/implementation caveats.

## Inference Workflow

For BRAT-D and BRAT-P, interval calls need two ingredients:

1. a fitted Boulevard-trained ensemble;
2. inference preparation, which estimates residual variance and caches the
   observed histogram-cell linear algebra used by confidence and prediction
   intervals.

The recommended workflow is:

```python
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

lower, upper, pred = model.predict_intervals(
    X_test,
    mode="confidence",
)
```

If `prepare_inference` is not called explicitly, BRAT-D and BRAT-P interval
methods prepare the cache on first use with the training data. A held-out
calibration set is usually preferable when you want the residual variance
estimate separated from model fitting.

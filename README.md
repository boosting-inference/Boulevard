# Boulevard

Boulevard is a research-oriented Python package for Boulevard / BRAT-style
boosting estimators with uncertainty intervals.

The first publishable target is the scikit-learn-compatible estimator family:

- `boulevard.BRATDHistGradientBoostingRegressor`
- `boulevard.BRATPHistGradientBoostingRegressor`
- `boulevard.IEBMRegressor`

XGBoost, LightGBM, and CatBoost integrations are development placeholders and
are not part of the stable public API yet.

## Install

The package name is `boulevard-boosting`; the import name is `boulevard`.

```bash
pip install boulevard-boosting
```

```python
import boulevard as bd
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import boulevard as bd

model = bd.BRATDHistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.45,
    dropout_rate=0.3,
    subsample_rate=0.8,
    max_depth=10,
    max_leaf_nodes=512,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    random_state=0,
)

model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
```

See `examples/brat_histogram_api_demo.py` for a complete BRAT-D / BRAT-P
diagnostic demo with RMSE, coverage, interval width, and timing summaries.

## Status

This package is currently a research preview. The scikit-learn histogram
estimators reuse scikit-learn private histogram-gradient-boosting internals, so
the dependency range is intentionally pinned to the tested scikit-learn private
API series.

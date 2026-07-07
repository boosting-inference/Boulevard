# Quickstart

This page shows the public API that a user should write today:

```python
import boulevard as bd
```

The current stable top-level estimators are:

```python
bd.BRATDHistGradientBoostingRegressor
bd.BRATPHistGradientBoostingRegressor
bd.IEBMRegressor
```

## BRAT-D Histogram Regressor

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

# Recommended for repeated interval calls. If omitted, interval methods use
# training data for inference preparation on first use.
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
```

`alpha=0.05` gives a 95% interval.

## BRAT-P Histogram Regressor

BRAT-P uses completed-history rounds and deterministic slots instead of random
dropout.

```python
import boulevard as bd

model = bd.BRATPHistGradientBoostingRegressor(
    n_rounds=100,
    trees_per_round=10,
    subsample_rate=0.8,
    max_depth=10,
    max_leaf_nodes=256,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    n_jobs=1,
    random_state=0,
)

model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
```

Use `n_jobs > 1` to parallelize slot fitting within eligible BRAT-P rounds.
Small `trees_per_round` values may not benefit because parallel scheduling has
fixed overhead.

## IEBM Regressor

`IEBMRegressor` is an experimental Inferable EBM-style additive model. It
currently supports numeric squared-error main effects.

```python
import boulevard as bd

model = bd.IEBMRegressor(
    max_rounds=100,
    max_bins=64,
    learning_rate=1.0,
    subsample_rate=0.8,
    max_depth=1,
    min_samples_leaf=10,
    random_state=0,
)

model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
```

For one feature's partial function:

```python
term_lower, term_upper, term_pred = model.predict_feature_intervals(
    0,
    X_test[:, 0],
    level=0.95,
    mode="confidence",
)
```

## Runnable Examples

From the repository root:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
python examples/iebm_quickstart.py
python examples/iebm_visual_check.py --output /tmp/iebm_visual_check.png
```

The BRAT histogram demo prints RMSE, coverage, interval width quantiles, and
wall-clock timings for BRAT-D, BRAT-P, and vanilla
`HistGradientBoostingRegressor`.

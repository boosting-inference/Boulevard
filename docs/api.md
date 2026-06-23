# API Reference

## Estimators

### `BRATDHistGradientBoostingRegressor`

Experimental sklearn histogram-tree backend for BRAT-D / Boulevard-style
regularized dropout boosting.

```python
import boulevard as bd

model = bd.BRATDHistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.45,
    dropout_rate=0.3,
    subsample_rate=0.8,
    max_depth=10,
    max_leaf_nodes=1024,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    random_state=0,
    nystrom_subsample_rate=1.0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

This estimator is the current package candidate for a faithful sklearn-based
BRAT-D training backend. It uses sklearn histogram tree internals for binning and
single-tree fitting, while replacing the boosting loop with BRAT-D residual
construction.

Supported scope:

- squared-error regression;
- continuous features;
- row subsampling through `subsample_rate`;
- asymptotic BRAT-D confidence, prediction, and reproduction intervals;
- observed histogram-cell inference with cached cell weight norms;
- optional Nyström sketching for interval weight norms through
  `nystrom_subsample_rate`.

Current limitations:

- no early stopping;
- no warm start;
- no categorical features;
- no monotone constraints;
- no interaction constraints;
- depends on sklearn private histogram-gradient-boosting internals.

Interval methods prepare their inference cache automatically on first use. By
default, the residual variance is estimated from the training data:

```python
lower, upper = model.confidence_interval(X_test)
```

To estimate residual variance from a held-out calibration set instead, pass the
calibration data directly to an interval method:

```python
lower, upper = model.confidence_interval(
    X_test,
    X_calib=X_calib,
    y_calib=y_calib,
)
```

For repeated interval calls with the same calibration set, users can still call
`prepare_inference` explicitly:

```python
model.prepare_inference(X_calib, y_calib)
lower, upper = model.confidence_interval(X_test)
```

For a combined BRAT-D / BRAT-P plotted API example with a vanilla histogram
gradient boosting baseline:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

To benchmark optional BRAT-P within-round parallel fitting, pass for example
`--bratp-n-jobs 2`.

The estimator defaults to `nystrom_subsample_rate=1.0`. In the current
histogram-cell implementation, this selects all observed cells as landmarks and
falls back to the stable exact observed-cell solve. Smaller values can be used
for sketched interval inference experiments.

The estimator also exposes `fit_diagnostics_` after fitting. This is intended for
research diagnostics and currently reports timing for binning, BRAT-D residual
construction, tree fitting, training-prediction caching, and interval-related
cache behavior.

### `BRATPHistGradientBoostingRegressor`

Experimental sklearn histogram-tree backend for BRAT-P parallelized Boulevard
training.

```python
import boulevard as bd

model = bd.BRATPHistGradientBoostingRegressor(
    n_rounds=100,
    trees_per_round=5,
    subsample_rate=0.8,
    n_jobs=2,
    max_depth=10,
    max_leaf_nodes=1024,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    random_state=0,
    nystrom_subsample_rate=1.0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

This estimator uses deterministic slot dropping in the residual construction
and exposes the same observed histogram-cell interval interface as the BRAT-D
histogram estimator. BRAT-P can fit slot trees within each completed-history
round in parallel with `n_jobs`; the default first round remains serial unless
`drop_first_round=True`, because the default first round is a sequential warm
start. The BRAT-P interval weights use the parallel KRR system `K^{-1} I + ((K -
1) / K) K_n`, where `K = trees_per_round`.

Like the BRAT-D histogram estimator, BRAT-P supports optional Nyström sketching
for interval weight norms through `nystrom_subsample_rate`. The default `1.0`
selects all observed cells and falls back to the exact observed-cell solve.

For a plotted diagnostic of the fitted signal, confidence and prediction bands,
weight norms, interval widths, calibration residuals, RMSE, coverage, interval
width quantiles, and wall-clock timings, use the combined BRAT-D / BRAT-P API
example:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

### `XGBRegressor`

Preliminary XGBoost wrapper scaffold. This should not yet be interpreted as a
faithful Boulevard-trained XGBoost backend.

# API Reference

## Estimators

### `BRATDHistGradientBoostingRegressor`

Sklearn histogram-tree backend for BRAT-D / Boulevard regularized dropout
boosting. This is the current package candidate for the faithful sklearn-based
BRAT-D backend: it reuses sklearn's histogram binning and single-tree grower,
but replaces the boosting loop with BRAT-D residual construction and applies
BRAT-D signal correction in `predict`.

```python
import boulevard as bd

model = bd.BRATDHistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.45,
    dropout_rate=0.3,
    subsample_rate=0.8,
    max_depth=10,
    max_leaf_nodes=512,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    nystrom_subsample_rate=1.0,
    random_state=0,
)
model.fit(X_train, y_train)

# Optional but recommended for repeated interval calls. If omitted, interval
# methods prepare the inference cache on first use with the training data.
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
ri_lower, ri_upper = model.reproduction_interval(X_test, alpha=0.05)
norms = model.weight_norms(X_test)
```

Important parameters:

- `max_iter`: number of BRAT-D trees;
- `learning_rate`: the Boulevard/BRAT-D `lambda` multiplier;
- `dropout_rate`: probability of dropping an old tree when constructing the
  next residual;
- `subsample_rate`: row subsampling before fitting each new tree;
- `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_bins`: sklearn
  histogram-tree complexity controls;
- `nystrom_subsample_rate`: fraction of observed histogram cells used as
  Nyström landmarks for interval weight norms. The default `1.0` uses all
  observed cells and gives the exact observed-cell solve.

Interval methods use `alpha`, so `alpha=0.05` gives a 95% interval. The three
main interval calls are:

```python
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
ri_lower, ri_upper = model.reproduction_interval(X_test, alpha=0.05)
```

Use a held-out calibration set when you want residual variance estimated away
from the training set:

```python
model.prepare_inference(X_calib, y_calib)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
```

For one-off calls, the calibration data can also be passed directly:

```python
ci_lower, ci_upper = model.confidence_interval(
    X_test,
    alpha=0.05,
    X_calib=X_calib,
    y_calib=y_calib,
)
```

Useful diagnostics:

```python
bins = model.apply_bin_indices(X_test)
cells = model.apply_cell_indices(X_test)
norms = model.weight_norms(X_test)
fit_diagnostics = model.fit_diagnostics_
```

Supported scope:

- squared-error regression;
- numeric features;
- row subsampling through `subsample_rate`;
- asymptotic BRAT-D confidence, prediction, and reproduction intervals;
- cached observed histogram-cell inference;
- optional Nyström sketching through `nystrom_subsample_rate`.

Current limitations:

- no early stopping;
- no warm start;
- no categorical features;
- no monotone constraints;
- no interaction constraints;
- depends on sklearn private histogram-gradient-boosting internals.

For a combined BRAT-D / BRAT-P plotted API example with a vanilla
`HistGradientBoostingRegressor` baseline:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

### `BRATPHistGradientBoostingRegressor`

Sklearn histogram-tree backend for BRAT-P parallelized Boulevard training.
BRAT-P trains trees in deterministic slots. Each round has `trees_per_round`
slots, and slot `k` is trained against the completed-history residual that drops
slot `k`. This is the parallelized paper variant, not random dropout.

```python
import boulevard as bd

total_tree_budget = 1000

model = bd.BRATPHistGradientBoostingRegressor(
    n_rounds=100,
    trees_per_round=total_tree_budget // 100,
    subsample_rate=0.8,
    max_depth=10,
    max_leaf_nodes=256,
    min_samples_leaf=5,
    max_bins=255,
    early_stopping=False,
    n_jobs=1,
    nystrom_subsample_rate=1.0,
    random_state=0,
)
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
ri_lower, ri_upper = model.reproduction_interval(X_test, alpha=0.05)
norms = model.weight_norms(X_test)
```

Important parameters:

- `n_rounds`: number of completed-history BRAT-P rounds;
- `trees_per_round`: number of deterministic slots per round. The total tree
  budget is `n_rounds * trees_per_round`;
- `subsample_rate`: row subsampling before fitting each slot tree;
- `n_jobs`: optional joblib thread count for fitting slot trees within a round.
  Use `1` or `None` for serial fitting;
- `drop_first_round`: if `False`, the first round uses a sequential warm start.
  If `True`, the first round can be fit with the same completed-history logic as
  later rounds;
- `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_bins`: sklearn
  histogram-tree complexity controls;
- `nystrom_subsample_rate`: same observed-cell Nyström control as BRAT-D.

BRAT-P exposes the same prediction, interval, calibration, bin/cell, and
diagnostic methods as BRAT-D:

```python
model.prepare_inference(X_calib, y_calib)
pred = model.predict(X_test)
ci_lower, ci_upper = model.confidence_interval(X_test, alpha=0.05)
pi_lower, pi_upper = model.prediction_interval(X_test, alpha=0.05)
ri_lower, ri_upper = model.reproduction_interval(X_test, alpha=0.05)
norms = model.weight_norms(X_test)
cells = model.apply_cell_indices(X_test)
fit_diagnostics = model.fit_diagnostics_
```

The BRAT-P interval weights use the parallel KRR system
`K^{-1} I + ((K - 1) / K) K_n`, where `K = trees_per_round`.

For a plotted diagnostic of the fitted signal, confidence and prediction bands,
weight norms, interval widths, calibration residuals, RMSE, coverage, interval
width quantiles, and wall-clock timings, use the combined BRAT-D / BRAT-P API
example:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

To benchmark optional BRAT-P within-round parallel fitting, pass for example:

```bash
python examples/brat_histogram_api_demo.py \
    --output /tmp/brat_histogram_api_demo.png \
    --bratp-n-jobs 2
```

### `IEBMRegressor`

Experimental sklearn-style Inferable EBM training backend.

```python
import boulevard as bd

model = bd.IEBMRegressor(
    max_rounds=100,
    max_bins=64,
    learning_rate=1.0,
    subsample_rate=0.8,
    warmup_rounds=20,
    truncation=10.0,
    max_depth=1,
    min_samples_leaf=10,
    leave_one_out=False,
    random_state=0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
bins = model.apply_bins(X_test)
lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
term_lower, term_upper, term_pred = model.predict_feature_intervals(
    0,
    X_test[:, 0],
    level=0.95,
    mode="confidence",
)
```

This class intentionally rebuilds the IEBM path inside boulevard instead of
patching InterpretML private modules. The first version is deliberately narrow:

- squared-error regression;
- numeric features only;
- main effects only;
- one-dimensional binned tree updates;
- additive warmup rounds followed by Boulevard-style averaging;
- one-dimensional tree complexity controlled by `max_depth`, or by the lower
  level `max_leaves` parameter;
- frozen full-model residuals by default, with `leave_one_out=True` available
  as an experimental residual variant;
- EBM-style experimental interval APIs, `predict_intervals` and
  `predict_feature_intervals`;
- sklearn-compatible `fit`, `predict`, `get_params`, and cloning behavior.

Current limitations:

- interval formulas are still experimental and should be treated as diagnostics
  until coverage behavior is audited;
- no interactions;
- no categorical feature handling;
- no `sample_weight`;
- no direct dependency on InterpretML internals yet.

For a minimal fit/predict example:

```bash
python examples/iebm_quickstart.py
```

For a multivariate visual diagnostic with partial dependence plots, confidence
bands, coverage annotations, residual variance, and bin-space weight norms:

```bash
python examples/iebm_visual_check.py --output /tmp/iebm_visual_check.png
```

The plotted bands use the experimental `predict_feature_intervals` API.

### `XGBRegressor`

Preliminary XGBoost wrapper scaffold. This should not yet be interpreted as a
faithful Boulevard-trained XGBoost backend.

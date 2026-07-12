# API Reference

The user-facing import style is:

```python
import boulevard as bd
```

The current stable top-level names are:

```python
bd.BRATDHistGradientBoostingRegressor
bd.BRATPHistGradientBoostingRegressor
bd.IEBMRegressor
```

XGBoost, LightGBM, and CatBoost namespaces exist only as development
placeholders. They are not stable top-level APIs yet.

## Common Interval Preparation

BRAT-D and BRAT-P interval methods need cached inference quantities. The
recommended explicit workflow is:

```python
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)
lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
```

`prepare_inference` does two things:

- estimates `sigma_hat2_` as the centered residual variance on the supplied
  calibration data, or on the training data if no calibration data is supplied;
- builds and caches the observed histogram-cell kernel system used by
  `weight_norms`, confidence intervals, prediction intervals, and reproduction
  intervals.

For one-off BRAT-D or BRAT-P interval calls, calibration data can also be
passed directly:

```python
lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
    X_calib=X_calib,
    y_calib=y_calib,
)
```

Explicit `prepare_inference` is clearer and avoids rebuilding the cache across
repeated interval calls.

## `BRATDHistGradientBoostingRegressor`

Histogram-tree BRAT-D / Boulevard regularized dropout boosting. The estimator
inherits scikit-learn's public `HistGradientBoostingRegressor` shape for
estimator compatibility, but replaces the boosting loop with BRAT-D residual
construction.

```python
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
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
pi_lower, pi_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="prediction",
)
ri_lower, ri_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="reproduction",
)
norms = model.weight_norms(X_test)
```

Important parameters:

- `max_iter`: number of BRAT-D trees.
- `learning_rate`: the Boulevard/BRAT-D `lambda` multiplier.
- `dropout_rate`: probability of dropping an old tree when constructing the
  next residual.
- `subsample_rate`: row subsampling before fitting each new tree.
- `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_bins`: histogram-tree
  complexity controls.
- `nystrom_subsample_rate`: fraction of observed histogram cells used as
  Nyström landmarks for interval weight norms. The default `1.0` uses all
  observed cells.

Useful diagnostics:

```python
bins = model.apply_bin_indices(X_test)
cells = model.apply_cell_indices(X_test)
norms = model.weight_norms(X_test)
fit_diagnostics = model.fit_diagnostics_
```

Current scope:

- squared-error regression;
- numeric features;
- row subsampling through `subsample_rate`;
- asymptotic confidence, prediction, and reproduction intervals;
- cached observed histogram-cell inference;
- optional Nyström sketching through `nystrom_subsample_rate`.

Current limitations:

- no early stopping;
- no warm start;
- no categorical features;
- no monotone constraints;
- no interaction constraints;
- depends on scikit-learn private histogram-gradient-boosting internals.

## `BRATPHistGradientBoostingRegressor`

Histogram-tree BRAT-P parallelized Boulevard training. BRAT-P trains trees in
deterministic slots. Each round has `trees_per_round` slots, and slot `k` is
trained against the completed-history residual that drops slot `k`.

```python
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
    nystrom_subsample_rate=1.0,
    random_state=0,
)
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
ci_lower, ci_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
pi_lower, pi_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="prediction",
)
ri_lower, ri_upper, _ = model.predict_intervals(
    X_test,
    level=0.95,
    mode="reproduction",
)
norms = model.weight_norms(X_test)
```

Important parameters:

- `n_rounds`: number of completed-history BRAT-P rounds.
- `trees_per_round`: number of deterministic slots per round. Total tree count
  is `n_rounds * trees_per_round`.
- `subsample_rate`: row subsampling before fitting each slot tree.
- `n_jobs`: optional joblib worker count for fitting slot trees within eligible
  rounds. Use `1` or `None` for serial fitting.
- `drop_first_round`: if `False`, the first round uses a sequential warm start.
  If `True`, the first round uses the same completed-history shape as later
  rounds.
- `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_bins`: histogram-tree
  complexity controls.
- `nystrom_subsample_rate`: same observed-cell Nyström control as BRAT-D.

BRAT-P exposes the same prediction, interval, calibration, bin/cell, and
diagnostic methods as BRAT-D:

```python
cells = model.apply_cell_indices(X_test)
norms = model.weight_norms(X_test)
fit_diagnostics = model.fit_diagnostics_
```

The BRAT-P interval weights use the parallel KRR system
`K^{-1} I + ((K - 1) / K) K_n`, where `K = trees_per_round`.

## `IEBMRegressor`

Experimental Inferable EBM-style additive regressor. This class rebuilds an
IEBM path inside Boulevard instead of patching InterpretML private modules.

```python
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
model.prepare_inference(X_calib, y_calib)
pi_scale = model.calibrate_intervals(
    X_calib,
    y_calib,
    level=0.95,
    mode="prediction",
)

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

Current scope:

- squared-error regression;
- numeric features only;
- main effects only;
- one-dimensional binned tree updates;
- additive warmup rounds followed by Boulevard-style averaging;
- tree complexity controlled by `max_depth`, or by the lower-level
  `max_leaves` parameter;
- EBM-style experimental interval APIs, `predict_intervals` and
  `predict_feature_intervals`;
- optional held-out interval calibration through `calibrate_intervals`;
- scikit-learn-compatible `fit`, `predict`, `get_params`, and cloning behavior.

Current limitations:

- interval formulas are experimental diagnostics until coverage behavior is
  audited more broadly;
- no interactions;
- no categorical feature handling;
- no `sample_weight`;
- no direct dependency on InterpretML internals yet.

## Examples

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
python examples/iebm_quickstart.py
python examples/iebm_visual_check.py --output /tmp/iebm_visual_check.png
```

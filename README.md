# Boulevard

Boulevard is a research-oriented Python package for sklearn-compatible
Boulevard / BRAT-style boosting estimators with uncertainty intervals.

The first release focuses on numeric squared-error regression with three public
estimators:

```python
import boulevard as bd

bd.DropoutBooster
bd.ParallelBooster
bd.ExplainableBooster
```

`DropoutBooster` and `ParallelBooster` are histogram-tree estimators built on
scikit-learn's histogram gradient boosting internals. `ExplainableBooster` is a
main-effect additive estimator in the EBM family, implemented directly in this
package.

There are no backwards-compatible aliases for older pre-release class names.

## Install

The package name is `boulevard-boosting`; the import name is `boulevard`.

```bash
pip install boulevard-boosting
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import boulevard as bd

model = bd.DropoutBooster(
    max_iter=700,
    learning_rate=0.8,
    dropout_rate=0.1,
    subsample_rate=0.8,
    max_depth=6,
    max_leaf_nodes=64,
    min_samples_leaf=2,
    max_bins=64,
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

`prepare_inference` estimates residual variance and builds the cached linear
algebra used by interval methods. If it is omitted, interval methods prepare
inference on first use with the training data. A held-out calibration set is
usually preferable for variance estimation.

Runnable examples:

```bash
python examples/quickstart.py
```

Open the broader demo notebook:

```text
examples/boulevard_boosters_demo.ipynb
```

## Recommended Use

Use `DropoutBooster` or `ParallelBooster` for low-dimensional smooth regression
signals. When dimension grows and an additive/main-effect model is scientifically
reasonable, start with `ExplainableBooster` and inspect its feature-level
intervals.

For interaction-heavy higher-dimensional functions, treat signal confidence
intervals as diagnostic unless you validate coverage in a problem-specific
simulation. Prediction intervals are currently the safer interval type for noisy
outcome uncertainty.

## Common Interval API

All three estimators support:

```python
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
```

Modes:

- `"confidence"`: uncertainty for the fitted signal.
- `"prediction"`: signal uncertainty plus residual noise variance.
- `"reproduction"`: repeated-training signal uncertainty.

`prepare_inference` estimates `sigma_hat2_` as centered residual variance on the
supplied calibration data. If no calibration data is supplied, it uses training
data.

## DropoutBooster

`DropoutBooster` is the sklearn-compatible BRAT-D / Boulevard dropout estimator.

```python
model = bd.DropoutBooster(
    max_iter=700,
    learning_rate=0.8,
    dropout_rate=0.1,
    subsample_rate=0.8,
    max_depth=6,
    max_leaf_nodes=64,
    min_samples_leaf=2,
    max_bins=64,
    random_state=0,
)
```

Important parameters:

- `max_iter`: number of trees.
- `learning_rate`: Boulevard/BRAT-D signal multiplier.
- `dropout_rate`: probability of dropping an old tree when constructing the
  next residual. Must be in `[0, 1)`.
- `subsample_rate`: row subsampling fraction for each new tree.
- `max_depth`, `max_leaf_nodes`: tree complexity controls.
- `min_samples_leaf`: minimum leaf size. Larger values smooth the fit.
- `max_bins`: number of histogram bins per feature. scikit-learn currently
  supports at most `255`.

Useful methods:

```python
pred = model.predict(X_test)
norms = model.weight_norms(X_test)
cells = model.apply_cell_indices(X_test)
ci_lower, ci_upper, _ = model.predict_intervals(X_test, mode="confidence")
pi_lower, pi_upper, _ = model.predict_intervals(X_test, mode="prediction")
ri_lower, ri_upper, _ = model.predict_intervals(X_test, mode="reproduction")
```

## ParallelBooster

`ParallelBooster` is the sklearn-compatible BRAT-P estimator. It uses
deterministic tree slots instead of random dropout.

```python
model = bd.ParallelBooster(
    n_rounds=70,
    trees_per_round=6,
    subsample_rate=0.8,
    max_depth=8,
    max_leaf_nodes=64,
    min_samples_leaf=4,
    max_bins=64,
    drop_first_round=True,
    n_jobs=1,
    random_state=0,
)
```

Important parameters:

- `n_rounds`: number of completed-history BRAT-P rounds.
- `trees_per_round`: number of deterministic tree slots per round. Total tree
  count is `n_rounds * trees_per_round`.
- `subsample_rate`: row subsampling fraction for each slot tree.
- `n_jobs`: optional joblib worker count for slot fitting. Use `1` or `None`
  for serial fitting.
- `drop_first_round`: if `False`, the first round is sequentially warm-started.
  If `True`, the first round uses the same completed-history shape as later
  rounds.
- `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_bins`: histogram-tree
  complexity controls.

The interval API matches `DropoutBooster`.

## ExplainableBooster

`ExplainableBooster` is an EBM-family additive estimator with feature-level
intervals. It is implemented in Boulevard and does not depend on InterpretML.

```python
model = bd.ExplainableBooster(
    max_rounds=160,
    max_bins=32,
    learning_rate=0.6,
    subsample_rate=1.0,
    warmup_rounds=10,
    max_depth=4,
    min_samples_leaf=8,
    leave_one_out=False,
    random_state=0,
)
```

Important parameters:

- `max_rounds`: number of additive update rounds.
- `max_bins`: number of one-dimensional bins per feature.
- `learning_rate`: additive update multiplier.
- `subsample_rate`: row subsampling for each feature update.
- `warmup_rounds`: number of early rounds before Boulevard-style averaging.
- `max_depth` or `max_leaves`: one-dimensional tree complexity. Specify only
  one.
- `min_samples_leaf`: minimum leaf size.
- `leave_one_out`: whether each feature update drops that feature's current
  contribution from the residual.

Feature-level interval API:

```python
term_lower, term_upper, term_pred = model.predict_feature_intervals(
    feature_idx=0,
    values=X_test[:, 0],
    level=0.95,
    mode="confidence",
)
```

## Hyperparameter Recommendations

These recommendations are empirical starting points, not formal guarantees.
They are based on synthetic diagnostics with known regression signals, Gaussian
noise with standard deviation `0.2`, and separate train / calibration / test
splits.

Coverage labels:

| Label | Meaning |
| --- | --- |
| `PASS` | signal CI coverage >= 0.90 |
| `OK` | signal CI coverage >= 0.80 |
| `LOW` | signal CI coverage < 0.80 |

Best trial per estimator and scenario:

| Scenario | Estimator | Status | RMSE vs Signal | Signal CI Coverage | PI Coverage | Median CI Width |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `smooth_1d` | `DropoutBooster` | PASS | 0.049 | 0.960 | 0.965 | 0.210 |
| `smooth_1d` | `ParallelBooster` | PASS | 0.053 | 0.980 | 0.970 | 0.250 |
| `smooth_1d` | `ExplainableBooster` | OK | 0.061 | 0.855 | 1.000 | 0.181 |
| `additive_3d` | `DropoutBooster` | OK | 0.146 | 0.824 | 0.976 | 0.400 |
| `additive_3d` | `ParallelBooster` | OK | 0.129 | 0.858 | 0.976 | 0.408 |
| `additive_3d` | `ExplainableBooster` | LOW | 0.071 | 0.782 | 1.000 | 0.176 |
| `interaction_5d` | `DropoutBooster` | LOW | 0.134 | 0.523 | 0.961 | 0.198 |
| `interaction_5d` | `ParallelBooster` | LOW | 0.149 | 0.606 | 0.958 | 0.253 |
| `interaction_5d` | `ExplainableBooster` | LOW | 0.077 | 0.603 | 1.000 | 0.139 |

The main takeaways are:

- 1D smooth problems are easy for all three estimators.
- 3D additive problems need more capacity than the original prototype defaults.
- 5D interaction problems remain hard for asymptotic signal CIs. Predictive RMSE
  can be reasonable while signal CI coverage is poor.
- Prediction intervals were close to or above 95% coverage in these diagnostics.
  This does not imply signal CIs are automatically calibrated.

### Dimension And Capacity

As dimension increases, use more samples and more model capacity.

| Effective Dimension | Suggested Samples | Tree/Partition Capacity |
| --- | ---: | --- |
| 1 | 500-1000 | shallow to moderate trees |
| 2-3 | 2000-4000 | deeper trees and more leaves |
| 4-8 | 5000+ | high capacity, then validate coverage |

Tune in this order:

1. Increase tree count or rounds.
2. Increase `max_depth` and `max_leaf_nodes`.
3. Decrease `min_samples_leaf` if the fit is over-smoothed.
4. Increase `max_bins` if the fitted function is too coarse.

If RMSE is high and signal CI coverage is low, tune the predictive model first.
The interval formula does not automatically correct approximation bias.

The estimators warn when tree-capacity settings are internally capped. For
example, increasing `max_leaf_nodes` will not change a tree if `max_depth`,
`min_samples_leaf`, subsampling, or one-dimensional `max_bins` already imposes a
smaller effective leaf budget.

### DropoutBooster Tuning

Low-dimensional starting point:

```python
model = bd.DropoutBooster(
    max_iter=700,
    learning_rate=0.8,
    dropout_rate=0.1,
    subsample_rate=0.8,
    max_depth=6,
    max_leaf_nodes=64,
    min_samples_leaf=2,
    max_bins=64,
    random_state=0,
)
```

Higher-dimensional additive starting point:

```python
model = bd.DropoutBooster(
    max_iter=1000,
    learning_rate=3.0,
    dropout_rate=0.0,
    subsample_rate=1.0,
    max_depth=10,
    max_leaf_nodes=256,
    min_samples_leaf=2,
    max_bins=128,
    random_state=0,
)
```

Recommended tuning:

- `max_iter`: use `200-700` for simple low-dimensional signals; use `700-1000`
  when dimension or curvature increases.
- `learning_rate`: start around `0.8-1.5` for 1D; larger values such as `3.0`
  were selected for the 3D additive search.
- `dropout_rate`: start around `0.0-0.1` for accuracy. Higher dropout is
  stronger regularization and can worsen signal bias.
- `subsample_rate`: start around `0.8`; use `1.0` when the model underfits.
- `max_depth` and `max_leaf_nodes`: increase together as dimension grows.
- `min_samples_leaf`: use small values such as `2-4` when bias dominates.
- `max_bins`: use `64` for smooth 1D signals; try `128-255` for higher
  dimension or sharper structure.

### ParallelBooster Tuning

Low-dimensional starting point:

```python
model = bd.ParallelBooster(
    n_rounds=70,
    trees_per_round=6,
    subsample_rate=0.8,
    max_depth=8,
    max_leaf_nodes=64,
    min_samples_leaf=4,
    max_bins=64,
    drop_first_round=True,
    n_jobs=1,
    random_state=0,
)
```

Higher-dimensional additive starting point:

```python
model = bd.ParallelBooster(
    n_rounds=70,
    trees_per_round=6,
    subsample_rate=1.0,
    max_depth=10,
    max_leaf_nodes=256,
    min_samples_leaf=4,
    max_bins=64,
    drop_first_round=True,
    n_jobs=1,
    random_state=0,
)
```

Recommended tuning:

- Total tree budget is `n_rounds * trees_per_round`.
- Increase `n_rounds` when the fit underfits.
- Increase `trees_per_round` when signal intervals are too narrow, but expect
  some tradeoff with fit time and possibly RMSE.
- `drop_first_round=True` keeps the first BRAT-P round closer to the frozen
  parallel training interpretation.
- Tune tree complexity the same way as `DropoutBooster`.
- Use `n_jobs=1` by default. Parallel scheduling only helps when
  `trees_per_round` is large enough.

### ExplainableBooster Tuning

Additive 3D starting point:

```python
model = bd.ExplainableBooster(
    max_rounds=160,
    max_bins=32,
    learning_rate=0.6,
    subsample_rate=0.8,
    warmup_rounds=10,
    max_depth=4,
    min_samples_leaf=8,
    leave_one_out=False,
    random_state=0,
)
```

Recommended tuning:

- `max_rounds`: start around `160`; increase toward `320` for harder additive
  functions.
- `max_depth`: start around `2-4`. Stumps are often too biased for nonlinear
  partial functions.
- `max_bins`: start with `32-64`; increase only when partial effects look too
  coarse and sample size is sufficient.
- `learning_rate`: start around `0.6-1.0`.
- `warmup_rounds`: `0-20` is a reasonable starting range.
- `leave_one_out=False` was selected in the best runs above.

`ExplainableBooster` can have strong RMSE while its signal CI coverage remains
below target. Treat current intervals as diagnostic and validate them on held-out
or simulated truth before relying on them.

## Practical Workflow

For a new dataset:

1. Split into train, calibration, and test or validation sets.
2. Fit a moderate starting model from the recommendations above.
3. Run `prepare_inference(X_calib, y_calib)`.
4. Check RMSE, residual variance, interval width quantiles, PI coverage for
   noisy outcomes, and CI coverage for known simulation truth if available.
5. If signal CI coverage is low and RMSE is high, increase model capacity.
6. If signal CI coverage is low but RMSE is already good, treat the signal CI as
   under-calibrated for that problem class.
7. Keep a held-out diagnostic script or notebook for every real analysis.

To rerun the recommendation search:

```bash
.venv/bin/python tests/diagnostics/booster_hyperparameter_search.py \
  --trials 30 \
  --random-search
```

## Development Notes

The current package target is a small sklearn-compatible release. This release
intentionally removes older pre-release aliases and backend placeholders.
XGBoost, LightGBM, and CatBoost work can return later in separate
backend-specific forks or integrations, but they are not part of this package
surface now.

Release cleanup decisions:

- The dropout BRAT estimator was renamed to `DropoutBooster`.
- The parallel BRAT estimator was renamed to `ParallelBooster`.
- The EBM-family estimator was renamed to `ExplainableBooster`.
- The earlier inference sketching prototype was removed from BRAT estimators.
- Truncation was removed from `ExplainableBooster`.
- Heavy diagnostic scripts were replaced by `examples/quickstart.py` and
  `examples/boulevard_boosters_demo.ipynb`.
- The release docs avoid advertising unsupported backend families.

Implementation notes:

- `DropoutBooster` and `ParallelBooster` use scikit-learn histogram-tree private
  internals for prebinning and tree fitting. This is why the scikit-learn
  dependency is pinned to the tested private API series.
- Both histogram estimators cache training tree predictions during fitting to
  avoid repeated old-tree traversal when constructing residuals.
- For inference, histogram estimators compress repeated binned rows into
  observed multidimensional cells. Interval weight norms are solved once for the
  observed cells, then query points reuse cached cell norms when they land in an
  observed cell.
- `ExplainableBooster` uses a pure Python/NumPy one-dimensional tree update.
  InterpretML's native EBM implementation remains faster because its update loop
  is compiled, but the Boulevard implementation is easier to audit and does not
  patch private InterpretML modules.

## Current Scope And Limitations

- Numeric squared-error regression only.
- sklearn-compatible estimator APIs.
- Experimental asymptotic intervals.
- No categorical feature handling.
- No XGBoost, LightGBM, or CatBoost backend in the first release.
- `DropoutBooster` and `ParallelBooster` depend on scikit-learn private
  histogram-gradient-boosting internals.
- The scikit-learn dependency range is intentionally pinned to the tested
  private API series.

## Release Checks

Before cutting a release:

```bash
.venv/bin/python -m ruff check src tests examples README.md
.venv/bin/python -m pytest -q
.venv/bin/python examples/quickstart.py
.venv/bin/jupyter nbconvert --to notebook --execute \
  examples/boulevard_boosters_demo.ipynb \
  --output /private/tmp/boulevard_boosters_demo_executed.ipynb \
  --ExecutePreprocessor.timeout=120
.venv/bin/python -m build --sdist --wheel
```

The wheel contents should contain only the release modules:

```text
boulevard/__init__.py
boulevard/estimators/interpretml/explainable.py
boulevard/estimators/sklearn/dropout.py
boulevard/estimators/sklearn/parallel.py
boulevard/intervals/asymptotic.py
```

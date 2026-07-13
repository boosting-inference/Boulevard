# API Reference

The release API is:

```python
import boulevard as bd

bd.DropoutBooster
bd.ParallelBooster
bd.ExplainableBooster
```

There are no backwards-compatible aliases for older pre-release BRAT class
names.

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

For empirical tuning guidance based on RMSE, coverage, width, and timing
diagnostics, see [Hyperparameter Recommendations](hyperparameters.md).

## `DropoutBooster`

Sklearn-compatible histogram-tree BRAT-D / Boulevard dropout boosting.

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
    early_stopping=False,
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
- `min_samples_leaf`: minimum leaf size. Larger values smooth the fit and often
  widen cells.
- `max_bins`: number of histogram bins per feature. scikit-learn currently
  supports at most `255`.

Short recommendations:

- Start with `dropout_rate` around `0.0` to `0.1`. Higher dropout is stronger
  regularization and can increase signal bias.
- Use `subsample_rate < 1` when you want extra regularization or faster tree
  fitting; use `1.0` when the model clearly underfits.
- Increase `max_iter` when the model underfits.
- Increase `max_depth` or `max_leaf_nodes` as dimension or curvature grows.
- Keep `early_stopping=False`; early stopping is not part of the current
  faithful BRAT-D training path.

Useful methods:

```python
pred = model.predict(X_test)
norms = model.weight_norms(X_test)
cells = model.apply_cell_indices(X_test)
ci_lower, ci_upper, _ = model.predict_intervals(X_test, mode="confidence")
pi_lower, pi_upper, _ = model.predict_intervals(X_test, mode="prediction")
ri_lower, ri_upper, _ = model.predict_intervals(X_test, mode="reproduction")
```

## `ParallelBooster`

Sklearn-compatible histogram-tree BRAT-P training. BRAT-P uses deterministic
slots instead of random dropout.

```python
model = bd.ParallelBooster(
    n_rounds=70,
    trees_per_round=6,
    subsample_rate=0.8,
    max_depth=8,
    max_leaf_nodes=128,
    min_samples_leaf=8,
    max_bins=64,
    drop_first_round=True,
    early_stopping=False,
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

Short recommendations:

- Use `trees_per_round` as the conceptual BRAT-P slot count. Values from `4` to
  `10` are reasonable starting points.
- Keep total tree count, `n_rounds * trees_per_round`, comparable to the tree
  budget you would use for `DropoutBooster`.
- Try `n_jobs > 1` only when `trees_per_round` is large enough for the joblib
  overhead to be worthwhile.
- Tune tree complexity the same way as `DropoutBooster`.

The interval API matches `DropoutBooster`.

## `ExplainableBooster`

EBM-family additive estimator with feature-level intervals. It is implemented in
Boulevard and does not depend on InterpretML.

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

Short recommendations:

- Start with `max_depth=1` or `max_depth=2` for simple main-effect fits.
- Increase `max_rounds` when the model underfits.
- Increase `max_bins` when partial functions look too coarse.
- Use held-out calibration data for `prepare_inference`.
- Treat interval coverage as experimental and verify on diagnostics relevant to
  the target problem.

Feature-level interval API:

```python
term_lower, term_upper, term_pred = model.predict_feature_intervals(
    feature_idx=0,
    values=X_test[:, 0],
    level=0.95,
    mode="confidence",
)
```

## Current Limitations

- Numeric squared-error regression only.
- Experimental asymptotic intervals.
- No categorical feature handling.
- No XGBoost, LightGBM, or CatBoost backend in the first release.
- `DropoutBooster` and `ParallelBooster` depend on scikit-learn private
  histogram-gradient-boosting internals.

## Examples

```bash
python examples/quickstart.py
```

Open the combined notebook:

```text
examples/boulevard_boosters_demo.ipynb
```

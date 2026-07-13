# Quickstart

Use the package through the top-level namespace:

```python
import boulevard as bd
```

## DropoutBooster

`DropoutBooster` is the sklearn-compatible BRAT-D estimator.

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
```

## ParallelBooster

`ParallelBooster` is the sklearn-compatible BRAT-P estimator. It trains
completed-history rounds with deterministic tree slots.

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

model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="confidence",
)
```

Use `n_jobs > 1` to parallelize slot fitting within eligible rounds. Small
`trees_per_round` values may not benefit because scheduling has fixed overhead.

## ExplainableBooster

`ExplainableBooster` is an EBM-family additive estimator with feature-level
intervals.

```python
model = bd.ExplainableBooster(
    max_rounds=160,
    max_bins=32,
    learning_rate=0.6,
    subsample_rate=1.0,
    warmup_rounds=10,
    max_depth=4,
    min_samples_leaf=8,
    random_state=0,
)

model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

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

## Runnable Examples

From the repository root:

```bash
python examples/quickstart.py
```

For the full visual comparison, open:

```text
examples/boulevard_boosters_demo.ipynb
```

For empirical starting points and coverage/accuracy tradeoffs, see
[Hyperparameter Recommendations](hyperparameters.md).

As a practical starting point, use `DropoutBooster` and `ParallelBooster` for
low-dimensional smooth signals. When dimension grows and an additive model is
appropriate, prefer `ExplainableBooster` and inspect its feature-level intervals.

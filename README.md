# Boulevard

Boulevard is a research-oriented Python package for sklearn-compatible
Boulevard / BRAT-style boosting estimators with asymptotic uncertainty
intervals.

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
    early_stopping=False,
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

`prepare_inference` estimates residual variance and builds the cached
histogram-cell linear algebra used by interval methods. If it is omitted,
interval methods prepare inference on first use with the training data. A
held-out calibration set is usually preferable for variance estimation.

For low-dimensional smooth signals, start with `DropoutBooster` or
`ParallelBooster`. For higher-dimensional additive signals, start with
`ExplainableBooster` and its feature-level intervals.

Runnable examples:

```bash
python examples/quickstart.py
```

The broader demo notebook is:

```text
examples/boulevard_boosters_demo.ipynb
```

For empirical tuning guidance, see:

```text
docs/hyperparameters.md
```

## Current Scope

- Numeric squared-error regression.
- sklearn-compatible estimator APIs.
- `predict_intervals(..., mode="confidence" | "prediction" | "reproduction")`.
- Experimental asymptotic intervals.
- No XGBoost, LightGBM, or CatBoost backend in the first release.

The histogram estimators reuse scikit-learn private internals, so the
scikit-learn dependency range is intentionally pinned to the tested private API
series.

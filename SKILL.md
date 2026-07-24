---
name: boulevard-boosting-inference
description: Use when an agent needs to help a user fit gradient-boosted regression models with uncertainty intervals using Boulevard. Covers estimator choice, train/calibration/test workflow, prediction and interval APIs, diagnostics, reporting language, and guardrails for sklearn-compatible DropoutBooster, ParallelBooster, and ExplainableBooster.
---

# Boulevard Boosting Inference

Use this guide when helping a user run Boulevard for numeric regression,
prediction, and uncertainty quantification. Treat it as an agent operating
manual, not as package marketing copy.

## Scope

Boulevard currently exposes:

```python
import boulevard as bd

bd.DropoutBooster
bd.ParallelBooster
bd.ExplainableBooster
```

Support only numeric squared-error regression. If the user asks for
classification, survival, causal inference, categorical handling without
preprocessing, or official TabArena leaderboard submission, say that this release
does not cover it yet and propose a scoped alternative.

## Choose The Estimator

Use this routing rule before writing code:

- Use `DropoutBooster` for low-dimensional smooth regression when the user wants
  a BRAT-D/dropout-style histogram-tree estimator.
- Use `ParallelBooster` for low-dimensional smooth regression when the user wants
  BRAT-P style grouped trees or a deterministic alternative to random dropout.
- Use `ExplainableBooster` when the signal is additive/main-effect, when the
  feature count grows, or when the user wants feature-level partial effects with
  intervals.
- Use external baselines such as HGBR, LightGBM, XGBoost, CatBoost, or EBM for
  pure predictive benchmarking. Boulevard is inference-aware boosting, not a
  guaranteed top leaderboard model.

Prefer effective modeling structure over raw column count. If the target is
well-described by main effects, `ExplainableBooster` is usually the better first
choice even with many input columns. If the target depends on dense interactions
among many features, validate signal confidence intervals with a problem-specific
simulation or held-out diagnostic before relying on them.

## Required Workflow

Prefer a train/calibration/test split. Do not use the final test set for
hyperparameter tuning or variance calibration when reporting test performance.

```python
model.fit(X_train, y_train)
model.prepare_inference(X_calib, y_calib)

pred = model.predict(X_test)
lower, upper, pred = model.predict_intervals(
    X_test,
    level=0.95,
    mode="prediction",
)
```

`prepare_inference` estimates residual variance and builds cached linear algebra
for interval queries. If it is omitted, interval methods will prepare inference
from training data on first use, but a held-out calibration set is usually the
better default.

If the user only has one training set, split it into model-training and
calibration subsets before touching the final test set.

## Interval Modes

Always state which interval target is being reported.

| Mode | Target | Use |
| --- | --- | --- |
| `"confidence"` | latent signal `f(x)` | Simulation or scientific mean-function uncertainty. |
| `"prediction"` | future noisy response `Y | X=x` | Default user-facing predictive uncertainty. |
| `"reproduction"` | repeated-training signal variability | Research diagnostic for model refits. |

For real data, do not report signal CI coverage unless the true signal is known.
Report prediction interval coverage against observed held-out responses instead.

## Code Patterns

Low-dimensional DropoutBooster start:

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

Low-dimensional ParallelBooster start:

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

Additive ExplainableBooster start:

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
```

Feature-level interval pattern for `ExplainableBooster`:

```python
lower, upper, partial = model.predict_feature_intervals(
    feature_idx=0,
    x_k=grid,
    level=0.95,
    mode="confidence",
)
```

Do not call `predict_feature_intervals` on `DropoutBooster` or
`ParallelBooster`.

## Diagnostics

For synthetic data with known signal, report:

- RMSE against the true signal.
- RMSE against noisy observed `y`.
- Signal CI coverage using `mode="confidence"`.
- Prediction interval coverage using `mode="prediction"`.
- Interval width quantiles, preferably q05/q50/q95.
- Fit time, `prepare_inference` time, prediction time, and interval-query time.

For real data without known signal, report:

- Validation or test RMSE/MAE against observed `y`.
- Prediction interval coverage against observed `y`.
- Calibration residual mean and variance.
- Interval width quantiles.
- A clear caveat that signal CIs are diagnostic unless validated separately.

Useful helpers:

```python
def coverage(y, lower, upper):
    return float(((y >= lower) & (y <= upper)).mean())


def width_quantiles(lower, upper):
    return np.quantile(upper - lower, [0.05, 0.5, 0.95])
```

## Tune By Symptoms

- If predictions are too flat and RMSE is high, increase tree count or rounds,
  `max_leaf_nodes`, `max_depth`, or `max_bins`.
- If signal CIs miss curved regions while prediction intervals cover noisy `y`,
  treat the issue as approximation bias and improve the predictive fit first.
- If intervals are too wide, reduce tree complexity, increase
  `min_samples_leaf`, or use more calibration/training data when possible.
- If high-dimensional interactions dominate, avoid claiming reliable signal CIs
  without a problem-specific simulation or validation study.

Tree controls cap each other. Raising `max_leaf_nodes` will not help if
`max_depth`, `min_samples_leaf`, subsampling size, or available histogram bins
already imposes a smaller effective partition.

## Reporting Language

Prefer:

> Boulevard returns asymptotic intervals. Prediction intervals are evaluated on
> held-out observed responses; signal confidence intervals require known truth
> or a simulation design to measure coverage.

Avoid:

> The interval has exact 95% coverage.

Do not claim conformal validity unless a conformal correction was explicitly
implemented and evaluated.

## Guardrails

- Do not tune on the final test set.
- Do not calibrate interval variance on the final test set when reporting final
  test performance.
- Do not present `mode="confidence"` as a prediction interval for noisy outcomes.
- Do not present `mode="prediction"` as uncertainty for the noiseless signal.
- Do not add TabArena, AutoGluon, LightGBM, XGBoost, CatBoost, or InterpretML as
  release dependencies unless the package scope changes.
- Keep benchmark work in a separate workspace from the release package.
- If package code is edited, run the local checks before claiming completion.

## Local Validation

When editing the Boulevard package, run:

```bash
python -m ruff check src tests examples
python -m pytest -q
python examples/quickstart.py
```

If notebook behavior matters, also validate that
`examples/boulevard_boosters_demo.ipynb` is valid JSON and run the cells when
notebook tooling is available.

## Reference Map

Read these files when more detail is needed:

- `README.md`: public API, install instructions, interval descriptions, and
  hyperparameter recommendations.
- `examples/quickstart.py`: shortest runnable API example.
- `examples/boulevard_boosters_demo.ipynb`: richer visual demonstration.
- `src/boulevard/estimators/sklearn/dropout.py`: `DropoutBooster`
  implementation.
- `src/boulevard/estimators/sklearn/parallel.py`: `ParallelBooster`
  implementation.
- `src/boulevard/estimators/interpretml/explainable.py`:
  `ExplainableBooster` implementation.


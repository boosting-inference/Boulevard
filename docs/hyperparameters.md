# Hyperparameter Recommendations

These recommendations are empirical starting points, not formal guarantees.
They are based on synthetic diagnostics with known regression signals, Gaussian
noise with standard deviation `0.2`, and separate train / calibration / test
splits. Each model was fitted on the training split, then
`prepare_inference(X_calib, y_calib)` was run before interval evaluation.

The current release targets numeric squared-error regression. Signal confidence
intervals can under-cover when approximation bias remains. Prediction intervals
are usually easier to cover because they include residual noise variance.

## Search Setup

The main recommendations were generated with:

```bash
.venv/bin/python tests/diagnostics/booster_hyperparameter_search.py \
  --trials 30 \
  --random-search
```

A follow-up high-dimensional stress run used:

```bash
.venv/bin/python tests/diagnostics/booster_hyperparameter_search.py \
  --trials 80 \
  --random-search \
  --scenarios interaction_5d
```

Default sample sizes grow with dimension:

| Scenario | Features | Samples |
| --- | ---: | ---: |
| `smooth_1d` | 1 | 1000 |
| `additive_3d` | 3 | 2500 |
| `interaction_5d` | 5 | 5000 |

Coverage labels:

| Label | Meaning |
| --- | --- |
| `PASS` | signal CI coverage >= 0.90 |
| `OK` | signal CI coverage >= 0.80 |
| `LOW` | signal CI coverage < 0.80 |

## Empirical Summary

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
- 3D additive problems need substantially more capacity than the original
  quickstart defaults. `DropoutBooster` and `ParallelBooster` reached the OK
  band in this search.
- 5D interaction problems remain hard for asymptotic signal CIs. The predictive
  RMSE can be reasonable while signal CI coverage is poor, which means the
  interval does not fully account for approximation bias.
- Prediction intervals were close to or above 95% coverage in this run. This
  does not imply signal CIs are automatically calibrated.

## High-Dimensional Interaction Stress Test

The 80-trial `interaction_5d` follow-up search confirms that the current signal
confidence intervals should not be advertised as calibrated for interaction-heavy
higher-dimensional functions. The best score and best coverage trials were still
below the OK threshold:

| Estimator | Best Type | RMSE vs Signal | Signal CI Coverage | PI Coverage | Median CI Width |
| --- | --- | ---: | ---: | ---: | ---: |
| `DropoutBooster` | best score | 0.134 | 0.523 | 0.961 | 0.198 |
| `DropoutBooster` | best coverage | 0.183 | 0.538 | 0.960 | 0.253 |
| `ParallelBooster` | best score | 0.149 | 0.606 | 0.958 | 0.253 |
| `ParallelBooster` | best coverage | 0.405 | 0.638 | 0.966 | 0.344 |
| `ExplainableBooster` | best score and coverage | 0.079 | 0.658 | 1.000 | 0.155 |

Practical interpretation:

- More hyperparameter search did not recover 90% signal coverage on this 5D
  interaction task.
- Low RMSE does not guarantee good signal CI coverage. For example,
  `ExplainableBooster` had the best RMSE but still covered only 65.8% of the
  true signal.
- Prediction intervals remained close to or above nominal coverage because they
  include noise variance. For outcome uncertainty, prediction intervals are the
  safer current default.
- For interaction-heavy problems, treat signal CIs as diagnostic unless you add
  an external calibration step or validate coverage in a simulation matching the
  target application.

## Dimension And Capacity

As dimension increases, use more samples and more model capacity. A useful
starting rule is:

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

## DropoutBooster

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
    early_stopping=False,
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
    early_stopping=False,
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
- `max_depth` and `max_leaf_nodes`: increase together as dimension grows. The
  3D additive search selected `max_depth=10` and `max_leaf_nodes=256`.
- `min_samples_leaf`: use small values such as `2-4` when bias dominates.
- `max_bins`: use `64` for smooth 1D signals; try `128-255` for higher
  dimension or sharper structure.

## ParallelBooster

Low-dimensional starting point:

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
    early_stopping=False,
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
- Tune `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, and `max_bins` the
  same way as `DropoutBooster`.
- Use `n_jobs=1` by default. Parallel scheduling only helps when
  `trees_per_round` is large enough.

## ExplainableBooster

Low-dimensional starting point:

```python
model = bd.ExplainableBooster(
    max_rounds=160,
    max_bins=32,
    learning_rate=0.6,
    subsample_rate=1.0,
    warmup_rounds=0,
    max_depth=4,
    min_samples_leaf=8,
    leave_one_out=False,
    random_state=0,
)
```

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
below target. Treat its current intervals as diagnostic and validate them on
held-out or simulated truth before relying on them.

## Practical Workflow

For a new dataset:

1. Split into train, calibration, and test or validation sets.
2. Fit a moderate starting model from the recommendations above.
3. Run `prepare_inference(X_calib, y_calib)`.
4. Check RMSE, residual variance, CI coverage for known simulation truth if
   available, PI coverage for noisy outcomes, and interval width quantiles.
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

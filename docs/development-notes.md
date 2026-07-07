# Development Notes

## Current sklearn-first release status

The current publishable package target is the sklearn-compatible estimator
surface:

```python
import boulevard as bd

bd.BRATDHistGradientBoostingRegressor
bd.BRATPHistGradientBoostingRegressor
bd.IEBMRegressor
```

XGBoost, LightGBM, and CatBoost remain development namespaces. They are not
top-level public APIs yet, and they should not be advertised as faithful
Boulevard-trained backends until each backend has its own implementation audit.

The package metadata, README, public API tests, and wheel build have been
updated around this sklearn-first release target. Remaining pre-release work is
mostly packaging and release hygiene:

- add continuous integration for lint, tests, and build checks;
- decide whether to ship or exclude tests in the source distribution;
- do one clean install-from-wheel smoke test on a fresh machine or CI runner;
- decide the first release version and tag policy;
- optionally publish to TestPyPI before PyPI;
- keep IEBM intervals marked experimental until more coverage diagnostics are
  run;
- keep non-sklearn backends namespaced as development work.

## 2026-06-23: Nyström sketching for histogram inference

BRAT-D and BRAT-P histogram estimators now support optional Nyström sketching
for interval weight norms:

```python
model = bd.BRATDHistGradientBoostingRegressor(
    nystrom_subsample_rate=1.0,
)
```

This is an inference-only approximation. It does not change the fitted
BRAT-D/BRAT-P tree ensemble or the prediction function. It only replaces the
full observed-cell kernel solve used for `weight_norms`, confidence intervals,
prediction intervals, and reproduction intervals.

The paper sketches the full row-level kernel with a subsampling matrix `S`. The
histogram implementation has already compressed repeated binned rows into
observed multidimensional cells, so the implementation samples landmark cells
with probability proportional to `cell_counts_`. This preserves the intended
row-subsampling interpretation while avoiding duplicate landmark columns from
identical binned rows.

The default remains exact through the all-landmark path:

```python
nystrom_subsample_rate=1.0
```

If the requested sketch would include all observed cells, the code falls back to
the exact observed-cell solve and reports `inference_method_ = "histogram_cell"`.
Otherwise it reports `inference_method_ = "histogram_cell_nystrom"` and stores
`nystrom_landmark_count_`.

The API demo relies on this estimator default:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

## 2026-06-23: Backend-oriented estimator layout

The estimator package is now organized by backend family. Sklearn-backed
implementations live under:

```text
boulevard.estimators.sklearn
```

The sklearn-first top-level public API is:

```python
import boulevard as bd

bd.BRATDHistGradientBoostingRegressor
bd.BRATPHistGradientBoostingRegressor
```

The old `boulevard.estimators.bratd` and `boulevard.estimators.bratp` import
paths were removed during release cleanup; canonical imports now go through
`boulevard.estimators.sklearn`. Backend families such as `xgboost`, `lightgbm`,
`catboost`, and `interpretml` remain namespaced development areas under
`boulevard.estimators`, but XGBoost/LightGBM/CatBoost are not top-level public
APIs for the sklearn-first release.

Early scaffolding modules that were not used by the current estimators were
removed: `boulevard.algorithms`, `boulevard.core`, empty `boulevard.inspection`,
and the unused early Nyström helper. Active shared code now lives in the places
that current estimators actually import from, mainly `backends`, `intervals`,
and `estimators.sklearn`.

The exact sample-space `BRATDRegressor` prototype was later removed from the
public package surface. The histogram BRAT-D estimator is now the sklearn-backed
BRAT-D implementation we intend to carry forward, so the prototype's private
decision-tree backend and leaf-kernel helpers were removed with it.

## 2026-06-21: Experimental histogram BRAT-P backend

This work adds the first serial histogram-tree BRAT-P estimator:

```python
BRATPHistGradientBoostingRegressor
```

The class lives in `boulevard.estimators.sklearn.bratp` and is exported from
top-level `boulevard` as `bd.BRATPHistGradientBoostingRegressor`. Its public
training API uses the BRAT-P round/slot structure:

```python
model = bd.BRATPHistGradientBoostingRegressor(
    n_rounds=60,
    trees_per_round=4,
    subsample_rate=0.8,
    early_stopping=False,
)
```

The first implementation trains the round/slot structure serially. It is not yet
parallelized, but it follows the deterministic BRAT-P residual construction:

- first round is warm-started by sequential boosting steps;
- later rounds train one tree per slot;
- the residual for slot `k` drops slot `k` from previous rounds and uses the
  other slots averaged over previous rounds.

The estimator reuses the observed histogram-cell inference machinery from the
BRAT-D histogram estimator, but the BRAT-P linear system is different. A
diagnostic found severe undercoverage in the first BRAT-P CI implementation:

```text
RMSE vs true signal: 0.0842
95% CI coverage vs true signal: 0.373
median 95% CI half-width: 0.0416
```

The issue was not prediction RMSE. It was an inference scaling bug. The code
initially used the unscaled system

```text
I + (K - 1) K_n
```

but the BRAT-P plug-in weights are based on

```text
K^{-1} I + ((K - 1) / K) K_n
```

where `K = trees_per_round`. Equivalently, the correct BRAT-P weights are `K`
times the weights from the unscaled system. After applying this scaling, the
same quickstart diagnostic reports:

```text
RMSE vs true signal: 0.0842
median abs error vs true signal: 0.0560
95% CI coverage vs true signal: 0.937
95% PI coverage vs noisy y: 0.977
median 95% CI half-width: 0.1666
```

The current API demo script is:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

It plots the fitted signal, confidence and prediction bands, interval widths,
BRAT-D and BRAT-P weight norms, signal error against CI half-width, and
calibration residuals. It also prints RMSE, coverage, interval-width quantiles,
and wall-clock timings.

Tests now include a regression check that compares the scaled BRAT-P cell system
against the unscaled system and verifies that the norm changes by the expected
factor `trees_per_round`.

## 2026-06-19: Experimental histogram BRAT-D backend

This commit adds the first working experimental histogram-tree BRAT-D estimator:

```python
BRATDHistGradientBoostingRegressor
```

The class lives in `boulevard.estimators.sklearn.bratd` and is exported from
top-level `boulevard` as `bd.BRATDHistGradientBoostingRegressor`. It inherits
sklearn's public `HistGradientBoostingRegressor` API shape, but replaces the
boosting loop with a custom BRAT-D loop using sklearn's private histogram
internals:

- `_BinMapper` for sklearn-compatible continuous-feature binning.
- `TreeGrower` for fitting one histogram tree from pseudo-residual gradients.
- `TreePredictor` for storing and predicting from fitted histogram trees.

Supported scope is deliberately narrow:

- squared-error regression only;
- no early stopping;
- no warm start;
- no categorical features;
- no monotone constraints;
- no interaction constraints.

The estimator now has a histogram-cell inference path. Training rows are
compressed into observed multidimensional bin cells, and the BRAT-D leaf-kernel
system is built in cell space with `cell_counts_` carrying training-row
multiplicity. This gives the same norm as the direct cell solve for observed
cells while avoiding repeated leaf traversal and linear solves for every query.

The current visual/API demo script is:

```bash
python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

It compares the histogram BRAT-D estimator with BRAT-P and vanilla sklearn
`HistGradientBoostingRegressor`. It also prints diagnostics for:

- prediction RMSE against the known synthetic truth;
- training, inference-preparation, prediction, and interval wall-clock time;
- cached versus direct-solve histogram-cell weight norms;
- signal-correction scaling;
- centered residual variance;
- observed-cell compression ratio;
- cache hit rates for observed cells;
- CI, PI, and reproduction-interval width relationships.

The current visual experiment shows that the histogram-cell norm cache is
numerically equivalent to the slow direct solve on the tested points, with max
absolute error near machine precision. It also confirms a large speedup in
`weight_norms`: cached observed-cell lookup is essentially instant compared with
the Python leaf-traversal path.

The main inference finding from this round is that low CI coverage can be driven
by tree resolution, not by the residual variance estimate. With the default
`max_leaf_nodes=31`, trees merge many of the 128 observed one-dimensional bins,
which can make the BRAT-D weight norm too small and narrow the CI. Increasing
effective tree complexity raises coverage substantially in the synthetic visual
check, reaching roughly 0.93 in the user's local run. Future diagnostics should
print per-tree leaf counts and cells-per-leaf summaries to make this visible.

Tests now cover the experimental estimator more directly:

- sklearn cloning;
- fit/predict smoke behavior;
- deterministic fitting;
- unsupported-mode validation;
- invalid sample weights;
- prediction signal correction;
- observed-cell compression;
- centered residual variance estimation;
- cached versus direct-solve weight norms;
- CI, PI, and reproduction-interval width relationships.

The current package test suite passes with:

```bash
.venv/bin/python -m pytest -q
```

Important unresolved issues:

- The BRAT-D residual denominator convention still needs a math audit. The code
  currently uses the number of already-fitted trees in the residual construction;
  we still need to confirm whether the paper's round index requires the
  post-addition denominator.
- The estimator depends on sklearn private histogram internals, so compatibility
  should be tested when sklearn versions change.
- The histogram-cell inference system should be compared against a tiny expanded
  sample-space calculation where duplicate binned rows are repeated explicitly.

## 2026-06-18: BRAT-D variance and CI diagnostics

The BRAT-D visual smoke script now checks more than interval plots. It also reports the variance estimate used by asymptotic intervals and plots where the confidence interval misses the known regression function in the synthetic example.

The current relevant example is:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/boulevard_mplconfig \
  .venv/bin/python examples/brat_histogram_api_demo.py --output /tmp/brat_histogram_api_demo.png
```

The diagnostic prints:

- `sigma_hat2`, the variance used by the estimator.
- the true simulation noise variance.
- the oracle calibration noise variance, using the known synthetic truth.
- centered calibration residual variance.
- uncentered calibration residual mean squared error.
- centered training residual variance.
- the range of `|prediction - truth| / CI half-width`.
- the number of grid points where the CI misses the truth.

This confirms that the current implementation estimates `sigma_hat2` as the centered calibration residual variance:

```python
sigma_hat2 = var(y_calib - predict(X_calib), ddof=1)
```

In the current synthetic visual check, `sigma_hat2` is close to the calibration residual variance by construction and can be larger than the oracle noise variance. Training residual variance is much smaller, which is expected because it is in-sample.

The confidence interval coverage shown in this script is a visual diagnostic, not the formal pointwise repeated-sampling coverage target. It measures the fraction of grid points in one fitted model where the interval covers the known synthetic regression function.

The current main discovery is that low CI coverage can occur even when `sigma_hat2` is not too small. The misses are better explained by the ratio

```text
|prediction - truth| / CI half-width
```

and by the geometry of the BRAT-D leaf kernel. In flat regions, many training points can land in the same leaves, spreading the test point influence across many samples. This makes the kernel weight norm `||r_n(x)||` small and narrows the CI. In high-slope regions, trees often split more aggressively, which can concentrate influence, increase `||r_n(x)||`, and widen the interval.

The next useful diagnostic is to compare the solved BRAT-D weight norm with the raw leaf-kernel vector norm. If the raw kernel vector is already small, the narrow interval comes from leaf geometry. If the raw kernel vector is moderate but the solved norm is small, the shrinkage in the linear solve is responsible.

## 2026-06-18: Route A histogram BRAT-D backend plan

The planned histogram backend will follow Route A: inherit from sklearn's public `HistGradientBoostingRegressor` for estimator API compatibility, but replace sklearn's internal boosting loop with BRAT-D residual construction.

The local development environment has two sklearn versions in use: the shell-level conda environment uses sklearn `1.3.0`, while the project `.venv` uses sklearn `1.8.0`. The initial histogram backend tests pass in both environments. The private internals identified for reuse are:

- `_BinMapper` for fitting and applying histogram bins.
- `TreeGrower` for fitting a single histogram tree from gradients and hessians.
- `TreePredictor` for storing fitted histogram tree predictors.

The first implementation step adds an experimental class:

```python
BRATDHistGradientBoostingRegressor
```

This class is intentionally not exported from `boulevard` yet. The first working version trains and predicts with sklearn's `_BinMapper` and `TreeGrower`, supporting squared-error regression with no early stopping, warm start, categorical features, monotone constraints, or interaction constraints.

The intended training semantics are:

1. Fit sklearn's bin mapper once on the training data.
2. For boosting round `b`, drop old trees independently with keep probability `q = 1 - dropout_rate`.
3. Construct the BRAT-D pseudo-response from the kept trees.
4. Fit one histogram tree to that pseudo-response through `TreeGrower`.
5. Store unscaled tree predictors; apply Boulevard/BRAT-D averaging and signal correction in prediction.

The first inference extension is observed-cell compression: use only observed multidimensional binned cells, build leaf blocks in cell-space, and compute the BRAT-D influence norm through a cell-count-weighted linear system instead of a full sample-space kernel. This is implemented in the experimental histogram class, but it should still be treated as provisional until we audit the residual-round denominator and compare the cell-space system against a small expanded sample-space calculation.

## 2026-06-24: wrap intervals in an Interval() method, arguments specifying which type of intervals they want

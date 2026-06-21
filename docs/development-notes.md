# Development Notes

## 2026-06-19: Experimental histogram BRAT-D backend

This commit adds the first working experimental histogram-tree BRAT-D estimator:

```python
BRATDHistGradientBoostingRegressor
```

The class lives in `boulevard.estimators.bratd` and is exported from top-level
`boulevard` as `bd.BRATDHistGradientBoostingRegressor`. It inherits sklearn's public
`HistGradientBoostingRegressor` API shape, but replaces the boosting loop with a
custom BRAT-D loop using sklearn's private histogram internals:

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

The visual check script is:

```bash
python examples/brat_d_hist_visual_check.py
```

It compares the histogram BRAT-D estimator with the exact sample-space
`BRATDRegressor` and vanilla sklearn `HistGradientBoostingRegressor`. It also
prints diagnostics for:

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

The relevant example is:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/boulevard_mplconfig \
  .venv/bin/python examples/brat_d_visual_check.py --output /tmp/brat_d_visual_check.png
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

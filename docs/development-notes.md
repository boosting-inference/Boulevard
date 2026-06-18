# Development Notes

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

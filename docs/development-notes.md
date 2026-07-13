# Development Notes

## Sklearn-First Release Target

The current package target is a small sklearn-compatible release:

```python
import boulevard as bd

bd.DropoutBooster
bd.ParallelBooster
bd.ExplainableBooster
```

This release intentionally removes older pre-release aliases and backend
placeholders. XGBoost, LightGBM, and CatBoost work can return later in separate
backend-specific forks or integrations, but they are not part of this package
surface now.

## Release Cleanup Decisions

- The dropout BRAT estimator was renamed to `DropoutBooster`.
- The parallel BRAT estimator was renamed to `ParallelBooster`.
- The EBM-family estimator was renamed to `ExplainableBooster`.
- The earlier inference sketching prototype was removed from BRAT estimators.
- Truncation was removed from `ExplainableBooster`.
- Heavy diagnostic scripts were replaced by `examples/quickstart.py` and
  `examples/boulevard_boosters_demo.ipynb`.
- The release docs avoid advertising unsupported backend families.

The old names were useful while mapping paper terminology into code, but the
release names are shorter and describe how a user should think about each
estimator.

## Implementation Notes

`DropoutBooster` and `ParallelBooster` use scikit-learn histogram-tree private
internals for prebinning and tree fitting. This is why the scikit-learn
dependency is pinned to the tested private API series.

Both histogram estimators cache training tree predictions during fitting. This
avoids repeated old-tree traversal when constructing residuals and is the main
fit-time speedup relative to the first prototype.

For inference, both histogram estimators compress repeated binned rows into
observed multidimensional cells. Interval weight norms are solved once for the
observed cells, then query points reuse cached cell norms when they land in an
observed cell.

`ExplainableBooster` uses a pure Python/NumPy one-dimensional tree update. The
current splitter uses a heap-based greedy search so that after a split only the
two child segments need new best-split calculations. InterpretML's native EBM
implementation remains faster because its update loop is compiled, but the
Boulevard implementation is easier to audit and does not patch private
InterpretML modules.

## Release Checks

Before cutting a release:

```bash
.venv/bin/python -m ruff check src tests examples docs
.venv/bin/python -m pytest -q
.venv/bin/python examples/quickstart.py
```

The combined notebook should be checked as valid JSON and, when practical,
executed manually before release.

# Boulevard

Boulevard is a Python package for Boulevard-style wrappers around popular boosted tree libraries, with a focus on regularization, stability, and uncertainty intervals.

The package is currently in early development.

The first publishable target is the sklearn-compatible estimator family:

- `boulevard.BRATDHistGradientBoostingRegressor`
- `boulevard.BRATPHistGradientBoostingRegressor`
- `boulevard.IEBMRegressor`

XGBoost, LightGBM, and CatBoost integrations are development placeholders and
are not part of the stable public API yet.

## Installation for development

```bash
pip install -e ".[dev]"
```

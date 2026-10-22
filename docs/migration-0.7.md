# Migrating from 0.7

0.8.0 replaced the 0.7 API of every model: constructors take hyperparameters
only, data goes to `fit` in a `PanelData`, `predict` returns DataFrames,
results live in `result_`, and runs are saved as directories. Nothing of the
old API is kept as a deprecated alias.

| 0.7                                             | 0.8                                                      |
|-------------------------------------------------|----------------------------------------------------------|
| `DFM(dataset, config=...)`, `DFMDataset(...)`   | `DFM(n_factors=..., clock=..., ...)`, `PanelData(y, freq=...)` |
| `fit()` / `fit(X)` returning `None` or a state  | `fit(data)` returning the model                          |
| `predict(horizon)` returning arrays, tuples or dicts | `predict(h)` and `predict_factors(h)`, DataFrames (AFM: `allocate()`) |
| `get_result()`, `result`, `training_state`      | `result_`, `n_iter_`, `converged_`, `factors_`, ...      |
| `hyperparameter_optimization(...)`              | `dfm_python.tuning.tune_hyperparameters(estimator, data, ...)` |
| single-file `.pkl` / `.pt` checkpoints          | run directories: `save(path)`, `Model.load(path)`, `dfm_python.load(path)` |
| flat YAML (`num_factors`, `threshold`, `seed`)  | `model:` / `data:` sections (`n_factors`, `tol`, `random_state`) |

The [CHANGELOG](https://github.com/minkeymouse/dfm-python/blob/main/CHANGELOG.md)
entry for 0.8.0 has the full old-to-new call table for every model, the
renamed hyperparameters and the shipped YAML values that now take effect.

Numbers: DFM's changed in 0.8.0 (the numerical-stability rewrite of the
E-step and M-step); DDFM's `fit` and `predict(h)`, iVDFM and AFM match 0.7.

Checkpoints: a single-file `.pkl` or `.pt` checkpoint from 0.7, or a save
directory from a development build of 0.8.0 before the DFM state-layout
change, raises `ConfigurationError` in `load`. Load it with the release that
wrote it. Scripts written for 0.7 keep working with
`pip install "dfm-python==0.7.0"`.

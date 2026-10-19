# Changelog

## Unreleased

- **Breaking (installation):** a bare `pip install dfm-python` no longer
  installs optuna, sktime, omegaconf, hydra-core, pyarrow, matplotlib or
  pytorch-lightning. Code that used them through dfm-python now raises an
  `ImportError` naming the extra to install: `calculate_rmse`/`calculate_mae`
  need `[metrics]`, `dfm_python.tuning` needs `[tune]`,
  `YamlSource.load()` needs `[yaml]`, polars input needs `[polars]` (polars
  and pyarrow), and the `tutorial/` scripts need `[examples]`.
  **Migration:** to keep the 0.7.0 behaviour, depend on `dfm-python[all]`
  (for example in a `requirements.txt` that pinned bare `dfm-python`).
- Runtime dependencies reduced to numpy, pandas, scipy, scikit-learn, torch,
  pykalman and einops. pytorch-lightning and matplotlib were never imported
  and are dropped; optuna, omegaconf, sktime, polars (with pyarrow) and
  hydra-core moved to the `tune`, `yaml`, `metrics`, `polars` and `examples`
  extras (`all` installs every extra).
- **Breaking (extras):** the `db`, `deep`, `transform`, `forecasting` and
  `hydra` extras are removed. pip only warns about an unknown extra, so
  `dfm-python[hydra]` now silently installs neither omegaconf nor hydra-core;
  use `[yaml]` or `[examples]` instead (`[metrics]` replaces `[transform]` and
  `[forecasting]`).
- The `dev` dependency group lists its packages directly (it works with
  `pip install --group dev` as well as uv) and includes the optional packages
  the test suite imports.
- `iVDFM` and the four config classes (`DFMConfig`, `DDFMConfig`,
  `iVDFMConfig`, `AFMConfig`) are exported from `dfm_python`. `iVDFMDataset`
  is no longer exported (from `dfm_python` or `dfm_python.dataset`); it is the
  iVDFM estimator's private window adapter in `dfm_python.dataset.ivdfm_dataset`.
- **Breaking (DDFM):** `DDFM` is now an estimator with sklearn conventions
  that takes `PanelData`: `DDFM(*, n_factors=4, encoder_layers=(16,),
  decoder_type="linear", activation="relu", learning_rate=0.005,
  optimizer="Adam", n_mc_samples=10, window_size=100, max_iter=200, tol=5e-4,
  interpolation_method="linear", interpolation_limit=10,
  interpolation_limit_direction="both", scaler=None, random_state=3,
  device="auto", verbose=0)`, then `fit(data)`, `predict(h, data=None)`,
  `predict_factors(h, data=None)`, `update(data)`, `save(path)` and
  `DDFM.load(path, device=None)` (or `dfm_python.load`). The targets are the
  panel's `y` and the encoder covariates its `exog`. `DDFMConfig` mirrors the
  constructor (`DDFM.from_config`, `get_config`) and reads the `model:`/`data:`
  YAML layout. The numbers are unchanged: the same data and hyperparameters
  give the same weights, factors, state-space matrices and forecasts.

  | Old | New |
  |---|---|
  | `DDFM(dataset, config, encoder_size, ..., seed=3)` (18 keywords; `config` was ignored except for the interpolation values); `DDFMDataset(data, time_idx, covariates, scaler)` | `DDFM(*, n_factors, encoder_layers, ..., random_state=3, device="auto", verbose=0)`; covariates in `PanelData.exog` |
  | `fit() -> None`, then a required `build_state_space()` | `fit(data) -> self` (builds the state space) |
  | `predict(horizon=None, *, data, update=True, return_series, return_factors)`: ndarray or tuple, mutates with `data` | `predict(h, data=None) -> DataFrame` (pure); `predict_factors(h, data=None) -> DataFrame` |
  | `update(DDFMDataset) -> None`; `torch.save` file, `load` rebuilt a minimal dataset; no device override | `update(data) -> self`; a directory with `weights.pt`; `DDFM.load(path, device=)` |
  | `get_result()`, `training_state` | `result_`, `factors_`, `n_iter_`, `converged_`, `scaler_`, `network_` |
  | `encoder_size=(*hidden, k)`, `seed`, `tolerance` | `encoder_layers=hidden`, `n_factors=k`, `random_state`, `tol` |

  - Removed: the `DDFM(dataset, config, ...)` constructor, `build_state_space`,
    `get_result`, `training_state`, `predict`'s `data`/`update`/`return_series`/
    `return_factors` flags, `torch.save` single-file checkpoints (loading one
    raises `ConfigurationError`; save a run with the release that wrote it and
    refit), the `utils.checkpoint` helpers `infer_ddfm_input_dim` and
    `infer_ddfm_params_from_state_dict`, the `DDFMDataset` export (from
    `dfm_python` and `dfm_python.dataset`; it is the estimator's private
    adapter in `dfm_python.dataset.ddfm_dataset`), the `DDFMTrainingState` and
    `DDFMStateSpaceParams` aliases and their exports, and
    `dfm_python.logger.ddfm_logger` (`DDFMTrainLogger`, `DDFMInferenceLogger`;
    `log_training_start("DDFM")` and `log_inference_start("DDFM")` return the
    base loggers).
  - `predict(h, data=...)` is pure: it conditions on `data` (the full panel,
    including the new rows) on a copy and leaves the model unchanged. The
    mutating path is the explicit `update(data)`.
  - `result_.C` is the loading matrix H, N x r (series x factors), the `DFM`
    orientation; it was H transposed (factors x N).
  - Known limitation, kept for identical numbers (the old `update` and
    `predict(data=)` did the same): refitting the target scaler
    inverse-transforms the training targets only for a scaler with `mean_`
    and `scale_`. With `"minmax"`, `"robust"`, `"maxabs"`, `"quantile"`
    or another transformer it refits on scaled training targets stacked with
    raw new ones, so `update(data)` and `predict(h, data=)` are only
    meaningful with `scaler="standard"` or `None`.
  - With `scaler=None` the forecasts are in model space (the engine's
    forecast, untransformed); the old `predict` raised without a scaler.
  - Dropped as dead (stored but never read): `max_epoch_pre_train`
    (pre-training runs `n_mc_samples` epochs) and `min_target_interpolate_ratio`,
    and the `DDFMConfig` fields `use_batch_norm`, `use_idiosyncratic`,
    `min_obs_idio` and `lags_input`. `DDFMConfig`'s `disp` is replaced by
    `verbose`, and its `num_factors`, `max_epoch`, `tolerance` and `seed` by
    `n_factors`, `max_iter`, `tol` and `random_state`.
  - Shipped YAML values that now take effect. `DDFM(dataset, config=...)`
    ignored the YAML except for the interpolation values, so a run with these
    files used the constructor defaults:
    - `config/ddfm_macro.yaml`: `learning_rate: 0.001` (ran as 0.005),
      `n_mc_samples: 1` (10), `encoder_layers: [64, 32]` ([16]),
      `n_factors: 2` (4). For the old tutorial, which passed the values,
      only `random_state: 3` differs (it ran unseeded).
    - `config/ddfm_finance.yaml`: `learning_rate: 0.001` (0.005),
      `encoder_layers: [64, 32]` ([16]), `n_factors: 5` (4),
      `decoder_type: mlp` (linear), `window_size: 200` (100), `max_iter: 10`
      (200). For the old tutorial only `random_state: 3` (unseeded) and
      `decoder_type: mlp` (it ran linear) differ.
    - `config/ddfm_exchange.yaml`: `window_size: 200` (100), `max_iter: 10`
      (200). The old tutorial passed every value and ran these.
- **Breaking (DFM):** `DFM` is now an estimator with sklearn conventions
  that takes `PanelData`: `DFM(*, n_factors=None, blocks=None, clock="m",
  tent_weights=None, max_iter=5000, tol=1e-5, ar_clip=None, data_clip=None,
  regularization=None, damping_factor=None, scaler=None, verbose=0)`, then
  `fit(data, *, checkpoint_callback=None)`, `predict(h, data=None)`,
  `predict_factors(h, data=None)`, `update(data, *, refit=False)`,
  `save(path)` and `DFM.load(path)` (or `dfm_python.load`), with no dataset.
  The series are the panel's `y` and their frequencies its `freq`
  (`{"gdp": "q"}` or `{"q": ["gdp"]}`; unlisted series use the clock).
  `DFMConfig` now mirrors the constructor (`DFM.from_config`, `get_config`)
  and reads the `model:`/`data:` YAML layout. The numbers are unchanged: the
  same data and hyperparameters give the same parameters, smoothed states,
  forecasts and log-likelihood.

  | Old | New |
  |---|---|
  | `DFM(dataset, config=None, scaler=None, **kwargs)` (kwargs mutated the caller's config); `DFMDataset(config, data, time_index)` | `DFM(*, n_factors, blocks=None, clock, tent_weights, max_iter, tol, ..., scaler=None, verbose=0)`; `PanelData(y, freq=...)` |
  | `fit(X=None, checkpoint_callback=None) -> DFMModelState`; series not in `[clock \| slower]` order raised | `fit(data, *, checkpoint_callback=None) -> self`; any column order (a stable reorder inside, results in your order) |
  | `predict(horizon=None, *, data=None) -> (X, Z)` | `predict(h, data=None)` and `predict_factors(h, data=None)`, both DataFrames; `h` is required |
  | `update(data, rescale=False, retrain=False) -> None` | `update(data, *, refit=False) -> self` |
  | `training_state`, `_result`, `get_result()`, `result` | `result_`, `n_iter_`, `converged_`, `loglik_`, `factors_`, `scaler_`, `engine_` |
  | pickle file; `load(path, config=None, dataset=<required>)`; `load_config` (broken) | a directory; `DFM.load(path)`; `from_config` |
  | `threshold`, `num_factors` (in a block), `frequency` | `tol`, `n_factors`, `PanelData.freq` |

  - Removed: the `DFM(dataset, config, ...)` constructor and its
    keyword-over-config merge, `get_result`, `result`, `training_state`,
    `reset`, `load_config` (it called a missing `_load_config_common` and could
    not run), the pickle `save` and the pickle/joblib `load` (a single-file
    checkpoint raises `ConfigurationError`; refit and save with this
    release), `update`'s `rescale` flag, `predict`'s `horizon=None` default
    (one year of clock periods) with `utils.misc.compute_default_horizon`, the
    `DFMDataset` export (from `dfm_python` and `dfm_python.dataset`; it is the
    estimator's private adapter in `dfm_python.dataset.dfm_dataset`), the
    `DFMStateSpaceParams` alias and its exports, and the `dfm_python.models.dfm`
    re-exports of the engine helpers (`initialize_parameters`,
    `setup_mixed_frequency_params`, `find_slower_frequency`, ...; import them
    from their submodules). The old `DFMConfig` (the engine's config) is
    private, and its never-read `idio_rho0` and `idio_min_var` are gone.
  - Fixed: `update()` always raised `ModuleNotFoundError` (it imported a
    validator from a module path that does not exist); it now runs, through
    `DFM.update(data)`.
  - Mixed frequency: one frequency slower than the clock; a series faster
    than the clock or a second slower frequency raises
    `UnsupportedInputError`. `tent_weights` is required for the slower
    frequency (there is no default kernel; mixed-frequency data without it
    raised `DataValidationError`): `fit` raises `ConfigurationError` naming
    the symmetric kernel to pass, for example
    `tent_weights={"q:m": [1, 2, 3, 2, 1]}`. Forecasts report every series at
    every clock date, slower series included, as before.
  - `scaler=None` stays the default (the value that ran when the argument was
    omitted); the engine rejects series with a standard deviation above 100
    or below 1e-4, so pass `scaler="standard"` for raw data. Forecasts are in
    data units.
  - The engine orders the series `[clock | slower]` internally;
    `internal_feature_names_` gives that order, which is the order of the
    series-indexed matrices of `result_` (`C`, `R`, `x_sm`). Forecasts and
    `freq_` are in your order.
  - `predict(h, data=new_rows)` (the old `predict(horizon, data=)`: a filter
    from `V_0` on a copy) and `update(new_rows)` (a filter and smoother from
    the last filtered covariance) take only the rows after the last
    conditioning date; an overlap raises `DataValidationError`. They give
    different numbers, as before. `update(data, refit=True)` is the old
    `retrain=True` (one EM iteration on all rows).
  - Shipped YAMLs: `config/dfm_{macro,finance,exchange}.yaml` use the
    `model:`/`data:` layout with the canonical names (`threshold` -> `tol`,
    `num_factors` -> `n_factors`, `frequency` -> `data: freq`). No value that
    ran changes: the dropped keys `clip_ar_coefficients`, `ar_clip_min`,
    `ar_clip_max`, `use_regularization`, `regularization_scale`,
    `augment_idio`, `augment_idio_slow`, `idio_rho0`, `idio_min_var` and
    `scaler` were never read (`ar_clip` and `regularization` ran as None,
    the default), and `scaler: standard` is the `StandardScaler` the tutorials
    passed. `dfm_finance.yaml` lists its series in the CSV's column order;
    the old tutorial ordered them by iterating over a set, which varied from
    run to run. The DFM tutorials no longer use Hydra.
- `__version__` is read from the installed package metadata.
- The package ships a `py.typed` marker.
- Fixed: `compute_cov_safe` with listwise deletion raised on every
  non-square input; `run_em_algorithm` returned non-finite parameters when an
  iteration diverged instead of the last finite ones (runs whose parameters
  stay finite, including those that stop because the log-likelihood is
  non-finite, are unchanged); `summary()` again prints
  the log-likelihood.
- Test suite repaired: tests for removed helpers deleted, the rest updated to
  the current API. It now collects and passes.
- CI (tests on Python 3.10-3.13, ruff, sdist/wheel build check), ruff and
  pre-commit configuration, and pytest `testpaths` added.
- Hygiene: undefined names on error paths fixed; debug `print`/`breakpoint`
  calls removed from `functional/krylov.py`; refactoring-history comment
  prefixes and leftover KDFM references removed; the unused
  `DEFAULT_KDFM_*` and `DEFAULT_INF_VALUE` constants removed.

## 0.7.0

- The Attention Factor Model (`dfm_python.models.afm`: `AFM`, attention and PCA
  factors, LongConv filter, Avellaneda-Lee baseline) now ships in the
  distribution; the 0.6.14 wheel on PyPI did not include it.
- README added and used as the package long description.
- Metadata: project URLs point to `github.com/minkeymouse/dfm-python`.
- `build` and `twine` moved from runtime dependencies to the `dev` extra.
- `__version__` now matches the release; the sdist no longer bundles the example CSVs or the lock file.

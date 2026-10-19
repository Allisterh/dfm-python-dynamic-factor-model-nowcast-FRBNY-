# Changelog

## Unreleased

- **Breaking: estimator API for all four models.** `DFM`, `DDFM`, `iVDFM`
  and `AFM` are estimators with scikit-learn conventions on one data
  container, `PanelData` (slots `y`, `exog`, `series_exog`, `freq`). The
  constructor takes keyword-only hyperparameters and only stores them;
  `fit(data)` returns the model; `predict(h)` and `predict_factors(h)` return
  DataFrames (AFM: `allocate()` and `backtest(data)`); fitted state lives in
  attributes ending in `_` (`result_`, `n_iter_`, `converged_`, `scaler_`,
  `factors_`, ...); `update(data, *, refit=False)` returns the model;
  `save(path)` writes a pickle-free directory that `Model.load(path)` and
  `dfm_python.load(path)` read without data; `get_params`, `set_params`,
  `clone`, `get_config` and `from_config` work on every model, and each
  `*Config` (`DFMConfig`, `DDFMConfig`, `iVDFMConfig`, `AFMConfig`) mirrors
  its constructor and reads the `model:`/`data:` YAML layout. The four
  models, their configs, `PanelData`, `load`, `set_verbosity`,
  `get_verbosity`, `NotFittedError` and `UnsupportedInputError` are exported
  from `dfm_python`. The numbers are
  unchanged: the same data and the same hyperparameters give the same
  parameters, factors, forecasts and diagnostics as 0.7 (golden fixtures
  captured from the old code enforce this, bitwise on CPU with pinned
  threads). The old API is removed, not deprecated; scripts and checkpoints
  written for 0.7 keep working with `pip install "dfm-python==0.7.0"`.

  Old calls and their replacements:

  | Model | Aspect | 0.7 | Now |
  |---|---|---|---|
  | DFM | construct | `DFM(dataset, config=None, scaler=None, **kwargs)` (kwargs mutated the caller's config); `DFMDataset(config, data, time_index)` | `DFM(*, n_factors=None, blocks=None, clock="m", tent_weights=None, max_iter=5000, tol=1e-5, ar_clip=None, data_clip=None, regularization=None, damping_factor=None, scaler=None, verbose=0)`; `PanelData(y, freq=...)` |
  | DFM | fit | `fit(X=None, checkpoint_callback=None) -> DFMModelState`; series not in `[clock \| slower]` order raised | `fit(data, *, checkpoint_callback=None) -> self`; any column order (a stable reorder inside, results in your order) |
  | DFM | predict | `predict(horizon=None, *, data=None) -> (X, Z)` | `predict(h, data=None)` and `predict_factors(h, data=None)`, both DataFrames |
  | DFM | update | `update(data, rescale=False, retrain=False) -> None` (always raised `ModuleNotFoundError`) | `update(data, *, refit=False) -> self` |
  | DFM | results | `training_state`, `_result`, `get_result()`, `result` | `result_`, `n_iter_`, `converged_`, `loglik_`, `factors_`, `scaler_`, `engine_` |
  | DFM | save/load | pickle file; `load(path, config=None, dataset=<required>)`; `load_config` (broken) | a directory; `DFM.load(path)`; `from_config` |
  | DDFM | construct | `DDFM(dataset, config, encoder_size, ..., seed=3)` (18 keywords; `config` ignored except for the interpolation values); `DDFMDataset(data, time_idx, covariates, scaler)` | `DDFM(*, n_factors=4, encoder_layers=(16,), decoder_type="linear", activation="relu", learning_rate=0.005, optimizer="Adam", n_mc_samples=10, window_size=100, max_iter=200, tol=5e-4, interpolation_method="linear", interpolation_limit=10, interpolation_limit_direction="both", scaler=None, random_state=3, device="auto", verbose=0)`; covariates in `PanelData.exog` |
  | DDFM | fit | `fit() -> None`, then a required `build_state_space()` | `fit(data) -> self` (builds the state space) |
  | DDFM | predict | `predict(horizon=None, *, data, update=True, return_series, return_factors)`: ndarray or tuple, mutates with `data` | `predict(h, data=None) -> DataFrame` (pure); `predict_factors(h, data=None) -> DataFrame` |
  | DDFM | update/save | `update(DDFMDataset) -> None`; `torch.save` file, `load` rebuilt a minimal dataset; no device override | `update(data) -> self`; a directory with `weights.pt`; `DDFM.load(path, device=)` |
  | DDFM | results | `get_result()`, `training_state` | `result_`, `factors_`, `n_iter_`, `converged_`, `scaler_`, `network_` |
  | iVDFM | construct | `iVDFM(dataset=None, data_dim, num_factors, context_dim, window, config, ..., seed=3, **kwargs)` (25 keywords; keyword defaults overwrote config values; seeded in `__init__`); `set_dataset()` | `iVDFM(*, n_factors=3, window=None, stride=None, ..., random_state=3, device="auto", verbose=0)`; pure `__init__`; context in `PanelData.exog` |
  | iVDFM | fit | `fit(*args, **kwargs) -> self` on the `set_dataset` data; `fit(data=)` ignored | `fit(data) -> self` |
  | iVDFM | predict | `predict(data=None, context_data=None, context=None, horizon=1, deterministic=True, num_samples=1) -> ndarray`, model space | `predict(h, data=None, *, futr_exog=None, deterministic=True, n_samples=1) -> DataFrame`, data units; `predict_factors(h, data=None, ...)` |
  | iVDFM | update/tune | `update(data_or_dataset, *, append, store_full_history, verbose)`; `hyperparameter_optimization(...)` | `update(data) -> self`; `dfm_python.tuning.tune_hyperparameters(estimator, data, ...)` (the same optuna code) |
  | iVDFM | save/load | `torch.save` checkpoint or weights-only file with a partial config; `load(path, *args, **kwargs)` | a directory with the full config; `iVDFM.load(path, device=)` |
  | AFM | construct | `AFM(dataset=None, config=None, *, n_char, n_factors, ..., seed)` (keyword over config over default through `pick()`); `set_dataset()`; `AFMDataset(returns, characteristics, window, stride, scaler)` | `AFM(*, n_factors=None, factor_model="attention", trading="longconv", hist_len=20, ..., random_state=None, device="auto", verbose=0)`; `PanelData(y, series_exog=...)` |
  | AFM | fit/predict | `fit(X=None) -> AFMModelState`; `predict(horizon=None, *, data=None) -> dict` with batch dimension 1 | `fit(data) -> self`; `allocate(data=None) -> DataFrame`; `backtest(data) -> AFMResult`; `predict` raises `TypeError` |
  | AFM | update/save | `update(data, retrain=False)`; `torch.save` dict, `load` without `map_location` | `update(data, *, refit=False) -> self`; a directory; `AFM.load(path, device=)` |

  - **Removed, not deprecated:**
    - the old constructors (`dataset=`, `config=`, positional datasets) and
      DFM's keyword-over-config merge;
    - `set_dataset`, `build_state_space`, `hyperparameter_optimization` (now
      `dfm_python.tuning.tune_hyperparameters`), `get_result`, `result`,
      `training_state`, and the engines' `reset`;
    - DDFM's `predict` flags `return_series`, `return_factors`, `data` and
      `update`; DFM's `update(rescale=)`; iVDFM's `update(append=,
      store_full_history=)`; DFM's `predict(horizon=None)` default (one year
      of clock periods) with `utils.misc.compute_default_horizon`;
    - `DFM.load_config` (it called a missing `_load_config_common` and could
      not run);
    - the public `DFMDataset`, `DDFMDataset`, `iVDFMDataset` and `AFMDataset`
      (from `dfm_python` and `dfm_python.dataset`, which now exports only
      `PanelData` and `TimeIndex`; they are the estimators' private adapters);
    - `ConfigSource`, `YamlSource`, `DictSource` and `make_config_source`
      (from `dfm_python`, `dfm_python.config` and `dfm_python.config.adapter`):
      use `XConfig.from_yaml(path)` and `X.from_config(config)`, or
      `dfm_python.config.load_yaml(path)` for the `model:` and `data:`
      sections; `detect_config_type` and `BaseModelConfig` are no longer
      exported;
    - the old abstract engine base `dfm_python.models.base.BaseFactorModel`
      (`dfm_python.BaseFactorModel` is the estimator base of all four
      models); `dfm_python.models` exports `DFM`, `DDFM`, `iVDFM`, `AFM` and
      the result classes;
    - the logger modules `train_logger`, `inference_logger`, `ddfm_logger`,
      `ivdfm_logger` and `dfm_logger`, with `BaseTrainLogger`,
      `DFMTrainLogger`, `DDFMTrainLogger`, `BaseInferenceLogger`,
      `DFMInferenceLogger`, `DDFMInferenceLogger`, `iVDFMInferenceLogger` and
      the `log_training_*`, `log_inference_*`, `log_prediction` and
      `log_training_epoch` helpers. `dfm_python.logger` keeps `get_logger`,
      `setup_logging`, `configure_logging`, `set_verbosity`, `get_verbosity`,
      `fit_verbosity` and `quiet_engine_loggers`; the engines' progress
      messages moved, unchanged, to `dfm_python.logger.progress`;
    - flat YAML files (without a `model:` section): `ConfigurationError`;
    - single-file `.pkl` and `.pt` checkpoints: loading one raises
      `ConfigurationError`; load it with the release that wrote it and refit
      and save with this one;
    - the `DFMStateSpaceParams`, `DDFMStateSpaceParams` and
      `DDFMTrainingState` aliases, the `AFMModelState` export, the
      `dfm_python.models.dfm` re-exports of the engine helpers (import them
      from their submodules), and the `utils.checkpoint` helpers
      `infer_ddfm_input_dim` and `infer_ddfm_params_from_state_dict`;
    - the unreachable first definition of `numeric.estimator.estimate_var`.
  - **Renamed hyperparameters** (old names are not accepted: an old keyword
    raises `TypeError`, an old YAML or config key raises
    `ConfigurationError` naming the canonical key): `num_factors` ->
    `n_factors`, `seed` -> `random_state`, `tolerance` and `threshold` ->
    `tol`, `num_regimes` -> `n_regimes`, `context` -> `PanelData.exog`,
    `sequence_length` -> `window`, iVDFM's `*_n_layers` ->
    `*_n_hidden_layers`, `context_data` -> `futr_exog`, `num_samples` ->
    `n_samples`, `target_scaler` -> `scaler`, `frequency` ->
    `PanelData.freq`, DDFM's `disp` -> `verbose` and `encoder_size=(*hidden,
    k)` -> `encoder_layers=hidden` with `n_factors=k`. Inferred in `fit`, no
    longer parameters: `data_dim`, `context_dim`, `n_char`.
  - **Behaviour to check when porting:**
    - `X, Z = model.predict(h)` silently unpacks the two column labels when
      the model has exactly two targets (any other count raises): use
      `predict(h)` and `predict_factors(h)`.
    - `h` is required everywhere; the old per-model defaults are gone.
    - `predict(h, data=...)` never changes the model; the mutating path is
      the explicit `update(data)`.
    - iVDFM forecasts are in data units (inverse-transformed with
      `scaler_`); they were in scaled model space.
    - Every value set in the constructor or YAML reaches the engine: iVDFM's
      keyword defaults no longer overwrite config values, DDFM no longer
      ignores its config, AFM has no `pick()` fallback.
    - Verbosity: the package is silent by default (a `NullHandler` only);
      `set_verbosity(...)`, `DFM_PYTHON_VERBOSITY` or `verbose=1` (progress)
      and `verbose=2` (debug) opt in.
    - Seeds keep their defaults (DFM none, DDFM 3, iVDFM 3, AFM None) and are
      recorded in `config.json`; saving with `random_state=None` warns.
  - **DFM:**
    - Series are `PanelData.y` in any column order, frequencies
      `PanelData.freq` (`{"gdp": "q"}` or `{"q": ["gdp"]}`; unlisted series
      use the clock). The engine orders the series by the stable partition
      `[clock | slower]` (`internal_feature_names_`, the order of the
      series-indexed matrices `C`, `R`, `x_sm` of `result_`); forecasts and
      `freq_` are in your order.
    - Mixed frequency: one frequency slower than the clock; a faster series
      or a second slower frequency raises `UnsupportedInputError`.
      `tent_weights` is required for the slower frequency (there is no default
      kernel; mixed-frequency data without it raised `DataValidationError`):
      `fit` raises `ConfigurationError` naming the symmetric kernel, for
      example `tent_weights={"q:m": [1, 2, 3, 2, 1]}`. Forecasts report
      every series at every clock date, slower series included, as before.
    - `scaler=None` stays the default (the value that ran when the argument
      was omitted; the 0.7 README's claim of internal standardization was
      wrong); the engine rejects series with a standard deviation above 100
      or below 1e-4, so pass `scaler="standard"` for raw data. Forecasts are
      in data units.
    - Fixed: `update()` always raised `ModuleNotFoundError` (it imported a
      validator from a module path that does not exist); it now runs.
    - `predict(h, data=new_rows)` (the old `predict(horizon, data=)`: a
      filter from `V_0` on a copy, forecasting from the last filtered state)
      and `update(new_rows)` (a filter and smoother from the last filtered
      covariance) take only the rows after the last conditioning date; an
      overlap raises `DataValidationError`. They give different numbers, as
      before. `update(data, refit=True)` is the old `retrain=True` (one EM
      iteration on all rows) and refreshes `result_`.
    - The old `DFMConfig` (the engine's config) is private, and its
      never-read `idio_rho0` and `idio_min_var` are gone.
  - **DDFM:**
    - `result_.C` is the loading matrix H, N x r (series x factors), the DFM
      orientation; it was H transposed (factors x N).
    - `update(data)` and `predict(h, data=)` take the full panel (training
      rows plus new ones). Known limitation, kept for identical numbers:
      refitting the target scaler inverse-transforms the training targets
      only for a scaler with `mean_` and `scale_`; with `"minmax"`,
      `"robust"`, `"maxabs"`, `"quantile"` or another transformer it refits
      on scaled training targets stacked with raw new ones, so `update` and
      `predict(h, data=)` are only meaningful with `scaler="standard"` or
      `None`. `update(data, refit=True)` raises `UnsupportedInputError`.
    - With `scaler=None` the forecasts are in model space (the engine's
      forecast, untransformed); the old `predict` raised without a scaler.
    - Dropped as dead (stored but never read): `max_epoch_pre_train`
      (pre-training runs `n_mc_samples` epochs), `min_target_interpolate_ratio`
      and the `DDFMConfig` fields `use_batch_norm`, `use_idiosyncratic`,
      `min_obs_idio` and `lags_input`.
  - **iVDFM:** `stride=None` means `window // 2`, the only stride that
    reached training. With `factor_order > 1` and `n_regimes == 1`,
    `n_factors` must not exceed the number of targets (`fit` raises
    `ConfigurationError`; the engine's AR initialization made the ELBO NaN).
    `update(data)` encodes with the fixed weights and appends the factors;
    `update(data, refit=True)` raises `UnsupportedInputError`.
  - **AFM:** `allocate(data=None)` returns the weights on
    `data.index[hist_len:]`; `update(data)` attaches data for `allocate()`,
    and `update(data, refit=True)` continues training from the current
    weights (the old `retrain=True`). A loaded model keeps no data:
    `allocate(data=...)`.
  - **Shipped YAML values that now take effect.** Every file in `config/`
    uses the `model:`/`data:` layout with the canonical names. Because
    constructor defaults no longer override the file, running a shipped
    YAML can give different numbers than the same file under 0.7:
    - `config/ivdfm_macro.yaml`: `n_factors: 3` (ran as 5),
      `factor_order: 3` (1), `encoder_n_hidden_layers` and
      `decoder_n_hidden_layers: 3` (2), `prior_n_hidden_layers: 2` (1),
      `tol: 1.0e-5` (5.0e-4).
    - `config/ivdfm_finance.yaml`: `n_factors: 1` (5),
      `encoder_hidden_dim` and `decoder_hidden_dim: 50` (200),
      `factor_order: 2` (1), `prior_n_hidden_layers: 2` (1), `tol: 1.0e-5`
      (5.0e-4).
    - `config/ivdfm_exchange.yaml`: `encoder_n_hidden_layers` and
      `decoder_n_hidden_layers: 3` (2), `prior_n_hidden_layers: 2` (1),
      `tol: 1.0e-5` (5.0e-4).
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
    - `config/dfm_{macro,finance,exchange}.yaml` and
      `config/afm_finance.yaml`: no value that ran changes. The dropped DFM
      keys `clip_ar_coefficients`, `ar_clip_min`, `ar_clip_max`,
      `use_regularization`, `regularization_scale`, `augment_idio`,
      `augment_idio_slow`, `idio_rho0`, `idio_min_var` and `scaler` were
      never read (`ar_clip` and `regularization` ran as None), `scaler:
      standard` is the `StandardScaler` the tutorials passed, and AFM's
      dropped `clock` and `scaler` were never read. `dfm_finance.yaml` lists
      its series in the CSV's column order (the old tutorial ordered them by
      iterating over a set, which varied from run to run).
  - The tutorials are rewritten on the new API (no Hydra; `--rows`,
    `--max-iter`/`--max-epochs`, `--run-dir`, `--device`), and the README
    documents the API for all four models; its examples run in the test
    suite.
- **Breaking (installation):** a bare `pip install dfm-python` no longer
  installs optuna, sktime, omegaconf, hydra-core, pyarrow, matplotlib or
  pytorch-lightning. Code that used them through dfm-python now raises an
  `ImportError` naming the extra to install: `calculate_rmse`/`calculate_mae`
  need `[metrics]`, `dfm_python.tuning` needs `[tune]`,
  reading YAML configs needs `[yaml]` (omegaconf; PyYAML also works),
  polars input needs `[polars]` (polars and pyarrow), and the `tutorial/`
  scripts need `[yaml]`.
  **Migration:** to keep the 0.7.0 behaviour, depend on `dfm-python[all]`
  (for example in a `requirements.txt` that pinned bare `dfm-python`).
- Runtime dependencies reduced to numpy, pandas, scipy, scikit-learn, torch,
  pykalman and einops. pytorch-lightning and matplotlib were never imported
  and are dropped; optuna, omegaconf, sktime and polars (with pyarrow) moved
  to the `tune`, `yaml`, `metrics` and `polars` extras (`all` installs every
  extra). hydra-core is no longer a dependency of any extra: the tutorials
  stopped using Hydra, and the short-lived `examples` extra is removed.
- **Breaking (extras):** the `db`, `deep`, `transform`, `forecasting` and
  `hydra` extras are removed. pip only warns about an unknown extra, so
  `dfm-python[hydra]` now silently installs neither omegaconf nor hydra-core;
  use `[yaml]` instead (`[metrics]` replaces `[transform]` and
  `[forecasting]`).
- The `dev` dependency group lists its packages directly (it works with
  `pip install --group dev` as well as uv) and includes the optional packages
  the test suite imports.
- CI runs every `tutorial/` script on truncated data with one iteration or
  epoch (the `tutorials` job).
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

# Changelog

## 0.8.0

First release with the estimator API. It also replaces the DFM E-step and
M-step for numerical stability and makes DDFM work under pandas 3.

### DFM numerical stability

DFM numerical-stability rewrite. **DFM numbers change**: fitted parameters,
factors, forecasts, log-likelihoods, iteration counts and convergence flags
all differ from earlier 0.8.0 development builds and from 0.7. **DDFM, iVDFM
and AFM numbers are unchanged**: their golden fixtures are bitwise identical.
Every change to the DFM numbers is checked against an independent oracle (a
reference Kalman filter in the test suite that does not import the package,
and statsmodels' `KalmanSmoother` when installed). The public estimator API is
unchanged.

- **DFM E-step rewritten.** `DFMKalmanFilter` is an in-house float64 Kalman
  filter and Durbin-Koopman smoother (the lag-one covariances come from the
  same recursion). Missing values are handled per element: at each row the
  observed entries are used and only the missing ones are skipped. Before,
  any NaN in a row dropped the whole row from the update, so on mixed-frequency
  panels the E-step ignored most of the data. The update uses a Cholesky
  factor of the innovation covariance and a Joseph-form covariance. The ad-hoc
  E-step stabilizations are removed: diagonal loading of the filtered and
  predicted covariances, eigenvalue caps, the condition-number guard (which
  never fired) and the clipping of smoothed states. A failing filter raises
  `NumericalError`.
- **Exact log-likelihood.** `result_.loglik` (and `loglik_`) is the exact
  Gaussian log-likelihood of the observed entries, computed in the filter
  pass, at the returned parameters. Before, every panel with a NaN reported
  exactly `-T/2` (`-60.0` for 120 rows, `-57.0` for 114, `-200.0` for 400),
  a side effect of passing a NaN-filled array to pykalman, and EM then
  "converged" after 3 iterations on every mixed-frequency panel. On fully
  observed data the old value came from the E-step before the last M-step.
- **pykalman removed** from the dependencies; the runtime dependencies are
  numpy, pandas, scipy, scikit-learn, torch and einops.
- **EM M-step is the closed-form Banbura-Modugno (2014) update**, with the
  tent restriction on slower-frequency loadings solved as restricted least
  squares. Defects fixed:
  - off-by-one alignment: the loading and `R` updates paired data row `t`
    with the smoothed state of row `t+1`;
  - the C-normalization "invariance" rescale went the wrong way and
    symmetrized `A`, so it changed the likelihood; normalization now runs
    once after EM, per factor, with the correct transform
    (`normalize_C=True`), and never inside the loop;
  - the `A` update zeroed the companion shift (lag identity) rows, which made
    the lag states deterministic zeros and the loading solves singular
    (rcond ~1e-21); only the current-factor rows are estimated now (a VAR(p)
    on the r*p lag columns);
  - the clock loading update subtracted the idiosyncratic cross-moments
    summed over all series from every series; they are per series now;
  - the idiosyncratic AR(1) update left out the lag-one smoothed covariance;
  - moment sums were capped at 1e3, `A`, `Q` and `V_0` eigenvalues at 0.99,
    `R` clipped, and the solves ridge-regularized and run in float32; all
    removed (float64, Cholesky solves, a singular system raises);
  - the single-frequency fast path for `C` dropped the idiosyncratic
    cross-moment and the missing-data weights; it is removed.
- **BM14 state layout for slower-frequency series.** A slower-frequency
  series now has only its tent-weighted idiosyncratic chain (a weekly AR(1)
  state is kept only for clock-frequency series), the chain loadings are the
  unnormalized tent weights, and the lag and chain shift states have exactly
  zero noise. The state dimension shrinks (for example 28 -> 26 states for the
  monthly/quarterly example, 44 -> 37 for weekly/monthly).
- **Initialization consistent with the model:** exact tent-restricted
  least-squares initial loadings, no column normalization of `C`, `R`
  initialized at the M-step floor (1e-4) instead of 1e-8, nonzero chain
  noise, `V_0` from the exact Lyapunov solve (with a finite diffuse part on
  unit-root directions). Two init bugs fixed: `build_lag_matrix` copied the
  current factor into every lag column, and the pairwise branch of
  `compute_cov_safe` indexed time points instead of variables.
- **EM driver on the exact log-likelihood.** Convergence is tested on the
  relative change of the exact log-likelihood, which does not decrease over
  the iterations. If, after the first iteration, it falls (by more than 1e-6
  relative), or a parameter or the filter becomes non-finite, EM stops and
  returns the best parameters seen, with `converged=False`; the metadata of
  `functional.em.run_em_algorithm` gains `stop_reason` and `best_loglik`. On `data/macro.csv` with
  `config/dfm_macro.yaml` (the last 300, 400 and 600 rows and the full panel)
  EM no longer diverges or leaves a non-finite filter, so `save` no longer
  writes a NaN `last_smoothed_cov` and `update` no longer raises there.
- **`update()` and `predict(h, data=)` continue the training filter**: the
  prior of the first new row is `(A x_T|T, A P_T|T A' + Q)` from the filter
  over the conditioning rows. Before, `update` skipped that transition step
  for several new rows (but not for one), and `predict(data=)` restarted from
  `V_0`. One new row and many new rows now take the same path, and
  `predict(h, data=new)` equals `update(new).predict(h)`. A single new row in
  `predict(h, data=)` used to raise a broadcasting `ValueError`.
  `update(data, refit=True)` still runs one EM iteration on all rows, now
  through the same EM driver.
- **Save directories:** DFM save directories written by development builds of
  0.8.0 before this change do not load (the state layout changed);
  `DFM.load` raises `ConfigurationError` asking to fit the model again.
  `FORMAT_VERSION` is unchanged because 0.8.0 was never released.
- **DDFM under pandas 3 (copy-on-write):** `DDFM.fit` raised "assignment
  destination is read-only" with pandas>=3 (allowed by `pandas>=2.0.0`); the
  writes now go through the frame, with unchanged numbers. Under pandas 3 the
  iVDFM time features now use nanoseconds as under pandas 2 (pandas 3 parses
  dates in microseconds, which moved them by ~1e-6).
- **Golden fixtures:** only the five DFM fixtures were recaptured, each
  checked against the reference filter (the reported log-likelihood equals
  the reference's to 6e-16 relative; all arrays finite):

  | Case | log-likelihood before (reported / true at those parameters) | after | EM iterations before -> after |
  |---|---|---|---|
  | `dfm_synthetic` (80 rows, no NaN) | -542.6755 / -542.6347 | -274.9745 | 20 -> 20 |
  | `dfm_macro_monthly_quarterly` (120 rows) | -60.0 / -1257.3704 | -998.5058 | 3 -> 10 |
  | `dfm_macro_monthly_quarterly_permuted` | -60.0 / -1257.3704 | -998.5058 | 3 -> 10 |
  | `dfm_macro_monthly_quarterly_update` (114 rows) | -57.0 / -1242.8559 | -941.1799 | 3 -> 10 |
  | ... its refit on 120 rows | -60.0 / -1566.0253 | -991.5799 | 1 -> 1 |
  | `dfm_macro_weekly_monthly` (400 rows) | -200.0 / -69446964.17 | -1639.5278 | 3 -> 10 |

  The old refit of `dfm_macro_monthly_quarterly_update` reached smoothed
  states of 9e41 and forecasts of 1e6; now 4.8 and 11. The five DFM cases
  agree within 1e-6 relative (2.5e-10 at most, norm-relative, on parameters
  and forecasts) between the default AVX-512 kernels and forced AVX2
  (OpenBLAS Haswell, Sandybridge) or SSE (Nehalem) kernels, so
  `dfm_macro_weekly_monthly` and `dfm_macro_monthly_quarterly_update` are
  compared across platforms again (`cross_platform=True`). Before, forcing
  AVX2 kernels moved `A` of the weekly/monthly case by 0.99 absolute.
- Tests: oracle tests for the E-step, EM monotonicity, the M-step Q-function
  and update/predict(data=); statsmodels is in the `dev` group for the
  optional cross-check.

### Estimator API

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
  from `dfm_python`. The migration itself left the
  numbers unchanged: the same data and the same hyperparameters give the same
  parameters, factors, forecasts and diagnostics as 0.7 (golden fixtures
  captured from the old code enforce this, bitwise on CPU with pinned
  threads). DFM's numbers then changed on purpose with the numerical-stability
  fixes above; DDFM's `update` and `predict(h, data=)` changed with the
  bug fixes below (its fit and `predict(h)` still match 0.7), and iVDFM and
  AFM still match 0.7. The old API is removed, not deprecated; scripts and checkpoints
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
      overlap raises `DataValidationError`. They gave different numbers, as
      before (now both continue the training filter and agree; see
      numerical-stability rewrite). `update(data, refit=True)` is the old `retrain=True` (one EM
      iteration on all rows) and refreshes `result_`.
    - The old `DFMConfig` (the engine's config) is private, and its
      never-read `idio_rho0` and `idio_min_var` are gone.
    - Known engine issues, left in place here (the engine was frozen for
      identical numbers) and fixed by the numerical-stability rewrite, except the one-iteration
      refit, which is kept: mixed-frequency fits
      report `result_.loglik` as exactly `-T/2` (`-57.0` for the README's
      114-row example), which is not a usable log-likelihood; on some
      mixed-frequency panels (truncated slices of `data/macro.csv`, or the
      full file with `config/dfm_macro.yaml`) EM diverges or leaves
      parameters under which the Kalman filter is not finite, so `save`
      warns and stores a NaN `last_smoothed_cov` and `update` raises; one or
      two EM iterations on short weekly/monthly slices can give explosive
      forecasts; `update(data, refit=True)` runs one EM iteration, not a
      full refit.
  - **DDFM:**
    - `result_.C` is the loading matrix H, N x r (series x factors), the DFM
      orientation; it was H transposed (factors x N).
    - `update(data)` and `predict(h, data=)` take only the rows after the
      last conditioning date and never refit the scalers (see "Bug fixes
      before release" below). `update(data, refit=True)` raises
      `UnsupportedInputError`.
    - With `scaler=None` the forecasts are in model space (the engine's
      forecast, untransformed); the old `predict` raised without a scaler.
    - Dropped as dead (stored but never read): `max_epoch_pre_train`
      (pre-training runs `n_mc_samples` epochs), `min_target_interpolate_ratio`
      and the `DDFMConfig` fields `use_batch_norm`, `use_idiosyncratic`,
      `min_obs_idio` and `lags_input`.
  - **iVDFM:** `stride=None` means `window // 2`, the only stride that
    reached training. `n_factors` may exceed the number of targets with any
    `factor_order` (see "Bug fixes before release" below).
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
  pykalman and einops (pykalman was then dropped; see DFM numerical stability above).
  pytorch-lightning and matplotlib were never imported and are dropped; optuna, omegaconf, sktime and polars (with pyarrow) moved
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

### Bug fixes before release

- **DFM non-stationarity warning.** The transition matrix `A` is no longer
  capped (the 0.99 eigenvalue cap biased the likelihood), so on
  non-stationary inputs the maximum-likelihood `A` can be explosive and
  long-horizon forecasts grow geometrically. `DFM.fit` now issues a
  `UserWarning` naming the spectral radius when that of the fitted
  `result_.A` is above 1, and the README says to difference or transform
  series to stationarity first. No numbers change; the stationary golden DFM
  fits (largest spectral radius 0.89) do not warn.
- **DFM `max_iter` default kept at 5000.** With the exact EM, the golden DFM
  panels converge at `tol=1e-5` in 67 to 566 iterations (`dfm_synthetic` 501,
  `dfm_macro_weekly_monthly` 566, the monthly/quarterly cases 67-68); a
  default of 100 would stop the first two before convergence, so the default
  is unchanged. `tol=1e-5` is the threshold of the EM's relative
  log-likelihood change test and is unchanged. A test now checks that the
  documented defaults (docstring, `DFMConfig`, this changelog) equal
  `DFM().get_params()`.
- **DDFM `update` and `predict(h, data=)` fixed.** **DDFM update numbers
  change; its fit and `predict(h)` do not** (the fit outputs of every DDFM
  golden fixture are bitwise identical). One semantics now, the DFM one
  (tags `update_kind="encoder_forward"`, `predict_data="new_rows"`, before
  `"encoder_forward_rescale"` and `"full_panel"`):
  - `data` holds only the rows after the last conditioning date; rows that
    overlap it raise `DataValidationError` (a `RangeIndex` must continue the
    model's), and `conditioning_index_` is extended by `data.index`. Before,
    `data` was the full panel and the factors of every row were appended
    after the training factors, so the training rows were appended twice.
  - The scalers fitted in `fit` scale the new rows and are never refitted:
    `scaler_` for the targets, and for the covariates the `type(scaler)()`
    instance fitted on the training covariates, which the engine now keeps
    (and `save` writes to `state.npz`/`state.json` under
    `covariate_scaler`). Before, the target scaler was refitted on the
    training rows plus the whole panel (repeating the training rows), and
    the training targets were inverse-transformed only for a scaler with
    `mean_` and `scale_`, so `"minmax"`, `"robust"`, `"maxabs"`,
    `"quantile"` and custom scalers were refitted on mixed units (with
    `"minmax"`, forecasts of -3.0, 11.2 and -14.0 for series between 46 and
    64; now 46.2, 52.1 and 54.8); the covariates were scaled with a scaler
    fitted on the update panel.
  - Missing values in the new rows are filled as `fit` fills its data:
    interpolated with the interpolation settings together with the
    conditioning rows (so a gap at the boundary is interpolated from the
    earlier rows), gaps left over carried forward. Before, the un-interpolated
    targets overwrote the interpolated ones, so a missing target reached the
    encoder as NaN, that row's factors were NaN and the forecast started from
    a zero state.
  - `predict(h, data=new)` equals `update(new).predict(h)`, bitwise, and
    leaves the model unchanged; successive updates give the same factors as
    one update with all their rows.
  - Golden fixtures: only the update outputs (`update_predict_X`,
    `update_predict_Z`, `update_factors`, `update_scaler_mean`,
    `update_scaler_scale`) were recaptured. `ddfm_exchange` and
    `ddfm_finance` now update with their 12 held-out rows instead of the
    whole panel; `ddfm_macro` has no rows after the 400 it fits, so it no
    longer has update outputs (its test checks that an update with the
    fitted rows raises).
- **iVDFM with more factors than targets and `factor_order > 1`.** The
  engine initialized the AR coefficients by OLS on the PCA factors of the
  last training window, zero-padded to `n_factors` when there were fewer PCA
  factors (`n_factors` above the number of targets, or above the window
  length). With `factor_order > 1` the padded factors got all-zero
  coefficient rows, which the companion SSM's L1 normalization divided by
  zero, so training stopped with "NaN/Inf in loss" in the first batch, and
  the estimator rejected the combination with `ConfigurationError`. The
  factors without a PCA factor now keep the SSM's own (non-data) random
  initialization of their AR coefficients and innovation scale (the
  initialization used when the engine does not initialize from data), the
  normalization leaves a row with zero norm unnormalized instead of
  dividing it by zero, and the `ConfigurationError` is removed. With
  `factor_order == 1` the padded factors keep their OLS values (a stable
  AR(1)), as before. `config/ivdfm_macro.yaml` (3 targets, `factor_order: 3`)
  and `config/ivdfm_finance.yaml` (1 target, `factor_order: 2`) are back to
  `n_factors: 5`; both train on the full data with finite ELBO and
  forecasts. No numbers change otherwise: every iVDFM golden fixture is
  bitwise identical (`ivdfm_macro` has 5 factors on 3 targets but
  `factor_order` 1; `ivdfm_regimes` and `ivdfm_exog_context` have no more
  factors than targets).
- **iVDFM tuning is reproducible.** `tune_hyperparameters` created its optuna
  study with the default, unseeded TPE sampler, so two searches with the same
  `random_state` drew different trials and could return different `best`
  values. The sampler is now `TPESampler(seed=random_state)`, seeded from the
  trials' `random_state` (`fixed["random_state"]` if given, else
  `estimator.random_state`); `random_state=None` keeps the unseeded sampler.
  The function signature is unchanged (it has no seed argument).
- **No optuna step warning for `window` and `stride`.** Both are suggested in
  steps of 10; when `max_window - min_window` (or `1000` minus the lowest
  stride, e.g. 12 for `fixed={"window": 25}`) was not a multiple of 10,
  optuna warned and lowered the upper bound to the last step. The bound is
  now lowered before the call, to the same value, so the search space and
  the draws are unchanged and nothing warns; ranges that are multiples of 10
  are untouched. No golden fixture covers tuning; none changed.

## 0.7.0

- The Attention Factor Model (`dfm_python.models.afm`: `AFM`, attention and PCA
  factors, LongConv filter, Avellaneda-Lee baseline) now ships in the
  distribution; the 0.6.14 wheel on PyPI did not include it.
- README added and used as the package long description.
- Metadata: project URLs point to `github.com/minkeymouse/dfm-python`.
- `build` and `twine` moved from runtime dependencies to the `dev` extra.
- `__version__` now matches the release; the sdist no longer bundles the example CSVs or the lock file.

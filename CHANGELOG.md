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

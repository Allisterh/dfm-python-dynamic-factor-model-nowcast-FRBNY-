# API redesign specification

This document specifies how the four models in `dfm_python` (DFM, DDFM, iVDFM, AFM) become one consistent library. It changes packaging, naming and the order of calls. It does not change numerical code: EM with pykalman, MCMC denoising, the ELBO loop and the Sharpe objective are called unchanged. Two published papers depend on these numbers, so every decision below either keeps today's numbers bit for bit on CPU or names the staged change and the test that guards it.

Releases are called "the next minor release" (both APIs work) and "the release after" (the old API is removed). No version numbers are assigned here.

## 1. Summary

The target follows sklearn fitted-attribute conventions, neuralforecast's `fit(data)`, and Hugging Face config and save conventions:

- `Model(**hyperparameters)` stores hyperparameters and does nothing else.
- `fit(data)` returns `self`. `data` is one shared container, `PanelData`.
- `predict(h)` returns a time-indexed `pandas.DataFrame`. AFM, which allocates rather than forecasts, uses `allocate()`.
- Fitted state lives in attributes ending in `_`. The fitted result has one home, `result_`.
- `save(dir)` writes a pickle-free directory. `Model.load(dir)` and `dfm_python.load(dir)` need no dataset.

The current classes are not rewritten. Each moves verbatim into a private engine module, and the public class is a thin estimator that builds the engine's inputs from `PanelData` and passes every hyperparameter explicitly. The new path therefore runs today's code, and golden fixtures captured before any change prove parity. In the next minor release, a constructor called the old way returns the untouched legacy object, so existing scripts, including the papers', run unchanged. A permanent `dfm_python.compat` module reproduces today's hyperparameter precedence, and saved runs record the values that actually ran.

### Decisions on contested points

| Question | Decision and reason |
|---|---|
| iVDFM keyword defaults (`factor_order=1`, `encoder_hidden_dim=200`, `decoder_var`, `tolerance`, `seed=3`, ...) overwrite config values; DDFM ignores its config except interpolation | Legacy mode keeps this. New mode passes every value explicitly, so what you set is what runs. `compat.legacy_params()` converts an old config plus keywords into effective parameters and lists each divergence. An old-style YAML loaded through the new API warns per diverging field. A migrated YAML therefore cannot silently start applying values that never ran. |
| iVDFM forecasts are in scaled model space | The next minor release keeps the values, labels them (`attrs["space"] = "model"`, tag `forecast_space`) and warns once. The release after inverse-transforms by default; `space="model"` keeps the old values permanently. |
| DFM column order `[clock | slower]` | `fit` applies a stable partition (relative order kept within each group) and reports in the user's order. Ordered input gives the identity permutation, so results are bitwise unchanged. Any other order raises today, so no result depends on it. The internal order is exactly what the error message asks users to build by hand, so no new permutation sensitivity appears. A golden test on permuted input pins this. |
| Meaning of `update()` | One promise: after `update(data)`, `predict` conditions on `data`, and fitted parameters change only with `refit=True`. The mechanism stays per model and is published as tag `update_kind` and one docstring table. Unifying the mechanism would change numbers. |
| Default horizon | `h` is required in new mode. Legacy mode keeps today's defaults (one year of clock periods, 6, 1). Code that omits it is legacy code, so no output shape changes. |
| AFM's `predict` | AFM gets `allocate()` and `backtest()`. In new mode `AFM.predict` raises `TypeError` naming `allocate`. `predict(h)` then means the same thing everywhere it exists. |
| iVDFM `stride` | `stride=None` means `window // 2`, the dataset default and the only stride that ever reached training (the config's `stride=1` is never read by `fit`). Owner to confirm. |
| Pre-fitted scaler | New mode clones and fits `scaler`, and warns if a fitted instance's statistics are discarded. Legacy mode keeps today's use-as-is path. |
| DFM load without a dataset | Each engine gains a private `_from_saved` that bypasses `__init__` (DFM's `__init__` derives blocks from dataset columns). Blocks, frequencies, column orders, `data_processed` and the last smoothed state are saved. |
| Parity scope | Bitwise on CPU with one BLAS thread, `torch.set_num_threads(1)` and deterministic algorithms. GPU runs are compared to a tolerance. |
| Seed defaults (DDFM 3, iVDFM 3, AFM None, DFM none) | Kept per model, documented in one table, and always recorded in `config.json`. |
| iVDFM tutorials (broken today) | Rewritten on the new API with effective-value YAMLs. Legacy `fit(dataset)` and `fit(data=dataset)` are forwarded with a warning. Nothing ran through them, so no number depends on them. |
| Tuple unpacking `X, Z = model.predict(h)` | Not grafted into new mode: a hybrid return type is what this redesign removes. Legacy objects keep returning tuples. New mode returns a DataFrame, and factors come from `predict_factors`. |

## 2. Goals and non-goals

Goals:

1. A user who knows sklearn or neuralforecast can guess the API.
2. One data container for all four models, AFM included.
3. One way to set each hyperparameter: a keyword-only constructor argument. The `kwargs > config > defaults` merge, AFM's `pick()` and the `getattr(config, ...)` fallbacks leave the public path.
4. One result location, one save format, and one meaning each for `device`, `random_state` and `verbose`.
5. Bitwise CPU parity with today's code, enforced by golden fixtures captured first.
6. An open door to a later sktime-like ecosystem through tags, a registry and a public `check_estimator`, with no sktime dependency or compatibility claim.

Non-goals:

- No new models, losses, inference algorithms, probabilistic forecasts, backtesting framework or trainer.
- No change to estimation code, initialization order or RNG consumption order.
- No sklearn `check_estimator` compliance; only `BaseEstimator` machinery is reused.
- No `safetensors`, no Hub upload, and no bundling of training data in saved runs.
- No cleanup of untyped access inside the frozen engines (most of the 76 `hasattr` calls).

## 3. Public API at a glance

### DFM (numpy, mixed frequency, EM + Kalman)

```python
import numpy as np, pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2010-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((120, 4)),
                  index=idx, columns=["gdp", "s0", "s1", "s2"])
df.loc[df.index.month % 3 != 0, "gdp"] = np.nan        # quarterly series, any column position

data = PanelData(y=df, freq={"gdp": "q"})                # unlisted series use the clock frequency
model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
            max_iter=50, tol=1e-5, scaler="standard")    # blocks=None: one global block
model.fit(data)                                          # returns model
fc = model.predict(6)                                    # DataFrame 6 x 4, columns in your order
zf = model.predict_factors(6)                            # DataFrame 6 x state_dim
model.result_.loglik, model.n_iter_, model.converged_
model.save("runs/dfm_macro")
model = DFM.load("runs/dfm_macro")                       # no dataset needed
```

### DDFM (torch autoencoder + MCMC denoising)

```python
from dfm_python import DDFM, PanelData

data = PanelData.from_wide(raw_df, time_col="date", exog_cols=["oil", "fx"])  # other columns are targets
model = DDFM(n_factors=4, encoder_layers=(16,), decoder_type="linear",
             learning_rate=0.005, n_mc_samples=10, max_iter=200, tol=5e-4,
             scaler="standard", random_state=3, device="cpu", verbose=1)
model.fit(data)                                          # builds the state space too
fc = model.predict(6)                                    # targets only
fc_new = model.predict(6, data=data_with_new_rows)       # conditions on new rows; model unchanged
model.update(data_with_new_rows)                         # explicit, mutating; returns model
model.save("runs/ddfm_macro")
```

### iVDFM (torch, identifiable VAE on innovations)

```python
from dfm_python import iVDFM, PanelData
from dfm_python.tuning import tune_hyperparameters       # needs the `tune` extra

data = PanelData.from_wide(raw_df, time_col="date", exog_cols=["policy_rate"])  # exog = context u_t
model = iVDFM(n_factors=5, factor_order=3, window=200, stride=None,
              innovation_distribution="laplace", learning_rate=1e-3,
              batch_size=256, max_epochs=10, scaler="standard",
              random_state=3, device="auto")
model.fit(data)
fc = model.predict(6, futr_exog=future_policy_df)        # fc.attrs["space"] == "model" for now
best = tune_hyperparameters(model, data, n_trials=30, horizon=96, metric="sMSE")
model.set_params(**best).fit(data)
model.save("runs/ivdfm_macro")
```

### AFM (torch, attention factors for statistical arbitrage)

```python
from dfm_python import AFM, PanelData

data = PanelData.from_arrays(y=returns, series_exog=characteristics,        # (T, N), (T, N, M)
                             index=dates, names=tickers, series_exog_names=char_names)
model = AFM(n_factors=3, hist_len=20, max_epochs=20, random_state=0, device="auto")
model.fit(data)
w = model.allocate()                                     # DataFrame (T - hist_len) x N of weights
res = model.backtest(test_data)                          # AFMResult: residuals, weights, book returns, Sharpe
model.result_.sharpe, model.residuals_, model.book_returns_
AFM(n_factors=3, factor_model="pca", trading="ou").fit(PanelData(y=returns_df))
model.save("runs/afm")
```

## 4. Data container

`dfm_python.PanelData` is a frozen, validated container that uses only numpy and pandas. It holds no device or scaler, and it never imputes, scales or reorders values.

| Slot | Type and shape | Meaning | Used by |
|---|---|---|---|
| `y` | DataFrame, T x N | observed target series (macro panel or asset returns) | all |
| `exog` | DataFrame, T x K, or None | per-time inputs that are not forecast | DDFM (encoder covariates), iVDFM (context u_t) |
| `series_exog` | DataFrame, T x (M*N), column MultiIndex `(feature, series)`, or None | per-time, per-series features | AFM (characteristics) |
| `freq` | `dict[str, str]` or None | per-series sampling frequency | DFM |

Constructors: `PanelData(y, exog=None, series_exog=None, freq=None)`, `from_wide(df, *, time_col, target_cols, exog_cols, freq)`, `from_long(df, *, id_col="unique_id", time_col="ds", target_col="y", series_exog_cols=(), exog_cols=())`, and `from_arrays(y, *, exog, series_exog, index, names, exog_names, series_exog_names, freq)`. Accessors: `index`, `series_names`, `exog_names`, `series_exog_names`, `n_timesteps`, `is_mixed_frequency`, `series_exog_array()` (T x N x M), `to_long()` and `loc(start, stop)`. Polars input works with the `polars` extra. `fit`, `predict` and `update` also accept a bare DataFrame (meaning `PanelData(y=df)`) or a 2-D array.

Validation:

- The index is unique and increasing, and is a `DatetimeIndex`, `PeriodIndex` or `RangeIndex`. Arrays get a `RangeIndex`.
- All slots share the index.
- `freq` accepts the per-series form `{"gdp": "q"}` and the grouped form `{"q": ["gdp"]}`, normalized in one place. Today the `BaseModelConfig` docstring recommends the grouped form, but the constructor rejects it.

AFM needs no special signature. In long format a firm characteristic is just another column on each `(unique_id, ds)` row, which makes it a per-series exogenous variable. `AFMDataset`'s `window`, `stride` and `scaler` are stored but never read by `fit` or `get_tensors`, so dropping them loses nothing.

A slot the model's tags exclude raises `UnsupportedInputError` and is never silently ignored. Examples: `exog` given to DFM, `series_exog` given to DDFM, a mixed-frequency panel given to iVDFM, or attention-mode AFM without `series_exog`.

The four `*Dataset` classes become private adapters that `fit` builds from `PanelData` and the hyperparameters. Each calls today's constructor with the same arguments in the same order:

- DDFM builds `[exog | y]` in that column order. The existing code scales the covariates with a fresh `type(scaler)()`.
- iVDFM runs today's interpolation, then forward and backward fill, then zero fill, then windowing.
- DFM applies the stable column partition and builds the `DFMConfig` that its dataset requires.

Scaling has one public owner, the `scaler` hyperparameter, and the fitted copy is exposed as `scaler_`. Inside the engines the fitted scaler stays where it is today, and `scaler_` points at it.

## 5. Config

### The constructor is the config

Every public model subclasses `sklearn.base.BaseEstimator`. scikit-learn is already a core dependency. `__init__` takes keyword-only hyperparameters and only stores them, with no validation, seeding, device work or data access. That gives `get_params`, `set_params`, `clone`, `__repr__` and Hydra `instantiate` for free. Validation runs at the start of `fit`.

The `*Config` dataclasses stay as the typed, serializable mirror of the constructor, in the role of Hugging Face's `PretrainedConfig`. `get_config()` returns one whose fields equal `get_params()`. `from_config(cfg)` is equivalent to `Model(**cfg.to_dict())`. `from_yaml` (with the `yaml` extra), `to_yaml`, `to_dict` and `from_dict` round-trip.

Unknown keys warn in the next minor release, naming the key and the closest canonical name, and raise `ConfigurationError` in the release after. One alias table maps legacy keys with a warning: `sequence_length` to `window`, `threshold` to `tol`, `num_factors` to `n_factors`, `encoder_n_layers` to `encoder_n_hidden_layers`, the flat DFM clipping keys, and so on.

### New-style and old-style YAML

A new-style YAML has a `model:` section (constructor parameters) and a `data:` section (`time_col`, `exog_cols`, `freq`). A flat file is old style. Loading one through the new API runs it through `compat.legacy_params` and emits a `LegacyPrecedenceWarning` for each field whose effective value differs from the file. For example, `ivdfm_macro`'s `factor_order: 3` ran as 1, and `ivdfm_finance`'s `encoder_hidden_dim: 50` ran as 200. The effective values are returned; `effective=False` takes the file literally.

The shipped YAMLs are regenerated in new style by a script that applies `compat.legacy_params`, so they hold the values that actually ran. A parity test checks that new YAML plus new API equals old YAML plus legacy API on the golden fixtures. The old flat files stay as the record of the published runs.

### Legacy precedence shim

`compat.legacy_params(model_cls, *, config=None, dataset=None, **kwargs)` reproduces today's resolution exactly:

- iVDFM: keyword defaults beat the config.
- DDFM: constructor keywords are used, and the config is ignored except interpolation.
- AFM: keyword, then config, then default.
- DFM: kwargs override the config.

It returns `params` (the effective new-mode parameters, including data-derived ones such as the dataset stride), `requested`, and `divergences`, a list of `(field, requested, effective)`. It is permanent, because it is the documented way to reproduce the papers.

### Canonical hyperparameters

Old names are in parentheses and are accepted with a warning during the transition. A default is the value that actually ran when the argument was omitted, which is the constructor default wherever it differs from the config default. For example, DDFM `learning_rate` is 0.005 and iVDFM `factor_order` is 1.

| Model | Parameters |
|---|---|
| DFM | `n_factors` (used when `blocks=None`), `blocks` (block key `n_factors`), `clock`, `tent_weights`, `max_iter`, `tol` (`threshold`), `ar_clip`, `data_clip`, `regularization`, `damping_factor`, `idio_rho0`, `idio_min_var`, `scaler`, `verbose` |
| DDFM | `n_factors`, `encoder_layers` (hidden widths; today's `encoder_size` is `(*encoder_layers, n_factors)`), `decoder_type`, `activation`, `learning_rate`, `optimizer`, `n_mc_samples`, `window_size`, `pretrain_epochs` (`max_epoch_pre_train`), `max_iter`, `tol` (`tolerance`), `min_target_interpolate_ratio` (misspelled today), `interpolation_method`/`_limit`/`_limit_direction`, `scaler`, `random_state` (`seed`), `device`, `verbose` (`disp`) |
| iVDFM | `n_factors` (`num_factors`), `window`, `stride`, `time_context`, encoder/decoder/prior `*_hidden_dim` and `*_n_hidden_layers`, `decoder_type`, `activation`, `slope`, `factor_order`, `innovation_distribution`, `decoder_var`, `n_regimes` (`num_regimes`), `regime_temperature`, `beta_kl`, `use_layer_norm`, `mixing`, `learning_rate`, `optimizer`, `optimizer_weight_decay`, `optimizer_momentum`, `batch_size`, `max_epochs`, `tol` (`tolerance`), `patience`, six `scheduler_*`, `scaler`, `random_state` (`seed`), `device`, `verbose` |
| AFM | `n_factors`, `factor_model`, `trading`, `embed_dim`, `hist_len`, `n_kernels`, `ridge`, `squash_lambda`, `pca_window`, `reestim`, `lambda_var`, `turnover_cost`, `short_cost`, `risk_free`, `learning_rate`, `max_epochs`, `patience`, `random_state` (`seed`), `device`, `verbose` |

Naming rules:

- `n_*` for counts.
- `max_iter` for outer algorithm iterations (EM, MCMC) and `max_epochs` for gradient passes.
- `tol` everywhere.
- `random_state` for seeds.

Removed parameters:

- Inferred in `fit`: `data_dim`, `context_dim`, `n_char`.
- Never read: `DDFMConfig.use_batch_norm`, `use_idiosyncratic`, `min_obs_idio` and `lags_input`; `DDFMConfig.num_factors`, which can contradict `encoder_size[-1]`; `iVDFMConfig.covariates`.
- Moved: `target_scaler` and `frequency` become `scaler` and `PanelData.freq`; `iVDFMConfig.context` becomes `PanelData.exog`.
- Deleted outright: `DFM.load_config`, which calls a missing `_load_config_common` and cannot run.

`blocks=None` gives one global block with `n_factors` factors. That is what `DFMConfig.from_dict` already does for a config without blocks. `tent_weights=None` follows today's resolution path unchanged, and a test pins it. If that path ends in an error, the message gives the symmetric kernel that `generate_tent_weights` produces and the exact argument to pass.

## 6. fit / predict / result_ contract

```python
class BaseFactorModel(sklearn.base.BaseEstimator):
    def fit(self, data, **fit_kwargs) -> Self
    def predict(self, h: int, data=None, **predict_kwargs) -> pd.DataFrame   # task == "forecast"
    def predict_factors(self, h: int, data=None) -> pd.DataFrame            # capability:factor_forecast
    def update(self, data, *, refit: bool = False) -> Self                  # capability:update
    def save(self, path, *, overwrite: bool = False) -> None
    @classmethod
    def load(cls, path, *, device=None, allow_pickle: bool = False) -> Self
    def get_config(self) -> BaseModelConfig
    @classmethod
    def from_config(cls, cfg) -> Self
    @classmethod
    def get_tags(cls) -> dict
```

### fit

`fit(data)` always returns `self`. It runs these steps:

1. Validate the parameters and data.
2. Build the private adapter.
3. Construct the engine with every parameter explicit. For iVDFM and AFM the engine's `__init__` seeds and builds the network, as today.
4. Train.
5. For DDFM, run `build_state_space()`, a deterministic least-squares step.
6. Set `result_` eagerly from today's `get_result()`.

Refitting starts from scratch. AFM's warm start is reached only through `update(data, refit=True)`. `fit` never mutates `data` or a hyperparameter. Derived values (`time_context_`, `n_char_`, internal column order) become underscore attributes. The only `fit_kwargs` is DFM's keyword-only `checkpoint_callback`.

### predict and allocate

`predict` never mutates fitted state, and no flag changes its return type:

- DDFM's `predict(data, update=True)` mutates the model today. The facade runs the same code on a deep copy of the engine, so the numbers are the same and the model is untouched.
- DDFM's `return_series` and `return_factors` flags become two methods, `predict` and `predict_factors`.
- `h` is a required integer. Anything else raises `TypeError`. Today `iVDFM.predict(6)` binds 6 to `data`.

What `data` means keeps today's per-model semantics, stated by tag `predict_data`:

| Model | `predict_data` | What `data` is |
|---|---|---|
| DFM | `new_rows` | rows strictly after `index_[-1]`; overlap raises `DataValidationError` |
| DDFM | `full_panel` | the full panel including the new rows, as today |
| iVDFM | `window` | every row passed is encoded, and the last state starts the forecast |

iVDFM's extra keywords keep today's defaults: `futr_exog` (was `context_data`), `deterministic=True` and `n_samples=1` (was `num_samples`). `space` is added in the release after.

`allocate(data=None)` exists only on models tagged `task="allocation"` (AFM) and returns weights. `backtest(data)` returns an `AFMResult` holding residuals, weights, book returns, Sharpe ratio and explained variance computed on `data`.

### Forecast DataFrame

- **Index.** The `h` timestamps after the last conditioning timestamp. The step comes from `index_.freq`, then from `pd.infer_freq`, then from the clock (`m` maps to `ME`, `q` to `QE`, `w` to `W`). A `RangeIndex` continues by position. The training index name is kept, or `"time"` if it had none. For `allocate`, the index is `data.index[hist_len:]`, and row *i* holds the weights applied to returns at that timestamp.
- **Columns.** `feature_names_in_` in the user's original order, even though DFM reorders internally.
- **Values.** float64.
- **Units.**
  - DFM: data units. The forecast is inverse-transformed when a scaler is set; with `scaler=None` the data is never scaled.
  - DDFM: data units.
  - iVDFM: model space until the release after.
- **attrs.** `{"model", "kind": "forecast" | "factors" | "weights", "origin": last conditioning timestamp, "space": "data" | "model"}`.
- **Mixed frequency.** DFM reports slower series at clock dates, as today.

### result_ and fitted attributes

`result_` holds today's `DFMResult`, `DDFMResult`, `iVDFMResult` or `AFMResult`. New-mode objects have no `training_state`, `_result`, `get_result()` or `result`. Legacy objects keep them and warn on public access. The `*ModelState` dataclasses become private and are used only by save and load.

Common fitted attributes: `result_`, `n_features_in_`, `feature_names_in_`, `exog_names_in_`, `index_`, `n_iter_`, `converged_`, `n_factors_`, `scaler_` and `random_state_`.

Tag-dependent fitted attributes:

- `factors_`: a DataFrame on `index_`. iVDFM uses a `(window, time)` MultiIndex, and DFM state columns are named `state_0..m`.
- `device_` and `network_`: torch models only. `network_` is the engine module.
- `residuals_` and `book_returns_`: AFM only.

### update

`update(data, *, refit=False) -> self` keeps today's mechanism for each model:

| Model | `update_kind` | `refit=False` | `refit=True` |
|---|---|---|---|
| DFM | `filter` | Kalman filter over new rows with fixed parameters | today's `retrain=True` |
| DDFM | `encoder_forward_rescale` | encoder forward pass; refits the target scaler (today's `_update`) | raises `UnsupportedInputError` |
| iVDFM | `encoder_forward` | encoder forward under `no_grad`; appends factors | raises `UnsupportedInputError` |
| AFM | `reattach` | attaches data for later `allocate()` | today's `retrain=True`: training continues from the current weights |

DFM's `rescale` and iVDFM's `append` and `store_full_history` remain as deprecated flags for one release.

### Tags

- Task and backend: `task` (`forecast` or `allocation`) and `backend` (`numpy` or `torch`).
- Data capabilities: `capability:exog`, `capability:series_exog`, `capability:mixed_frequency`, `capability:missing_values`.
- Method capabilities: `capability:update`, `capability:factor_forecast`, `capability:device`.
- Behaviour: `update_kind`, `predict_data`, `forecast_space`, `non_deterministic`.

`dfm_python.registry.all_estimators()` lists the four models.

## 7. Save/load layout

`save(path)` writes to a sibling temporary directory, reloads it to verify (keeping DFM's write-and-verify safety), and then renames it into place. `overwrite=False` refuses an existing directory.

| File | Contents |
|---|---|
| `config.json` | `format_version` (a file-format counter, independent of releases); `model_type`; `library_version` (read at save time); `params` (effective `get_params()`, with estimators written as `{"class", "params"}`); `random_state_effective` (including the implicit 3); `requested` and `divergences` (only when a legacy resolution or the converter diverged); `tags`; `runtime` (device, torch and numpy versions, determinism flag, thread count) |
| `data_spec.json` | target names in user order and internal order, exog and series-exog names, normalized `freq`, index kind, freq, name, first, last and length |
| `state.npz` | every array that is not a parameter or buffer, saved with `allow_pickle=False` and namespaced keys. DFM: `ssm/A`, `ssm/C`, `ssm/Q`, `ssm/R`, `ssm/Z_0`, `ssm/V_0`, `mixed_freq/*` (constraint matrix and vector, idio indicator), `structure/blocks`, `data/data_processed`, `data/last_smoothed_state`, `data/last_smoothed_cov`. DDFM: `factors` (MC x T x r), `eps`, `last_neurons`, `ssm/F`, `ssm/H`, `ssm/Q`, `ssm/R`, `ssm/mu_0`, `ssm/Sigma_0`. iVDFM: `factors`, `innovations`, regime weights. AFM: `residuals`, `book_returns`. All models: `scaler/<attr>` arrays |
| `state.json` | non-array `*ModelState` fields: `r`, `p`, block names, `mixed_freq`, tent weights, frequencies, `num_iter`, `converged`, `loglik` or `sharpe` |
| `result.npz`, `result.json` | arrays and scalars of `result_`, so a loaded model reports exactly what the fitted one did. `model_state_dict` and `fit_params` are rebuilt from the weights and state files, not duplicated |
| `weights.pt` | torch models only: `network_.state_dict()`, loaded with `torch.load(path, map_location=device, weights_only=True)` |
| `scaler.json` | class name and scalar parameters for the Standard, Robust, MinMax, MaxAbs and Quantile scalers |
| `scaler.pkl` | only for other scalers; `load` requires `allow_pickle=True` |

Numpy and torch state sit side by side. Non-parameter arrays go to `state.npz` for all four models, and torch tensors go only to `weights.pt`. DFM has no `weights.pt`, and `load` checks the `backend` tag against the files present.

`Model.load(path, device=None)` works in four steps:

1. Check `model_type` in `config.json`.
2. Rebuild the engine through `_from_saved(spec, state)`, which bypasses `__init__`. This removes DFM's required `dataset=` and DDFM's roughly 50-line minimal-dataset rebuild.
3. Load the weights with `map_location`, so a GPU-saved run loads on a CPU-only machine.
4. Restore `scaler_` and `result_`.

`dfm_python.load(path)` dispatches on `model_type`. The default path never unpickles.

Legacy files:

- `Model.load("old.pkl" | "old.pt")` reads a single file with today's code and emits a `FutureWarning`.
- `compat.convert_checkpoint(old_file, new_dir, model_type=None)` writes the new layout. For DFM it takes column names from the stored `dataset_metadata`.
- Effective values missing from old iVDFM checkpoints (`decoder_var`, `beta_kl`, `regime_temperature`, `use_layer_norm`, the learning rate and the seed) are filled from today's defaults, and each is recorded in `divergences` with a warning.

## 8. Device, seed, logging, verbosity

**Device.** The torch models take `device: str | torch.device = "auto"`, meaning CUDA if available and otherwise CPU. That is today's default for all three, so default runs do not move. It is resolved in `fit` to `device_`. `load(device=...)` overrides it. Today DDFM and AFM cannot be overridden at all. `PanelData` and the adapters have no device. DFM has no `device` parameter, because an argument that does nothing would surprise users.

**Seed.**

| Model | Parameter | Default | What is seeded (unchanged) |
|---|---|---|---|
| DFM | none | n/a | nothing; EM is deterministic |
| DDFM | `random_state` | 3 | local `np.random.RandomState` in fit; `torch.manual_seed` in the encoder and decoder builders and `layer/mlp.py` |
| iVDFM | `random_state` | 3 | `torch.manual_seed` then `np.random.seed` in the engine `__init__`, just before building components |
| AFM | `random_state` | None | `torch.manual_seed` in the engine `__init__` |

The eleven seeding call sites stay where they are and call one helper, `utils.seed.apply_seed(seed, *, torch_global, numpy_global)`, which does exactly what each site does today. Moving any of them would change RNG consumption order. `fit` constructs the engine (which seeds, then builds) and then trains, which is the order of today's dataset-in-constructor path. AFM's `set_dataset` path builds after seeding, and parity covers both orders. A grep test forbids seeding outside the helper. `random_state_effective` is always saved.

**Logging and verbosity.**

- The package logger gets only a `NullHandler`, and nothing is attached at import.
- `dfm_python.set_verbosity("debug" | "info" | "warning" | "error")` and `get_verbosity()` are the opt-in. `DFM_PYTHON_VERBOSITY` is honoured.
- Every model takes `verbose: int = 0`: 0 is silent, 1 shows progress and 2 is debug. The progress handler is attached only during `fit`.
- DDFM's `disp` interval maps to `verbose=1`. iVDFM's hard-coded `verbose=True` becomes the parameter.
- The five logger modules collapse into `logger.py` and `progress.py`.

None of this changes numbers.

## 9. Error handling conventions

| Exception | Bases | Raised when |
|---|---|---|
| `DFMError` | `Exception` | root of the library's errors (name kept) |
| `ConfigurationError`, `ConfigValidationError` | `DFMError`, `ValueError` | invalid hyperparameters; unknown keys (in the release after) |
| `DataValidationError` | `DFMError`, `ValueError` | malformed `PanelData`, misaligned index, overlapping `new_rows` data |
| `UnsupportedInputError` | `DataValidationError` | a slot, frequency layout or `refit` that the tags exclude |
| `NotFittedError` | `DFMError`, `sklearn.exceptions.NotFittedError` | `predict`, `allocate`, `update`, `save` or `result_` before `fit` |
| `ModelNotTrainedError` | alias of `NotFittedError` | kept for existing `except` clauses |
| `NumericalError`, `NumericalStabilityError` | `DFMError`, `ArithmeticError` | unchanged engine sites |

Warnings:

- `FutureWarning` for each deprecated call pattern, once per instance or call site.
- `LegacyPrecedenceWarning(UserWarning)` for each field whose effective value differs from the requested one.
- `UserWarning` when a fitted scaler's statistics are discarded, or when a run is saved with `random_state=None`.

Messages name the argument and the fix, such as the closest canonical key or the excluding tag. No input is silently ignored. Today's silent cases, DDFM `predict(data=..., update=False)` and iVDFM `fit(data=...)`, now either work or raise.

## 10. Current to target mapping

| Model | Aspect | Current | Target |
|---|---|---|---|
| DFM | construct | `DFM(dataset, config=None, scaler=None, **kwargs)`; kwargs mutate the caller's config; `DFMDataset(config, data, time_index)` | `DFM(*, n_factors, blocks=None, clock, tent_weights, max_iter, tol, ..., scaler=None, verbose=0)`; `PanelData(y, freq=...)` |
| DFM | fit | `fit(X=None, checkpoint_callback=None) -> DFMModelState`; wrong column order raises | `fit(data, *, checkpoint_callback=None) -> self`; stable reorder |
| DFM | predict | `predict(horizon=None, *, data=None) -> (X, Z)` | `predict(h, data=None)` and `predict_factors(h, data=None)`, both DataFrames |
| DFM | update | `update(data, rescale=False, retrain=False) -> None` | `update(data, *, refit=False) -> self` |
| DFM | results | `training_state`, `_result`, `get_result()`, `result` | `result_`, `n_iter_`, `converged_`, `factors_` |
| DFM | save/load | pickle file; `load(path, config=None, dataset=<required>)`; `load_config` broken | directory; `DFM.load(path)`; `from_config` |
| DDFM | construct | `DDFM(dataset, config, encoder_size, ..., seed=3)` (18 keywords), config ignored; `DDFMDataset(data, time_idx, covariates, scaler)` | `DDFM(*, n_factors, encoder_layers, ..., random_state=3, device="auto", verbose=0)`; covariates in `exog` |
| DDFM | fit | `fit() -> None`, then a required `build_state_space()` | `fit(data) -> self` |
| DDFM | predict | `predict(horizon=None, *, data, update=True, return_series, return_factors)`, ndarray or tuple, mutates | `predict(h, data=None) -> DataFrame` (pure); `predict_factors(h)` |
| DDFM | update / save | `update(DDFMDataset) -> None`; `torch.save` dict, load rebuilds a minimal dataset; no device override | `update(data) -> self`; directory with `weights.pt`; `device=` |
| iVDFM | construct | `iVDFM(dataset=None, data_dim, num_factors, context_dim, window, config, ..., seed=3, **kwargs)` (25 keywords); defaults clobber config; seeds in `__init__`; `set_dataset()` | `iVDFM(*, n_factors, window, stride=None, ..., random_state=3, device="auto", verbose=0)`; pure `__init__`; context in `exog` |
| iVDFM | fit | `fit(*args, **kwargs) -> self` on the `set_dataset` data; `fit(data=)` ignored | `fit(data) -> self` |
| iVDFM | predict | `predict(data=None, context_data=None, context=None, horizon=1, deterministic=True, num_samples=1) -> ndarray`, model space | `predict(h, data=None, *, futr_exog=None, deterministic=True, n_samples=1) -> DataFrame`, labelled; data space in the release after |
| iVDFM | update / tune | `update(data_or_dataset, *, append, store_full_history, verbose)`; `hyperparameter_optimization(...)` | `update(data) -> self`; `tuning.tune_hyperparameters(estimator, data, ...)` (same optuna code) |
| iVDFM | save/load | `torch.save` checkpoint or weights-only file with a partial config; `load(path, *args, **kwargs)` | directory with the full effective config; `iVDFM.load(path, device=)` |
| AFM | construct | `AFM(dataset=None, config=None, *, n_char, n_factors, ..., seed)`; `pick()`; `set_dataset()`; `AFMDataset(returns, characteristics, window, stride, scaler)` | `AFM(*, n_factors, factor_model, trading, hist_len, ..., random_state=None, device="auto", verbose=0)`; `PanelData(y, series_exog)` |
| AFM | fit / predict | `fit(X=None) -> AFMModelState`; `predict(horizon=None, *, data=None) -> dict` with batch dimension 1 | `fit(data) -> self`; `allocate(data=None) -> DataFrame`; `backtest(data) -> AFMResult` |
| AFM | update / save | `update(data, retrain=False)`; `torch.save` dict, `load` without `map_location` | `update(data, *, refit=False) -> self`; directory; `AFM.load(path, device=)` |

## 11. Contract tests

The contract tests live in `src/test/contract/`, parametrized over `registry.all_estimators()` with one tiny CPU fixture per tag combination. `dfm_python.testing.check_estimator(est)` exposes the same checks for future models. Every model must pass the following:

1. **Pure construction.** The global torch and numpy RNG states are unchanged, no CUDA memory is allocated, and `vars(est)` equals `get_params()`.
2. **Parameter machinery.** `get_params`/`set_params` and `clone` round-trip. Config fields equal the constructor parameters. `from_config(get_config())` gives an equal estimator, YAML and JSON round-trips are lossless, and unknown keys warn (later raise).
3. **Unfitted access.** `predict`, `allocate`, `update`, `save` and `result_` raise `NotFittedError`, and `hasattr(model, "result_")` is false.
4. **fit returns self and is repeatable.** Two fits on CPU give bitwise-identical `result_` arrays.
5. **fit adds only underscore attributes**, and mutates neither the input data nor any hyperparameter.
6. **Output shape.** Forecasters return `h` rows starting one step after the conditioning data, columns equal to `feature_names_in_` in user order, finite float64 values, and `attrs["space"]` equal to the `forecast_space` tag. Allocators return index `data.index[hist_len:]` with asset columns, and their `predict` raises `TypeError`.
7. **Purity of predict and allocate.** The fitted-state hash is unchanged after repeated calls, including with `data=`.
8. **Save/load round-trip.** The file set matches section 7 for the backend. `allow_pickle=False` and `weights_only=True` loads succeed, and no pickle appears for default scalers. Reloaded forecasts and `result_` are bitwise equal, `dfm_python.load` dispatches correctly, and a CUDA-saved model loads on CPU (skipped without CUDA).
9. **Legacy checkpoints.** One committed checkpoint per model, written by the pre-change commit, loads through the legacy reader and the converter and reproduces the golden forecast.
10. **Capability enforcement.** Each slot, layout or `refit` that the tags exclude raises `UnsupportedInputError`.
11. **Device** (tag-gated). `device="cpu"` puts every parameter on CPU.
12. **Seeds.** The same `random_state` reproduces results bitwise on CPU, `random_state_effective` is saved, and the grep test finds no seeding outside `apply_seed`.
13. **Silence.** Importing the package adds no handlers. With `verbose=0`, `fit` prints nothing and logs nothing at INFO.
14. **update.** It returns `self`, `predict` afterwards conditions on the new rows, and parameters are unchanged when `refit=False`.
15. **Legacy precedence pinned.** `compat.legacy_params` reports iVDFM's effective `factor_order=1` against a requested 3, and DDFM's 0.005 learning rate against a requested 0.002. Legacy construction gives the same effective values.
16. **Parity suite** (`src/test/parity/`). Golden fixtures from the pre-change commit cover each tutorial configuration on truncated shipped data, a synthetic panel, DFM mixed frequency, DFM on permuted columns, iVDFM with several regimes, and AFM in attention and PCA modes. The suite compares the legacy path, the new path, the new path from a regenerated YAML, and save-then-load with `assert_array_equal`. The unmodified DFM and DDFM tutorials also run in CI on truncated data.

## 12. Migration and deprecation plan

### The next minor release

- **Construction style picks the API.** Each public class defines `__new__`. A positional argument, `dataset=` or `config=` returns the private legacy class: today's body unchanged, registered as a virtual subclass so `isinstance` still holds. It emits one `FutureWarning` naming the new call. The new class therefore carries no hidden constructor arguments and no dual-mode branches.
- **Legacy objects keep everything** they do today: signatures, return types, horizon defaults, precedence, single-file saves, `set_dataset`, `build_state_space` and `hyperparameter_optimization`. They also gain read-only `result_`. iVDFM legacy `fit(dataset)` and `fit(data=dataset)` now forward the dataset with a `DeprecationWarning`. These calls fail today, so no number moves.
- **The new API is the documented one.** README, docstrings and the CHANGELOG switch to it. The CHANGELOG lists every divergence per shipped YAML.
- **Aliases** for old parameter names and old YAML keys work with a `FutureWarning`. `ConfigSource`, `YamlSource` and `make_config_source` become deprecated wrappers.
- **Loading.** `load` reads both formats, and `convert_checkpoint` is available.
- **Logging.** Import-time handlers are removed. Only output changes.
- **iVDFM forecasts** warn once about the coming switch to data units.
- **Examples.** YAMLs are regenerated in new style with effective values, and the old flat files are kept. The iVDFM tutorials are rewritten. The DFM and DDFM tutorials stay unchanged as the CI legacy check.

### The release after

- **Removed:** the legacy dispatch; the public `*Dataset` classes (their code lives on inside the adapters); `set_dataset`, `build_state_space`, `hyperparameter_optimization`, `get_result`, `result` and `training_state`; the `return_*` and deprecated `update` flags; all aliases; and old-format writing.
- **Changed:** iVDFM forecasts default to data units, with `space="model"` available. Unknown config keys raise. The DFM and DDFM tutorials move to the new API and keep their golden comparison.
- **Kept:** `compat.legacy_params` (permanent), legacy-file reading through `convert_checkpoint` (lifetime is question 7 in section 15), the old flat YAMLs, and `BaseFactorModel` as the public base class.

## 13. Implementation plan

Each item fits one agent session, ends with the suite green, and touches at most one model. Golden capture comes before any production change. Shared infrastructure comes next, and AFM is the first model, because it is the smallest and has the simplest state.

1. **Golden capture.** Add `src/test/parity/make_golden.py` and fixtures for all four models from the current commit, with pinned threads. Commit one legacy checkpoint per model. Add tests asserting that the legacy path reproduces the fixtures, and a CI job that runs the DFM and DDFM tutorials on truncated data. No production change.
2. **Errors and logging.** Rebase the exceptions (section 9). Add `NotFittedError`, `UnsupportedInputError` and `LegacyPrecedenceWarning`. Install the `NullHandler`, add `set_verbosity`, collapse the logger modules, and add the silence test.
3. **Seed helper.** Add `apply_seed` and route the eleven sites through it in place. Add the grep test and confirm parity is unchanged.
4. **PanelData.** Add the container, validation, the `from_*` constructors and `to_long`, `freq` normalization and polars input, with unit tests.
5. **Base class and contract scaffolding.** Add the new `BaseFactorModel`, tags, the registry, the `__new__` dispatch helper, and `testing.check_estimator` with the contract-test module, initially empty.
6. **Save/load primitives.** Add the atomic directory writer, the JSON and npz codecs, the known-scaler codec, the gated pickle fallback, `dfm_python.load`, the converter skeleton, and the strict-key and alias machinery for configs.
7. **AFM.** Move the class to `models/afm/_engine.py` unchanged and add `_from_saved`. Write the facade (`fit`, `allocate`, `backtest`, `update`, `save`, `load`, tags, `AFMConfig` mirror). Add the legacy dispatch, the AFM entries in `compat.legacy_params` and the converter, parity for both the `set_dataset` and the constructor-dataset order, and contract registration.
8. **iVDFM facade.** Move the class to the engine. Build it with explicit parameters, `stride=None` semantics, labelled `predict`, `update`, save and load, and the config mirror. Add `compat.legacy_params` reproducing the clobbering, the converter fill-in, contract tests and parity.
9. **iVDFM tuning and examples.** Extract `tuning.tune_hyperparameters` around today's optuna code. Write the YAML regeneration script and regenerate the iVDFM YAMLs. Rewrite the iVDFM tutorials. Add YAML parity tests.
10. **DDFM facade.** Move the class to the engine. Add the `n_factors`/`encoder_layers` split, the `[exog | y]` adapter, `fit` including the state space, deep-copy `predict(data=)`, `predict_factors`, `update`, and `_from_saved` replacing the minimal-dataset rebuild. Add the config mirror without the dead fields, `compat.legacy_params`, contract tests and parity.
11. **DFM facade.** Move the class to the engine. Write the adapter that builds `DFMConfig` and `DFMDataset`, with the stable partition and output in user order. Add `_from_saved` with `data_processed` and the last smoothed state, `new_rows` validation and `update`. Remove `load_config`. Add the permuted-input golden test, contract tests and parity.
12. **DFM and DDFM examples.** Regenerate the YAMLs in new style and add new-API example scripts beside the unchanged tutorials, with YAML parity tests.
13. **Documentation.** Rewrite the README, and add a migration guide (old call, new call, unpacking replacements, `compat.legacy_params`) and the CHANGELOG divergence tables.

## 14. Risks

- **Parity hazards.** The known ones are:
  - DDFM's `[exog | y]` order and its `type(scaler)()` covariate scaler
  - iVDFM's in-dataset imputation
  - the seed-then-build order
  - eager `get_result()`, which must not consume RNG ahead of a stochastic `predict`
  - the deep copy behind DDFM's pure `predict(data=)`

  Each has a golden case. Because the engines are today's classes called with explicit arguments, the hazards sit in small adapters.
- **CPU-only bitwise parity.** It holds only with pinned threads. A GPU-produced paper result can be matched only to a tolerance.
- **Migrating by hand bypasses the precedence check.** Copying an old YAML's values into new-mode keywords skips the divergence warning, which covers only `from_yaml` and `from_config`. The regenerated YAMLs and the migration guide are the mitigation.
- **Rename churn.** `tol`, `random_state`, `n_factors` and `encoder_layers` touch every migrated script. Aliases help for one release only.
- **Virtual-subclass dispatch is unusual.** A user subclass that overrides `__init__` would be routed incorrectly. It lasts one release.
- **Two-column unpacking.** `X, Z = model.predict(h)` on a new-mode model with exactly two targets silently yields the column labels. The migration guide flags this. Any other column count raises.
- **Eager `result_`** may lengthen `fit` on large DFM panels. If so, it can become lazy without an API change.

## 15. Open questions for the owner

1. **Which path and effective hyperparameters produced the published iVDFM and DDFM numbers?** For example, `factor_order` 1 or 3 for macro, and hidden width 200 or 50 for finance. Running `compat.legacy_params` on the paper scripts answers it. The answer decides the regenerated YAMLs and whether the papers' stated settings need a note.
2. **iVDFM forecast units.** Approve switching iVDFM to data units in the release after (with `space="model"` kept permanently), or keep model space and only label it.
3. **iVDFM stride.** Confirm the paper runs used the dataset default `window // 2` (or an explicit value), not the unused config `stride=1`.
4. **Naming.** Accept the sklearn names (`n_factors`, `n_regimes`, `random_state`, `tol`), or keep `num_factors`, `seed` and `tolerance`? Should the YAML aliases outlive the release after?
5. **Seed defaults.** Keep 3, 3 and None permanently, or unify to None later and accept changed default-run numbers?
6. **RNG isolation.** Should `fit` gain an opt-in `torch.random.fork_rng`? Fitted numbers stay the same, but later stochastic iVDFM draws would change.
7. **Legacy checkpoint lifetime.** Keep `convert_checkpoint` and the legacy readers permanently for paper-era files?
8. **AFM verb.** Is `allocate` the right name, and should `AFM.predict` raise or alias it?
9. **DFM tent-weight default.** If today's path fails without `tent_weights`, should the symmetric `generate_tent_weights` kernel become the documented default? It would affect only inputs that fail today.
10. **DFM default scaler.** Keep `None` as the code does, and correct the README's claim of internal standardization (proposed), or change the default?
11. **Mixed-frequency output.** Keep reporting slower series at every clock date, or mask them to observation dates?
12. **`DDFMResult.C` orientation.** It is factors x N, unlike DFM's N x state. Fix it in the release after, or leave it?
13. **sklearn coupling.** Rely on sklearn's `_get_param_names`, or vendor about 60 lines?

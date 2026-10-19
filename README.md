# dfm-python

Dynamic factor models in Python.

- **DFM**: linear, mixed-frequency Dynamic Factor Model estimated by EM with
  Kalman filtering and smoothing.
- **DDFM**: Deep Dynamic Factor Model with a nonlinear (PyTorch) encoder.
- **iVDFM**: Identifiable Variational Dynamic Factor Model.
- **AFM**: the Attention Factor Model of Epstein, Wang, Choi and Pelger (2025),
  a cross-sectional conditional factor model whose residuals are traded
  portfolios, with PCA and Avellaneda-Lee benchmarks.

Mixed frequencies are handled with a clock: latent factors evolve at one
frequency (for example monthly) and lower-frequency series load on them through
tent-kernel aggregation in the observation equation. Series faster than the
clock are not supported.

## Install

```bash
pip install dfm-python        # or: uv add dfm-python
```

Requires Python 3.10 or later; PyTorch is a required dependency.

Optional extras:

| Extra      | Adds        | Needed for                                              |
|------------|-------------|---------------------------------------------------------|
| `yaml`     | omegaconf   | loading configs from YAML (`YamlSource`)                |
| `tune`     | optuna      | iVDFM hyperparameter search (`dfm_python.tuning`)       |
| `metrics`  | sktime      | `calculate_rmse`, `calculate_mae`                       |
| `polars`   | polars, pyarrow | passing polars DataFrames to the dataset classes    |
| `examples` | hydra-core  | running the scripts in `tutorial/`                      |
| `all`      | all of the above |                                                    |

```bash
pip install "dfm-python[yaml,tune]"
```

## DFM

DFM is an estimator on `PanelData`: the series in `y`, in any column order,
and the sampling frequency of the slower ones in `freq` (unlisted series use
the clock). One frequency slower than the clock is supported; its tent kernel
is required.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, PanelData

idx = pd.date_range("2010-01-31", periods=120, freq="ME", name="date")
df = pd.DataFrame(np.random.default_rng(0).standard_normal((120, 4)),
                  index=idx, columns=["gdp", "s0", "s1", "s2"])
df.loc[df.index.month % 3 != 0, "gdp"] = np.nan        # quarterly series, any column position

data = PanelData(y=df, freq={"gdp": "q"})
model = DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
            max_iter=50, scaler="standard")             # blocks=None: one global block
model.fit(data)                                         # EM + Kalman smoother
fc = model.predict(6)                                   # DataFrame 6 x 4, columns in your order
zf = model.predict_factors(6)                           # DataFrame 6 x state dim
model.result_.loglik, model.n_iter_, model.converged_
```

`scaler=None` (the default) fits on the data as given; pass
`scaler="standard"` to standardize the series. Forecasts are in data units and
report every series at every clock date, slower series included. The engine
orders the series `[clock | slower]` internally (`internal_feature_names_`),
and the series-indexed matrices of `result_` (`C`, `R`, `x_sm`) are in that
order. `blocks={"name": {"n_factors": 2, "series": [...]}, ...}` replaces the
single global block.

AFM, iVDFM and DDFM below are estimators on the same `PanelData` container.
Every model and config class is importable from `dfm_python`. YAML
configurations for each model are in
[`config/`](https://github.com/minkeymouse/dfm-python/tree/main/config) and
end-to-end scripts in
[`tutorial/`](https://github.com/minkeymouse/dfm-python/tree/main/tutorial)
(the DFM scripts use Hydra: install the `examples` extra to run them).

## AFM

AFM takes asset returns `(T, N)` and, for learned attention factors, firm
characteristics `(T, N, M)` in one `PanelData` container. It is trained on an
after-cost Sharpe ratio plus an explained-variance term (`lambda_var`). AFM
allocates rather than forecasts: `allocate()` returns portfolio weights and
`backtest(data)` evaluates them on other data.

```python
import numpy as np
import pandas as pd
from dfm_python import AFM, PanelData

rng = np.random.default_rng(0)
T, N, M = 120, 20, 8                        # dates, assets, characteristics
returns = 0.01 * rng.standard_normal((T, N))
characteristics = rng.standard_normal((T, N, M))
dates = pd.bdate_range("2020-01-01", periods=T, name="date")
tickers = [f"a{i}" for i in range(N)]

data = PanelData.from_arrays(y=returns, series_exog=characteristics, index=dates,
                             names=tickers, series_exog_names=[f"c{j}" for j in range(M)])
model = AFM(n_factors=3, hist_len=20, max_epochs=20, random_state=0, device="cpu")
model.fit(data)                             # returns model

weights = model.allocate()                  # DataFrame (T - hist_len) x N, index dates[hist_len:]
model.result_.sharpe, model.result_.explained_variance
model.residuals_                            # DataFrame T x N of residual returns
model.book_returns_                         # Series of after-cost returns on dates[hist_len:]

res = model.backtest(data.loc(dates[60], None))   # AFMResult on other data, no training
res.sharpe, res.weights.shape               # (1, n_eval, N)

model.save("runs/afm")                      # pickle-free directory
model = AFM.load("runs/afm")                # or dfm_python.load("runs/afm"); device= overrides
weights = model.allocate(data=data)         # a loaded model keeps no data: pass it

# Benchmarks: PCA factors need no characteristics
pca = AFM(n_factors=3, hist_len=20, factor_model="pca", trading="ou")  # or trading="longconv"
pca.fit(PanelData(y=pd.DataFrame(returns, index=dates, columns=tickers)))
```

In long format (`unique_id`, `ds`, `y` plus one column per characteristic),
build the same panel with
`PanelData.from_long(df, series_exog_cols=["c0", ..., "c7"])`. The
hyperparameters can also come from YAML: `config/afm_finance.yaml` holds a
`model:` section that `AFM.from_config(AFMConfig.from_yaml(path))` reads.

## iVDFM

iVDFM, the identifiable variational dynamic factor model, learns nonlinear
factors with AR(`factor_order`) dynamics from sliding windows of the panel and
is trained on the ELBO. The prior of the innovations is conditioned on a context
u_t, which makes the factors identifiable. u_t is the `exog` columns of
`PanelData` plus `time_context` generated time features. Missing target and exog
values are interpolated. Forecasts are returned in data units: the model's
output is inverse-transformed with the fitted `scaler_`.

```python
import numpy as np
import pandas as pd
from dfm_python import iVDFM, PanelData

rng = np.random.default_rng(0)
dates = pd.date_range("2010-01-31", periods=156, freq="ME", name="date")
raw = pd.DataFrame(rng.standard_normal((156, 4)), columns=["s0", "s1", "s2", "policy_rate"])
raw.insert(0, "date", dates)

data = PanelData.from_wide(raw.iloc[:144], time_col="date", exog_cols=["policy_rate"])  # exog = context u_t
model = iVDFM(n_factors=2, factor_order=2, window=48, stride=None,  # stride=None means window // 2
              time_context=1, scaler="standard", max_epochs=20,
              random_state=3, device="auto")                      # random_state=3 is the default
model.fit(data)                                                   # returns model
model.result_.training_elbo, model.n_iter_, model.converged_
model.factors_                              # window factors, (window, time) MultiIndex

future = raw.set_index("date")[["policy_rate"]]                # future values of the exog
fc = model.predict(6, futr_exog=future.iloc[144:150])          # DataFrame 6 x 3, data units
zf = model.predict_factors(6, futr_exog=future.iloc[144:150])  # DataFrame 6 x 2 (factor_0, factor_1)

new = PanelData.from_wide(raw.iloc[96:150], time_col="date", exog_cols=["policy_rate"])
fc_new = model.predict(6, new, futr_exog=future.iloc[150:156])  # conditions on new; model unchanged
model.update(new)               # encoder forward pass with the fixed weights; appends the factors
model.predict(6, futr_exog=future.iloc[150:156])                # equals fc_new

model.save("runs/ivdfm", overwrite=True)    # pickle-free directory
model = iVDFM.load("runs/ivdfm")            # or dfm_python.load("runs/ivdfm"); device= overrides
```

`update(data)` never trains: `update(data, refit=True)` raises
`UnsupportedInputError`, so call `fit` to retrain. Without `futr_exog`, a model
fitted with exog warns and fills the future context with time features. A
`futr_exog` DataFrame is matched by column name and its index must be the
forecast dates (a default `RangeIndex` or an array is taken by position). With
`factor_order > 1` and one regime, `n_factors` cannot exceed the number of
targets: `fit` raises `ConfigurationError`.

Hyperparameter search uses optuna (the `tune` extra). It fits clones of the
estimator on a holdout split and returns the best values under the estimator's
names:

```python
from dfm_python.tuning import tune_hyperparameters   # pip install "dfm-python[tune]"

best = tune_hyperparameters(model, data, n_trials=30, horizon=12, min_window=24, max_window=48,
                            metric="sMSE")
model.set_params(**best).fit(data)
```

The shipped YAMLs (`config/ivdfm_macro.yaml`, `ivdfm_finance.yaml`,
`ivdfm_exchange.yaml`) hold a `model:` section with the constructor parameters
and a `data:` section with the `PanelData.from_wide` arguments:

```python
from dfm_python.config import iVDFMConfig, load_yaml

model = iVDFM.from_config(iVDFMConfig.from_yaml("config/ivdfm_exchange.yaml"))
sections = load_yaml("config/ivdfm_exchange.yaml")
data = PanelData.from_wide(pd.read_csv("data/exchange_rate.csv"), **sections["data"])
```

`tutorial/tutorial_{macro,finance,exchange}_ivdfm.py` run these configurations
end to end (`--rows` and `--max-epochs` shorten a run).

## DDFM

DDFM, the deep dynamic factor model, encodes each time step with a nonlinear
encoder into `n_factors` factors and reconstructs the targets with a linear (or
MLP) decoder. The autoencoder is pre-trained, then trained with MCMC denoising
(`n_mc_samples` corrupted copies of the targets per iteration, at most
`max_iter` iterations); `fit` then builds a linear state space on the factors,
an AR(1) transition and the decoder's last linear layer, which the forecasts
use. The `exog` columns of `PanelData` are the encoder's covariates: they are
encoded with the targets but not reconstructed or forecast. Missing values are
interpolated. Forecasts are returned in data units: the model's output is
inverse-transformed with the fitted `scaler_`.

```python
import numpy as np
import pandas as pd
from dfm_python import DDFM, PanelData

rng = np.random.default_rng(0)
dates = pd.date_range("2010-01-31", periods=132, freq="ME", name="date")
raw = pd.DataFrame(rng.standard_normal((132, 5)), columns=["s0", "s1", "s2", "oil", "fx"])
raw.insert(0, "date", dates)

data = PanelData.from_wide(raw.iloc[:120], time_col="date", exog_cols=["oil", "fx"])  # exog = covariates
model = DDFM(n_factors=2, encoder_layers=(16,), decoder_type="linear",  # encoder 5 -> 16 -> 2
             learning_rate=0.005, n_mc_samples=10, max_iter=50, tol=5e-4,
             scaler="standard", random_state=3, device="auto")     # random_state=3 is the default
model.fit(data)                             # returns model; builds the state space too
model.n_iter_, model.converged_
model.result_.A, model.result_.C            # AR(1) transition (2 x 2), loadings (3 x 2: series x factors)
model.factors_                              # DataFrame 120 x 2 (factor_0, factor_1)

fc = model.predict(6)                       # DataFrame 6 x 3 (s0, s1, s2), data units
zf = model.predict_factors(6)               # DataFrame 6 x 2

full = PanelData.from_wide(raw, time_col="date", exog_cols=["oil", "fx"])   # all 132 rows
fc_new = model.predict(6, data=full)        # conditions on the full panel; model unchanged
model.update(full)                          # mutates: encodes every row, refits the target scaler
model.predict(6)                            # equals fc_new

model.save("runs/ddfm", overwrite=True)     # pickle-free directory (weights.pt, state.npz, ...)
model = DDFM.load("runs/ddfm")              # or dfm_python.load("runs/ddfm"); device= overrides
```

`data=` and `update` take the full panel, the training rows plus any new ones,
with the same series and exog columns: every row is encoded with the trained
weights and the forecasts start after its last date. The target scaler is
refitted on the training targets and the panel's, so `scaler_` changes;
`predict(h, data=)` does all of this on a copy. `update(data)` never trains:
`update(data, refit=True)` raises `UnsupportedInputError`, so call `fit` to
retrain. `result_.C` is N x r (series by factors), as in DFM.
`update` and `predict(h, data=)` are only meaningful with `scaler="standard"`
or `None`: for the other scalers the engine refits on scaled training targets
stacked with raw new ones (a known engine bug, kept for identical numbers).

The shipped YAMLs (`config/ddfm_macro.yaml`, `ddfm_finance.yaml`,
`ddfm_exchange.yaml`) hold a `model:` section with the constructor parameters
and a `data:` section with the `PanelData.from_wide` arguments; the macro and
finance files name the covariates in `exog_cols`:

```python
from dfm_python.config import DDFMConfig, load_yaml

model = DDFM.from_config(DDFMConfig.from_yaml("config/ddfm_exchange.yaml"))
sections = load_yaml("config/ddfm_exchange.yaml")
data = PanelData.from_wide(pd.read_csv("data/exchange_rate.csv"), **sections["data"])
```

`tutorial/tutorial_{macro,finance,exchange}_ddfm.py` run these configurations
end to end (`--rows` and `--max-iter` shorten a run).

## Layout

```
src/dfm_python/
  models/      dfm/, ddfm/, ivdfm/, afm/ model classes
  dataset/     PanelData, DFMDataset, and the private data adapters of DDFM, iVDFM and AFM
  config/      model configs (DFMConfig, ..., AFMConfig), YAML adapter, results
  ssm/         state-space form and Kalman filtering
  functional/  EM algorithm
  numeric/, layer/, utils/, logger/   shared numerics, network layers, helpers
src/test/      pytest suite
config/        example YAML configurations (model: and data: sections; DFM's are flat Hydra files)
data/          example datasets used by the tutorials
tutorial/      end-to-end scripts per model and dataset
```

Development, from a clone (the `dev` dependency group holds pytest, ruff, the
build tools and the optional packages the tests import):

```bash
uv sync --group dev      # or: pip install -e . --group dev
uv run pytest            # test suite (src/test)
uv run ruff check        # lint; also run by pre-commit and CI
```

Add `--all-extras` to the sync to also run the tests that skip without
sktime (as CI does).

The paper repository [minkeymouse/nfe](https://github.com/minkeymouse/nfe) uses
AFM as a baseline.

## License

MIT. See [LICENSE](https://github.com/minkeymouse/dfm-python/blob/main/LICENSE).

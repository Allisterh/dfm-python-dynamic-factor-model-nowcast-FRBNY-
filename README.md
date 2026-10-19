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

Pass preprocessed data; the model standardizes it internally.

```python
import numpy as np
import pandas as pd
from dfm_python import DFM, DFMDataset
from dfm_python.config import DFMConfig

rng = np.random.default_rng(0)
cols = ["s0", "s1", "s2", "s3"]
df = pd.DataFrame(rng.standard_normal((120, 4)), columns=cols)
df["date"] = pd.date_range("2010-01-31", periods=120, freq="ME")

config = DFMConfig(
    blocks={"global": {"num_factors": 1, "series": cols}},
    frequency={c: "m" for c in cols},
    clock="m",
    max_iter=50,
)
dataset = DFMDataset(config=config, data=df, time_index="date")
model = DFM(dataset=dataset, config=config)
model.fit()                                     # EM + Kalman smoother
X_forecast, Z_forecast = model.predict(horizon=6)
print(X_forecast.shape, Z_forecast.shape)       # (6, 4) series, (6, state dim)
```

`DDFM` (`from dfm_python import DDFM, DDFMDataset`) follows the same
dataset-then-model pattern; every model and config class is importable from
`dfm_python`. Hydra YAML configurations for each model are in
[`config/`](https://github.com/minkeymouse/dfm-python/tree/main/config) and
end-to-end scripts in
[`tutorial/`](https://github.com/minkeymouse/dfm-python/tree/main/tutorial)
(install the `examples` extra to run them).

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
fitted with exog warns and fills the future context with time features.

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

## Layout

```
src/dfm_python/
  models/      dfm/, ddfm/, ivdfm/, afm/ model classes
  dataset/     DFMDataset, DDFMDataset, iVDFMDataset, AFMDataset
  config/      model configs (DFMConfig, ..., AFMConfig), YAML adapter, results
  ssm/         state-space form and Kalman filtering
  functional/  EM algorithm
  numeric/, layer/, utils/, logger/   shared numerics, network layers, helpers
src/test/      pytest suite
config/        example Hydra YAML configurations
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

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
| `tune`     | optuna      | iVDFM hyperparameter search                             |
| `metrics`  | sktime      | `calculate_rmse`, `calculate_mae`                       |
| `polars`   | polars      | passing polars DataFrames to the dataset classes        |
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

`DDFM` (`from dfm_python import DDFM, DDFMDataset`) and `iVDFM`
(`from dfm_python.models.ivdfm import iVDFM`) follow the same
dataset-then-model pattern. Hydra YAML configurations for each model are in
[`config/`](https://github.com/minkeymouse/dfm-python/tree/main/config) and
end-to-end scripts in
[`tutorial/`](https://github.com/minkeymouse/dfm-python/tree/main/tutorial).

## AFM

AFM takes a returns panel `(T, N)` and, for learned attention factors, firm
characteristics `(T, N, M)`. It is trained on an after-cost Sharpe ratio plus an
explained-variance term (`lambda_var`).

```python
import numpy as np
import torch
from dfm_python import AFM, AFMDataset
from dfm_python.models.afm.factors import residuals

rng = np.random.default_rng(0)
T, N, M = 120, 20, 8                        # dates, assets, characteristics
returns = 0.01 * rng.standard_normal((T, N))
characteristics = rng.standard_normal((T, N, M))

ds = AFMDataset(returns=returns, characteristics=characteristics)
model = AFM(dataset=ds, n_factors=3, hist_len=20, max_epochs=20, seed=0)
model.fit()

out = model.predict()
out["residuals"].shape                      # (1, T, N) residual returns
out["weights"].shape                        # (1, T - hist_len, N) portfolio weights
out["book_returns"].shape                   # (1, T - hist_len) after-cost returns
result = model.get_result()
print(result.sharpe, result.explained_variance)

# Residuals from the fitted factor block directly
R, X = ds.get_tensors(model.device)
with torch.no_grad():
    omega_f, beta_t = model.factor_block(X)  # (T, K, N), (T, N, K)
    eps = residuals(omega_f, beta_t, R)      # (T, N)

# Benchmarks: PCA factors need no characteristics
pca = AFM(dataset=AFMDataset(returns=returns), n_factors=3, hist_len=20,
          factor_model="pca", trading="ou")  # or trading="longconv"
pca.fit()
```

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
tutorial/      end-to-end scripts per model and dataset
```

Tests, from a clone: `uv sync --group dev && uv run pytest src/test -q`.

The paper repository [minkeymouse/nfe](https://github.com/minkeymouse/nfe) uses
AFM as a baseline.

## License

MIT. See [LICENSE](https://github.com/minkeymouse/dfm-python/blob/main/LICENSE).

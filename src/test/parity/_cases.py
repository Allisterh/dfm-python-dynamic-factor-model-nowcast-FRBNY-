"""Golden cases: each one fits a model through today's public API.

A case builds its data from a recipe, constructs the model, trains it and
returns a :class:`CaseRun` with

* ``outputs``: the engine numbers that the fixture pins,
* ``hyperparameters``: the values the constructed object actually holds, read
  off its attributes right after construction (not the values passed in), and
* ``data``: the recipe plus a digest of the exact input that was fitted.

``make_golden.py`` writes these to ``fixtures/<case>.npz`` and ``<case>.json``;
the ``test_golden_*`` modules re-run the same case and compare. When a model is
migrated to the new estimator API its test switches to the estimator and passes
the recorded hyperparameters to the new constructor.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MACRO_CSV = REPO_ROOT / "data" / "macro.csv"
CONFIG_DIR = REPO_ROOT / "config"

# Global RNGs are reset to this value before every case, so a case never depends
# on what ran before it. The models then seed themselves as they do today.
GLOBAL_RNG_PRESEED = 20251019


# --------------------------------------------------------------------- types
@dataclasses.dataclass
class CaseRun:
    outputs: dict[str, np.ndarray]
    hyperparameters: dict[str, Any]
    data: dict[str, Any]
    raised: dict[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    model: str
    run: Callable[[], CaseRun]

    def __repr__(self) -> str:  # readable pytest ids
        return self.name


# ------------------------------------------------------------------- helpers
def jsonable(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON types (lists, dicts, str, float, int, bool, None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"class": type(obj).__name__,
                "fields": jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})}
    return {"class": type(obj).__name__}


def normalized(obj: Any) -> Any:
    """``jsonable`` followed by a JSON round trip, for comparing with a manifest."""
    return json.loads(json.dumps(jsonable(obj), sort_keys=True))


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, pd.DataFrame):
            h.update(json.dumps([str(c) for c in a.columns]).encode())
            h.update(json.dumps([str(i) for i in a.index]).encode())
            for col in a.columns:
                if np.issubdtype(a[col].dtype, np.number):
                    h.update(np.ascontiguousarray(a[col].to_numpy(np.float64)).tobytes())
                else:
                    h.update(json.dumps([str(v) for v in a[col]]).encode())
        else:
            arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attrs(obj: Any, names: list[str]) -> dict[str, Any]:
    return {n: getattr(obj, n) for n in names}


def _class_name(obj: Any) -> str | None:
    return None if obj is None else type(obj).__name__


def _np(t: Any) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _reset_global_rngs() -> None:
    random.seed(GLOBAL_RNG_PRESEED)
    np.random.seed(GLOBAL_RNG_PRESEED)
    torch.manual_seed(GLOBAL_RNG_PRESEED)


@contextlib.contextmanager
def force_cpu() -> Iterator[None]:
    """Make today's ``cuda if available else cpu`` device choice resolve to CPU."""
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        yield


def _load_yaml(name: str) -> dict[str, Any]:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / name), resolve=True)


def _macro_tail(n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(MACRO_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.iloc[-n_rows:].reset_index(drop=True)


def _macro_recipe(**extra: Any) -> dict[str, Any]:
    return {"source": "data/macro.csv", "source_sha256": _file_digest(MACRO_CSV), **extra}


# ----------------------------------------------------------------------- AFM
AFM_ATTRS = ["factor_model", "trading", "n_factors", "n_char", "embed_dim", "hist_len",
              "n_kernels", "ridge", "squash_lambda", "pca_window", "reestim", "lambda_var",
              "turnover_cost", "short_cost", "risk_free", "learning_rate", "max_epochs",
              "patience", "seed", "device"]


def afm_inputs(factor_model: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """The AFM case's returns (T, N), characteristics (T, N, M) or None, and data recipe."""
    # Same panel as src/test/models/test_afm.py::_make_data.
    T, N, M, seed = 60, 6, 8, 0
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, N)) * 0.01
    characteristics = rng.standard_normal((T, N, M))
    use_chars = factor_model == "attention"
    data = {
        "generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N, "M": M,
        "returns": "rng.standard_normal((T, N)) * 0.01",
        "characteristics": "rng.standard_normal((T, N, M)), drawn after returns",
        "characteristics_passed": use_chars,
        "dataset": "AFMDataset(returns=returns, characteristics=characteristics if passed else omitted)",
        "input_sha256": _digest(returns, characteristics) if use_chars else _digest(returns),
    }
    return returns, (characteristics if use_chars else None), data


def _afm_case(factor_model: str, trading: str, **overrides: Any) -> CaseRun:
    # The engine path: the pre-redesign AFM class, now the private AFMEngine,
    # called exactly as the old public constructor was (dataset in __init__).
    # test_golden_afm.py also runs the public AFM estimator against the same fixtures.
    # ``overrides`` replace the common keywords (the early-stopping case).
    from dfm_python.dataset.afm_dataset import AFMDataset
    from dfm_python.models.afm._engine import AFMEngine

    returns, characteristics, data = afm_inputs(factor_model)

    with force_cpu():
        _reset_global_rngs()
        ds = AFMDataset(returns=returns, characteristics=characteristics)
        kwargs = {"n_factors": 3, "hist_len": 10, "max_epochs": 5, "seed": 0, **overrides}
        model = AFMEngine(dataset=ds, factor_model=factor_model, trading=trading, **kwargs)
        hyper = _attrs(model, AFM_ATTRS)
        state = model.fit()
        weights = model.predict()["weights"]

    outputs = {
        "residuals": _np(state.residuals),
        "book_returns": _np(state.book_returns),
        # float() of a float32 tensor: stored as float32 (exact) so the fallback
        # tolerance matches the precision the value was computed in.
        "sharpe": np.float32(state.sharpe),
        "explained_variance": np.float32(state.explained_variance),
        "num_iter": np.int64(state.num_iter),
        "converged": np.bool_(state.converged),
        # The last epoch's objective, a Python float (float64 of the float32 loss).
        "loss_now": np.float64(state.loss_now),
        "predict_weights": _np(weights),
    }
    for k, v in model.state_dict().items():
        outputs[f"sd/{k}"] = _np(v)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


# ----------------------------------------------------------------------- DFM
def _dfm_hyper(model: Any) -> dict[str, Any]:
    cfg = model._config
    config_fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
                     if not f.name.startswith("_")}
    return {
        "config": config_fields,
        "em_config": dataclasses.asdict(cfg.to_em_config()),
        "num_factors": model.num_factors,
        "r": model.r,
        "p": model.p,
        "blocks": model.blocks,
        "scaler": _class_name(model.scaler),
    }


def _dfm_fit(config_dict: dict[str, Any], df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.preprocessing import StandardScaler

    from dfm_python import DFM, DFMDataset
    from dfm_python.config import DFMConfig

    # Same calls as tutorial/tutorial_macro_dfm.py.
    _reset_global_rngs()
    config = DFMConfig.from_dict(dict(config_dict))
    dataset = DFMDataset(config=config, data=df, time_index="date")
    model = DFM(dataset=dataset, config=config, scaler=StandardScaler())
    hyper = _dfm_hyper(model)
    model.fit()
    result = model.result
    X, Z = model.predict(horizon=6)
    outputs = {
        "A": model.A, "C": model.C, "Q": model.Q, "R": model.R, "Z_0": model.Z_0, "V_0": model.V_0,
        "loglik": np.float64(result.loglik),
        "num_iter": np.int64(result.num_iter),
        "predict_X": X, "predict_Z": Z,
        "columns": np.array(list(dataset.variables.columns), dtype=str),
    }
    return {k: np.asarray(v) for k, v in outputs.items()}, hyper


def _dfm_synthetic() -> CaseRun:
    T, N, r, seed = 80, 5, 2, 0
    rng = np.random.default_rng(seed)
    f = np.zeros((T, r))
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal(r)
    loadings = rng.standard_normal((N, r))
    y = f @ loadings.T + 0.5 * rng.standard_normal((T, N))
    cols = [f"s{i}" for i in range(N)]
    df = pd.DataFrame(y, columns=cols)
    df.insert(0, "date", pd.date_range("2010-01-31", periods=T, freq="ME"))
    config = {
        "blocks": {"Block_Global": {"num_factors": r, "series": cols}},
        "frequency": {c: "m" for c in cols},
        "clock": "m",
        "max_iter": 20,
        "threshold": 1e-5,
    }
    data = {
        "generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N,
        "recipe": ("f[0]=0; f[t] = 0.7 f[t-1] + rng.standard_normal(2) for t=1..T-1; "
                   "loadings = rng.standard_normal((N, 2)); "
                   "y = f @ loadings.T + 0.5 * rng.standard_normal((T, N))"),
        "columns": cols, "index": "pd.date_range('2010-01-31', periods=T, freq='ME') in column 'date'",
        "config_dict": config,
        "input_sha256": _digest(df),
    }
    outputs, hyper = _dfm_fit(config, df)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_DFM_WM_ROWS = 400


def _dfm_macro_weekly_monthly() -> CaseRun:
    # tutorial_macro_dfm.py on the last rows of data/macro.csv: weekly clock with
    # monthly series, config/dfm_macro.yaml unchanged.
    config = _load_yaml("dfm_macro.yaml")
    df = _macro_tail(_DFM_WM_ROWS)
    data = _macro_recipe(rows=f"last {_DFM_WM_ROWS} rows", config="config/dfm_macro.yaml",
                         input_sha256=_digest(df))
    outputs, hyper = _dfm_fit(config, df)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


_MQ_MONTHLY = ["KOROCOINF", "KOEMPTOTO", "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOWRCCNSE"]
_MQ_QUARTERLY = ["KOEQUIPTE", "KOIMPCONA"]
_MQ_MONTHS = 120
_MQ_PERMUTED = ["KOEQUIPTE", "KOROCOINF", "KOWRCDURE", "KOIMPCONA",
                "KOEMPTOTO", "KOCONPRCF", "KOWRCCNSE", "KOHWRWEMP"]


def _mq_panel() -> pd.DataFrame:
    raw = pd.read_csv(MACRO_CSV)
    raw["date"] = pd.to_datetime(raw["date"])
    cols = _MQ_MONTHLY + _MQ_QUARTERLY
    df = raw.loc[raw["date"].dt.is_month_end, ["date"] + cols]
    df = df.dropna(how="all", subset=cols).iloc[-_MQ_MONTHS:].reset_index(drop=True)
    not_quarter_end = df["date"].dt.month % 3 != 0
    df.loc[not_quarter_end, _MQ_QUARTERLY] = np.nan
    return df


def _mq_config() -> dict[str, Any]:
    config = _load_yaml("dfm_macro.yaml")
    config.update({
        "clock": "m",
        "tent_weights": {"q:m": [1, 2, 3, 2, 1]},
        "blocks": {"Block_Global": {"num_factors": 2, "series": _MQ_MONTHLY + _MQ_QUARTERLY}},
        "frequency": {"m": list(_MQ_MONTHLY), "q": list(_MQ_QUARTERLY)},
    })
    return config


def _mq_recipe(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    return _macro_recipe(
        rows=(f"rows whose date is a month end, all-NaN rows dropped, last {_MQ_MONTHS}; "
              "quarterly series set to NaN where month % 3 != 0"),
        monthly=_MQ_MONTHLY, quarterly=_MQ_QUARTERLY, column_order=columns,
        config="config/dfm_macro.yaml with clock='m', tent_weights={'q:m': [1, 2, 3, 2, 1]}, "
               "one global block of 2 factors over monthly + quarterly series, grouped frequency",
        input_sha256=_digest(df))


def _dfm_macro_monthly_quarterly() -> CaseRun:
    df = _mq_panel()
    outputs, hyper = _dfm_fit(_mq_config(), df)
    return CaseRun(outputs=outputs, hyperparameters=hyper,
                   data=_mq_recipe(df, _MQ_MONTHLY + _MQ_QUARTERLY))


def _dfm_macro_monthly_quarterly_permuted() -> CaseRun:
    # Same panel and config, input columns permuted (quarterly interleaved with
    # monthly). Records the error raised today, or the outputs if none is raised.
    df = _mq_panel()[["date"] + _MQ_PERMUTED]
    data = _mq_recipe(df, _MQ_PERMUTED)
    try:
        outputs, hyper = _dfm_fit(_mq_config(), df)
    except Exception as exc:  # recorded, not swallowed: the test asserts the same error
        return CaseRun(outputs={}, hyperparameters={}, data=data,
                       raised={"type": type(exc).__name__, "message": str(exc)})
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


# ---------------------------------------------------------------------- DDFM
_DDFM_ATTRS = ["encoder_size", "decoder_size", "decoder_type", "activation", "learning_rate",
               "optimizer_type", "n_mc_samples", "window_size", "max_epoch_pre_train", "max_iter",
               "tolerance", "disp", "min_target_interporate_ratio", "interpolation_method",
               "interpolation_limit", "interpolation_limit_direction", "num_series", "input_dim",
               "output_dim", "initializer_seed", "lags_input", "device"]
_DDFM_ROWS = 400
_DDFM_MAX_ITER = 3
_DDFM_SEED = 3


def _ddfm_macro() -> CaseRun:
    from sklearn.preprocessing import StandardScaler

    from dfm_python import DDFM
    from dfm_python.config.schema.model import DDFMConfig
    from dfm_python.dataset.ddfm_dataset import DDFMDataset

    # tutorial_macro_ddfm.py on the last rows of data/macro.csv, with a few MCMC
    # iterations and an explicit seed (the YAML's seed is null, which would not
    # be reproducible).
    df = _macro_tail(_DDFM_ROWS)
    target = "KOEQUIPTE"
    covariates = [c for c in df.columns if c not in ("date", target)]
    data = _macro_recipe(rows=f"last {_DDFM_ROWS} rows", target=target, covariates=covariates,
                         config="config/ddfm_macro.yaml", input_sha256=_digest(df))

    with force_cpu():
        _reset_global_rngs()
        config = DDFMConfig.from_dict(_load_yaml("ddfm_macro.yaml"))
        dataset = DDFMDataset(data=df, time_idx="date", covariates=covariates, scaler=StandardScaler())
        model = DDFM(
            dataset=dataset, config=config,
            encoder_size=tuple(config.encoder_layers), decoder_type="linear",
            activation=config.activation, learning_rate=config.learning_rate, optimizer="Adam",
            n_mc_samples=config.n_mc_samples, window_size=config.window_size,
            max_iter=_DDFM_MAX_ITER, tolerance=config.tolerance, disp=config.disp, seed=_DDFM_SEED,
            interpolation_method=config.interpolation_method,
            interpolation_limit=config.interpolation_limit,
            interpolation_limit_direction=config.interpolation_limit_direction,
        )
        hyper = _attrs(model, _DDFM_ATTRS)
        hyper["scaler"] = _class_name(model.scaler)
        model.fit()
        model.build_state_space()
        X, Z = model.predict(horizon=6)

    st = model.training_state
    outputs = {
        "factors": _np(model.factors),
        "eps": _np(model.eps),
        "F": _np(st.F), "H": _np(st.H), "Q": _np(st.Q), "R": _np(st.R),
        "mu_0": _np(st.mu_0), "Sigma_0": _np(st.Sigma_0),
        "num_iter": np.int64(model._num_iter),
        "predict_X": _np(X), "predict_Z": _np(Z),
    }
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


# --------------------------------------------------------------------- iVDFM
_IVDFM_ATTRS = ["data_dim", "latent_dim", "context_dim", "time_context", "window",
                "encoder_hidden_dim", "encoder_n_hidden_layers", "decoder_type",
                "decoder_hidden_dim", "decoder_n_hidden_layers", "prior_hidden_dim",
                "prior_n_hidden_layers", "activation", "slope", "factor_order",
                "innovation_distribution", "decoder_var", "num_regimes", "regime_temperature",
                "beta_kl", "use_layer_norm", "mixing", "learning_rate", "optimizer_type",
                "optimizer_weight_decay", "optimizer_momentum", "batch_size", "max_epochs",
                "tolerance", "patience", "scheduler_type", "scheduler_step_size",
                "scheduler_gamma", "scheduler_patience", "scheduler_factor", "scheduler_min_lr",
                "seed", "device"]


def _ivdfm_hyper(model: Any, dataset: Any) -> dict[str, Any]:
    hyper = _attrs(model, _IVDFM_ATTRS)
    hyper["dataset"] = {"window": dataset.window, "stride": dataset.stride,
                        "context_dim": dataset.context_dim,
                        "scaler": _class_name(dataset.scaler)}
    return hyper


def _ivdfm_outputs(model: Any, horizon: int = 6) -> dict[str, np.ndarray]:
    pred = model.predict(horizon=horizon)
    outputs = {
        "factors": _np(model.factors),
        "innovations": _np(model.innovations),
        "num_iter": np.int64(model._num_iter),
        "predict": _np(pred),
    }
    regime_weights = getattr(model, "regime_weights", None)
    if regime_weights is not None:
        outputs["regime_weights"] = _np(regime_weights)
    return outputs


_IVDFM_ROWS = 400
_IVDFM_MAX_EPOCHS = 3


def _ivdfm_macro() -> CaseRun:
    from dfm_python import iVDFM
    from dfm_python.config.schema.model import iVDFMConfig
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset

    # tutorial_macro_ivdfm.py on the last rows of data/macro.csv. The tutorial
    # itself does not run today (it passes context_dim to iVDFMDataset), so the
    # dataset gets the tutorial's data, covariates, window (sequence_length) and
    # scaler, and the model is built in the constructor-dataset order.
    yaml_cfg = _load_yaml("ivdfm_macro.yaml")
    df = _macro_tail(_IVDFM_ROWS)
    targets = ["KOEQUIPTE", "KOEMPTOTO", "KOHWRWEMP"]
    covariates = [c for c in df.columns if c != "date" and c not in targets][:32]
    df_data = df[["date"] + targets + covariates].copy()
    window = int(yaml_cfg.get("sequence_length") or 100)
    data = _macro_recipe(rows=f"last {_IVDFM_ROWS} rows", targets=targets, covariates=covariates,
                         window=window, config="config/ivdfm_macro.yaml",
                         input_sha256=_digest(df_data))

    with force_cpu():
        _reset_global_rngs()
        config = iVDFMConfig.from_dict(yaml_cfg)
        dataset = iVDFMDataset(data=df_data, time_idx="date", covariates=covariates,
                               window=window, scaler=config.scaler)
        model = iVDFM(dataset=dataset, config=config, max_epochs=_IVDFM_MAX_EPOCHS)
        hyper = _ivdfm_hyper(model, dataset)
        model.fit()
        outputs = _ivdfm_outputs(model)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


def _ivdfm_regimes() -> CaseRun:
    from dfm_python import iVDFM
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset

    T, N, seed = 60, 4, 0
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((T, N))
    data = {"generator": "numpy.random.default_rng", "seed": seed, "T": T, "N": N,
            "recipe": "y = rng.standard_normal((T, N)); iVDFMDataset(data=y, window=24, stride=1, time_context=1)",
            "input_sha256": _digest(y)}

    with force_cpu():
        _reset_global_rngs()
        dataset = iVDFMDataset(data=y, window=24, stride=1, time_context=1)
        model = iVDFM(dataset=dataset, num_factors=3, num_regimes=2, factor_order=2,
                      max_epochs=2, batch_size=16)
        hyper = _ivdfm_hyper(model, dataset)
        model.fit()
        outputs = _ivdfm_outputs(model)
    return CaseRun(outputs=outputs, hyperparameters=hyper, data=data)


# ------------------------------------------------------------------ registry
CASES: dict[str, Case] = {c.name: c for c in [
    Case("afm_attention_longconv", "afm", lambda: _afm_case("attention", "longconv")),
    Case("afm_pca_longconv", "afm", lambda: _afm_case("pca", "longconv")),
    Case("afm_pca_ou", "afm", lambda: _afm_case("pca", "ou")),
    # patience stops training early (at epoch 17 of 40); the learning rate is
    # high enough that the loss stalls.
    Case("afm_pca_longconv_patience", "afm",
         lambda: _afm_case("pca", "longconv", max_epochs=40, learning_rate=0.1, patience=3)),
    Case("dfm_synthetic", "dfm", _dfm_synthetic),
    Case("dfm_macro_weekly_monthly", "dfm", _dfm_macro_weekly_monthly),
    Case("dfm_macro_monthly_quarterly", "dfm", _dfm_macro_monthly_quarterly),
    Case("dfm_macro_monthly_quarterly_permuted", "dfm", _dfm_macro_monthly_quarterly_permuted),
    Case("ddfm_macro", "ddfm", _ddfm_macro),
    Case("ivdfm_macro", "ivdfm", _ivdfm_macro),
    Case("ivdfm_regimes", "ivdfm", _ivdfm_regimes),
]}


def cases_for(model: str) -> list[Case]:
    return [c for c in CASES.values() if c.model == model]


def check_case(name: str, run: CaseRun) -> None:
    """Compare a fresh run of case ``name`` with its fixture."""
    from ._golden import assert_golden, load_golden

    manifest, _ = load_golden(name)
    assert normalized(run.data) == manifest["data"], f"golden '{name}': input data differs from the fixture"
    assert normalized(run.raised) == manifest["raised"], (
        f"golden '{name}': raised {run.raised}, fixture recorded {manifest['raised']}")
    assert normalized(run.hyperparameters) == manifest["hyperparameters"], (
        f"golden '{name}': constructed hyperparameters differ from the fixture")
    assert_golden(run.outputs, name)

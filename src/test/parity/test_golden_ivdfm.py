"""Golden tests for iVDFM.

``test_golden_ivdfm_engine`` runs the engine (the pre-redesign class, now
``iVDFMEngine``) exactly as the fixtures were captured.
``test_golden_ivdfm_estimator`` runs the public ``iVDFM`` estimator, built from
the constructor, on the same data with the recorded hyperparameters (mapped to
the canonical names, ``device="cpu"``). The engine it builds must hold the
recorded hyperparameters when training starts, reproduce every golden output,
and the estimator's ``predict`` must equal the fixture's model-space forecast
inverse-transformed with ``scaler_``, bitwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from dfm_python import PanelData, iVDFM
from dfm_python.models.ivdfm._engine import iVDFMEngine

from ._cases import (
    IVDFM_EXOG_HORIZON,
    _IVDFM_ROWS,
    _ivdfm_hyper,
    _ivdfm_outputs,
    _macro_tail,
    _reset_global_rngs,
    cases_for,
    check_case,
    ivdfm_exog_context_inputs,
    ivdfm_time_features,
    normalized,
)
from ._golden import assert_golden, load_golden

_SCALER_NAMES = {None: None, "StandardScaler": "standard"}


@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the hyperparameters the engine held in the fixture."""
    ds = hyper["dataset"]
    window = hyper["window"]
    return {
        "n_factors": hyper["latent_dim"],
        "window": window,
        # stride=None is window // 2, the dataset default.
        "stride": None if ds["stride"] == max(1, ds["window"] // 2) else ds["stride"],
        "time_context": hyper["time_context"],
        "encoder_hidden_dim": hyper["encoder_hidden_dim"],
        "encoder_n_hidden_layers": hyper["encoder_n_hidden_layers"],
        "decoder_type": hyper["decoder_type"],
        "decoder_hidden_dim": hyper["decoder_hidden_dim"],
        "decoder_n_hidden_layers": hyper["decoder_n_hidden_layers"],
        "prior_hidden_dim": hyper["prior_hidden_dim"],
        "prior_n_hidden_layers": hyper["prior_n_hidden_layers"],
        "activation": hyper["activation"],
        "slope": hyper["slope"],
        "factor_order": hyper["factor_order"],
        "innovation_distribution": hyper["innovation_distribution"],
        "decoder_var": hyper["decoder_var"],
        "n_regimes": hyper["num_regimes"],
        "regime_temperature": hyper["regime_temperature"],
        "beta_kl": hyper["beta_kl"],
        "use_layer_norm": hyper["use_layer_norm"],
        "mixing": hyper["mixing"],
        "learning_rate": hyper["learning_rate"],
        "optimizer": hyper["optimizer_type"],
        "optimizer_weight_decay": hyper["optimizer_weight_decay"],
        "optimizer_momentum": hyper["optimizer_momentum"],
        "batch_size": hyper["batch_size"],
        "max_epochs": hyper["max_epochs"],
        "tol": hyper["tolerance"],
        "patience": hyper["patience"],
        "scheduler_type": hyper["scheduler_type"],
        "scheduler_step_size": hyper["scheduler_step_size"],
        "scheduler_gamma": hyper["scheduler_gamma"],
        "scheduler_patience": hyper["scheduler_patience"],
        "scheduler_factor": hyper["scheduler_factor"],
        "scheduler_min_lr": hyper["scheduler_min_lr"],
        "scaler": _SCALER_NAMES[ds["scaler"]],
        "random_state": hyper["seed"],
        "device": "cpu",
    }


def _panel(case: str, manifest: dict) -> tuple[PanelData, Any]:
    """The case's data as PanelData, and the future exog (or None)."""
    recipe = manifest["data"]
    if case == "ivdfm_macro":
        # The case also passed covariates, which the dataset drops from the
        # targets and context; the arrays it builds are the same without them.
        df = _macro_tail(_IVDFM_ROWS).set_index("date")
        assert recipe["rows"] == f"last {_IVDFM_ROWS} rows"
        return PanelData(y=df[recipe["targets"]]), None
    if case == "ivdfm_regimes":
        y = np.random.default_rng(recipe["seed"]).standard_normal((recipe["T"], recipe["N"]))
        return PanelData.from_arrays(y), None
    if case == "ivdfm_exog_context":
        df, x_future, data = ivdfm_exog_context_inputs()
        assert normalized(data) == recipe
        df = df.set_index("date")
        return PanelData(y=df[["y0", "y1", "y2"]], exog=df[["x0", "x1"]]), x_future
    raise AssertionError(f"no PanelData recipe for {case}")


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's hyperparameters (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = iVDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_ivdfm_hyper(self, self._dataset)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(iVDFMEngine, "fit", fit)
    return seen


@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm_estimator(case, pinned, engine_at_fit):
    manifest, fixture = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data, x_future = _panel(case.name, manifest)

    model = iVDFM(**params)
    assert model.get_params() == {**iVDFM().get_params(), **params}

    _reset_global_rngs()
    assert model.fit(data) is model

    # Every hyperparameter reached the engine unchanged (read when its training starts).
    assert engine_at_fit == [hyper]

    engine = model.network_
    outputs = _ivdfm_outputs(engine, horizon=IVDFM_EXOG_HORIZON)
    if x_future is not None:
        context = np.hstack([ivdfm_time_features(IVDFM_EXOG_HORIZON, hyper["time_context"], hyper["window"]), x_future])
        outputs["predict_futr"] = engine.predict(horizon=IVDFM_EXOG_HORIZON, context_data=context)
    assert_golden(outputs, case.name)

    # The estimator's forecasts are the engine's, in data units.
    def data_units(model_space: np.ndarray) -> np.ndarray:
        values = model_space.astype(np.float64)
        return values if model.scaler_ is None else model.scaler_.inverse_transform(values)

    if x_future is None:
        forecast = model.predict(IVDFM_EXOG_HORIZON)
    else:
        with pytest.warns(UserWarning, match="futr_exog"):
            forecast = model.predict(IVDFM_EXOG_HORIZON)
        futr = model.predict(IVDFM_EXOG_HORIZON, futr_exog=pd.DataFrame(x_future, columns=["x0", "x1"]))
        assert futr.to_numpy().tobytes() == data_units(fixture["predict_futr"]).tobytes()
        np.testing.assert_array_equal(
            model.predict(IVDFM_EXOG_HORIZON, futr_exog=x_future).to_numpy(), futr.to_numpy(), strict=True
        )
    assert forecast.to_numpy().tobytes() == data_units(fixture["predict"]).tobytes()
    assert list(forecast.columns) == list(data.series_names)
    assert all(dtype == np.float64 for dtype in forecast.dtypes)

    # Fitted attributes come from the same run.
    assert model.n_iter_ == int(fixture["num_iter"])
    assert model.converged_ == bool(fixture["converged"])
    assert model.n_factors_ == hyper["latent_dim"]
    assert (model.window_, model.stride_) == (hyper["dataset"]["window"], hyper["dataset"]["stride"])
    assert model.context_dim_ == hyper["dataset"]["context_dim"]
    assert model.time_context_ == hyper["time_context"]
    n_windows, window, r = fixture["factors"].shape
    assert model.factors_.shape == (n_windows * window, r)
    # factors_ holds the engine's windows, reordered chronologically.
    rows = model.factors_.to_numpy().reshape(n_windows, window, r)
    engine_rows = {fixture["factors"][k].astype(np.float64).tobytes() for k in range(n_windows)}
    assert {rows[w].tobytes() for w in range(n_windows)} == engine_rows

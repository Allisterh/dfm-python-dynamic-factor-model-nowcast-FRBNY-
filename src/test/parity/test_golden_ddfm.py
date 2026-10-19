"""Golden tests for DDFM.

``test_golden_ddfm_engine`` runs the engine (the pre-redesign class, now
``DDFMEngine``) exactly as the fixtures were captured.
``test_golden_ddfm_estimator`` runs the public ``DDFM`` estimator, built from
the constructor, on the same data as ``PanelData`` (covariates in ``exog``)
with the recorded hyperparameters mapped to the canonical names
(``device="cpu"``). The engine it builds must hold the recorded
hyperparameters when training starts and reproduce every golden output of the
fit bitwise; the estimator's ``predict``, ``predict_factors``, ``result_.C``
and ``factors_`` must equal the fixture's forecasts, loadings and factors
bitwise. The fixture's update outputs are not compared here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfm_python import DDFM, PanelData
from dfm_python.models.ddfm._engine import DDFMEngine

from ._cases import (
    DDFM_UPDATE_KEYS,
    _DDFM_HOLDOUT,
    _DDFM_HORIZON,
    _DDFM_ROWS,
    _FINANCE_TARGET,
    _ddfm_hyper,
    _digest,
    _macro_tail,
    _reset_global_rngs,
    cases_for,
    check_case,
    ddfm_engine_outputs,
    ddfm_exchange_panel,
    ddfm_finance_panel,
    normalized,
)
from ._golden import assert_golden, load_golden

_SCALER_NAMES = {None: None, "StandardScaler": "standard"}


@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the hyperparameters the engine held in the fixture."""
    encoder_size = hyper["encoder_size"]
    return {
        "n_factors": encoder_size[-1],
        "encoder_layers": tuple(encoder_size[:-1]),
        "decoder_type": hyper["decoder_type"],
        "activation": hyper["activation"],
        "learning_rate": hyper["learning_rate"],
        "optimizer": hyper["optimizer_type"],
        "n_mc_samples": hyper["n_mc_samples"],
        "window_size": hyper["window_size"],
        "max_iter": hyper["max_iter"],
        "tol": hyper["tolerance"],
        "interpolation_method": hyper["interpolation_method"],
        "interpolation_limit": hyper["interpolation_limit"],
        "interpolation_limit_direction": hyper["interpolation_limit_direction"],
        "scaler": _SCALER_NAMES[hyper["scaler"]],
        "random_state": hyper["initializer_seed"],
        "device": "cpu",
    }


def _panel(case: str, manifest: dict) -> PanelData:
    """The data the case fitted, as PanelData: the targets in ``y`` and the covariates in ``exog``."""
    recipe = manifest["data"]
    if case == "ddfm_macro":
        df = _macro_tail(_DDFM_ROWS)
        assert _digest(df) == recipe["input_sha256"]
        frame = df.set_index("date")
        return PanelData(y=frame[[recipe["target"]]], exog=frame[recipe["covariates"]])
    if case == "ddfm_exchange":
        fit_df = ddfm_exchange_panel().iloc[:-_DDFM_HOLDOUT]
        assert _digest(fit_df) == recipe["fit_input_sha256"]
        assert recipe["covariates"] == [] and recipe["targets"] == list(fit_df.columns)
        return PanelData(y=fit_df)
    if case == "ddfm_finance":
        panel, covariates = ddfm_finance_panel()
        fit_df = panel.iloc[:-_DDFM_HOLDOUT]
        assert _digest(fit_df) == recipe["fit_input_sha256"]
        assert recipe["targets"] == [_FINANCE_TARGET] and recipe["covariates"] == covariates
        frame = fit_df.set_index("date")
        return PanelData(y=frame[[_FINANCE_TARGET]], exog=frame[covariates])
    raise AssertionError(f"no PanelData recipe for {case}")


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's hyperparameters (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = DDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_ddfm_hyper(self)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "fit", fit)
    return seen


@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_estimator(case, pinned, engine_at_fit):
    manifest, fixture = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data = _panel(case.name, manifest)

    model = DDFM(**params)
    assert model.get_params() == {**DDFM().get_params(), **params}

    _reset_global_rngs()
    assert model.fit(data) is model

    # Every hyperparameter reached the engine unchanged (read when its training starts).
    assert engine_at_fit == [hyper]

    # The engine reproduces every pinned output of the fit; the update outputs
    # (an engine predict(data=...), which mutates it) are left out here.
    assert_golden(ddfm_engine_outputs(model.network_), case.name, exclude=DDFM_UPDATE_KEYS)

    # The estimator's forecasts are the engine's, in data units.
    forecast = model.predict(_DDFM_HORIZON)
    assert forecast.to_numpy().tobytes() == fixture["predict_X"].astype(np.float64).tobytes()
    assert forecast.shape == fixture["predict_X"].shape
    assert list(forecast.columns) == list(data.series_names)
    assert all(dtype == np.float64 for dtype in forecast.dtypes)
    factors = model.predict_factors(_DDFM_HORIZON)
    assert factors.to_numpy().tobytes() == fixture["predict_Z"].astype(np.float64).tobytes()
    assert factors.shape == fixture["predict_Z"].shape
    assert forecast.index.equals(factors.index) and forecast.index[0] > data.index[-1]

    # result_ and the fitted attributes come from the same run.
    np.testing.assert_array_equal(model.result_.C, fixture["H"], strict=True)
    np.testing.assert_array_equal(model.result_.Z, fixture["result_Z"], strict=True)
    np.testing.assert_array_equal(
        model.factors_.to_numpy(), fixture["result_Z"].astype(np.float64), strict=True
    )
    assert model.factors_.index.equals(data.index)
    assert model.n_iter_ == int(fixture["num_iter"])
    assert model.converged_ == bool(fixture["converged"])
    assert model.n_factors_ == hyper["encoder_size"][-1]
    np.testing.assert_array_equal(model.scaler_.mean_, fixture["scaler_mean"], strict=True)
    np.testing.assert_array_equal(model.scaler_.scale_, fixture["scaler_scale"], strict=True)

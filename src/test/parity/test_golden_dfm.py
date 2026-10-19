"""Golden tests for DFM.

``test_golden_dfm_engine`` runs the engine (the pre-redesign class, now
``DFMEngine``) exactly as the fixtures were captured, update path included
(the fixtures were captured with a test-only import shim around ``update``,
whose broken import the engine move fixed), and must reproduce every output
bitwise on the capture platform.

``test_golden_dfm_estimator`` runs the public ``DFM`` estimator on the same
data as ``PanelData`` (the series in the user's column order, their
frequencies in ``freq``), built from the constructor with the recorded
hyperparameters under the canonical names. The engine it builds must hold the
recorded config (every ``DFMConfig`` field, the EM config, the block
structure) when EM starts and reproduce every golden output of the fit
bitwise; the estimator's ``result_``, ``factors_``, ``scaler_`` and
``internal_feature_names_`` must equal the fixture's, ``predict(6)`` must equal
the fixture's forecast reordered to the user's columns, and
``predict_factors(6)`` the fixture's state forecast. The permuted case passes
the series in the permuted order of its fixture; the estimator's internal
order is the fixture's dataset order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfm_python import DFM, PanelData
from dfm_python.models.dfm._engine import DFMEngine

from ._cases import (
    _DFM_WM_ROWS,
    _MQ_MONTHLY,
    _MQ_PERMUTED,
    _MQ_QUARTERLY,
    _MQ_UPDATE_ROWS,
    DFM_HORIZON,
    DFM_UPDATE_KEYS,
    _dfm_hyper,
    _digest,
    _macro_tail,
    _mq_panel,
    cases_for,
    check_case,
    dfm_fit_outputs,
    dfm_synthetic_inputs,
    normalized,
)
from ._golden import assert_golden, load_golden


@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_engine(case, pinned):
    check_case(case.name, case.run())


def _frame(case: str, recipe: dict) -> pd.DataFrame:
    """The frame the case fitted (a ``date`` column, then the series in the case's order)."""
    frames = {
        "dfm_synthetic": lambda: dfm_synthetic_inputs()[0],
        "dfm_macro_weekly_monthly": lambda: _macro_tail(_DFM_WM_ROWS),
        "dfm_macro_monthly_quarterly": _mq_panel,
        "dfm_macro_monthly_quarterly_permuted": lambda: _mq_panel()[["date"] + _MQ_PERMUTED],
        "dfm_macro_monthly_quarterly_update": lambda: _mq_panel().iloc[:-_MQ_UPDATE_ROWS],
    }
    df = frames[case]()
    assert _digest(df) == recipe["input_sha256"], f"{case}: the frame differs from the fixture's input"
    return df


def _panel(case: str, manifest: dict) -> PanelData:
    """The fitted frame as PanelData: the series in the case's column order, their frequencies in ``freq``."""
    df = _frame(case, manifest["data"])
    y = df.set_index("date")
    frequency = manifest["hyperparameters"]["config"]["frequency"]
    return PanelData(y=y, freq={name: frequency[name] for name in y.columns})


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the config the engine held in the fixture."""
    config = hyper["config"]
    assert hyper["scaler"] == "StandardScaler"
    return {
        "blocks": {name: {"n_factors": block["num_factors"], "series": list(block["series"])}
                   for name, block in config["blocks"].items()},
        "clock": config["clock"],
        "tent_weights": config["tent_weights"],
        "max_iter": config["max_iter"],
        "tol": config["threshold"],
        "ar_clip": config["ar_clip"],
        "data_clip": config["data_clip"],
        "regularization": config["regularization"],
        "damping_factor": config["damping_factor"],
        "scaler": "standard",
    }


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's config and structure (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = DFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_dfm_hyper(self)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFMEngine, "fit", fit)
    return seen


@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_estimator(case, pinned, engine_at_fit):
    manifest, fixture = load_golden(case.name)
    assert manifest["raised"] is None
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data = _panel(case.name, manifest)
    user_order = list(data.series_names)
    if case.name == "dfm_macro_monthly_quarterly_permuted":
        assert user_order == _MQ_PERMUTED
    elif case.name.startswith("dfm_macro_monthly_quarterly"):
        assert user_order == _MQ_MONTHLY + _MQ_QUARTERLY

    model = DFM(**params)
    assert model.get_params() == {**DFM().get_params(), **params}
    assert model.fit(data) is model

    # The engine held the recorded config, EM config and block structure when EM started.
    assert engine_at_fit == [hyper]

    # The engine reproduces every pinned output of the fit (the update path is the next item's).
    engine = model.engine_
    X, Z = engine.predict(horizon=DFM_HORIZON)
    outputs = dfm_fit_outputs(engine, X, Z)
    outputs["columns"] = np.array(list(model.internal_feature_names_), dtype=str)
    assert_golden(outputs, case.name, exclude=[key for key in DFM_UPDATE_KEYS if key in fixture])

    # The internal order is the fixture's dataset order.
    columns = [str(c) for c in fixture["columns"]]
    assert list(model.internal_feature_names_) == columns
    assert list(model.feature_names_in_) == user_order

    # result_ is a copy of the engine's result, in the internal order.
    result = model.result_
    assert result is not engine.result
    for name in ("A", "C", "Q", "R", "Z_0", "V_0"):
        np.testing.assert_array_equal(getattr(result, name), fixture[name], strict=True)
    np.testing.assert_array_equal(result.Z, fixture["result_Z"], strict=True)
    np.testing.assert_array_equal(result.x_sm, fixture["result_x_sm"], strict=True)
    assert np.float64(result.loglik).tobytes() == fixture["loglik"].tobytes()
    np.testing.assert_array_equal(model.factors_.to_numpy(), fixture["result_Z"].astype(np.float64), strict=True)
    assert list(model.factors_.columns) == [f"state_{i}" for i in range(fixture["result_Z"].shape[1])]
    assert model.factors_.index.equals(data.index)
    np.testing.assert_array_equal(model.scaler_.mean_, fixture["scaler_mean"], strict=True)
    np.testing.assert_array_equal(model.scaler_.scale_, fixture["scaler_scale"], strict=True)
    assert model.n_iter_ == int(fixture["num_iter"])
    assert model.converged_ == bool(fixture["converged"])
    assert np.float64(model.loglik_).tobytes() == fixture["loglik"].tobytes()
    assert model.n_factors_ == hyper["num_factors"]
    assert model.freq_ == {name: hyper["config"]["frequency"][name] for name in user_order}

    # predict(h): the engine's forecast with the columns in the user's order.
    forecast = model.predict(DFM_HORIZON)
    expected = pd.DataFrame(fixture["predict_X"], columns=columns)[user_order]
    assert list(forecast.columns) == user_order
    assert forecast.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert all(dtype == np.float64 for dtype in forecast.dtypes)
    assert forecast.index[0] > data.index[-1] and len(forecast.index) == DFM_HORIZON

    factors = model.predict_factors(DFM_HORIZON)
    np.testing.assert_array_equal(factors.to_numpy(), fixture["predict_Z"], strict=True)
    assert list(factors.columns) == list(model.factors_.columns)
    assert factors.index.equals(forecast.index)

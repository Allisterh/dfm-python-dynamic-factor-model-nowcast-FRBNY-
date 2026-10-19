"""Tests for the private DFM engine (models.dfm._engine.DFMEngine).

The engine is the pre-redesign DFM class, moved and renamed, with one import
fixed: ``update()`` imported ``..numeric.validator`` (the non-existent
``dfm_python.models.numeric``) and always raised ``ModuleNotFoundError``.
These tests cover ``update()`` without any import shim (several rows, through
the filter-and-smooth branch, and one row, through pykalman's
``filter_update``), the purity of ``predict(horizon, data=)``, and the public
names that still point at the engine until the estimator exists.
"""

import copy
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import dfm_python
from dfm_python.config import DFMConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.models.dfm._engine import DFMEngine
from dfm_python.ssm.kalman import DFMKalmanFilter

_PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")


def _panel(T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    loadings = rng.standard_normal(4)
    y = np.outer(f, loadings) + 0.5 * rng.standard_normal((T, 4))
    df = pd.DataFrame(y, columns=[f"s{i}" for i in range(4)])
    df.insert(0, "date", pd.date_range("2015-01-31", periods=T, freq="ME"))
    return df


@pytest.fixture
def fitted():
    df = _panel(60, seed=0)
    cols = [c for c in df.columns if c != "date"]
    config = DFMConfig.from_dict({
        "blocks": {"Block_Global": {"num_factors": 1, "series": cols}},
        "frequency": {c: "m" for c in cols},
        "clock": "m",
        "max_iter": 5,
        "threshold": 1e-5,
    })
    dataset = DFMDataset(config=config, data=df, time_index="date")
    model = DFMEngine(dataset=dataset, config=config, scaler=StandardScaler())
    model.fit()
    model.get_result()  # computes and caches the result, as update() and predict() do
    return model


def _new_rows(n: int) -> np.ndarray:
    return _panel(60 + n, seed=1).iloc[-n:, 1:].to_numpy(np.float64)


def _start_state(model: DFMEngine) -> tuple[np.ndarray, np.ndarray]:
    """The state update() starts from: last smoothed mean, last filtered covariance."""
    cov = model._create_kalman_filter().filter(np.ma.masked_invalid(model.data_processed))[1]
    return model.result.Z[-1].copy(), cov[-1].copy()


def _params(model: DFMEngine) -> dict[str, np.ndarray]:
    return {k: getattr(model, k).copy() for k in _PARAMS}


def _assert_params_unchanged(model: DFMEngine, before: dict[str, np.ndarray]) -> None:
    for k in _PARAMS:
        np.testing.assert_array_equal(getattr(model, k), before[k], err_msg=k, strict=True)


def test_update_multiple_rows(fitted):
    model = fitted
    params = _params(model)
    Z_before = model.result.Z.copy()
    data_before = model.data_processed.copy()
    Z_last, V_last = _start_state(model)
    new = _new_rows(4)
    new_std = model.scaler.transform(new)
    expected = model._create_kalman_filter(Z_last, V_last).filter_and_smooth(
        np.ma.masked_invalid(new_std), compute_loglik=False)[0]

    assert model.update(new) is None

    result = model.result
    assert result.Z.shape == (Z_before.shape[0] + 4, Z_before.shape[1])
    np.testing.assert_array_equal(result.Z[:-4], Z_before, strict=True)
    np.testing.assert_array_equal(result.Z[-4:], expected, strict=True)
    np.testing.assert_array_equal(result.x_sm, result.Z @ result.C.T, strict=True)
    np.testing.assert_array_equal(model.data_processed, np.vstack([data_before, new_std]), strict=True)
    _assert_params_unchanged(model, params)
    # The validator import resolved on its own, with no module aliased in.
    assert "dfm_python.models.numeric" not in sys.modules


def test_update_single_row_uses_filter_update(fitted):
    model = fitted
    params = _params(model)
    Z_before = model.result.Z.copy()
    Z_last, V_last = _start_state(model)
    new = _new_rows(1)
    new_std = model.scaler.transform(new)
    pk = model._create_kalman_filter(Z_last, V_last)._pykalman
    expected, _ = pk.filter_update(filtered_state_mean=Z_last, filtered_state_covariance=V_last,
                                   observation=np.ma.masked_invalid(new_std[0]))

    with mock.patch.object(DFMKalmanFilter, "filter_and_smooth",
                           side_effect=AssertionError("one row must not use filter_and_smooth")):
        model.update(new)

    Z = model.result.Z
    assert Z.shape == (Z_before.shape[0] + 1, Z_before.shape[1])
    np.testing.assert_array_equal(Z[:-1], Z_before, strict=True)
    np.testing.assert_array_equal(Z[-1], np.asarray(expected, dtype=np.float64), strict=True)
    assert model.data_processed.shape[0] == Z.shape[0]
    _assert_params_unchanged(model, params)


def test_update_retrain_runs_em_on_all_rows(fitted):
    model = fitted
    params = _params(model)
    T = model.data_processed.shape[0]
    model.update(_new_rows(4), retrain=True)
    assert model.data_processed.shape[0] == T + 4
    assert model.result.Z.shape[0] == T + 4
    assert model.result.num_iter == 1
    assert not np.array_equal(model.A, params["A"])


def test_predict_with_data_leaves_the_model_unchanged(fitted):
    model = fitted
    state = copy.deepcopy(model.training_state)
    data_before = model.data_processed.copy()
    Z_before = model.result.Z.copy()
    X0, Z0 = model.predict(horizon=3)

    X_d, Z_d = model.predict(horizon=3, data=_new_rows(4))
    assert X_d.shape == (3, 4) and Z_d.shape == (3, Z_before.shape[1])
    assert not np.array_equal(Z_d, Z0)

    for k in _PARAMS:
        np.testing.assert_array_equal(getattr(model.training_state, k), getattr(state, k), err_msg=k, strict=True)
    np.testing.assert_array_equal(model.data_processed, data_before, strict=True)
    np.testing.assert_array_equal(model._result.Z, Z_before, strict=True)
    X1, Z1 = model.predict(horizon=3)
    np.testing.assert_array_equal(X1, X0, strict=True)
    np.testing.assert_array_equal(Z1, Z0, strict=True)


def test_public_dfm_names_the_engine_until_the_estimator_exists():
    import dfm_python.models
    import dfm_python.models.dfm

    assert dfm_python.DFM is DFMEngine
    assert dfm_python.models.DFM is DFMEngine
    assert dfm_python.models.dfm.DFM is DFMEngine

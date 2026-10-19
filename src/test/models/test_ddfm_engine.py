"""Tests for the private DDFM engine (models.ddfm._engine.DDFMEngine).

The engine is the pre-redesign DDFM class with three edits that change no
math: a ``device`` keyword that overrides the automatic CUDA/CPU choice, the
pure ``_forecast`` extraction from ``predict``, and ``target_series`` /
``covariates`` copied from the dataset so that ``_update`` needs no dataset.
"""

import copy
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.utils.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _frame(T: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((T, 4)), columns=["cov", "a", "b", "c"])


def _dataset(frame: pd.DataFrame, scaler="standard") -> DDFMDataset:
    return DDFMDataset(
        data=frame,
        time_idx="index",
        covariates=["cov"],
        scaler=StandardScaler() if scaler == "standard" else scaler,
    )


def _engine(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    params = dict(encoder_size=(4, 2), n_mc_samples=2, max_iter=2, device="cpu")
    params.update(kwargs)
    return DDFMEngine(dataset, **params)


def _fitted(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    engine = _engine(dataset, **kwargs)
    engine.fit()
    engine.build_state_space()
    return engine


def test_target_series_and_covariates_are_copied_from_the_dataset():
    dataset = _dataset(_frame())
    engine = _engine(dataset)
    assert engine.target_series == ["a", "b", "c"]
    assert engine.covariates == ["cov"]
    assert engine.target_series is not dataset.target_series
    assert engine.covariates is not dataset.covariates


def test_device_keyword_overrides_the_automatic_choice():
    dataset = _dataset(_frame())
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        engine = _engine(dataset, device="cpu")
    assert engine.device == torch.device("cpu")
    # Fit outside the patch: torch's optimizer step itself queries CUDA when
    # it believes CUDA is available, which fails on a machine without it.
    engine.fit()
    params = list(engine.parameters())
    assert params
    assert all(p.device.type == "cpu" for p in params)


def test_device_none_keeps_the_automatic_choice():
    dataset = _dataset(_frame())
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert _engine(dataset, device=None).device == torch.device("cpu")
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        assert _engine(dataset, device=None).device == torch.device("cuda")


def test_forecast_equals_predict_bitwise():
    engine = _fitted(_dataset(_frame()))
    y_forecast, Z_forecast = engine._forecast(6)
    y_pred, Z_pred = engine.predict(horizon=6)
    assert y_forecast.shape == (6, 3)
    assert Z_forecast.shape == (6, 2)
    for got, want in ((y_forecast, y_pred), (Z_forecast, Z_pred)):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()


def test_forecast_without_scaler_is_the_identity_and_predict_still_raises():
    engine = _fitted(_dataset(_frame(), scaler=None))
    assert engine.scaler is None
    y_forecast, Z_forecast = engine._forecast(6)
    H = np.nan_to_num(engine.training_state.H, nan=0.0, posinf=0.0, neginf=0.0)
    expected = np.nan_to_num(Z_forecast @ H.T)
    assert y_forecast.tobytes() == expected.tobytes()
    with pytest.raises(ConfigurationError):
        engine.predict(horizon=6)


def test_update_needs_no_dataset():
    frame = _frame(T=72)
    engine = _fitted(_dataset(frame.iloc[:60]))
    detached = copy.deepcopy(engine)
    detached._dataset = None

    engine._update(_dataset(frame))
    detached._update(_dataset(frame))

    assert engine.factors.shape[-2] == 60 + 72
    assert detached.factors.tobytes() == engine.factors.tobytes()
    assert detached.scaler.mean_.tobytes() == engine.scaler.mean_.tobytes()
    assert detached.scaler.scale_.tobytes() == engine.scaler.scale_.tobytes()

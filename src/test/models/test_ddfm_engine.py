"""Tests for the private DDFM engine (models.ddfm._engine.DDFMEngine).

The engine is the pre-redesign DDFM class with three edits that change no
math: a ``device`` keyword that overrides the automatic CUDA/CPU choice, the
pure ``_forecast`` extraction from ``predict``, and ``target_series`` /
``covariates`` copied from the dataset so that ``_update`` needs no dataset.

The engine internals that the old ``test_ddfm.py`` covered (encoder sizes,
the pre-training inputs, the decoder helpers, the variance diagnostics, the
dimension constants and the ``infer_*`` checkpoint helpers) are tested here
against the engine; its workflow tests moved to ``test_ddfm_estimator.py``.
"""

import copy
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from dfm_python.config.constants import DEFAULT_ENCODER_LAYERS, MIN_DDFM_TIME_STEPS, MIN_VARIABLES
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.numeric.statistic import diagnose_variance_collapse
from dfm_python.utils.checkpoint import infer_ddfm_input_dim, infer_input_dim_from_data
from dfm_python.utils.errors import (
    ConfigurationError,
    DataError,
    DataValidationError,
    ModelNotTrainedError,
)


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


# -- engine internals (ported from the old test_ddfm.py) ----------------------------


def _all_targets(num_series: int = 5, time_steps: int = 10, scaler=None, seed: int = 42) -> DDFMDataset:
    """A dataset whose columns are all targets (no covariates), as the old tests built it."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame(rng.randn(time_steps, num_series), columns=[f"series_{i}" for i in range(num_series)])
    return DDFMDataset(data=data, time_idx="index", scaler=scaler)


def _default_engine(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    return DDFMEngine(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), device="cpu", **kwargs)


@pytest.mark.parametrize("encoder_size", [(16, 4), (64, 32, 4), (4,)])
def test_initialization_with_encoder_sizes(encoder_size):
    engine = DDFMEngine(dataset=_all_targets(), encoder_size=encoder_size, device="cpu")
    assert engine.encoder_size == encoder_size
    assert engine.num_series == 5
    assert engine.encoder_size[-1] == encoder_size[-1]  # the last width is the number of factors


def test_window_size_and_n_mc_samples_are_stored():
    engine = _default_engine(_all_targets(), n_mc_samples=10, window_size=100)
    assert engine.window_size == 100
    assert engine.n_mc_samples == 10


def test_build_inputs_for_pretrain_is_the_data_optionally_interpolated():
    """_build_inputs_for_pretrain returns dataset.data, interpolated on request, without feature engineering."""
    data = pd.DataFrame(np.random.RandomState(0).randn(10, 5), columns=[f"series_{i}" for i in range(5)])
    data.iloc[1, 0] = np.nan
    data.iloc[3, 2] = np.nan
    dataset = DDFMDataset(data=data, time_idx="index", scaler=None)
    engine = _default_engine(dataset, interpolation_method="linear", interpolation_limit=10,
                             interpolation_limit_direction="both", max_iter=1)

    raw = engine._build_inputs_for_pretrain(interpolate=False)
    assert raw.shape == dataset.data.shape
    assert list(raw.columns) == list(dataset.data.columns)
    assert raw.isna().sum().sum() > 0

    filled = engine._build_inputs_for_pretrain(interpolate=True)
    assert filled.shape == dataset.data.shape
    assert list(filled.columns) == list(dataset.data.columns)
    assert filled.isna().sum().sum() == 0


def test_get_result_and_predict_before_fit_raise_not_trained():
    engine = _default_engine(_all_targets())
    with pytest.raises(ModelNotTrainedError):
        engine.get_result()
    assert getattr(engine, "_result", None) is None
    with pytest.raises(ModelNotTrainedError, match="model has not been trained"):
        engine.predict(horizon=5)


@pytest.mark.parametrize("invalid_input", ["not a dict", [1, 2, 3], 42, None])
def test_infer_input_dim_rejects_non_dicts(invalid_input):
    with pytest.raises(DataValidationError, match="state_dict must be a dictionary"):
        infer_ddfm_input_dim(invalid_input)


def test_infer_input_dim_of_an_empty_dict_is_none():
    assert infer_ddfm_input_dim({}) is None


def test_infer_input_dim_without_matching_keys_is_none():
    assert infer_ddfm_input_dim({"some.other.key": torch.zeros(10, 5)}) is None


def test_infer_input_dim_from_the_first_encoder_layer():
    assert infer_ddfm_input_dim({"autoencoder.encoder.layers.0.weight": torch.zeros(32, 64)}) == 64


def test_infer_input_dim_from_the_decoder_weight():
    assert infer_ddfm_input_dim({"autoencoder.decoder.decoder.weight": torch.zeros(10, 5)}) == 10


@pytest.mark.parametrize("make", [lambda: np.array([[1, 2, 3], [4, 5, 6]]),
                                  lambda: torch.tensor([[1, 2, 3], [4, 5, 6]])])
def test_infer_input_dim_from_2d_data(make):
    assert infer_input_dim_from_data(make()) == 3


@pytest.mark.parametrize("make", [lambda: np.array([1, 2, 3]), lambda: torch.tensor([1, 2, 3])])
def test_infer_input_dim_from_1d_data_raises(make):
    with pytest.raises(DataError, match="Data must be at least 2D"):
        infer_input_dim_from_data(make())


def test_dimension_constants():
    assert MIN_VARIABLES == 1
    assert MIN_DDFM_TIME_STEPS == 2


def test_decoder_helpers():
    engine = _default_engine(_all_targets(time_steps=50), max_iter=1)
    engine.fit()
    linear = engine.decoder.get_last_linear_layer()
    assert isinstance(linear, torch.nn.Linear)
    assert linear in [m for m in engine.decoder.modules() if isinstance(m, torch.nn.Linear)]
    assert engine.decoder_type == "linear"
    assert engine.decoder.get_intermediate() is None  # the linear decoder has no hidden layer


def test_variance_diagnostics_attributes_after_fit():
    engine = _default_engine(_all_targets(time_steps=50), max_iter=1)
    engine.fit()
    assert hasattr(engine, "prediction_std")
    assert hasattr(engine, "factor_std")


def test_variance_diagnostics_report_standardization():
    dataset_std = _all_targets(time_steps=50, scaler=StandardScaler())
    dataset_std.scaler.fit(dataset_std.data.values)
    dataset_std.data = pd.DataFrame(dataset_std.scaler.transform(dataset_std.data.values),
                                    columns=dataset_std.data.columns, index=dataset_std.data.index)
    engine_std = _default_engine(dataset_std, max_iter=1)
    engine_std.fit()

    dataset_robust = _all_targets(time_steps=50, scaler=RobustScaler())
    engine_robust = _default_engine(dataset_robust, max_iter=1)
    engine_robust.fit()

    rng = np.random.RandomState(42)
    prediction_std = rng.rand(50, 5) * 0.1
    prediction_mean = rng.rand(50, 5)
    factors_mean = rng.rand(50, 4)
    for engine, dataset in ((engine_std, dataset_std), (engine_robust, dataset_robust)):
        diagnostics = diagnose_variance_collapse(
            prediction_std=prediction_std, prediction_mean=prediction_mean, factors_mean=factors_mean,
            y_actual=engine.y_actual, target_scaler=dataset.scaler,
            encoder=engine.encoder, decoder=engine.decoder,
        )
        assert "is_standardized" in diagnostics


def test_variance_diagnostics_validate_shapes():
    dataset = _all_targets(time_steps=50)
    engine = _default_engine(dataset, max_iter=1)
    engine.fit()

    rng = np.random.RandomState(42)
    prediction_mean = rng.rand(50, 5)
    factors_mean = rng.rand(50, 4)

    def diagnose(prediction_std):
        return diagnose_variance_collapse(
            prediction_std=prediction_std, prediction_mean=prediction_mean, factors_mean=factors_mean,
            y_actual=engine.y_actual, target_scaler=dataset.scaler,
            encoder=engine.encoder, decoder=engine.decoder,
        )

    invalid = diagnose([[0.1] * 5] * 50)
    assert any("Invalid prediction_std type" in w for w in invalid["warnings"])
    one_d = diagnose(rng.rand(50))
    assert any("1D prediction_std array" in w for w in one_d["warnings"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        empty = diagnose(np.array([]).reshape(0, 5))
    assert any("Invalid prediction_std shape" in w for w in empty["warnings"])
    valid = diagnose(rng.rand(50, 5) * 0.1)
    assert "prediction_std_mean" in valid
    assert "variance_collapse_detected" in valid

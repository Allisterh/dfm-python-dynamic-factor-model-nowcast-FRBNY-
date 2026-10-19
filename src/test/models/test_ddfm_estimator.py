"""Tests for the public DDFM estimator (dfm_python.models.ddfm.estimator)."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import inspect
import json
import logging

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import dfm_python
import dfm_python.models
import dfm_python.models.ddfm
from dfm_python import PanelData
from dfm_python.config import DDFMConfig, DFMConfig
from dfm_python.config.schema.params import DDFMModelState
from dfm_python.config.schema.results import DDFMResult
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.models.ddfm.estimator import _TIME_COLUMN, _TRAINING_BUFFERS, DDFM
from dfm_python.registry import all_estimators
from dfm_python.testing import _state_hash
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None

TINY = {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 2,
        "random_state": 0, "device": "cpu"}


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _panel(T: int = 48, N: int = 3, K: int = 1, *, index: str = "datetime", seed: int = 0) -> PanelData:
    rng = np.random.default_rng(seed)
    if index == "datetime":
        idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    else:
        idx = pd.RangeIndex(T)
    y = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=[f"s{i}" for i in range(N)])
    y.iloc[5, 0] = np.nan
    exog = None
    if K:
        exog = pd.DataFrame(rng.standard_normal((T, K)), index=idx, columns=[f"x{k}" for k in range(K)])
    return PanelData(y=y, exog=exog)


@pytest.fixture(scope="module")
def fitted() -> DDFM:
    with torch.random.fork_rng():
        return DDFM(**TINY, scaler="standard").fit(_panel())


# -- construction ---------------------------------------------------------------


def test_defaults_are_the_values_that_ran_when_omitted():
    assert DDFM().get_params() == {
        "n_factors": 4, "encoder_layers": (16,), "decoder_type": "linear", "activation": "relu",
        "learning_rate": 0.005, "optimizer": "Adam", "n_mc_samples": 10, "window_size": 100, "max_iter": 200,
        "tol": 5e-4, "interpolation_method": "linear", "interpolation_limit": 10,
        "interpolation_limit_direction": "both", "scaler": None, "random_state": 3, "device": "auto", "verbose": 0,
    }


def test_constructor_only_stores_and_is_keyword_only():
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    est = DDFM(n_factors=3, encoder_layers=[8, 4], device="cuda:7", scaler="nonsense")
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert vars(est) == est.get_params()
    assert est.encoder_layers == [8, 4]
    with pytest.raises(TypeError):
        DDFM(4)  # noqa
    for old in ("encoder_size", "seed", "tolerance", "disp", "dataset", "config", "max_epoch_pre_train",
                "pretrain_epochs", "min_target_interporate_ratio"):
        with pytest.raises(TypeError):
            DDFM(**{old: 1})


def test_tags():
    tags = DDFM.get_tags()
    assert tags["task"] == "forecast" and tags["backend"] == "torch"
    for cap in ("exog", "missing_values", "update", "factor_forecast", "device"):
        assert tags[f"capability:{cap}"] is True
    assert not tags["capability:series_exog"] and not tags["capability:mixed_frequency"]
    assert tags["update_kind"] == "encoder_forward_rescale" and tags["predict_data"] == "full_panel"
    assert tags["non_deterministic"] is False
    assert DDFM().get_tags()["non_deterministic"] is False
    assert DDFM(random_state=None).get_tags()["non_deterministic"] is True


def test_exports():
    assert dfm_python.DDFM is DDFM
    assert dfm_python.models.DDFM is DDFM
    assert dfm_python.models.ddfm.DDFM is DDFM
    assert issubclass(DDFM, dfm_python.BaseFactorModel) and DDFM is not DDFMEngine


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params, name",
    [
        ({"n_factors": 0}, "n_factors"), ({"n_factors": 2.0}, "n_factors"), ({"n_factors": True}, "n_factors"),
        ({"n_mc_samples": 0}, "n_mc_samples"), ({"window_size": -1}, "window_size"), ({"max_iter": 0}, "max_iter"),
        ({"encoder_layers": 16}, "encoder_layers"), ({"encoder_layers": (8, 0)}, "encoder_layers"),
        ({"encoder_layers": (8.0,)}, "encoder_layers"), ({"encoder_layers": "16"}, "encoder_layers"),
        ({"encoder_layers": (), "decoder_type": "mlp"}, "encoder_layers"),
        ({"decoder_type": "cnn"}, "decoder_type"), ({"activation": "gelu"}, "activation"),
        ({"optimizer": "RMSprop"}, "optimizer"), ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"), ({"tol": -1.0}, "tol"), ({"tol": float("inf")}, "tol"),
        ({"interpolation_limit": 0}, "interpolation_limit"), ({"interpolation_limit": 2.5}, "interpolation_limit"),
        ({"interpolation_limit_direction": "up"}, "interpolation_limit_direction"),
        ({"interpolation_method": ""}, "interpolation_method"), ({"interpolation_method": None}, "interpolation_method"),
        ({"scaler": "zscore"}, "scaler"), ({"scaler": object()}, "scaler"),
        ({"random_state": 1.5}, "random_state"), ({"verbose": -1}, "verbose"), ({"verbose": 1.0}, "verbose"),
        ({"device": "tpu"}, "device"),
    ],
)
def test_invalid_parameters_raise_configuration_error_naming_them(params, name, monkeypatch):
    def no_engine(*args, **kwargs):
        raise AssertionError("the engine must not be built")

    monkeypatch.setattr("dfm_python.models.ddfm.estimator.DDFMEngine", no_engine)
    est = DDFM(**{**TINY, **params})
    with pytest.raises(ConfigurationError, match=name):
        est.fit(_panel())
    assert not hasattr(est, "result_")


def test_valid_edge_parameters_are_accepted(monkeypatch):
    built = []

    class Stop(Exception):
        pass

    def engine(*args, **kwargs):
        built.append(kwargs)
        raise Stop

    monkeypatch.setattr("dfm_python.models.ddfm.estimator.DDFMEngine", engine)
    for params in ({"encoder_layers": ()}, {"encoder_layers": [np.int64(4)]}, {"interpolation_limit": None},
                   {"tol": 0}, {"random_state": None}, {"scaler": "Standard"}, {"scaler": RobustScaler()},
                   {"device": torch.device("cpu")}, {"n_factors": np.int64(2)}):
        with pytest.raises(Stop):
            DDFM(**{**TINY, **params}).fit(_panel())
    assert len(built) == 9


def test_unexpected_fit_keywords_raise():
    with pytest.raises(TypeError, match="unexpected"):
        DDFM(**TINY).fit(_panel(), epochs=3)


def test_series_exog_and_mixed_frequency_are_rejected():
    panel = _panel(K=0)
    rng = np.random.default_rng(1)
    with_series_exog = PanelData.from_arrays(rng.standard_normal((20, 2)), series_exog=rng.standard_normal((20, 2, 3)))
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        DDFM(**TINY).fit(with_series_exog)
    mixed = PanelData(y=panel.y, freq={"s0": "q"})
    with pytest.raises(UnsupportedInputError, match="mixed"):
        DDFM(**TINY).fit(mixed)


# -- the adapter and the engine call ----------------------------------------------


def test_every_hyperparameter_reaches_the_engine(monkeypatch):
    seen = {}
    original = DDFMEngine.__init__

    def init(self, *args, **kwargs):
        seen.update(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "__init__", init)
    params = {
        "n_factors": 3, "encoder_layers": (6, 5), "decoder_type": "mlp", "activation": "tanh",
        "learning_rate": 0.01, "optimizer": "AdamW", "n_mc_samples": 2, "window_size": 12, "max_iter": 1,
        "tol": 1e-3, "interpolation_method": "nearest", "interpolation_limit": 3,
        "interpolation_limit_direction": "forward", "scaler": "minmax", "random_state": 11, "device": "cpu",
    }
    model = DDFM(**params).fit(_panel())
    dataset = seen.pop("dataset")
    assert isinstance(dataset, DDFMDataset)
    assert seen == {
        "config": None, "encoder_size": (6, 5, 3), "decoder_type": "mlp", "seed": 11, "activation": "tanh",
        "learning_rate": 0.01, "optimizer": "AdamW", "n_mc_samples": 2, "window_size": 12, "max_iter": 1,
        "tolerance": 1e-3, "disp": 10, "interpolation_method": "nearest", "interpolation_limit": 3,
        "interpolation_limit_direction": "forward", "device": torch.device("cpu"),
    }
    engine = model.network_
    assert engine.encoder_size == (6, 5, 3) and engine.decoder_size == (5, 6)
    assert engine.optimizer_type == "AdamW" and engine.tolerance == 1e-3 and engine.initializer_seed == 11
    assert model.scaler_ is engine.scaler and type(model.scaler_) is MinMaxScaler
    assert model.n_factors_ == 3 and model.factors_.shape == (48, 3)


def test_adapter_builds_exog_then_y_on_a_range_index():
    est = DDFM(**TINY, scaler="standard")
    # y listed first in PanelData; the engine's frame is still [exog | y].
    panel = _panel(N=2, K=2)
    ds = est._adapter(panel)
    assert list(ds.data.columns) == ["x0", "x1", "s0", "s1"]
    assert ds.covariates == ["x0", "x1"] and ds.target_series == ["s0", "s1"]
    assert isinstance(ds.data.index, pd.RangeIndex) and ds.time_idx == _TIME_COLUMN
    assert isinstance(ds.scaler, StandardScaler)
    # Missing values pass through to the engine (it interpolates and imputes them).
    assert ds.missing_y[5, 0] and ds.missing_y.sum() == 1
    # The inputs are not mutated.
    assert np.isnan(panel.y.iloc[5, 0]) and isinstance(panel.y.index, pd.DatetimeIndex)


def test_adapter_time_column_never_collides():
    y = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 2)), columns=[_TIME_COLUMN, "b"])
    ds = DDFM(**TINY)._adapter(PanelData(y=y))
    assert ds.target_series == [_TIME_COLUMN, "b"] and ds.covariates == []
    assert ds.time_idx == _TIME_COLUMN + "_"


def test_adapter_without_exog_and_without_scaler():
    ds = DDFM(**TINY)._adapter(_panel(K=0))
    assert list(ds.data.columns) == ["s0", "s1", "s2"] and ds.covariates == [] and ds.scaler is None


def test_fit_matches_the_engine_on_the_same_frame():
    """The estimator runs the engine on [exog | y]: same numbers as the engine on that frame."""
    panel = _panel(N=2, K=2)
    torch.manual_seed(123)
    model = DDFM(**TINY, scaler="standard").fit(panel)
    frame = pd.concat([panel.exog, panel.y], axis=1).reset_index(drop=True)
    torch.manual_seed(123)
    dataset = DDFMDataset(data=frame, time_idx="none", covariates=["x0", "x1"], scaler=StandardScaler())
    engine = DDFMEngine(dataset, encoder_size=(4, 2), n_mc_samples=2, window_size=16, max_iter=2, seed=0,
                        device="cpu")
    engine.fit()
    engine.build_state_space()
    np.testing.assert_array_equal(model.network_.factors, engine.factors, strict=True)
    np.testing.assert_array_equal(model.result_.C, engine.training_state.H, strict=True)
    for (k, a), (k2, b) in zip(model.network_.state_dict().items(), engine.state_dict().items()):
        assert k == k2
        assert torch.equal(a, b), k
    X, Z = engine.predict(horizon=4)
    assert model.predict(4).to_numpy().tobytes() == X.tobytes()
    assert model.predict_factors(4).to_numpy().tobytes() == Z.tobytes()


def test_scaler_instance_is_cloned_and_a_fitted_one_warns():
    scaler = StandardScaler()
    model = DDFM(**{**TINY, "scaler": scaler}).fit(_panel())
    assert model.scaler is scaler and not hasattr(scaler, "mean_")
    assert model.scaler_ is not scaler and isinstance(model.scaler_, StandardScaler)

    fitted_scaler = StandardScaler().fit(np.ones((3, 3)))
    with pytest.warns(UserWarning, match="already fitted") as record:
        model = DDFM(**{**TINY, "scaler": fitted_scaler}).fit(_panel())
    assert record[0].filename == __file__
    np.testing.assert_array_equal(fitted_scaler.mean_, np.ones(3))
    assert model.scaler_ is not fitted_scaler


def test_device_cpu_overrides_cuda(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_available", lambda: True)
        assert DDFM(device="auto")._resolve_device() == torch.device("cuda")
        assert DDFM(device="cpu")._resolve_device() == torch.device("cpu")
    model = DDFM(**TINY).fit(_panel())
    assert model.device_ == torch.device("cpu") and model.network_.device == torch.device("cpu")
    assert all(p.device.type == "cpu" for p in model.network_.parameters())


# -- fitted state -----------------------------------------------------------------


def test_fitted_attributes(fitted):
    engine = fitted.network_
    assert isinstance(engine, DDFMEngine) and isinstance(fitted.result_, DDFMResult)
    assert fitted.scaler_ is engine.scaler and isinstance(fitted.scaler_, StandardScaler)
    assert fitted.n_iter_ == engine._num_iter == 2 and fitted.converged_ is False
    assert fitted.n_factors_ == 2 and fitted.random_state_ == 0
    assert list(fitted.feature_names_in_) == ["s0", "s1", "s2"] and list(fitted.exog_names_in_) == ["x0"]
    assert fitted.n_features_in_ == 3
    assert fitted.conditioning_index_.equals(fitted.index_) and fitted.index_.equals(_panel().index)
    assert fitted.factors_.index.equals(fitted.index_)
    assert list(fitted.factors_.columns) == ["factor_0", "factor_1"]
    assert all(dtype == np.float64 for dtype in fitted.factors_.dtypes)
    np.testing.assert_array_equal(fitted.factors_.to_numpy(), fitted.result_.Z.astype(np.float64))


def test_result_c_is_series_by_factors(fitted):
    H = fitted.network_.training_state.H
    assert fitted.result_.C.shape == (3, 2) == H.shape
    np.testing.assert_array_equal(fitted.result_.C, H)
    np.testing.assert_allclose(fitted.result_.x_sm, fitted.result_.Z @ fitted.result_.C.T, rtol=1e-6)


def test_engine_keeps_no_training_data(fitted):
    engine = fitted.network_
    assert engine._dataset is None
    for name in _TRAINING_BUFFERS:
        assert name not in vars(engine), name
    # The scaled training frame stays: the engine's update refits the scaler on it.
    assert list(engine.data.columns) == ["x0", "s0", "s1", "s2"]
    assert engine.target_series == ["s0", "s1", "s2"] and engine.covariates == ["x0"]


def test_mlp_decoder_fit_drops_its_last_iteration_datasets():
    model = DDFM(**{**TINY, "decoder_type": "mlp"}).fit(_panel())
    assert "_last_iter_datasets" not in vars(model.network_)
    assert model.network_.last_neurons.shape == (2, 48, 4)
    assert model.predict(2).shape == (2, 3)


def test_unfitted_access_raises_not_fitted():
    est = DDFM(**TINY)
    assert not hasattr(est, "result_") and not hasattr(est, "network_")
    for call in (lambda: est.predict(3), lambda: est.predict_factors(3)):
        with pytest.raises(NotFittedError):
            call()
    with pytest.raises(NotFittedError):
        _ = est.result_


def test_failed_fit_leaves_the_model_unfitted():
    model = DDFM(**TINY).fit(_panel())
    model.set_params(n_factors=0)
    with pytest.raises(ConfigurationError):
        model.fit(_panel())
    assert not hasattr(model, "result_") and not hasattr(model, "network_")


def test_fit_is_repeatable():
    a = DDFM(**TINY, scaler="standard").fit(_panel())
    b = DDFM(**TINY, scaler="standard").fit(_panel())
    np.testing.assert_array_equal(a.result_.Z, b.result_.Z)
    assert a.predict(3).equals(b.predict(3))


def test_fit_logs_nothing_below_warning_with_verbose_zero(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DDFM(**TINY).fit(_panel())
    assert [r for r in caplog.records if r.levelno < logging.WARNING] == []


def test_fit_with_verbose_logs_the_engine_progress(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DDFM(**{**TINY, "verbose": 1}).fit(_panel())
    assert any(r.name.startswith("dfm_python.models.ddfm") and r.levelno == logging.INFO for r in caplog.records)


# -- forecasting --------------------------------------------------------------------


def test_predict_frame(fitted):
    fc = fitted.predict(4)
    assert fc.shape == (4, 3) and list(fc.columns) == ["s0", "s1", "s2"]
    assert all(dtype == np.float64 for dtype in fc.dtypes)
    expected_index = pd.date_range("2014-01-31", periods=4, freq="ME", name="date")
    assert fc.index.equals(expected_index)
    assert fc.attrs == {"model": "DDFM", "kind": "forecast", "origin": fitted.index_[-1]}
    assert np.isfinite(fc.to_numpy()).all()

    zf = fitted.predict_factors(4)
    assert zf.shape == (4, 2) and list(zf.columns) == ["factor_0", "factor_1"]
    assert zf.index.equals(expected_index)
    assert zf.attrs == {"model": "DDFM", "kind": "factors", "origin": fitted.index_[-1]}


def test_predict_is_the_engine_forecast_in_data_units(fitted):
    y_forecast, z_forecast = fitted.network_._forecast(5)
    H = fitted.network_.training_state.H
    model_space = z_forecast @ H.T
    np.testing.assert_array_equal(fitted.predict(5).to_numpy(), y_forecast)
    np.testing.assert_allclose(fitted.predict(5).to_numpy(), fitted.scaler_.inverse_transform(model_space))
    np.testing.assert_array_equal(fitted.predict_factors(5).to_numpy(), z_forecast)


def test_predict_does_not_change_the_model(fitted):
    factors = fitted.network_.factors.copy()
    first = fitted.predict(3)
    fitted.predict_factors(3)
    assert fitted.predict(3).equals(first)
    np.testing.assert_array_equal(fitted.network_.factors, factors)


def test_scaler_none_forecasts_in_model_space():
    model = DDFM(**TINY).fit(_panel(index="range"))
    assert model.scaler_ is None
    fc = model.predict(3)
    _, z_forecast = model.network_._forecast(3)
    np.testing.assert_array_equal(fc.to_numpy(), z_forecast @ model.network_.training_state.H.T)
    assert fc.index.equals(pd.RangeIndex(48, 51, name="time"))


@pytest.mark.parametrize("h", [2.0, "3", None, True, np.float64(2)])
def test_horizon_must_be_an_integer(fitted, h):
    with pytest.raises(TypeError, match="h must be"):
        fitted.predict(h)
    with pytest.raises(TypeError, match="h must be"):
        fitted.predict_factors(h)


def test_horizon_must_be_positive(fitted):
    with pytest.raises(ValueError):
        fitted.predict(0)
    assert fitted.predict(np.int64(2)).shape == (2, 3)


# -- update ---------------------------------------------------------------------


def _weights(est: DDFM) -> dict:
    return {k: v.detach().clone() for k, v in est.network_.state_dict().items()}


def _same_weights(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def _full(T: int = 60, K: int = 1) -> PanelData:
    """A full panel: 60 rows, the training rows' dates followed by 12 new ones."""
    return _panel(T=T, K=K, seed=1)


def test_update_appends_the_encoded_panel_and_refits_the_scaler(fitted):
    est = copy.deepcopy(fitted)
    weights, result, factors = _weights(est), est.result_, est.factors_.copy()
    engine = est.network_
    old_scaler = copy.deepcopy(est.scaler_)
    training_targets = engine.data[engine.target_series].to_numpy() * old_scaler.scale_ + old_scaler.mean_
    full = _full()
    assert est.update(full) is est
    # Every row of the panel is encoded and appended to each MC sample's factors.
    assert engine.factors.shape == (2, 48 + 60, 2)
    np.testing.assert_array_equal(engine.factors[:, :48], fitted.network_.factors)
    # The target scaler is refitted in place on the training targets (data units) and the panel's.
    expected = StandardScaler().fit(np.vstack([training_targets, full.y.to_numpy()]))
    np.testing.assert_array_equal(est.scaler_.mean_, expected.mean_)
    np.testing.assert_array_equal(est.scaler_.scale_, expected.scale_)
    assert est.scaler_ is engine.scaler and est.result_.target_scaler is est.scaler_
    # The fit is unchanged; the forecasts now start after the panel.
    assert _same_weights(weights, _weights(est))
    assert est.result_ is result and est.factors_.equals(factors) and est.index_.equals(_panel().index)
    assert est.conditioning_index_.equals(full.index)
    forecast = est.predict(3)
    assert forecast.index.equals(pd.date_range(full.index[-1], periods=4, freq="ME", name="date")[1:])
    assert forecast.attrs["origin"] == full.index[-1] and np.isfinite(forecast.to_numpy()).all()
    assert est.get_params() == fitted.get_params()


def test_update_is_the_engine_update_on_the_adapter_dataset(fitted):
    est = copy.deepcopy(fitted)
    reference = copy.deepcopy(fitted.network_)
    full = _full()
    dataset = est._adapter(full, fitted=True)
    # The dataset scales with an unfitted clone of scaler_; scaler_ itself is not touched by building it.
    assert type(dataset.scaler) is StandardScaler and dataset.scaler is not est.scaler_
    np.testing.assert_array_equal(est.scaler_.mean_, fitted.scaler_.mean_)
    assert list(dataset.data.columns) == ["x0", "s0", "s1", "s2"] and isinstance(dataset.data.index, pd.RangeIndex)
    est.update(full)
    reference._update(dataset)
    np.testing.assert_array_equal(est.network_.factors, reference.factors, strict=True)
    np.testing.assert_array_equal(est.scaler_.mean_, reference.scaler.mean_, strict=True)
    y_forecast, z_forecast = reference._forecast(3)
    assert est.predict(3).to_numpy().tobytes() == y_forecast.tobytes()
    assert est.predict_factors(3).to_numpy().tobytes() == z_forecast.tobytes()


def test_update_without_a_scaler_and_without_exog():
    est = DDFM(**TINY).fit(_panel(K=0, index="range"))
    assert est._adapter(_full(K=0), fitted=True).scaler is None
    full = PanelData(y=_full(K=0).y.reset_index(drop=True))
    est.update(full)
    assert est.scaler_ is None and est.network_.factors.shape == (2, 48 + 60, 2)
    assert est.predict(2).index.equals(pd.RangeIndex(60, 62, name="time"))


def test_update_refit_raises(fitted):
    est = copy.deepcopy(fitted)
    with pytest.raises(UnsupportedInputError, match="refit"):
        est.update(_full(), refit=True)
    assert est.conditioning_index_.equals(fitted.conditioning_index_)
    np.testing.assert_array_equal(est.network_.factors, fitted.network_.factors)


@pytest.mark.parametrize("change", ["series", "order", "exog", "no_exog"])
def test_update_and_predict_data_check_the_names(fitted, change):
    panel = _full()
    if change == "series":
        panel = PanelData(y=panel.y.rename(columns={"s0": "other"}), exog=panel.exog)
    elif change == "order":
        panel = PanelData(y=panel.y[["s1", "s0", "s2"]], exog=panel.exog)
    elif change == "exog":
        panel = _full(K=2)
    else:
        panel = PanelData(y=panel.y)
    est = copy.deepcopy(fitted)
    for call in (lambda: est.update(panel), lambda: est.predict(3, data=panel), lambda: est.predict_factors(3, data=panel)):
        with pytest.raises(DataValidationError):
            call()
    assert est.conditioning_index_.equals(fitted.conditioning_index_)


def test_update_rejects_excluded_inputs(fitted):
    rng = np.random.default_rng(0)
    with_series_exog = PanelData.from_arrays(rng.standard_normal((20, 3)), series_exog=rng.standard_normal((20, 3, 2)))
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        copy.deepcopy(fitted).update(with_series_exog)


def test_update_before_fit_raises():
    with pytest.raises(NotFittedError):
        DDFM(**TINY).update(_full())


# -- predict(data=...) ----------------------------------------------------------


def test_predict_data_is_update_on_a_copy(fitted):
    est = fitted
    full = _full()
    before = _state_hash(est)
    weights, factors, mean = _weights(est), est.network_.factors.copy(), est.scaler_.mean_.copy()
    state = torch.get_rng_state()
    forecast = est.predict(4, data=full)
    factor_forecast = est.predict_factors(4, data=full)
    assert torch.equal(state, torch.get_rng_state())
    assert est.predict(4, data=full).equals(forecast)
    assert _state_hash(est) == before
    assert _same_weights(weights, _weights(est)) and np.array_equal(est.network_.factors, factors)
    np.testing.assert_array_equal(est.scaler_.mean_, mean)
    assert est.conditioning_index_.equals(_panel().index)

    reference = copy.deepcopy(est).update(full)
    assert forecast.to_numpy().tobytes() == reference.predict(4).to_numpy().tobytes()
    assert forecast.index.equals(reference.predict(4).index) and forecast.attrs == reference.predict(4).attrs
    assert forecast.index[0] > full.index[-1]
    assert factor_forecast.equals(reference.predict_factors(4))


def test_predict_data_with_the_training_panel_starts_after_it(fitted):
    forecast = fitted.predict(3, data=_panel())
    assert forecast.index.equals(fitted.predict(3).index)
    assert np.isfinite(forecast.to_numpy()).all()


# -- save / load ----------------------------------------------------------------


def _assert_same_model(loaded: DDFM, est: DDFM) -> None:
    assert type(loaded) is DDFM and loaded.get_params() == est.get_params()
    assert _same_weights(_weights(loaded), _weights(est))
    for name in ("n_features_in_", "n_iter_", "converged_", "n_factors_", "random_state_"):
        assert getattr(loaded, name) == getattr(est, name), name
    for name in ("feature_names_in_", "exog_names_in_"):
        assert list(getattr(loaded, name)) == list(getattr(est, name)), name
    assert loaded.index_.equals(est.index_) and getattr(loaded.index_, "freq", None) == getattr(est.index_, "freq", None)
    assert loaded.conditioning_index_.equals(est.conditioning_index_)
    assert loaded.factors_.equals(est.factors_)
    assert loaded.device_ == torch.device("cpu") and loaded.network_.device == torch.device("cpu")
    engine, original = loaded.network_, est.network_
    for name in ("factors", "eps", "last_neurons"):
        a, b = getattr(engine, name), getattr(original, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    for name in ("F", "H", "Q", "R", "mu_0", "Sigma_0"):
        a, b = getattr(engine.training_state, name), getattr(original.training_state, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    assert engine.data.equals(original.data) and list(engine.data.columns) == list(original.data.columns)
    assert isinstance(engine.data.index, pd.RangeIndex)
    assert engine.target_series == original.target_series and engine.covariates == original.covariates
    np.testing.assert_array_equal(engine.target_indices, original.target_indices, strict=True)
    if original.covariates:
        assert torch.equal(engine._target_col_tensor, original._target_col_tensor)
    assert engine._num_iter == original._num_iter
    assert engine._converged == getattr(original, "_converged", False)
    assert getattr(engine, "loss_now", None) == getattr(original, "loss_now", None)
    assert type(getattr(engine, "loss_now", None)) is type(getattr(original, "loss_now", None))
    for name in ("encoder_size", "decoder_size", "input_dim", "output_dim", "num_series", "initializer_seed",
                 "interpolation_limit", "learning_rate", "tolerance", "disp"):
        assert getattr(engine, name) == getattr(original, name), name
    assert engine.scaler is loaded.scaler_
    result, fitted_result = loaded.result_, est.result_
    assert result.target_scaler is loaded.scaler_
    for f in dataclasses.fields(DDFMResult):
        if f.name == "target_scaler":
            continue
        a, b = getattr(result, f.name), getattr(fitted_result, f.name)
        if isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f.name
        else:
            assert a == b and type(a) is type(b), f.name


def test_save_load_round_trip(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"config.json", "data_spec.json", "state.npz", "state.json", "result.npz", "result.json",
                     "weights.pt", "scaler.json"}
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["num_iter"] == 2 and state["loss_now"] is None and state["converged"] is False
    assert state["encoder_size"] == [4, 2] and state["decoder_size"] is None
    assert state["data_columns"] == ["x0", "s0", "s1", "s2"]
    assert state["target_series"] == ["s0", "s1", "s2"] and state["covariates"] == ["x0"]
    assert (state["input_dim"], state["output_dim"], state["num_series"]) == (4, 3, 4)
    assert state["conditioning_index"]["kind"] == "datetime"
    with np.load(tmp_path / "run" / "state.npz") as npz:
        keys = set(npz.files)
        assert np.isnan(npz["data/frame"]).sum() == 1  # the training frame keeps its missing value
    assert {"factors", "eps", "last_neurons", "data/frame"} | {f"ssm/{n}" for n in ("F", "H", "Q", "R", "mu_0", "Sigma_0")} <= keys
    assert "target_scaler" in json.loads((tmp_path / "run" / "result.json").read_text())["excluded"]
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["model_type"] == "DDFM" and config["params"]["random_state"] == 0
    full = _full()
    for loader in (DDFM.load, dfm_python.load):
        loaded = loader(tmp_path / "run", device="cpu")
        _assert_same_model(loaded, fitted)
        assert loaded.predict(3).equals(fitted.predict(3))
        assert loaded.predict_factors(3).equals(fitted.predict_factors(3))
        assert loaded.predict(3, data=full).equals(fitted.predict(3, data=full))
        assert isinstance(loaded.scaler_, StandardScaler)


def test_save_load_after_update(fitted, tmp_path):
    est = copy.deepcopy(fitted).update(_full())
    est.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run")
    _assert_same_model(loaded, est)
    assert loaded.network_.factors.shape == (2, 108, 2)
    assert loaded.predict(3).equals(est.predict(3))
    np.testing.assert_array_equal(loaded.scaler_.mean_, est.scaler_.mean_)
    # Updating the loaded model further works like updating the original.
    more = _full(T=72)
    assert loaded.update(more).predict(3).equals(copy.deepcopy(est).update(more).predict(3))


def test_save_load_mlp_without_scaler_or_exog(tmp_path):
    est = DDFM(**{**TINY, "decoder_type": "mlp", "encoder_layers": (6, 4), "max_iter": 3}).fit(
        _panel(K=0, index="range")
    )
    assert isinstance(est.network_.loss_now, float)
    est.save(tmp_path / "run")
    assert json.loads((tmp_path / "run" / "state.json").read_text())["decoder_size"] == [4, 6]
    loaded = DDFM.load(tmp_path / "run")
    _assert_same_model(loaded, est)
    assert loaded.scaler_ is None and loaded.network_.last_neurons.shape == (2, 48, 6)
    assert loaded.predict(3).equals(est.predict(3))
    full = PanelData(y=_full(K=0).y.reset_index(drop=True))
    assert loaded.predict(3, data=full).equals(est.predict(3, data=full))


def test_load_does_not_touch_the_global_rngs(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    DDFM.load(tmp_path / "run")
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))


def test_from_saved_sets_every_attribute_init_sets(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run").network_
    built = DDFMEngine(dataset=fitted._adapter(_panel()), config=None, device="cpu", **fitted._engine_params())
    missing = sorted(set(vars(built)) - set(vars(loaded)))
    assert not missing
    for name in ("encoder_size", "decoder_size", "decoder_type", "activation", "n_mc_samples", "window_size",
                 "max_epoch_pre_train", "max_iter", "tolerance", "disp", "min_target_interporate_ratio",
                 "interpolation_method", "interpolation_limit", "interpolation_limit_direction", "num_series",
                 "input_dim", "output_dim", "initializer_seed", "learning_rate", "optimizer_type", "device",
                 "lags_input", "target_series", "covariates", "_config"):
        assert getattr(loaded, name) == getattr(built, name), name
    assert loaded._dataset is None and loaded.optimizer is None and loaded.scheduler is None
    assert isinstance(loaded.training_state, DDFMModelState) and loaded.training_state.F is not None
    assert dict(loaded.named_modules()).keys() == dict(fitted.network_.named_modules()).keys()
    assert list(loaded.state_dict()) == list(fitted.network_.state_dict())
    assert not loaded.autoencoder.training


def test_from_saved_rejects_unknown_engine_parameters(fitted):
    dims = {"input_dim": 4, "output_dim": 3, "num_series": 4, "target_series": ["s0", "s1", "s2"], "covariates": ["x0"]}
    for bad in ({"dataset": None}, {"device": "cpu"}, {"pretrain_epochs": 3}):
        with pytest.raises(TypeError, match="unexpected"):
            DDFMEngine._from_saved({**fitted._engine_params(), **bad}, dims, "cpu")


def test_load_rejects_a_state_that_does_not_match_the_config(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    path = tmp_path / "run" / "state.json"
    state = json.loads(path.read_text())
    state["encoder_size"] = [8, 2]
    path.write_text(json.dumps(state))
    with pytest.raises(ConfigurationError, match="encoder"):
        DDFM.load(tmp_path / "run")


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(NotFittedError):
        DDFM(**TINY).save(tmp_path / "run")
    assert not (tmp_path / "run").exists()


# -- config mirror and registry -------------------------------------------------


def test_config_mirrors_the_constructor():
    cfg = DDFM().get_config()
    assert isinstance(cfg, DDFMConfig) and cfg.to_dict() == DDFM().get_params()
    assert DDFM._config_class is DDFMConfig and dfm_python.DDFMConfig is DDFMConfig
    assert DDFMConfig.field_names() == tuple(inspect.signature(DDFM.__init__).parameters)[1:]
    est = DDFM.from_config({"n_factors": 3, "encoder_layers": [16, 8], "scaler": "robust", "tol": 1e-3})
    assert est.n_factors == 3 and est.encoder_layers == (16, 8) and est.scaler == "robust" and est.tol == 1e-3


@pytest.mark.parametrize(
    "old",
    ["encoder_size", "num_factors", "tolerance", "seed", "disp", "max_epoch", "use_batch_norm", "use_idiosyncratic",
     "min_obs_idio", "lags_input", "max_epoch_pre_train", "frequency", "clock"],
)
def test_config_rejects_old_names(old):
    with pytest.raises(ConfigurationError, match=old):
        DDFMConfig.from_dict({old: 1})


@pytest.mark.parametrize("params", [{"encoder_layers": [64, 32]}, {"model_type": "ddfm"}, {"learning_rate": 0.01}])
def test_dfm_config_from_dict_points_ddfm_configs_to_the_mirror(params):
    with pytest.raises(ConfigurationError, match=r"DDFMConfig\.from_dict.*DDFM\.from_config"):
        DFMConfig.from_dict({"clock": "m", **params})


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_interpolation_values_reach_the_engine(tmp_path, monkeypatch):
    """Ported from the engine test of config interpolation precedence: what the YAML sets is what runs."""
    seen = {}
    original = DDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.update(method=self.interpolation_method, limit=self.interpolation_limit,
                    direction=self.interpolation_limit_direction)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "fit", fit)
    path = tmp_path / "ddfm.yaml"
    path.write_text(
        "model:\n  n_factors: 2\n  encoder_layers: [4]\n  n_mc_samples: 2\n  window_size: 16\n  max_iter: 2\n"
        "  interpolation_method: nearest\n  interpolation_limit: 7\n  interpolation_limit_direction: forward\n"
        "  random_state: 0\n  device: cpu\n",
        encoding="utf-8",
    )
    model = DDFM.from_config(DDFMConfig.from_yaml(path))
    assert model.interpolation_limit == 7
    engine = model.fit(_panel()).network_
    assert seen == {"method": "nearest", "limit": 7, "direction": "forward"}
    assert engine._config is None and engine.interpolation_limit == 7


def test_ddfm_is_registered():
    assert ("DDFM", DDFM) in all_estimators()

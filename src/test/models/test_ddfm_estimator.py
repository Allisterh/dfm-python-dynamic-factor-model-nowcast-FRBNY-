"""Tests for the public DDFM estimator (dfm_python.models.ddfm.estimator)."""

from __future__ import annotations

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
from dfm_python.config.schema.results import DDFMResult
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.models.ddfm.estimator import _TIME_COLUMN, _TRAINING_BUFFERS, DDFM
from dfm_python.utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError

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


def test_predict_with_data_is_not_implemented_yet(fitted):
    with pytest.raises(NotImplementedError, match="data"):
        fitted.predict(3, data=_panel())
    with pytest.raises(NotImplementedError, match="data"):
        fitted.predict_factors(3, data=_panel())

"""Tests for the public iVDFM estimator (dfm_python.models.ivdfm.estimator)."""

from __future__ import annotations

import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

import dfm_python
import dfm_python.models
import dfm_python.models.ivdfm
from dfm_python import PanelData
from dfm_python.config.schema.results import iVDFMResult
from dfm_python.models.ivdfm._engine import iVDFMEngine
from dfm_python.models.ivdfm.estimator import _TIME_COLUMN, iVDFM
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

TINY = {
    "n_factors": 2, "window": 8, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
    "batch_size": 4, "max_epochs": 2, "random_state": 0, "device": "cpu",
}


def _panel(T: int = 40, N: int = 3, K: int = 0, *, index: str = "datetime", seed: int = 0) -> PanelData:
    rng = np.random.default_rng(seed)
    if index == "datetime":
        idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    elif index == "period":
        idx = pd.period_range("2010-01", periods=T, freq="M", name="date")
    else:
        idx = pd.RangeIndex(T)
    y = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=[f"s{i}" for i in range(N)])
    exog = None
    if K:
        exog = pd.DataFrame(rng.standard_normal((T, K)), index=idx, columns=[f"u{k}" for k in range(K)])
    return PanelData(y=y, exog=exog)


@pytest.fixture(scope="module")
def fitted() -> iVDFM:
    return iVDFM(**TINY, scaler="standard").fit(_panel())


@pytest.fixture(scope="module")
def fitted_exog() -> iVDFM:
    return iVDFM(**TINY, time_context=2).fit(_panel(K=2))


# -- construction ---------------------------------------------------------------


def test_defaults_are_the_values_that_ran_when_omitted():
    assert iVDFM().get_params() == {
        "n_factors": 3, "window": None, "stride": None, "time_context": 1,
        "encoder_hidden_dim": 200, "encoder_n_hidden_layers": 2, "decoder_type": "mlp",
        "decoder_hidden_dim": 200, "decoder_n_hidden_layers": 2, "prior_hidden_dim": 100,
        "prior_n_hidden_layers": 1, "activation": "lrelu", "slope": 0.1, "factor_order": 1,
        "innovation_distribution": "laplace", "decoder_var": 0.01, "n_regimes": 1, "regime_temperature": 1.0,
        "beta_kl": 1.0, "use_layer_norm": False, "mixing": False, "learning_rate": 1e-3, "optimizer": "Adam",
        "optimizer_weight_decay": 0.0, "optimizer_momentum": 0.9, "batch_size": 32, "max_epochs": 100,
        "tol": 5e-4, "patience": 20, "scheduler_type": "step", "scheduler_step_size": None,
        "scheduler_gamma": 0.5, "scheduler_patience": 10, "scheduler_factor": 0.1, "scheduler_min_lr": 0.0,
        "scaler": None, "random_state": 3, "device": "auto", "verbose": 0,
    }


def test_constructor_only_stores_and_is_keyword_only():
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    est = iVDFM(n_factors=4, window=10)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert vars(est) == est.get_params()
    with pytest.raises(TypeError):
        iVDFM(4)  # noqa
    with pytest.raises(TypeError):
        iVDFM(num_factors=4)  # old names are not accepted


def test_tags():
    tags = iVDFM.get_tags()
    assert tags["task"] == "forecast" and tags["backend"] == "torch"
    for cap in ("exog", "missing_values", "update", "factor_forecast", "device"):
        assert tags[f"capability:{cap}"] is True
    assert not tags["capability:series_exog"] and not tags["capability:mixed_frequency"]
    assert tags["update_kind"] == "encoder_forward" and tags["predict_data"] == "window"
    assert tags["non_deterministic"] is False
    assert iVDFM(random_state=None).get_tags()["non_deterministic"] is True


def test_exports():
    assert dfm_python.iVDFM is iVDFM
    assert dfm_python.models.ivdfm.iVDFM is iVDFM
    assert dfm_python.models.iVDFM is iVDFM
    assert issubclass(iVDFM, dfm_python.BaseFactorModel)


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"n_factors": 0}, {"n_factors": 2.0}, {"window": 0}, {"stride": 0}, {"time_context": 0},
        {"patience": 0}, {"factor_order": 0}, {"n_regimes": 0}, {"regime_temperature": 0.0},
        {"innovation_distribution": "cauchy"}, {"decoder_type": "cnn"}, {"activation": "gelu"},
        {"beta_kl": -1.0}, {"optimizer": "RMSprop"}, {"scheduler_type": "linear"}, {"scheduler_gamma": 0.0},
        {"scheduler_gamma": 1.5}, {"scheduler_patience": -1}, {"scheduler_factor": 2.0}, {"scheduler_min_lr": -1.0},
        {"encoder_hidden_dim": [8, 0]}, {"learning_rate": 0.0}, {"batch_size": 0}, {"max_epochs": 0},
        {"scaler": "zscore"}, {"scaler": object()}, {"random_state": 1.5}, {"verbose": -1}, {"device": "tpu"},
        {"use_layer_norm": "yes"},
    ],
)
def test_invalid_parameters_raise_configuration_error(params):
    est = iVDFM(**{**TINY, **params})
    with pytest.raises(ConfigurationError):
        est.fit(_panel())
    assert not hasattr(est, "result_")


def test_unsupported_inputs():
    panel = _panel()
    series_exog = pd.DataFrame(
        np.zeros((40, 3)), index=panel.index, columns=pd.MultiIndex.from_product([["c"], panel.series_names])
    )
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        iVDFM(**TINY).fit(PanelData(y=panel.y, series_exog=series_exog))
    with pytest.raises(TypeError, match="unexpected keyword"):
        iVDFM(**TINY).fit(panel, epochs=3)


# -- every hyperparameter reaches the engine ---------------------------------


def test_precedence_values_reach_the_engine():
    # factor_order=3 used to run as 1 from a config, and patience=None as 20.
    est = iVDFM(**{**TINY, "factor_order": 3, "patience": None, "scheduler_type": None, "n_regimes": 2,
                   "tol": 1e-3, "beta_kl": 0.5, "random_state": 11})
    est.fit(_panel())
    engine = est.network_
    assert engine.factor_order == 3 and engine._config.factor_order == 3
    assert engine.patience is None and engine._config.patience is None
    assert engine.scheduler_type is None and engine._config.scheduler_type is None and engine.scheduler is None
    assert engine.num_regimes == 2 and engine.tolerance == 1e-3 and engine.beta_kl == 0.5 and engine.seed == 11
    assert est.result_.config["patience"] is None and est.result_.config["factor_order"] == 3


def test_patience_none_disables_early_stopping():
    # A tiny learning rate stalls the ELBO (only the sampling noise moves it), so
    # patience=2 stops early; patience=None (which the engine used to turn into
    # 20) trains for every epoch.
    common = {**TINY, "max_epochs": 30, "tol": 0.0, "learning_rate": 1e-12, "scheduler_type": None}
    stopped = iVDFM(**common, patience=2).fit(_panel())
    full = iVDFM(**common, patience=None).fit(_panel())
    assert stopped.converged_ and stopped.n_iter_ < 30
    assert not full.converged_ and full.n_iter_ == 30


def test_explicit_tuples_and_all_architecture_values_reach_the_engine():
    est = iVDFM(**{**TINY, "encoder_hidden_dim": (8, 6), "decoder_type": "residual", "activation": "tanh",
                   "slope": 0.2, "decoder_var": 0.05, "mixing": True, "use_layer_norm": True,
                   "optimizer": "SGD", "optimizer_momentum": 0.5, "scheduler_type": "plateau",
                   "scheduler_patience": 3, "scheduler_factor": 0.5, "scheduler_min_lr": 1e-5})
    engine = est.fit(_panel()).network_
    assert engine.encoder_hidden_dim == [8, 6]
    assert (engine.decoder_type, engine.activation, engine.slope, engine.decoder_var) == ("residual", "tanh", 0.2, 0.05)
    assert engine.mixing is True and engine.use_layer_norm is True and engine.factor_mixing is not None
    assert engine.optimizer_type == "SGD" and engine.optimizer_momentum == 0.5
    assert isinstance(engine.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert (engine.scheduler_patience, engine.scheduler_factor, engine.scheduler_min_lr) == (3, 0.5, 1e-5)


def test_stride_none_is_half_the_window():
    est = iVDFM(**{**TINY, "window": 10}).fit(_panel())
    assert est.stride_ == 5 and est.window_ == 10
    assert est.network_._config.stride == 5
    est = iVDFM(**{**TINY, "window": 10, "stride": 3}).fit(_panel())
    assert est.stride_ == 3 and est.factors_.index.get_level_values("window").nunique() == (40 - 10) // 3 + 1
    whole = iVDFM(**{**TINY, "window": None}).fit(_panel(T=20))
    assert whole.window_ == 20 and whole.factors_.shape == (20, 2)


# -- fit ------------------------------------------------------------------------


def test_fit_sets_the_fitted_attributes(fitted):
    est = fitted
    panel = _panel()
    assert isinstance(est.result_, iVDFMResult)
    assert isinstance(est.network_, iVDFMEngine) and est.network_._dataset is None
    assert est.device_ == torch.device("cpu")
    assert isinstance(est.scaler_, StandardScaler) and est.scaler == "standard"
    assert est.n_iter_ == 2 and est.converged_ is False
    assert est.n_factors_ == 2 and est.random_state_ == 0
    assert (est.window_, est.stride_, est.time_context_, est.context_dim_) == (8, 4, 1, 1)
    assert est.conditioning_index_.equals(panel.index) and est.index_.equals(panel.index)
    assert list(est.feature_names_in_) == ["s0", "s1", "s2"] and list(est.exog_names_in_) == []
    np.testing.assert_allclose(est.scaler_.mean_, panel.y.to_numpy().mean(axis=0))


def test_factors_frame_is_chronological(fitted):
    frame = fitted.factors_
    n_windows = (40 - 8) // 4 + 1
    assert frame.shape == (n_windows * 8, 2)
    assert list(frame.columns) == ["factor_0", "factor_1"]
    assert frame.index.names == ["window", "date"]
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    index = _panel().index
    for w in (0, 3, n_windows - 1):
        assert frame.loc[w].index.equals(index[w * 4: w * 4 + 8].rename("date"))


def test_factors_frame_undoes_the_extraction_order():
    est = iVDFM(**TINY)
    est.stride_ = 2
    n_windows, window = 4, 3
    order = [2, 0, 3, 1]  # engine row k holds window order[k]
    factors = np.zeros((n_windows, window, 1), dtype=np.float32)
    for k, w in enumerate(order):
        factors[k] = w
    accessed = [0, 1, 2, 3, 3, 2, 1, 0] + order  # training accesses, then the final extraction
    frame = est._factors_frame(factors, accessed, pd.RangeIndex(10))
    for w in range(n_windows):
        assert (frame.loc[w].to_numpy() == w).all()
        assert list(frame.loc[w].index) == [2 * w, 2 * w + 1, 2 * w + 2]
    natural = est._factors_frame(factors, [], pd.RangeIndex(10))
    np.testing.assert_array_equal(natural.to_numpy().reshape(n_windows, window), factors[:, :, 0])


def test_fit_is_repeatable_and_does_not_mutate_inputs():
    panel = _panel()
    y = panel.y.copy()
    est = iVDFM(**TINY)
    first = est.fit(panel).factors_.copy()
    assert est.fit(panel).factors_.equals(first)
    assert iVDFM(**TINY).fit(panel).factors_.equals(first)
    assert panel.y.equals(y) and est.get_params() == iVDFM(**TINY).get_params()


def test_missing_values_are_interpolated():
    panel = _panel()
    y = panel.y.copy()
    y.iloc[[3, 10], [0, 2]] = np.nan
    est = iVDFM(**TINY).fit(PanelData(y=y))
    assert np.isfinite(est.factors_.to_numpy()).all()


def test_time_columns_from_the_index():
    est = iVDFM(**TINY)
    device = torch.device("cpu")
    dt = est._adapter(_panel(), device)
    period = est._adapter(_panel(index="period"), device)
    positional = est._adapter(_panel(index="range"), device)
    assert dt.time_idx == _TIME_COLUMN and period.time_idx == _TIME_COLUMN and positional.time_idx is None
    # A PeriodIndex gives the time features of its start times.
    start_times = PanelData(y=_panel().y.set_axis(_panel(index="period").index.to_timestamp()))
    np.testing.assert_array_equal(period.context, est._adapter(start_times, device).context)
    np.testing.assert_array_equal(positional.context[:, 0], np.arange(40, dtype=np.float32) / 39)
    np.testing.assert_array_equal(dt.data, positional.data)


def test_time_features_ignore_index_resolution_and_time_zone():
    est = iVDFM(**TINY)
    device = torch.device("cpu")
    panel = _panel()
    reference = est._adapter(panel, device).context
    for index in (panel.index.as_unit("s"), panel.index.tz_localize("UTC"),
                  panel.index.tz_localize("UTC").tz_convert("Asia/Seoul")):
        other = est._adapter(PanelData(y=panel.y.set_axis(index)), device).context
        np.testing.assert_array_equal(other, reference)


def test_time_column_name_never_clashes_with_a_series():
    panel = _panel()
    y = panel.y.rename(columns={"s0": _TIME_COLUMN})
    est = iVDFM(**TINY)
    dataset = est._adapter(PanelData(y=y), torch.device("cpu"))
    assert dataset.time_idx == _TIME_COLUMN + "_"
    assert dataset.target_series == [_TIME_COLUMN, "s1", "s2"]


def test_scaler_instances_are_cloned():
    scaler = StandardScaler()
    est = iVDFM(**TINY, scaler=scaler)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.fit(_panel())
    assert est.scaler is scaler and not hasattr(scaler, "mean_")
    assert est.scaler_ is not scaler and hasattr(est.scaler_, "mean_")
    assert est.network_._config.scaler == "standard"

    fitted_scaler = StandardScaler().fit(np.ones((3, 3)))
    with pytest.warns(UserWarning, match="already fitted"):
        est = iVDFM(**TINY, scaler=fitted_scaler).fit(_panel())
    np.testing.assert_array_equal(fitted_scaler.mean_, np.ones(3))
    assert est.scaler_ is not fitted_scaler


def test_get_result_reads_no_random_numbers(fitted):
    state = torch.get_rng_state()
    fitted.network_.get_result()
    assert torch.equal(state, torch.get_rng_state())


# -- silence --------------------------------------------------------------------


def _fit_logged(verbose: int) -> tuple[list, str]:
    package_logger = logging.getLogger("dfm_python")
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.DEBUG)
    saved = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            iVDFM(**{**TINY, "verbose": verbose}).fit(_panel())
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(saved)
    return [r for r in records if r.levelno >= logging.INFO], out.getvalue()


def test_verbose_0_is_silent_even_with_the_package_logger_at_info():
    records, out = _fit_logged(0)
    assert records == [] and out == ""


def test_verbose_1_shows_the_engine_progress():
    records, out = _fit_logged(1)
    messages = [r.getMessage() for r in records]
    assert any("Epoch" in m for m in messages)
    assert "Epoch" in out


# -- predict --------------------------------------------------------------------


def test_predict_is_in_data_units(fitted):
    forecast = fitted.predict(5)
    model_space = fitted.network_.predict(horizon=5)
    expected = fitted.scaler_.inverse_transform(model_space.astype(np.float64))
    assert forecast.to_numpy().tobytes() == expected.tobytes()
    assert list(forecast.columns) == ["s0", "s1", "s2"]
    assert all(dtype == np.float64 for dtype in forecast.dtypes)
    assert forecast.index.equals(pd.date_range("2013-05-31", periods=5, freq="ME", name="date"))
    assert forecast.attrs == {"model": "iVDFM", "kind": "forecast", "origin": pd.Timestamp("2013-04-30")}


def test_predict_without_scaler_is_the_engine_forecast():
    est = iVDFM(**TINY).fit(_panel(index="range"))
    forecast = est.predict(3)
    np.testing.assert_array_equal(forecast.to_numpy(), est.network_.predict(horizon=3).astype(np.float64))
    assert forecast.index.equals(pd.RangeIndex(40, 43, name="time"))


def test_predict_factors(fitted):
    factors = fitted.predict_factors(4)
    assert factors.shape == (4, 2) and list(factors.columns) == ["factor_0", "factor_1"]
    assert all(dtype == np.float64 for dtype in factors.dtypes)
    assert factors.attrs["kind"] == "factors" and factors.index.equals(fitted.predict(4).index)
    engine_factors, _ = fitted.network_._forecast(4, None, True, 1)
    np.testing.assert_array_equal(factors.to_numpy(), engine_factors[0].numpy().astype(np.float64))


def test_predict_is_pure_and_repeatable(fitted):
    before = {k: v.clone() for k, v in fitted.network_.state_dict().items()}
    first = fitted.predict(3)
    assert fitted.predict(3).equals(first)
    after = fitted.network_.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_stochastic_predict(fitted):
    torch.manual_seed(0)
    a = fitted.predict(3, deterministic=False, n_samples=3)
    torch.manual_seed(0)
    b = fitted.predict(3, deterministic=False, n_samples=3)
    assert a.equals(b) and np.isfinite(a.to_numpy()).all()
    with pytest.raises(ValueError, match="n_samples"):
        fitted.predict(3, deterministic=False, n_samples=0)
    with pytest.raises(TypeError, match="deterministic"):
        fitted.predict(3, deterministic="no")


@pytest.mark.parametrize("bad_h", [1.5, "3", None, True])
def test_non_integer_horizon_raises_type_error(fitted, bad_h):
    with pytest.raises(TypeError):
        fitted.predict(bad_h)
    with pytest.raises(TypeError):
        fitted.predict_factors(bad_h)


def test_horizon_below_one_raises(fitted):
    with pytest.raises(ValueError):
        fitted.predict(0)


def test_not_yet_implemented_paths(fitted):
    with pytest.raises(NotImplementedError):
        fitted.predict(3, data=_panel())
    with pytest.raises(NotImplementedError):
        fitted.predict_factors(3, data=_panel())
    with pytest.raises(NotImplementedError):
        fitted.update(_panel())
    panel = _panel()
    series_exog = pd.DataFrame(
        np.zeros((40, 3)), index=panel.index, columns=pd.MultiIndex.from_product([["c"], panel.series_names])
    )
    with pytest.raises(UnsupportedInputError):  # excluded inputs are rejected before anything else
        fitted.predict(3, data=PanelData(y=panel.y, series_exog=series_exog))


def test_unfitted_access_raises():
    est = iVDFM(**TINY)
    for call in (lambda: est.predict(3), lambda: est.predict_factors(3), lambda: est.update(_panel())):
        with pytest.raises(NotFittedError):
            call()
    assert not hasattr(est, "result_")


# -- futr_exog ------------------------------------------------------------------


def test_futr_exog_follows_the_time_features(fitted_exog):
    est = fitted_exog
    assert est.context_dim_ == 4 and est.time_context_ == 2
    future = np.arange(12, dtype=np.float64).reshape(6, 2) / 10
    forecast = est.predict(6, futr_exog=future)
    context = np.hstack([est.network_._future_time_features(6, 2), future])
    np.testing.assert_array_equal(forecast.to_numpy(), est.network_.predict(horizon=6, context_data=context).astype(np.float64))
    by_name = est.predict(6, futr_exog=pd.DataFrame(future[:, ::-1], columns=["u1", "u0"]))
    assert by_name.equals(forecast)
    factors = est.predict_factors(6, futr_exog=future)
    assert factors.shape == (6, 2)


def test_futr_exog_missing_warns_and_uses_time_features(fitted_exog):
    with pytest.warns(UserWarning, match="futr_exog"):
        forecast = fitted_exog.predict(3)
    np.testing.assert_array_equal(forecast.to_numpy(), fitted_exog.network_.predict(horizon=3).astype(np.float64))


@pytest.mark.parametrize(
    "futr_exog, match",
    [
        (np.zeros((5, 2)), "shape"),
        (np.zeros((6, 3)), "shape"),
        (pd.DataFrame(np.zeros((6, 2)), columns=["u0", "x"]), "lacks"),
        (np.full((6, 2), np.nan), "NaN"),
    ],
)
def test_bad_futr_exog_raises(fitted_exog, futr_exog, match):
    with pytest.raises(DataValidationError, match=match):
        fitted_exog.predict(6, futr_exog=futr_exog)


def test_futr_exog_without_exog_raises(fitted):
    with pytest.raises(DataValidationError, match="without exog"):
        fitted.predict(3, futr_exog=np.zeros((3, 1)))

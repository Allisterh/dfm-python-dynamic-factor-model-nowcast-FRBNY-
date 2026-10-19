"""Tests for the public DFM estimator (dfm_python.models.dfm.estimator)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

import dfm_python
import dfm_python.models
import dfm_python.models.dfm
from dfm_python import PanelData
from dfm_python.config import DFMConfig
from dfm_python.config.schema.results import DFMResult
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.models.dfm._engine import DFMEngine
from dfm_python.models.dfm.estimator import _TIME_COLUMN, DFM
from dfm_python.models.dfm.tent import generate_tent_weights
from dfm_python.testing import _CHECKS, _Context, _instance_tags, _split, _state_hash, make_panel
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

TENT = {"q:m": [1, 2, 3, 2, 1]}
TINY = {"max_iter": 5, "tent_weights": TENT}
_MONTHLY = ["m0", "m1", "m2", "m3"]


def _frame(T: int = 60, *, seed: int = 0, index: str = "datetime") -> pd.DataFrame:
    """Four monthly series and one quarterly series (NaN off quarter ends) driven by one AR(1) factor."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    y = np.outer(f, rng.standard_normal(5)) + 0.5 * rng.standard_normal((T, 5))
    idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date") if index == "datetime" else pd.RangeIndex(T)
    df = pd.DataFrame(y, index=idx, columns=_MONTHLY + ["gdp"])
    if index == "datetime":
        df.loc[df.index.month % 3 != 0, "gdp"] = np.nan
    else:
        df.loc[df.index % 3 != 2, "gdp"] = np.nan
    df.iloc[7, 1] = np.nan
    return df


def _mixed(order=None, **kwargs) -> PanelData:
    df = _frame(**kwargs)
    if order is not None:
        df = df[order]
    return PanelData(y=df, freq={"gdp": "q"})


def _monthly(**kwargs) -> PanelData:
    return PanelData(y=_frame(**kwargs)[_MONTHLY])


@pytest.fixture(scope="module")
def fitted() -> DFM:
    return DFM(**TINY, scaler="standard").fit(_mixed(["gdp", "m0", "m1", "m2", "m3"]))


# -- construction ---------------------------------------------------------------


def test_defaults_are_the_values_that_ran_when_omitted():
    assert DFM().get_params() == {
        "n_factors": None, "blocks": None, "clock": "m", "tent_weights": None, "max_iter": 5000, "tol": 1e-5,
        "ar_clip": None, "data_clip": None, "regularization": None, "damping_factor": None, "scaler": None,
        "verbose": 0,
    }
    # The engine config defaults these came from.
    assert DFMConfig(blocks={"b": {"num_factors": 1, "series": []}}).max_iter == 5000
    assert DFMConfig(blocks={"b": {"num_factors": 1, "series": []}}).threshold == 1e-5


def test_constructor_only_stores_and_is_keyword_only():
    np_state = np.random.get_state()
    est = DFM(n_factors=0, clock="x", tent_weights="nonsense", scaler="nonsense", blocks=[1])
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert vars(est) == est.get_params()
    assert est.blocks == [1]
    with pytest.raises(TypeError):
        DFM(1)  # noqa
    for old in ("dataset", "config", "threshold", "num_factors", "frequency", "idio_rho0", "idio_min_var",
                "target_scaler", "random_state", "device"):
        with pytest.raises(TypeError):
            DFM(**{old: 1})


def test_tags():
    tags = DFM.get_tags()
    assert tags["task"] == "forecast" and tags["backend"] == "numpy"
    for cap in ("mixed_frequency", "missing_values", "update", "factor_forecast"):
        assert tags[f"capability:{cap}"] is True
    for cap in ("exog", "series_exog", "device"):
        assert tags[f"capability:{cap}"] is False
    assert tags["update_kind"] == "filter" and tags["predict_data"] == "new_rows"
    assert tags["non_deterministic"] is False


def test_exports():
    assert dfm_python.DFM is DFM
    assert dfm_python.models.DFM is DFM
    assert dfm_python.models.dfm.DFM is DFM
    assert "DFM" in dfm_python.__all__ and "DFM" in dfm_python.models.__all__
    assert issubclass(DFM, dfm_python.BaseFactorModel) and DFM is not DFMEngine
    assert not hasattr(dfm_python.models.dfm, "DFMEngine")


@pytest.mark.parametrize("params", DFM.get_test_params())
@pytest.mark.parametrize("check", ["tags", "pure_construction", "unfitted_result", "fit", "fit_attributes", "silence"])
def test_contract_checks_that_apply_before_update_and_save(params, check):
    """The contract checks the estimator passes already (update, predict(data=) and save/load come next)."""
    checks = {name: fn for name, fn, _ in _CHECKS}
    panel = make_panel(DFM)
    train, new = _split(panel)
    ctx = _Context(cls=DFM, params=params, data=panel, train=train, new=new,
                   tags=_instance_tags(DFM, params, DFM.get_tags()), label="DFM")
    if check == "unfitted_result":
        est = ctx.make()
        assert not hasattr(est, "result_")
        for call in (lambda: est.predict(3), lambda: est.predict_factors(3)):
            with pytest.raises(NotFittedError):
                call()
        return
    if check == "fit_attributes":
        checks["fit"](ctx)
    checks[check](ctx)


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params, name",
    [
        ({"n_factors": 0}, "n_factors"), ({"n_factors": 1.0}, "n_factors"), ({"n_factors": True}, "n_factors"),
        ({"blocks": {}}, "blocks"), ({"blocks": [("b", 1)]}, "blocks"), ({"blocks": {"b": 1}}, "blocks"),
        ({"blocks": {"b": {"n_factors": 0, "series": ["m0"]}}}, "n_factors"),
        ({"blocks": {"b": {"n_factors": 1}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": []}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": "m0"}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": ["m0", "m0"]}}}, "series"),
        ({"blocks": {1: {"n_factors": 1, "series": ["m0"]}}}, "block names"),
        ({"clock": "x"}, "clock"), ({"clock": None}, "clock"),
        ({"tent_weights": [1, 2, 1]}, "tent_weights"),
        ({"tent_weights": {"x:m": [1, 2, 1]}}, "tent_weights"),
        ({"tent_weights": {"q:w": [1, 2, 1]}}, "clock"),
        ({"tent_weights": {"w": [1, 2, 1]}}, "not slower"),
        ({"tent_weights": {"q": [1, 2, 1], "q:m": [1, 2, 1]}}, "twice"),
        ({"tent_weights": {"q:m": []}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1, 0, 1]}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1, float("nan")]}}, "tent_weights"),
        ({"tent_weights": {"q:m": "121"}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1] * 13}}, "at most 12"),
        ({"max_iter": 0}, "max_iter"), ({"max_iter": 2.0}, "max_iter"),
        ({"tol": -1.0}, "tol"), ({"tol": float("inf")}, "tol"), ({"tol": "1e-5"}, "tol"),
        ({"ar_clip": {"low": -1}}, "min"), ({"ar_clip": [-1, 1]}, "ar_clip"), ({"ar_clip": {}}, "ar_clip"),
        ({"ar_clip": {"min": 1.0, "max": -1.0}}, "ar_clip"), ({"ar_clip": {"min": float("nan")}}, "ar_clip"),
        ({"data_clip": 0}, "data_clip"), ({"data_clip": -3.0}, "data_clip"),
        ({"regularization": {"scal": 1e-6}}, "scale"), ({"regularization": 1e-6}, "regularization"),
        ({"regularization": {"min_eigenvalue": 0.0}}, "min_eigenvalue"),
        ({"regularization": {"scale": -1.0}}, "scale"),
        ({"damping_factor": 0.0}, "damping_factor"), ({"damping_factor": 1.5}, "damping_factor"),
        ({"scaler": "zscore"}, "scaler"), ({"scaler": object()}, "scaler"),
        ({"verbose": -1}, "verbose"), ({"verbose": 1.0}, "verbose"),
    ],
)
def test_invalid_parameters_raise_configuration_error_naming_them(params, name, monkeypatch):
    def no_engine(*args, **kwargs):
        raise AssertionError("the engine must not be built")

    monkeypatch.setattr("dfm_python.models.dfm.estimator.DFMEngine", no_engine)
    est = DFM(**{**TINY, **params})
    with pytest.raises(ConfigurationError, match=name):
        est.fit(_mixed())
    assert not hasattr(est, "result_")


def test_n_factors_and_blocks_are_exclusive():
    est = DFM(n_factors=2, blocks={"b": {"n_factors": 1, "series": _MONTHLY}})
    with pytest.raises(ConfigurationError, match="set n_factors or blocks"):
        est.fit(_monthly())


def test_unknown_block_key_names_n_factors():
    est = DFM(blocks={"b": {"num_factors": 1, "series": _MONTHLY}})
    with pytest.raises(ConfigurationError, match="num_factors.*n_factors"):
        est.fit(_monthly())


def test_unknown_block_series_raise_data_validation_error():
    est = DFM(blocks={"b": {"n_factors": 1, "series": ["m0", "gnp"]}}, max_iter=3)
    with pytest.raises(DataValidationError, match="gnp"):
        est.fit(_monthly())


def test_valid_edge_parameters_are_accepted(monkeypatch):
    built = []

    class Stop(Exception):
        pass

    def engine(*args, **kwargs):
        built.append(kwargs)
        raise Stop

    monkeypatch.setattr("dfm_python.models.dfm.estimator.DFMEngine", engine)
    for params in ({"n_factors": np.int64(2)}, {"tol": 0}, {"scaler": "Standard"}, {"scaler": RobustScaler()},
                   {"tent_weights": {"q": np.array([1.0, 2.0, 1.0])}}, {"tent_weights": {"q:m": (1, 1, 1)}},
                   {"ar_clip": {"max": 0.99}}, {"regularization": {"scale": 0.0}}, {"damping_factor": 1},
                   {"data_clip": 5}, {"blocks": {"g": {"series": ["gdp", "m1"]}, "h": {"n_factors": 2, "series": ("m0",)}}}):
        with pytest.raises(Stop):
            DFM(**{**TINY, **params}).fit(_mixed())
    assert len(built) == 11


def test_unexpected_fit_keywords_raise():
    with pytest.raises(TypeError):
        DFM(**TINY).fit(_mixed(), max_iter=3)


def test_exog_series_exog_and_bad_layouts_are_rejected():
    df = _frame()
    with pytest.raises(UnsupportedInputError, match="exog"):
        DFM(**TINY).fit(PanelData(y=df[_MONTHLY], exog=df[["gdp"]].fillna(0.0)))
    rng = np.random.default_rng(1)
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        DFM(**TINY).fit(PanelData.from_arrays(rng.standard_normal((20, 2)), series_exog=rng.standard_normal((20, 2, 3))))
    with pytest.raises(UnsupportedInputError, match="faster than the clock"):
        DFM(**TINY).fit(PanelData(y=df, freq={"m0": "w"}))
    with pytest.raises(UnsupportedInputError, match="one frequency slower"):
        DFM(**TINY).fit(PanelData(y=df, freq={"gdp": "q", "m0": "a"}))
    with pytest.raises(DataValidationError, match="strings"):
        DFM(max_iter=3).fit(pd.DataFrame(df[_MONTHLY].to_numpy(), index=df.index))


def test_slower_frequency_without_tent_weights_names_the_symmetric_kernel():
    kernel = generate_tent_weights(5).tolist()
    assert kernel == [1, 2, 3, 2, 1]
    with pytest.raises(ConfigurationError) as info:
        DFM(max_iter=3).fit(_mixed())
    message = str(info.value)
    assert "['gdp']" in message and "no default kernel" in message
    assert "generate_tent_weights(5)" in message and "tent_weights={'q:m': [1, 2, 3, 2, 1]}" in message
    # The kernel the message names fits.
    model = DFM(max_iter=3, tent_weights={"q:m": kernel}).fit(_mixed())
    assert model.engine_._config.tent_weights == {"q:m": [1, 2, 3, 2, 1]}
    # Weekly clock with a monthly series: 4 weeks per month, 7 weights.
    with pytest.raises(ConfigurationError, match=r"tent_weights=\{'m:w': \[1, 2, 3, 4, 3, 2, 1\]\}"):
        DFM(clock="w").fit(PanelData(y=_frame()[_MONTHLY], freq={"m0": "m"}))
    # Annual on a monthly clock: the symmetric kernel would exceed the supported size.
    with pytest.raises(ConfigurationError, match="23 weights, more than the 12 supported"):
        DFM().fit(PanelData(y=_frame(), freq={"gdp": "a"}))
    # A tent key for the series' frequency without the clock part works too.
    DFM(max_iter=3, tent_weights={"q": kernel}).fit(_mixed())


# -- the adapter and the engine call ----------------------------------------------


def test_every_hyperparameter_reaches_the_engine(monkeypatch):
    seen = {}
    original = DFMEngine.fit

    def fit(self, *args, **kwargs):
        seen["config"], seen["scaler"], seen["kwargs"] = self._config, self.scaler, kwargs
        seen["columns"] = list(self._dataset.variables.columns)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFMEngine, "fit", fit)

    def callback(*args, **kwargs):
        return None

    params = {
        "blocks": {"g": {"n_factors": 2, "series": ["m0", "m1", "m2", "m3", "gdp"]}, "r": {"series": ["m2"]}},
        "clock": "m", "tent_weights": {"q:m": [1, 2, 1]}, "max_iter": 4, "tol": 1e-3,
        "ar_clip": {"min": -0.9, "max": 0.9}, "data_clip": 6.0,
        "regularization": {"scale": 1e-5, "min_eigenvalue": 1e-7, "max_eigenvalue": 0.95},
        "damping_factor": 0.9, "scaler": "robust",
    }
    model = DFM(**params).fit(_mixed(["gdp", "m0", "m1", "m2", "m3"]), checkpoint_callback=callback)
    config = seen["config"]
    assert config.blocks == {"g": {"num_factors": 2, "series": ["m0", "m1", "m2", "m3", "gdp"]},
                             "r": {"num_factors": 1, "series": ["m2"]}}
    assert config.frequency == {"m0": "m", "m1": "m", "m2": "m", "m3": "m", "gdp": "q"}
    assert list(config.frequency) == ["m0", "m1", "m2", "m3", "gdp"]
    assert config.clock == "m" and config.tent_weights == {"q:m": [1, 2, 1]}
    assert config.max_iter == 4 and config.threshold == 1e-3
    assert config.ar_clip == {"min": -0.9, "max": 0.9} and config.data_clip == 6.0
    assert config.regularization == params["regularization"] and config.damping_factor == 0.9
    em = config.to_em_config()
    assert em.regularization == 1e-5 and em.min_norm == 1e-7 and em.max_eigenval == 0.95
    assert type(seen["scaler"]) is RobustScaler and seen["kwargs"] == {"checkpoint_callback": callback}
    assert seen["columns"] == ["m0", "m1", "m2", "m3", "gdp"]
    # The engine got copies: the hyperparameters are unchanged objects of the caller.
    assert config.tent_weights["q:m"] is not params["tent_weights"]["q:m"]
    assert config.ar_clip is not params["ar_clip"]
    assert model.get_params()["blocks"] is params["blocks"]
    assert model.n_factors_ == 3


def test_default_block_is_one_global_block():
    model = DFM(max_iter=3).fit(_monthly())
    assert model.engine_._config.blocks == {"Block_Global": {"num_factors": 1, "series": []}}
    assert model.n_factors_ == 1
    model = DFM(n_factors=2, max_iter=3).fit(_monthly())
    assert model.engine_._config.blocks == {"Block_Global": {"num_factors": 2, "series": []}}
    assert model.n_factors_ == 2


def test_fit_matches_the_engine_on_the_internal_order():
    data = _mixed(["gdp", "m0", "m1", "m2", "m3"])
    model = DFM(**TINY, scaler="standard").fit(data)

    frame = _frame()[_MONTHLY + ["gdp"]].reset_index(drop=True)
    frame.insert(0, "t", np.arange(len(frame)))
    config = DFMConfig.from_dict({
        "blocks": {"Block_Global": {"num_factors": 1, "series": []}},
        "frequency": {**{c: "m" for c in _MONTHLY}, "gdp": "q"},
        "clock": "m", "max_iter": 5, "threshold": 1e-5, "tent_weights": TENT,
    })
    engine = DFMEngine(dataset=DFMDataset(config=config, data=frame, time_index="t"), config=config,
                       scaler=StandardScaler())
    engine.fit()
    result = engine.get_result()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z", "x_sm"):
        np.testing.assert_array_equal(getattr(model.result_, name), getattr(result, name), err_msg=name, strict=True)
    assert model.result_.loglik == result.loglik and model.n_iter_ == result.num_iter
    X, Z = engine.predict(horizon=4)
    np.testing.assert_array_equal(model.predict(4)[_MONTHLY + ["gdp"]].to_numpy(), X, strict=True)
    np.testing.assert_array_equal(model.predict_factors(4).to_numpy(), Z, strict=True)


def test_time_column_never_collides():
    df = _frame()[_MONTHLY].rename(columns={"m0": _TIME_COLUMN})
    model = DFM(max_iter=3).fit(df)
    assert list(model.internal_feature_names_) == [_TIME_COLUMN, "m1", "m2", "m3"]
    assert list(model.predict(2).columns) == [_TIME_COLUMN, "m1", "m2", "m3"]


def test_scaler_instance_is_cloned_and_a_fitted_one_warns():
    scaler = StandardScaler()
    model = DFM(**TINY, scaler=scaler).fit(_mixed())
    assert model.scaler_ is not scaler and not hasattr(scaler, "mean_")
    assert model.scaler_ is model.engine_.scaler and type(model.scaler_) is StandardScaler

    fitted_scaler = StandardScaler().fit(np.arange(10.0).reshape(-1, 1))
    with pytest.warns(UserWarning, match="already fitted") as record:
        model = DFM(**TINY, scaler=fitted_scaler).fit(_mixed())
    assert record[0].filename == __file__
    assert fitted_scaler.mean_.shape == (1,) and model.scaler_.mean_.shape == (5,)


def test_scaler_none_fits_on_the_data_as_given():
    model = DFM(**TINY).fit(_mixed())
    assert model.scaler_ is None and model.engine_.scaler is None
    np.testing.assert_array_equal(model.engine_.data_processed, _frame()[_MONTHLY + ["gdp"]].to_numpy(), strict=True)


# -- fitted attributes ----------------------------------------------------------------


def test_fitted_attributes(fitted):
    user = ["gdp", "m0", "m1", "m2", "m3"]
    assert list(fitted.feature_names_in_) == user and fitted.n_features_in_ == 5
    assert list(fitted.internal_feature_names_) == ["m0", "m1", "m2", "m3", "gdp"]
    assert fitted.freq_ == {"gdp": "q", "m0": "m", "m1": "m", "m2": "m", "m3": "m"}
    assert list(fitted.freq_) == user
    assert list(fitted.exog_names_in_) == []
    assert isinstance(fitted.engine_, DFMEngine) and fitted.engine_._dataset is None
    assert isinstance(fitted.result_, DFMResult) and fitted.result_ is not fitted.engine_.result
    np.testing.assert_array_equal(fitted.result_.Z, fitted.engine_.result.Z, strict=True)
    assert fitted.n_iter_ == fitted.result_.num_iter and isinstance(fitted.n_iter_, int)
    assert fitted.converged_ == bool(fitted.result_.converged) and isinstance(fitted.converged_, bool)
    assert fitted.loglik_ == fitted.result_.loglik and isinstance(fitted.loglik_, float)
    assert fitted.n_factors_ == 1
    assert fitted.index_.equals(_frame().index) and fitted.conditioning_index_.equals(fitted.index_)
    # scaler_ holds the statistics of the series in the internal order.
    np.testing.assert_allclose(fitted.scaler_.mean_, np.nanmean(_frame()[list(fitted.internal_feature_names_)], axis=0))
    # result_ matrices are in the internal order: one row of C per series.
    assert fitted.result_.C.shape[0] == 5 and fitted.result_.x_sm.shape == (60, 5)


def test_factors_frame(fitted):
    factors = fitted.factors_
    m = fitted.result_.Z.shape[1]
    assert list(factors.columns) == [f"state_{i}" for i in range(m)]
    assert factors.index.equals(fitted.index_) and factors.index.name == "date"
    assert all(dtype == np.float64 for dtype in factors.dtypes)
    np.testing.assert_array_equal(factors.to_numpy(), fitted.result_.Z, strict=True)


def test_unfitted_access_raises_not_fitted():
    est = DFM(**TINY)
    assert not hasattr(est, "result_") and not hasattr(est, "engine_")
    for call in (lambda: est.predict(3), lambda: est.predict_factors(3)):
        with pytest.raises(NotFittedError):
            call()


def test_failed_fit_leaves_the_model_unfitted():
    model = DFM(**TINY).fit(_mixed())
    model.set_params(tent_weights=None)
    with pytest.raises(ConfigurationError):
        model.fit(_mixed())
    assert not hasattr(model, "result_") and not hasattr(model, "engine_")


def test_fit_is_repeatable_and_does_not_mutate_inputs():
    data = _mixed()
    before = data.y.copy()
    params = {**TINY, "tent_weights": {"q:m": [1, 2, 3, 2, 1]}}
    a = DFM(**params, scaler="standard").fit(data)
    b = DFM(**params, scaler="standard").fit(data)
    np.testing.assert_array_equal(a.result_.Z, b.result_.Z, strict=True)
    assert a.predict(3).equals(b.predict(3))
    assert data.y.equals(before) and params["tent_weights"] == {"q:m": [1, 2, 3, 2, 1]}


def test_fit_logs_nothing_with_verbose_zero(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        model = DFM(**TINY).fit(make_panel(DFM))
        model.predict(3)
        model.predict_factors(3)
    assert [r for r in caplog.records if r.levelno < logging.ERROR] == []


def test_fit_with_verbose_logs_the_engine_progress(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DFM(**{**TINY, "verbose": 1}).fit(_mixed())
    assert any(r.name.startswith("dfm_python.models.dfm") and r.levelno == logging.INFO for r in caplog.records)


# -- forecasting --------------------------------------------------------------------


def test_predict_frame_in_the_user_order(fitted):
    forecast = fitted.predict(4)
    assert list(forecast.columns) == ["gdp", "m0", "m1", "m2", "m3"]
    assert forecast.shape == (4, 5) and all(dtype == np.float64 for dtype in forecast.dtypes)
    expected_index = pd.date_range("2015-01-31", periods=4, freq="ME", name="date")
    assert forecast.index.equals(expected_index)
    assert forecast.attrs == {"model": "DFM", "kind": "forecast", "origin": fitted.index_[-1]}
    # The quarterly series is reported at every clock date.
    assert np.isfinite(forecast["gdp"].to_numpy()).all()
    X, _ = fitted.engine_.predict(horizon=4)
    internal = list(fitted.internal_feature_names_)
    np.testing.assert_array_equal(forecast.to_numpy(), pd.DataFrame(X, columns=internal)[list(forecast.columns)].to_numpy())


def test_predict_factors_frame(fitted):
    factors = fitted.predict_factors(4)
    assert list(factors.columns) == list(fitted.factors_.columns)
    assert factors.index.equals(fitted.predict(4).index)
    assert factors.attrs == {"model": "DFM", "kind": "factors", "origin": fitted.index_[-1]}
    _, Z = fitted.engine_.predict(horizon=4)
    np.testing.assert_array_equal(factors.to_numpy(), Z, strict=True)


def test_range_index_continues_by_position():
    model = DFM(**TINY).fit(PanelData(y=_frame(index="range"), freq={"gdp": "q"}))
    assert model.predict(3).index.equals(pd.RangeIndex(60, 63, name="time"))
    assert model.factors_.index.equals(pd.RangeIndex(60))


def test_permuted_input_is_reported_in_the_user_order():
    """The internal order is the stable partition [clock | slower]; outputs follow the user's order."""
    ordered = DFM(**TINY, scaler="standard").fit(_mixed(_MONTHLY + ["gdp"]))
    assert list(ordered.internal_feature_names_) == _MONTHLY + ["gdp"]
    expected = ordered.predict(5)

    # Moving the slower series anywhere keeps the internal order, so the numbers are the same.
    for order in (["gdp"] + _MONTHLY, ["m0", "m1", "gdp", "m2", "m3"]):
        model = DFM(**TINY, scaler="standard").fit(_mixed(order))
        assert list(model.internal_feature_names_) == _MONTHLY + ["gdp"]
        assert list(model.feature_names_in_) == order
        np.testing.assert_array_equal(model.result_.Z, ordered.result_.Z, strict=True)
        forecast = model.predict(5)
        assert list(forecast.columns) == order
        assert forecast[_MONTHLY + ["gdp"]].equals(expected)
        assert model.predict_factors(5).equals(ordered.predict_factors(5))

    # Within the clock group the user's relative order is kept.
    permuted_order = ["m2", "gdp", "m0", "m3", "m1"]
    permuted = DFM(**TINY, scaler="standard").fit(_mixed(permuted_order))
    assert list(permuted.internal_feature_names_) == ["m2", "m0", "m3", "m1", "gdp"]
    assert list(permuted.predict(5).columns) == permuted_order
    assert list(permuted.freq_) == permuted_order


def test_predict_does_not_change_the_model(fitted):
    before = _state_hash(fitted)
    first = fitted.predict(3)
    fitted.predict_factors(3)
    assert _state_hash(fitted) == before
    assert fitted.predict(3).equals(first)


def test_predict_data_is_not_available_yet(fitted):
    for method in (fitted.predict, fitted.predict_factors):
        with pytest.raises(NotImplementedError, match="data="):
            method(3, data=_mixed())


@pytest.mark.parametrize("h", [1.0, "3", None, True])
def test_horizon_must_be_an_integer(fitted, h):
    with pytest.raises(TypeError):
        fitted.predict(h)
    with pytest.raises(TypeError):
        fitted.predict_factors(h)


def test_horizon_must_be_positive(fitted):
    with pytest.raises(ValueError):
        fitted.predict(0)

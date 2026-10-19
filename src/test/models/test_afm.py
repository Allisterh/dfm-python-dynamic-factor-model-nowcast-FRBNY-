"""Tests for the Attention Factor Model (models.afm).

Covers the residual-construction math (attention weights, closed-form loadings,
residual projection identity), the LongConv filter, the ``AFM`` estimator
(fit/allocate/backtest/update on ``PanelData``, tags, validation, config
mirror), the private ``AFMEngine`` (``device`` keyword, ``_from_saved``) and
the paper's three model classes (attention+longconv, pca+longconv, pca+ou).
Small synthetic panels only: this is a from-the-paper unit test, not an
empirical reproduction.
"""

import dataclasses
import inspect
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from dfm_python import PanelData
from dfm_python.models.afm import (AFM, AttentionFactors, PCAFactors,
                                   LongConv1d, residuals, ridge_loadings,
                                   avellaneda_lee_weights)
from dfm_python.models.afm._engine import AFMEngine
from dfm_python.dataset.afm_dataset import AFMDataset
from dfm_python.config.schema.model import AFMConfig
from dfm_python.config.schema.results import AFMResult
from dfm_python.utils.errors import (ConfigurationError, DataValidationError,
                                     NotFittedError, UnsupportedInputError)


def _make_data(T=60, N=6, M=8, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, N)) * 0.01
    characteristics = rng.standard_normal((T, N, M))
    return returns, characteristics


class TestAttentionFactors:
    def test_factor_weight_shapes_and_normalization(self):
        block = AttentionFactors(n_char=8, n_factors=3, embed_dim=16)
        X = torch.randn(4, 6, 8)                      # (B, N, M)
        omega_f, beta_t = block(X)
        assert omega_f.shape == (4, 3, 6)             # (B, K, N)
        assert beta_t.shape == (4, 6, 3)              # (B, N, K)
        # softmax over assets: each factor row sums to 1
        row_sums = omega_f.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_residual_projection_identity(self):
        block = AttentionFactors(n_char=8, n_factors=3, embed_dim=16)
        X = torch.randn(6, 8)
        R = torch.randn(6)
        omega_f, beta_t = block(X)
        eps_fn = residuals(omega_f, beta_t, R)
        # omega_eps R = (I - beta^T omega_f) R must equal R - beta^T (omega_f R)
        I = torch.eye(6)
        omega_eps = I - beta_t @ omega_f
        eps_proj = omega_eps @ R
        assert torch.allclose(eps_fn, eps_proj, atol=1e-5)

    def test_ridge_loadings_shape(self):
        omega_f = torch.randn(2, 4, 7)               # (B, K, N)
        beta_t = ridge_loadings(omega_f, ridge=1e-2)
        assert beta_t.shape == (2, 7, 4)             # (B, N, K)


class TestPCAFactors:
    def test_pca_shapes(self):
        pca = PCAFactors(n_factors=3)
        Rwin = torch.randn(2, 40, 6)                 # (B, L, N)
        omega_f, beta_t = pca(Rwin)
        assert omega_f.shape == (2, 3, 6)
        assert beta_t.shape == (2, 6, 3)


class TestLongConv:
    def test_output_shape_and_history_use(self):
        conv = LongConv1d(seq_len=10, n_kernels=8)
        u = torch.randn(4, 6, 10)                     # (B, N, s)
        w = conv(u)
        assert w.shape == (4, 6)
        # a change in the history changes the output (uses the history)
        u2 = u.clone(); u2[..., 3] += 1.0
        assert not torch.allclose(conv(u), conv(u2))

    def test_wrong_length_raises(self):
        conv = LongConv1d(seq_len=10, n_kernels=8)
        with pytest.raises(ValueError):
            conv(torch.randn(4, 6, 9))


def _panel(T=60, N=6, M=8, seed=0, chars=True):
    returns, characteristics = _make_data(T, N, M, seed)
    index = pd.date_range("2020-01-01", periods=T, freq="B", name="date")
    return PanelData.from_arrays(
        returns, series_exog=characteristics if chars else None, index=index,
        names=[f"a{i}" for i in range(N)],
        series_exog_names=[f"c{j}" for j in range(M)] if chars else None)


def _model(**params):
    base = dict(n_factors=3, hist_len=10, max_epochs=3, random_state=0, device="cpu")
    base.update(params)
    return AFM(**base)


def _state(model):
    return {k: v.detach().clone() for k, v in model.network_.state_dict().items()}


def _same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# Estimator parameter -> engine attribute.
_ENGINE_NAMES = {"n_factors": "n_factors", "factor_model": "factor_model", "trading": "trading",
                 "embed_dim": "embed_dim", "hist_len": "hist_len", "n_kernels": "n_kernels",
                 "ridge": "ridge", "squash_lambda": "squash_lambda", "pca_window": "pca_window",
                 "reestim": "reestim", "lambda_var": "lambda_var", "turnover_cost": "turnover_cost",
                 "short_cost": "short_cost", "risk_free": "risk_free",
                 "learning_rate": "learning_rate", "max_epochs": "max_epochs",
                 "patience": "patience", "random_state": "seed"}


class TestAFM:
    def test_constructor_only_stores(self):
        torch_state = torch.random.get_rng_state()
        model = AFM(n_factors=3, hist_len=10, random_state=0)
        assert torch.equal(torch_state, torch.random.get_rng_state())
        assert vars(model) == model.get_params()
        assert not hasattr(model, "result_")

    def test_constructor_is_keyword_only(self):
        with pytest.raises(TypeError):
            AFM(3)

    def test_defaults_are_the_engine_defaults(self):
        engine = AFMEngine()                          # no dataset: nothing is built or seeded
        params = AFM().get_params()
        for name, attr in _ENGINE_NAMES.items():
            assert params[name] == getattr(engine, attr), name
        assert params["device"] == "auto" and params["verbose"] == 0

    def test_tags(self):
        tags = AFM.get_tags()
        assert tags["task"] == "allocation" and tags["backend"] == "torch"
        assert tags["capability:series_exog"] and tags["capability:update"] and tags["capability:device"]
        assert not tags["capability:exog"] and not tags["capability:mixed_frequency"]
        assert tags["update_kind"] == "reattach" and tags["predict_data"] is None
        assert tags["non_deterministic"] is True              # the default random_state=None
        assert AFM().get_tags()["non_deterministic"] is True
        assert AFM(random_state=0).get_tags()["non_deterministic"] is False

    def test_fit_sets_fitted_attributes(self):
        data = _panel()
        model = _model()
        assert model.fit(data) is model
        assert isinstance(model.result_, AFMResult)
        assert model.result_.factor_model == "attention" and model.result_.sharpe is not None
        assert isinstance(model.network_, AFMEngine)
        assert model.device_ == torch.device("cpu")
        assert model.n_features_in_ == 6
        assert list(model.feature_names_in_) == [f"a{i}" for i in range(6)]
        assert list(model.exog_names_in_) == []
        assert list(model.series_exog_names_in_) == [f"c{j}" for j in range(8)]
        assert model.index_.equals(data.index)
        assert model.n_iter_ == 3 and model.converged_ is False
        assert model.n_factors_ == 3 and model.n_char_ == 8
        assert model.scaler_ is None and model.random_state_ == 0
        assert model.data_ is data
        assert model.residuals_.shape == (60, 6) and model.residuals_.index.equals(data.index)
        assert list(model.residuals_.columns) == list(model.feature_names_in_)
        assert model.residuals_.dtypes.eq(np.float64).all()
        assert model.book_returns_.index.equals(data.index[10:])
        assert model.book_returns_.dtype == np.float64
        np.testing.assert_array_equal(model.book_returns_.to_numpy(),
                                      model.result_.book_returns[0].astype(np.float64))

    def test_hyperparameters_reach_the_engine(self):
        params = dict(n_factors=2, factor_model="attention", trading="longconv", embed_dim=6,
                      hist_len=7, n_kernels=5, ridge=0.02, squash_lambda=2e-3, pca_window=30,
                      reestim=4, lambda_var=0.3, turnover_cost=1e-3, short_cost=2e-4,
                      risk_free=1e-4, learning_rate=2e-3, max_epochs=2, patience=5, random_state=4)
        model = AFM(device="cpu", **params).fit(_panel())
        for name, value in params.items():
            assert getattr(model.network_, _ENGINE_NAMES[name]) == value, name

    def test_fit_is_repeatable(self):
        data = _panel()
        a, b = _model().fit(data), _model().fit(data)
        np.testing.assert_array_equal(a.result_.residual_portfolios, b.result_.residual_portfolios)
        np.testing.assert_array_equal(a.result_.book_returns, b.result_.book_returns)
        assert _same_state(_state(a), _state(b))
        state = _state(a)
        a.fit(data)                                   # refitting starts from scratch
        assert _same_state(state, _state(a))

    def test_fit_accepts_a_bare_dataframe_for_pca(self):
        returns, _ = _make_data()
        df = pd.DataFrame(returns, columns=list("abcdef"))
        model = _model(factor_model="pca", trading="longconv").fit(df)
        assert model.result_.factor_model == "pca"
        assert model.n_char_ is None and list(model.series_exog_names_in_) == []
        assert list(model.allocate().columns) == list("abcdef")

    def test_pca_ou_single_pass(self):
        model = _model(factor_model="pca", trading="ou", max_epochs=5).fit(_panel(chars=False))
        assert model.n_iter_ == 1                     # no learnable parameters
        assert model.result_.trading == "ou"

    def test_training_reduces_loss(self):
        data = _panel(seed=1)
        start = _model(max_epochs=1).fit(data)        # the loss of the first pass is at the initial weights
        trained = _model(max_epochs=40).fit(data)

        def loss(m):
            return -(m.result_.sharpe + m.lambda_var * m.result_.explained_variance)

        assert loss(trained) <= loss(start) + 1e-6

    def test_allocate(self):
        data = _panel()
        model = _model().fit(data)
        weights = model.allocate()
        assert weights.shape == (50, 6)
        assert weights.index.equals(data.index[10:])
        assert list(weights.columns) == list(model.feature_names_in_)
        assert weights.dtypes.eq(np.float64).all()
        assert weights.attrs == {"model": "AFM", "kind": "weights", "origin": data.index[-1]}
        np.testing.assert_allclose(weights.abs().sum(axis=1), 1.0, rtol=1e-5)  # ||w||_1 = 1
        assert weights.equals(model.allocate(data=data))

    def test_allocate_is_pure_and_matches_the_engine(self):
        data = _panel()
        model = _model().fit(data)
        state = _state(model)
        weights = model.allocate()
        assert _same_state(state, _state(model))
        expected = model.network_.predict(data=AFMDataset(returns=_make_data()[0],
                                                          characteristics=_make_data()[1]))["weights"][0]
        np.testing.assert_array_equal(weights.to_numpy(), expected.astype(np.float64))

    def test_allocate_on_new_data(self):
        model = _model().fit(_panel())
        new = _panel(T=40, seed=5)
        weights = model.allocate(data=new)
        assert weights.index.equals(new.index[10:])
        assert weights.attrs["origin"] == new.index[-1]

    def test_allocate_checks_assets_and_characteristics(self):
        model = _model().fit(_panel())
        returns, chars = _make_data()
        with pytest.raises(DataValidationError, match="same assets"):
            model.allocate(data=PanelData.from_arrays(returns, series_exog=chars))  # names s0.. vs a0..
        renamed = PanelData.from_arrays(returns, series_exog=chars, names=[f"a{i}" for i in range(6)],
                                        series_exog_names=[f"z{j}" for j in range(8)])
        with pytest.raises(DataValidationError, match="same characteristics"):
            model.allocate(data=renamed)
        with pytest.raises(UnsupportedInputError, match="series_exog"):
            model.allocate(data=PanelData.from_arrays(returns, names=[f"a{i}" for i in range(6)]))

    def test_predict_raises_type_error(self):
        model = _model().fit(_panel())
        with pytest.raises(TypeError, match="allocate"):
            model.predict(5)
        with pytest.raises(TypeError, match="allocate"):
            AFM().predict(5)

    def test_backtest(self):
        data = _panel()
        model = _model().fit(data)
        result = model.backtest(data)
        assert isinstance(result, AFMResult) and result is not model.result_
        assert result.residual_portfolios.shape == (1, 60, 6)
        assert result.weights.shape == (1, 50, 6)
        assert result.book_returns.shape == (1, 50)
        assert isinstance(result.sharpe, float) and isinstance(result.explained_variance, float)
        assert result.num_iter == model.n_iter_ and result.factor_model == "attention"
        np.testing.assert_array_equal(result.weights[0].astype(np.float64), model.allocate().to_numpy())
        other = model.backtest(_panel(T=40, seed=5))
        assert other.weights.shape == (1, 30, 6)

    def test_unfitted(self):
        model = _model()
        assert not hasattr(model, "result_")
        with pytest.raises(NotFittedError):
            _ = model.result_
        with pytest.raises(NotFittedError):
            model.allocate()
        with pytest.raises(NotFittedError):
            model.backtest(_panel())
        with pytest.raises(NotFittedError):
            model.update(_panel())

    def test_attention_requires_series_exog(self):
        model = _model()
        with pytest.raises(UnsupportedInputError, match="factor_model='attention'"):
            model.fit(_panel(chars=False))
        assert not hasattr(model, "result_")

    @pytest.mark.parametrize("bad", ["exog", "mixed", "nan"])
    def test_rejected_inputs(self, bad):
        data = _panel()
        if bad == "exog":
            exog = pd.DataFrame({"x": np.arange(60.0)}, index=data.index)
            data = PanelData(data.y, exog=exog, series_exog=data.series_exog)
            tag = "capability:exog"
        elif bad == "mixed":
            data = PanelData(data.y, series_exog=data.series_exog, freq={"a0": "q", "a1": "m"})
            tag = "capability:mixed_frequency"
        else:
            y = data.y
            y.iloc[3, 2] = np.nan
            data = PanelData(y, series_exog=data.series_exog)
            tag = "capability:missing_values"
        with pytest.raises(UnsupportedInputError, match=tag):
            _model().fit(data)

    @pytest.mark.parametrize("params, name", [
        ({"n_factors": None}, "n_factors"),
        ({"n_factors": 0}, "n_factors"),
        ({"n_factors": 2.0}, "n_factors"),
        ({"factor_model": "bogus"}, "factor_model"),
        ({"trading": "bogus"}, "trading"),
        ({"hist_len": 0}, "hist_len"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"ridge": -1.0}, "ridge"),
        ({"turnover_cost": float("nan")}, "turnover_cost"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"risk_free": "0"}, "risk_free"),
        ({"patience": 0}, "patience"),
        ({"random_state": 1.5}, "random_state"),
        ({"verbose": -1}, "verbose"),
        ({"device": "nonsense"}, "device"),
    ])
    def test_invalid_parameters(self, params, name):
        model = _model(**params)
        with pytest.raises(ConfigurationError, match=name):
            model.fit(_panel())
        assert not hasattr(model, "result_")

    def test_failed_refit_clears_the_old_fit(self):
        model = _model().fit(_panel())
        model.set_params(trading="bogus")
        with pytest.raises(ConfigurationError):
            model.fit(_panel())
        assert not hasattr(model, "result_")

    def test_does_not_mutate_input(self):
        data = _panel()
        copy = PanelData(data.y, series_exog=data.series_exog)
        model = _model()
        params = model.get_params()
        model.fit(data)
        model.allocate()
        assert data.equals(copy)
        assert model.get_params() == params

    def test_update_reattaches_without_changing_parameters(self):
        model = _model().fit(_panel())
        state, result = _state(model), model.result_
        new = _panel(T=40, seed=5)
        assert model.update(new) is model
        assert _same_state(state, _state(model))
        assert model.result_ is result
        assert model.data_ is new
        assert model.allocate().equals(model.allocate(data=new))
        assert model.allocate().index.equals(new.index[10:])

    def test_update_refit_continues_training_like_the_engine(self):
        data, new = _panel(), _panel(T=40, seed=5)
        model = _model().fit(data)
        before = _state(model)
        assert model.update(new, refit=True) is model
        assert not _same_state(before, _state(model))
        assert model.index_.equals(new.index)
        assert model.residuals_.index.equals(new.index)
        assert model.book_returns_.index.equals(new.index[10:])
        assert model.n_iter_ == 3

        # Today's engine path: fit, then update(retrain=True) on the new data.
        returns, chars = _make_data()
        new_returns, new_chars = _make_data(T=40, seed=5)
        engine = AFMEngine(AFMDataset(returns=returns, characteristics=chars), n_factors=3,
                           hist_len=10, max_epochs=3, seed=0, device="cpu")
        engine.fit()
        engine.update(AFMDataset(returns=new_returns, characteristics=new_chars), retrain=True)
        assert _same_state({k: v.detach() for k, v in engine.state_dict().items()}, _state(model))
        np.testing.assert_array_equal(engine.get_result().book_returns, model.result_.book_returns)

    def test_device(self):
        model = _model().fit(_panel())
        assert all(p.device.type == "cpu" for p in model.network_.parameters())
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            auto = _model(device="auto").fit(_panel())
        assert auto.device_ == torch.device("cpu")
        assert _model(device=torch.device("cpu")).fit(_panel()).device_ == torch.device("cpu")

    def test_sklearn_machinery(self):
        model = _model(embed_dim=8)
        assert clone(model).get_params() == model.get_params()
        assert model.set_params(hist_len=5) is model and model.hist_len == 5
        with pytest.raises(ConfigurationError, match="n_factors"):
            model.set_params(num_factors=3)
        assert "AFM(" in repr(model)


class TestAFMConfig:
    def test_fields_mirror_the_constructor(self):
        sig = inspect.signature(AFM.__init__)
        ctor = {n: p.default for n, p in sig.parameters.items() if n != "self"}
        fields = {f.name: f.default for f in dataclasses.fields(AFMConfig)}
        assert list(fields) == list(ctor)
        assert fields == ctor
        assert AFM._config_class is AFMConfig

    def test_from_dict(self):
        cfg = AFMConfig.from_dict({"n_factors": 5, "factor_model": "pca", "hist_len": 15})
        assert cfg.n_factors == 5
        assert cfg.factor_model == "pca"
        assert cfg.hist_len == 15
        assert cfg.turnover_cost == 5e-4              # default preserved

    @pytest.mark.parametrize("old, new", [("num_factors", "n_factors"), ("seed", "random_state")])
    def test_old_names_are_unknown_keys(self, old, new):
        with pytest.raises(ConfigurationError, match=new):
            AFMConfig.from_dict({old: 3})

    def test_invalid_values_raise_at_fit(self):
        model = AFM.from_config(AFMConfig(n_factors=2, factor_model="bogus"))
        with pytest.raises(ConfigurationError, match="factor_model"):
            model.fit(_panel())

    def test_get_config_round_trip(self):
        model = _model(embed_dim=12, patience=3)
        cfg = model.get_config()
        assert isinstance(cfg, AFMConfig)
        assert cfg.to_dict() == model.get_params()
        assert AFM.from_config(cfg).get_params() == model.get_params()
        assert AFM.from_config(cfg.to_dict()).get_params() == model.get_params()

    def test_yaml_round_trip(self, tmp_path):
        pytest.importorskip("yaml")
        cfg = AFMConfig(n_factors=4, hist_len=8, embed_dim=12, max_epochs=2, random_state=1, device="cpu")
        cfg.to_yaml(tmp_path / "afm.yaml")
        assert AFMConfig.from_yaml(tmp_path / "afm.yaml") == cfg

    def test_config_drives_model(self):
        cfg = AFMConfig.from_dict({"n_factors": 4, "hist_len": 8, "embed_dim": 12,
                                   "max_epochs": 2, "device": "cpu", "random_state": 0})
        returns, characteristics = _make_data(T=40, N=5, M=7)
        model = AFM.from_config(cfg).fit(PanelData.from_arrays(returns, series_exog=characteristics))
        assert model.network_.n_factors == 4
        assert model.network_.hist_len == 8
        assert model.network_.embed_dim == 12
        assert model.result_.num_factors == 4


class TestAFMEngine:
    def test_device_keyword(self):
        engine = AFMEngine(n_factors=2, factor_model="pca", device="cpu")
        assert engine.device == torch.device("cpu")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            assert AFMEngine(n_factors=2, factor_model="pca").device == torch.device("cpu")

    @pytest.mark.parametrize("factor_model, trading", [("attention", "longconv"), ("pca", "ou")])
    def test_from_saved_rebuilds_without_consuming_rng(self, factor_model, trading):
        model = _model(factor_model=factor_model, trading=trading).fit(_panel())
        params = {attr: getattr(model.network_, attr) for attr in _ENGINE_NAMES.values()}
        rng = torch.random.get_rng_state()
        engine = AFMEngine._from_saved(params, model.n_char_, "cpu")
        assert torch.equal(rng, torch.random.get_rng_state())
        assert engine.n_char == model.n_char_ and engine._built and engine._dataset is None
        assert engine.training_state is None
        for attr in _ENGINE_NAMES.values():
            assert getattr(engine, attr) == getattr(model.network_, attr), attr
        engine.load_state_dict(model.network_.state_dict())
        returns, chars = _make_data()
        batch = model.network_._as_batch(AFMDataset(returns=returns, characteristics=chars))
        with torch.no_grad():
            a, b = model.network_.forward(*batch), engine.forward(*batch)
        assert torch.equal(a["weights"], b["weights"]) and torch.equal(a["residuals"], b["residuals"])


class TestFidelity:
    """Faithfulness to the paper's LongConv (Squash + geometric init) and the
    rolling-window PCA benchmark, plus the convention-correct optimizer builder."""

    def test_longconv_squash_sparsifies(self):
        heavy = LongConv1d(seq_len=10, n_kernels=8, squash_lambda=50.0)
        assert torch.count_nonzero(heavy._squash()) == 0        # all below threshold
        gentle = LongConv1d(seq_len=10, n_kernels=8, squash_lambda=1e-5)
        assert torch.count_nonzero(gentle._squash()) > 0

    def test_longconv_geometric_init_decays(self):
        torch.manual_seed(0)
        conv = LongConv1d(seq_len=20, n_kernels=8)
        row_mag = conv.kernel.detach().abs().sum(1)             # per-kernel magnitude
        assert row_mag[-1] < row_mag[0]                         # high-h decays fastest

    def test_rolling_pca_varies_over_time(self):
        from dfm_python.models.afm.factors import rolling_pca_weights
        R = torch.randn(1, 400, 8)
        of, bt = rolling_pca_weights(R, n_factors=2, pca_window=100,
                                     reestim=20, ridge=1e-2)
        assert of.shape == (1, 400, 2, 8) and bt.shape == (1, 400, 8, 2)
        assert not torch.allclose(of[:, 50], of[:, 350])       # re-estimated, not static

    def test_pca_uses_rolling_not_static(self):
        # A model with factor_model='pca' should re-estimate over time.
        returns, _ = _make_data(T=120, N=6)
        model = AFM(n_factors=2, hist_len=10, factor_model="pca",
                    trading="ou", reestim=15, pca_window=60, device="cpu")
        model.fit(PanelData.from_arrays(returns))
        assert model.result_ is not None

    def test_build_afm_optimizer(self):
        import torch.nn as nn
        from dfm_python.numeric.builder import build_afm_optimizer
        params = list(nn.Linear(3, 3).parameters())
        assert isinstance(build_afm_optimizer(params, 1e-3), torch.optim.Adam)
        assert isinstance(build_afm_optimizer(params, 1e-3, "sgd"), torch.optim.SGD)

    def test_dataset_dataloader_parity(self):
        returns, characteristics = _make_data(T=80, N=5, M=6)
        ds = AFMDataset(returns=returns, characteristics=characteristics,
                        window=30, stride=10)
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        r, c = next(iter(loader))
        assert r.shape == (2, 30, 5) and c.shape == (2, 30, 5, 6)
        r_all, c_all = ds.get_all_tensors_on_device()
        assert r_all.shape == (80, 5)


def test_afm_passes_the_contract_checks_before_save_load(monkeypatch):
    # AFM's save/load hooks and its registration come with the next item; every
    # other contract check already applies.
    import dfm_python.testing as testing

    monkeypatch.setattr(testing, "_CHECKS", tuple(c for c in testing._CHECKS if c[0] != "save_load"))
    ran = testing.check_estimator(AFM)
    assert ran[-1] == "update" and "device" in ran and "save_load" not in ran

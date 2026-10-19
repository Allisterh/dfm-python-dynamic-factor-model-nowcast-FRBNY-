"""Tests for dfm_python.tuning: the subset, metric and suggest helpers and tune_hyperparameters."""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from dfm_python import AFM, PanelData, iVDFM
from dfm_python.tuning import (
    compute_forecast_metric,
    get_optuna_subset,
    suggest_ivdfm_hyperparameters,
    suggest_ivdfm_hyperparameters_causal,
    tune_hyperparameters,
)

needs_optuna = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="optuna (the tune extra) is not installed"
)

# Tiny networks and structure, so a trial trains in well under a second.
_TINY_FIXED = {
    "n_factors": 2,
    "n_regimes": 1,
    "factor_order": 1,
    "mixing": False,
    "encoder_hidden_dim": 8,
    "decoder_hidden_dim": 8,
    "prior_hidden_dim": 8,
    "encoder_n_hidden_layers": 1,
    "decoder_n_hidden_layers": 1,
    "prior_n_hidden_layers": 1,
}
_TINY_SEARCH = {
    "min_window": 10,
    "max_window": 20,
    "horizon": 6,
    "max_epochs_per_trial": 1,
    "min_batch_size": 8,
}


def _panel(exog: bool = False, T: int = 60) -> PanelData:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-31", periods=T, freq="ME", name="date")
    y = pd.DataFrame(rng.standard_normal((T, 3)).cumsum(axis=0), index=index, columns=["a", "b", "c"])
    if not exog:
        return PanelData(y=y)
    x = pd.DataFrame(rng.standard_normal((T, 2)), index=index, columns=["x0", "x1"])
    x.iloc[5, 0] = np.nan  # filled like fit fills it
    return PanelData(y=y, exog=x)


def _make_mock_trial(suggest_ints=None, suggest_floats=None, suggest_categoricals=None):
    """Return a mock Optuna trial that records suggested values."""
    suggest_ints = suggest_ints or {}
    suggest_floats = suggest_floats or {}
    suggest_categoricals = suggest_categoricals or {}

    def suggest_int(name, low, high, step=1):
        return suggest_ints.get(name, low)

    def suggest_float(name, low, high, log=False):
        return suggest_floats.get(name, (low + high) / 2)

    def suggest_categorical(name, choices):
        return suggest_categoricals.get(name, choices[0])

    trial = MagicMock()
    trial.suggest_int = suggest_int
    trial.suggest_float = suggest_float
    trial.suggest_categorical = suggest_categorical
    return trial


class TestGetOptunaSubset:
    """Tests for get_optuna_subset."""

    def test_get_optuna_subset_array(self):
        """Subset from array: shapes and split are correct."""
        np.random.seed(42)
        T, N = 200, 5
        data = np.random.randn(T, N).astype(np.float32)
        max_window, min_stride, horizon = 50, 10, 12  # min_stride >= 10 in get_optuna_subset
        num_windows = (T - max_window) // min_stride + 1  # 16
        train_window_ratio = 10 / num_windows  # use 10 windows
        train_data, test_true = get_optuna_subset(
            data, max_window=max_window, min_stride=min_stride, horizon=horizon,
            train_window_ratio=train_window_ratio,
        )
        T_subset = (10 - 1) * min_stride + max_window + horizon  # 152
        assert train_data.shape[0] == T_subset - horizon
        assert train_data.shape[1] == N
        assert test_true.shape[0] == horizon
        assert test_true.shape[1] == N
        np.testing.assert_array_almost_equal(
            test_true,
            data[-horizon:, :],
        )

    def test_get_optuna_subset_dataframe(self):
        """Subset from DataFrame: numeric columns only, shapes correct."""
        np.random.seed(42)
        T, N = 200, 4
        data = pd.DataFrame(
            np.random.randn(T, N),
            columns=[f"x{i}" for i in range(N)],
        )
        max_window, min_stride, horizon = 40, 10, 10  # min_stride >= 10 in get_optuna_subset
        num_windows = (T - max_window) // min_stride + 1  # 17
        train_window_ratio = 8 / num_windows  # use 8 windows
        train_data, test_true = get_optuna_subset(
            data, max_window=max_window, min_stride=min_stride, horizon=horizon,
            train_window_ratio=train_window_ratio,
        )
        assert isinstance(train_data, pd.DataFrame)
        assert list(train_data.columns) == [f"x{i}" for i in range(N)]
        T_subset = (8 - 1) * min_stride + max_window + horizon  # 130
        assert len(train_data) == T_subset - horizon
        assert test_true.shape == (horizon, N)

    def test_get_optuna_subset_raises_short_data(self):
        """Too short data raises ValueError."""
        data = np.random.randn(5, 3)  # T=5, horizon+2 would need at least 3+2=5
        with pytest.raises(ValueError, match="Data too short"):
            get_optuna_subset(
                data, max_window=100, min_stride=10, horizon=10, train_window_ratio=1.0,
            )


class TestComputeForecastMetric:
    """Tests for compute_forecast_metric."""

    def test_compute_forecast_metric_smse(self):
        """sMSE returns finite value and is scale-normalized."""
        np.random.seed(42)
        horizon, N = 10, 4
        preds = np.random.randn(horizon, N)
        trues = np.random.randn(horizon, N)
        smse = compute_forecast_metric(preds, trues, metric="sMSE")
        assert np.isfinite(smse)
        assert smse >= 0.0

    def test_compute_forecast_metric_smae(self):
        """sMAE returns finite value."""
        np.random.seed(42)
        horizon, N = 10, 4
        preds = np.random.randn(horizon, N)
        trues = np.random.randn(horizon, N)
        smae = compute_forecast_metric(preds, trues, metric="sMAE")
        assert np.isfinite(smae)
        assert smae >= 0.0

    def test_compute_forecast_metric_perfect_prediction(self):
        """Perfect prediction gives sMSE ~0, sMAE ~0."""
        horizon, N = 10, 4
        trues = np.random.randn(horizon, N)
        preds = trues.copy()
        smse = compute_forecast_metric(preds, trues, metric="sMSE")
        smae = compute_forecast_metric(preds, trues, metric="sMAE")
        assert abs(smse) < 1e-10
        assert abs(smae) < 1e-10

    def test_compute_forecast_metric_unknown_raises(self):
        """Unknown metric raises ValueError."""
        preds = np.random.randn(5, 2)
        trues = np.random.randn(5, 2)
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_forecast_metric(preds, trues, metric="invalid")


@needs_optuna
class TestSuggestIvdfmHyperparameters:
    """Tests for suggest_ivdfm_hyperparameters."""

    def test_suggest_ivdfm_hyperparameters_returns_dict(self):
        """Suggested params contain expected keys and types."""
        trial = _make_mock_trial()
        params = suggest_ivdfm_hyperparameters(trial, max_window=200, max_regimes=5)
        assert "n_factors" in params
        assert "window" in params
        assert "stride" in params
        assert "n_regimes" in params
        assert "regime_temperature" in params
        assert "mixing" in params
        assert "factor_order" in params
        assert "encoder_hidden_dim" in params
        assert "decoder_hidden_dim" in params
        assert "learning_rate" in params
        assert "max_epochs" in params
        assert "decoder_var" in params
        assert "beta_kl" in params
        assert "dropout" in params
        assert isinstance(params["n_factors"], int)
        assert isinstance(params["window"], int)
        assert isinstance(params["mixing"], bool)

    def test_suggest_ivdfm_hyperparameters_respects_fixed(self):
        """Fixed params are in output; meta keys are not passed through."""
        trial = _make_mock_trial()
        fixed = {"innovation_distribution": "laplace", "max_regimes": 3}
        params = suggest_ivdfm_hyperparameters(trial, max_window=100, max_regimes=7, fixed=fixed)
        assert params.get("innovation_distribution") == "laplace"
        assert "max_regimes" not in params



    def test_suggest_uses_estimator_names(self):
        """Trial keys are the estimator's n_factors and n_regimes; fixed values under those names are used."""
        names = []
        trial = _make_mock_trial()
        inner = trial.suggest_int

        def suggest_int(name, low, high, step=1):
            names.append(name)
            return inner(name, low, high, step)

        trial.suggest_int = suggest_int
        params = suggest_ivdfm_hyperparameters(trial, max_window=200, max_regimes=5)
        assert {"n_factors", "n_regimes"} <= set(names)
        assert not any(name.startswith("num_") for name in names)
        assert set(params) <= set(iVDFM().get_params()) | {"dropout"}

        names.clear()
        params = suggest_ivdfm_hyperparameters(trial, max_window=200, fixed={"n_factors": 4, "n_regimes": 2})
        assert params["n_factors"] == 4 and params["n_regimes"] == 2
        assert "n_factors" not in names and "n_regimes" not in names

    def test_suggest_causal_skips_structural_params(self):
        """The causal space suggests no structural parameter and passes fixed non-structural keys through."""
        trial = _make_mock_trial()
        params = suggest_ivdfm_hyperparameters_causal(trial, fixed={"n_factors": 3, "n_regimes": 2, "decoder_type": "linear"})
        assert "n_factors" not in params and "n_regimes" not in params
        assert params["decoder_type"] == "linear"
        assert set(params) <= set(iVDFM().get_params()) | {"dropout"}


@needs_optuna
class TestTuneHyperparameters:
    """Tests for tune_hyperparameters on the iVDFM estimator."""

    def test_one_trial_returns_params_set_params_accepts(self):
        data = _panel()
        y_before = data.y.copy()
        est = iVDFM(scaler="standard", random_state=0, device="cpu")
        params_before = est.get_params()
        best = tune_hyperparameters(est, data, n_trials=1, fixed=dict(_TINY_FIXED), **_TINY_SEARCH)
        assert isinstance(best, dict) and best
        assert "dropout" not in best
        assert not any(k.startswith("num_") for k in best)
        for key, value in _TINY_FIXED.items():
            assert best[key] == value
        assert best["time_context"] == est.time_context
        assert best["batch_size"] == 8
        # The estimator and the data are unchanged, and nothing was fitted in place.
        assert est.get_params() == params_before
        assert not hasattr(est, "network_")
        pd.testing.assert_frame_equal(data.y, y_before)
        est.set_params(**best)
        assert est.get_params()["n_factors"] == 2

    def test_return_study(self):
        import optuna

        best, study = tune_hyperparameters(
            iVDFM(random_state=0, device="cpu"), _panel(), n_trials=1, fixed=dict(_TINY_FIXED),
            return_study=True, **_TINY_SEARCH,
        )
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 1
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert {k: v for k, v in study.best_params.items() if k != "dropout"}.items() <= best.items()

    def test_exog_passes_futr_exog(self, monkeypatch):
        data = _panel(exog=True)
        calls = []
        predict = iVDFM.predict

        def spy(self, h, data=None, **kwargs):
            calls.append((h, kwargs.get("futr_exog"), self.conditioning_index_, list(self.exog_names_in_)))
            return predict(self, h, data, **kwargs)

        monkeypatch.setattr(iVDFM, "predict", spy)
        best = tune_hyperparameters(
            iVDFM(random_state=0, device="cpu"), data, n_trials=1, fixed=dict(_TINY_FIXED), **_TINY_SEARCH,
        )
        assert best
        assert len(calls) == 1
        h, futr_exog, conditioning_index, exog_names = calls[0]
        assert h == 6
        assert exog_names == ["x0", "x1"]
        np.testing.assert_array_equal(futr_exog, data.exog.to_numpy()[-6:])
        # Trained on the rows before the holdout, with the real index.
        assert conditioning_index[-1] == data.index[-7]
        assert conditioning_index.equals(data.index[-6 - len(conditioning_index):-6])

    def test_no_completed_trial_returns_empty_with_warning(self):
        with pytest.warns(UserWarning, match="no trial completed"):
            best = tune_hyperparameters(
                iVDFM(random_state=0, device="cpu"), _panel(), n_trials=1,
                fixed={**_TINY_FIXED, "n_factors": 0}, **_TINY_SEARCH,
            )
        assert best == {}

    def test_rejects_unknown_fixed_key_and_metric(self):
        est = iVDFM(device="cpu")
        with pytest.raises(ValueError, match="num_factors"):
            tune_hyperparameters(est, _panel(), n_trials=1, fixed={"num_factors": 2}, **_TINY_SEARCH)
        with pytest.raises(ValueError, match="Unknown metric"):
            tune_hyperparameters(est, _panel(), n_trials=1, metric="rmse", **_TINY_SEARCH)


def test_non_ivdfm_estimator_raises_type_error():
    with pytest.raises(TypeError, match="iVDFM"):
        tune_hyperparameters(AFM(), _panel(), n_trials=1)
    with pytest.raises(TypeError, match="iVDFM"):
        tune_hyperparameters(iVDFM, _panel(), n_trials=1)


def test_import_without_optuna(monkeypatch):
    """dfm_python.tuning imports without optuna; tune_hyperparameters then names the tune extra."""
    for name in [m for m in sys.modules if m == "dfm_python.tuning" or m.startswith("dfm_python.tuning.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "optuna", None)
    tuning = importlib.import_module("dfm_python.tuning")
    assert "optuna" not in vars(tuning)
    with pytest.raises(ImportError, match=r"pip install 'dfm-python\[tune\]'"):
        tuning.tune_hyperparameters(iVDFM(device="cpu"), _panel(), n_trials=1)
    # The helpers that do not need optuna still work.
    assert tuning.compute_forecast_metric(np.zeros((2, 1)), np.array([[0.0], [1.0]]), metric="MSE") == 0.5

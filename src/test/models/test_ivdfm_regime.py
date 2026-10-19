"""Regime tests for the iVDFM estimator: K=1 versus K>1 fitted through fit/predict/update/save on PanelData.

The engine-level regime math (A_diag/B_diag shapes, effective coefficients as
a convex combination, pi at the window origin, the K=1 global SSM) is in
test_ivdfm_engine.py.
"""

import numpy as np
import pandas as pd
import pytest

from dfm_python import PanelData, iVDFM

SMALL = {
    "n_factors": 3, "window": 24, "stride": 1, "encoder_hidden_dim": 16, "decoder_hidden_dim": 16,
    "prior_hidden_dim": 16, "batch_size": 16, "max_epochs": 2, "random_state": 0, "device": "cpu",
}


def _panel(T=60, N=4, seed=42) -> PanelData:
    values = np.random.default_rng(seed).standard_normal((T, N))
    index = pd.date_range("2015-01-31", periods=T, freq="ME", name="date")
    return PanelData(y=pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(N)]))


class TestRegimeK1:
    def test_k1_fit_predict_p4(self):
        """K=1, p=4: fit and predict run; no regime structure; no NaNs."""
        model = iVDFM(**SMALL, n_regimes=1, factor_order=4).fit(_panel())
        assert model.network_.A_diag is None and model.network_.B_diag is None
        forecast = model.predict(5)
        assert forecast.shape == (5, 4)
        assert np.isfinite(forecast.to_numpy()).all()
        assert model.result_.num_regimes == 1


class TestRegimeKGreaterThan1:
    def test_k2_fit_predict_p4(self):
        """K=2, p=4: fit and predict run with regime-indexed AR coefficients; no NaNs."""
        model = iVDFM(**SMALL, n_regimes=2, factor_order=4).fit(_panel())
        engine = model.network_
        assert engine.A_diag.shape == (2, 4, 3) and engine.B_diag.shape == (2, 3)
        forecast = model.predict(5)
        assert forecast.shape == (5, 4)
        assert np.isfinite(forecast.to_numpy()).all()
        assert np.isfinite(model.predict_factors(5).to_numpy()).all()
        result = model.result_
        assert result.num_regimes == 2 and result.regime_weights is not None
        np.testing.assert_allclose(np.asarray(result.regime_weights).sum(axis=-1), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("temperature", [0.5, 2.0])
    def test_regime_temperature_reaches_the_engine(self, temperature):
        model = iVDFM(**SMALL, n_regimes=3, regime_temperature=temperature).fit(_panel())
        assert model.network_.regime_temperature == temperature
        assert model.result_.regime_temperature == temperature

    def test_k2_update_and_save_load(self, tmp_path):
        model = iVDFM(**SMALL, n_regimes=2, factor_order=2).fit(_panel())
        new = PanelData(y=_panel(T=72, seed=7).y.iloc[30:])
        model.update(new)
        forecast = model.predict(5)
        assert forecast.index[0] == new.index[-1] + pd.offsets.MonthEnd(1)
        model.save(tmp_path / "run")
        loaded = iVDFM.load(tmp_path / "run")
        assert loaded.predict(5).equals(forecast)
        np.testing.assert_array_equal(loaded.network_.regime_weights, model.network_.regime_weights)

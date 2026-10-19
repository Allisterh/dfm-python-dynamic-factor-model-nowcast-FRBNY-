"""Engine-math tests for the private iVDFM engine (models.ivdfm._engine.iVDFMEngine).

The engine is built with explicit dimensions (``data_dim``, ``num_factors``,
``context_dim``, ``window``), which builds its networks without a dataset.
These tests cover the forward pass, the ELBO, and the regime structure: K=1
has no regime-indexed AR coefficients, K>1 has ``A_diag`` (K, p, r) and
``B_diag`` (K, r), the effective coefficients are the convex combination of
the regime coefficients under pi, and the dynamics use pi at the window
origin. Fitting, forecasting, updating and saving go through the public
estimator and are tested in test_ivdfm.py and test_ivdfm_regime.py. The
removed pre-redesign entry points (``set_dataset``, ``save``, ``load``,
``hyperparameter_optimization``, raw data in ``update``) stay removed.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from dfm_python.models.ivdfm._engine import iVDFMEngine
from dfm_python.utils.errors import ModelNotTrainedError


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch and numpy RNGs in __init__; restore them after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _engine(**params) -> iVDFMEngine:
    base = {"data_dim": 5, "num_factors": 3, "context_dim": 1, "window": 20, "device": torch.device("cpu")}
    base.update(params)
    return iVDFMEngine(**base)


def _inputs(engine: iVDFMEngine, batch: int = 2, T: int = 20, N: int = 5, context_dim: int = 1):
    device = next(engine.parameters()).device
    return torch.randn(batch, T, N, device=device), torch.randn(batch, T, context_dim, device=device)


class TestEngineConstruction:
    def test_explicit_dimensions_build_the_components(self):
        engine = _engine()
        assert engine.data_dim == 5
        assert engine.latent_dim == 3
        assert engine.context_dim == 1
        assert engine.window == 20
        assert engine.innovation_encoder is not None
        assert engine.decoder is not None
        assert engine.prior_network is not None
        assert engine.ssm is not None
        assert all(p.device.type == "cpu" for p in engine.parameters())

    def test_without_dimensions_nothing_is_built(self):
        engine = iVDFMEngine(num_factors=3)
        assert engine.data_dim is None
        assert engine.context_dim is None
        assert engine.latent_dim == 3
        assert len(list(engine.parameters())) == 0


class TestEngineSurface:
    """The pre-redesign entry points were removed, not deprecated; they must not come back."""

    @pytest.mark.parametrize("name", ["set_dataset", "save", "load", "hyperparameter_optimization"])
    def test_old_entry_point_is_removed(self, name):
        assert not hasattr(iVDFMEngine, name)
        assert not hasattr(_engine(), name)

    def test_engine_does_not_inherit_the_old_base(self):
        import dfm_python.models.base

        assert not hasattr(dfm_python.models.base, "BaseFactorModel")
        assert iVDFMEngine.__bases__ == (torch.nn.Module,)

    def test_init_sets_the_old_base_attributes(self):
        engine = _engine()
        assert engine._result is None and engine.data_processed is None and engine._dataset is None
        assert engine.training_state is None and engine._config is not None

    def test_update_requires_training(self):
        with pytest.raises(ModelNotTrainedError):
            _engine().update(np.zeros((30, 5), dtype=np.float32))

    @pytest.mark.parametrize("data", [np.zeros((30, 5), dtype=np.float32), pd.DataFrame(np.zeros((30, 5)))])
    def test_update_takes_a_dataset_only(self, data):
        engine = _engine()
        engine.training_state = {}  # past the training check; raw data is refused before it is read
        with pytest.raises(TypeError, match="iVDFMDataset"):
            engine.update(data)
        assert engine.factors is None and engine.innovations is None


class TestEngineForward:
    def test_forward_shapes(self):
        engine = _engine()
        y_1T, u_1T = _inputs(engine)
        outputs = engine.forward(y_1T, u_1T)
        assert outputs["y_pred"].shape == (2, 20, 5)
        assert outputs["eta"].shape == (2, 20, 3)
        assert outputs["factors"].shape == (2, 20, 3)

    def test_elbo(self):
        engine = _engine()
        y_1T, u_1T = _inputs(engine)
        elbo, loss_dict = engine.elbo(y_1T, u_1T, N=100)
        assert isinstance(elbo, torch.Tensor) and elbo.dim() == 0 and torch.isfinite(elbo)
        assert isinstance(loss_dict, dict)
        assert "reconstruction" in loss_dict
        assert "kl" in loss_dict

    def test_context_width_follows_context_dim(self):
        engine = _engine(context_dim=3)
        y_1T, u_1T = _inputs(engine, context_dim=3)
        assert engine.forward(y_1T, u_1T)["factors"].shape == (2, 20, 3)


class TestEngineRegimeK1:
    """K=1: no regime structure; A_diag and B_diag are None."""

    def test_k1_has_no_a_diag_b_diag(self):
        engine = _engine(num_regimes=1, factor_order=4)
        engine._build_components()
        assert engine.A_diag is None
        assert engine.B_diag is None

    def test_k1_forward_uses_the_global_ssm(self):
        engine = _engine(num_regimes=1, factor_order=2)
        engine._build_components()
        out = engine.forward(*_inputs(engine))
        assert out["factors"].shape == (2, 20, 3)
        assert out.get("regime_weights") is None


class TestEngineRegimeKGreaterThan1:
    """K>1: A_diag (K, p, r), B_diag (K, r); effective coefficients; pi at the window origin."""

    def test_k2_has_a_diag_b_diag(self):
        engine = _engine(num_regimes=2, factor_order=4)
        engine._build_components()
        assert engine.A_diag is not None
        assert engine.B_diag is not None
        assert engine.A_diag.shape == (2, 4, 3)
        assert engine.B_diag.shape == (2, 3)

    def test_effective_regime_coeffs_shape(self):
        engine = _engine(num_regimes=3, factor_order=4)
        engine._build_components()
        device = next(engine.parameters()).device
        pi_w = torch.softmax(torch.randn(4, 3, device=device), dim=-1)
        A_eff, B_eff = engine._effective_regime_coeffs(pi_w)
        assert A_eff.shape == (4, 3, 4)
        assert B_eff.shape == (4, 3)

    def test_effective_regime_coeffs_convex_combination(self):
        """A one-hot pi_w gives that regime's A_diag[k] and B_diag[k]; a mixture gives the weighted sum."""
        engine = _engine(num_regimes=3, factor_order=4)
        engine._build_components()
        device = next(engine.parameters()).device
        A_eff, B_eff = engine._effective_regime_coeffs(torch.tensor([[0.0, 1.0, 0.0]], device=device))
        torch.testing.assert_close(A_eff[0], engine.A_diag[1].permute(1, 0))  # (p, r) -> (r, p)
        torch.testing.assert_close(B_eff[0], engine.B_diag[1])
        pi = torch.tensor([[0.2, 0.3, 0.5]], device=device)
        A_mix, B_mix = engine._effective_regime_coeffs(pi)
        expected_A = sum(pi[0, k] * engine.A_diag[k] for k in range(3)).permute(1, 0)
        expected_B = sum(pi[0, k] * engine.B_diag[k] for k in range(3))
        torch.testing.assert_close(A_mix[0], expected_A)
        torch.testing.assert_close(B_mix[0], expected_B)

    def test_k2_forward_returns_regime_weights(self):
        engine = _engine(num_regimes=2, factor_order=2)
        engine._build_components()
        out = engine.forward(*_inputs(engine))
        assert out["factors"].shape == (2, 20, 3)
        assert out["regime_weights"].shape == (2, 20, 2)
        torch.testing.assert_close(out["regime_weights"].sum(-1), torch.ones(2, 20))
        assert not torch.isnan(out["factors"]).any()

    def test_dynamics_use_pi_at_the_window_origin(self):
        """With K>1 the AR coefficients come from regime_weights[:, 0, :], frozen for the whole window."""
        engine = _engine(num_regimes=2, factor_order=1)
        engine._build_components()
        engine.eval()
        y_1T, u_1T = _inputs(engine)
        seen = []
        original = engine._effective_regime_coeffs

        def spy(pi_w):
            seen.append(pi_w.detach().clone())
            return original(pi_w)

        engine._effective_regime_coeffs = spy
        out = engine.forward(y_1T, u_1T)
        assert out["regime_weights"].shape == (2, 20, 2)
        assert out["factors"].shape == (2, 20, 3)
        assert len(seen) == 1 and seen[0].shape == (2, 2)
        torch.testing.assert_close(seen[0], out["regime_weights"][:, 0, :])

"""Tests for numeric.estimator module."""

import numpy as np
from dfm_python.config.constants import DEFAULT_DTYPE


class TestEstimator:
    """Test suite for numeric estimator utilities."""
    
    def test_estimate_ar1_basic(self):
        """Test estimate_ar1 with basic input."""
        from dfm_python.numeric.estimator import estimate_ar1
        
        T = 100
        N = 2
        y = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        x = y[:-1]  # Lagged values (T-1 x N)
        y_current = y[1:]  # Current values (T-1 x N) to match x
        
        A_diag, Q_diag = estimate_ar1(y_current, x=x)
        # Returns diagonal arrays, not matrices
        assert A_diag.shape == (N,)
        assert Q_diag.shape == (N,)
        # Should be 1D arrays
        assert A_diag.ndim == 1
        assert Q_diag.ndim == 1
    
    def test_estimate_var_basic(self):
        """Test estimate_var with basic input."""
        from dfm_python.numeric.estimator import estimate_var
        
        T = 100
        N = 2
        y = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        x = y[:-1]  # Lagged values (T-1 x N)
        y_current = y[1:]  # Current values (T-1 x N) to match x
        
        A, Q = estimate_var(y_current, x)
        # Verify return types and basic properties
        assert A.ndim == 2
        assert Q.ndim == 2
        # A should have m rows where m = y_current.shape[1] = N
        assert A.shape[0] == N
        # Q should be square matrix (implementation may vary, but should be 2D)
        assert Q.shape[0] == Q.shape[1]
    
    def test_estimate_variance_basic(self):
        """Test estimate_variance with basic input."""
        from dfm_python.numeric.estimator import estimate_variance
        
        T = 100
        N = 3
        residuals = np.random.randn(T, N).astype(DEFAULT_DTYPE)
        
        R = estimate_variance(residuals=residuals)
        # Returns diagonal matrix (N x N), not vector
        assert R.shape == (N, N)
        # Should be 2D matrix
        assert R.ndim == 2
        # Diagonal elements should be non-negative
        assert np.all(np.diag(R) >= 0)



def test_get_idio_constant_residual_has_zero_ar_coefficient():
    """A constant residual series (std 0) gets Phi = 0 instead of a division by zero."""
    import warnings
    from dfm_python.numeric.estimator import get_idio

    eps = np.column_stack([np.full(20, 0.5), np.linspace(-1, 1, 20)])
    observed = np.ones_like(eps, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Phi, mu, std = get_idio(eps, observed)
    assert Phi[0, 0] == 0.0 and np.isfinite(Phi).all()
    assert mu[0] == 0.5 and std[0] == 0.0

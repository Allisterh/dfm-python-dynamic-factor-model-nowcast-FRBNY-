"""Tests for ssm.kalman module."""

import pytest
import numpy as np
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.numeric.stability import create_scaled_identity
from dfm_python.config.constants import DEFAULT_PROCESS_NOISE, DEFAULT_TRANSITION_COEF, DEFAULT_IDENTITY_SCALE
from dfm_python.utils.errors import ModelNotInitializedError


class TestDFMKalmanFilter:
    """Test suite for DFMKalmanFilter."""
    
    def test_dfm_kalman_filter_initialization(self):
        """Test DFMKalmanFilter can be initialized."""
        # Test initialization without parameters (lazy initialization)
        kf = DFMKalmanFilter()
        assert kf is not None
        with pytest.raises(ModelNotInitializedError):
            kf.filter(np.zeros((2, 5)))
        
        # Test initialization with all parameters
        m, n = 3, 5  # state dim, observation dim
        A = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf2 = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        assert kf2 is not None
        np.testing.assert_array_equal(kf2.transition_matrices, A)
        np.testing.assert_array_equal(kf2.observation_matrices, C)
    
    def test_dfm_kalman_filter_predict(self):
        """Test DFMKalmanFilter prediction step."""
        # Setup filter with parameters
        m, n = 2, 3
        A = create_scaled_identity(m, DEFAULT_TRANSITION_COEF, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # Test filter() method
        T = 10
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        
        assert filtered_means is not None
        assert filtered_covs is not None
        assert filtered_means.shape == (T, m)
        assert filtered_covs.shape == (T, m, m)
        
        # Test that filter() raises error when not initialized
        kf_uninit = DFMKalmanFilter()
        with pytest.raises(ModelNotInitializedError, match="parameters not initialized"):
            kf_uninit.filter(observations)
    
    def test_dfm_kalman_filter_update(self):
        """Test DFMKalmanFilter update step."""
        # Test update_parameters() method
        m, n = 2, 3
        A = create_scaled_identity(m, DEFAULT_TRANSITION_COEF, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        R = create_scaled_identity(n, DEFAULT_PROCESS_NOISE, dtype=np.float64)
        Z0 = np.zeros(m)
        V0 = create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=np.float64)
        
        kf = DFMKalmanFilter()
        
        # Update parameters
        kf.update_parameters(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # The parameters are stored exactly as given (float64 copies, no regularization)
        np.testing.assert_array_equal(kf.transition_matrices, A)
        np.testing.assert_array_equal(kf.observation_matrices, C)
        np.testing.assert_array_equal(kf.transition_covariance, Q)
        np.testing.assert_array_equal(kf.observation_covariance, R)
        np.testing.assert_array_equal(kf.initial_state_mean, Z0)
        np.testing.assert_array_equal(kf.initial_state_covariance, V0)
        
        # Test that filter works after update
        T = 5
        observations = np.random.randn(T, n)
        filtered_means, filtered_covs = kf.filter(observations)
        assert filtered_means.shape == (T, m)
        assert filtered_covs.shape == (T, m, m)

    def test_progress_bar_no_spam(self):
        """Test that progress bar doesn't spam repeated prints when stuck at 100%."""
        import time
        
        # Setup filter with parameters
        m, n = 2, 3
        A = np.eye(m, dtype=np.float64)
        C = np.random.randn(n, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(n, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # Create observations
        T = 100
        observations = np.random.randn(T, n)
        
        # Run filter_and_smooth which triggers progress bar
        # The key test: it should complete quickly without hanging
        start_time = time.time()
        result = kf.filter_and_smooth(observations)
        elapsed = time.time() - start_time
        
        # Verify it completes (no infinite loop)
        assert elapsed < 5.0, f"Filter took {elapsed:.2f}s, possible infinite loop in progress bar"
        
        # Verify we got results
        assert result is not None
        assert len(result) == 4  # Should return (smoothed_means, smoothed_covs, cross_covs, loglik)
    
    def test_lag_one_covariances_match_the_reference(self):
        """VVsmooth[t] = Cov(x_t, x_{t-1} | Y) for t >= 1 (zero at t = 0), as the reference RTS smoother."""
        from . import _reference_kf as ref

        m = 3
        T = 10
        rng = np.random.default_rng(5)
        A = np.eye(m, dtype=np.float64) * 0.9
        C = rng.standard_normal((5, m))
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(5, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)

        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )

        observations = rng.standard_normal((T, 5))
        smoothed_means, smoothed_covs, cross_covs, loglik = kf.filter_and_smooth(observations)

        assert cross_covs.shape == (T, m, m)
        np.testing.assert_array_equal(cross_covs[0], np.zeros((m, m)))
        expected = ref.filter_smooth(A, C, Q, R, Z0, V0, observations, smoother="rts")
        np.testing.assert_allclose(cross_covs[1:], expected["pair"][1:], rtol=1e-10, atol=1e-12)

    def test_no_inplace_mutation(self):
        """filter_and_smooth leaves the caller's arrays and the stored parameters unchanged."""
        m = 3
        T = 10
        rng = np.random.default_rng(6)
        A = np.eye(m, dtype=np.float64) * 0.9
        C = rng.standard_normal((5, m))
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(5, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        given = [M.copy() for M in (A, C, Q, R, Z0, V0)]

        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        stored = [kf.transition_matrices.copy(), kf.transition_covariance.copy(), kf.initial_state_covariance.copy()]

        observations = rng.standard_normal((T, 5))
        observations[2, 1] = np.nan
        obs_copy = observations.copy()
        kf.filter_and_smooth(observations)

        for before, after in zip(given, (A, C, Q, R, Z0, V0)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(observations, obs_copy)
        np.testing.assert_array_equal(kf.transition_matrices, stored[0])
        np.testing.assert_array_equal(kf.transition_covariance, stored[1])
        np.testing.assert_array_equal(kf.initial_state_covariance, stored[2])

    def test_parameters_are_not_regularized(self):
        """No PSD repair: Q, R and V_0 are only symmetrized; the filter equals the reference with the same Q."""
        from . import _reference_kf as ref

        m = 3
        rng = np.random.default_rng(7)
        A = np.eye(m, dtype=np.float64) * 0.9
        C = rng.standard_normal((5, m))
        # Not symmetric and singular (zero noise on the last state): stored symmetrized, not loaded.
        Q = np.array([[0.2, 0.05, 0.0],
                      [0.03, 0.1, 0.0],
                      [0.0, 0.0, 0.0]], dtype=np.float64)
        R = np.eye(5, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)

        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        np.testing.assert_array_equal(kf.transition_covariance, 0.5 * (Q + Q.T))

        observations = rng.standard_normal((10, 5))
        means, covs = kf.filter(observations)
        expected = ref.kalman_filter(A, C, 0.5 * (Q + Q.T), R, Z0, V0, observations)
        np.testing.assert_allclose(means, expected["fm"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(covs, expected["fP"], rtol=1e-10, atol=1e-12)

    def test_cached_factors_invalidation_on_update(self):
        """Test Bug Fix 3.2: Cached factors invalidated when parameters update."""
        m = 3
        A = np.eye(m, dtype=np.float64) * 0.9
        C = np.random.randn(5, m)
        Q = np.eye(m, dtype=np.float64) * 0.1
        R = np.eye(5, dtype=np.float64) * 0.1
        Z0 = np.zeros(m)
        V0 = np.eye(m, dtype=np.float64)
        
        kf = DFMKalmanFilter(
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # Manually set cache to simulate it being set during EM
        kf._cached_smoothed_factors = np.random.randn(10, m)
        
        # Verify cache exists
        assert kf._cached_smoothed_factors is not None
        
        # Update parameters (should invalidate cache)
        A_new = np.eye(m, dtype=np.float64) * 0.95
        kf.update_parameters(
            transition_matrices=A_new,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=Z0,
            initial_state_covariance=V0
        )
        
        # Cache should be None after parameter update (Bug Fix 3.2)
        assert kf._cached_smoothed_factors is None


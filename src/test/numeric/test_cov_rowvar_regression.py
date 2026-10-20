"""Regression tests for covariance axis semantics.

This guards against confusing time steps (T) with variables (m) when rowvar=True.
"""

import numpy as np

from dfm_python.numeric.stability import compute_cov_safe
from dfm_python.config.constants import DEFAULT_DTYPE


def test_compute_cov_safe_rowvar_true_uses_rows_as_variables() -> None:
    """rowvar=True must interpret rows as variables, returning (m, m) covariance."""
    rng = np.random.default_rng(0)
    T = 10_000
    m = 5
    X = rng.normal(size=(T, m)).astype(DEFAULT_DTYPE)  # (time, variables)

    cov = compute_cov_safe(X.T, rowvar=True, pairwise_complete=True, fallback_to_identity=False)
    assert cov.shape == (m, m)
    assert np.allclose(cov, cov.T, atol=1e-6)
    # Rows of X.T are the variables: the covariance of the m variables over the T observations.
    np.testing.assert_allclose(cov, np.cov(X, rowvar=False, bias=True), rtol=1e-12, atol=1e-12)
    cov_cols = compute_cov_safe(X, rowvar=False, pairwise_complete=True, fallback_to_identity=False)
    np.testing.assert_allclose(cov_cols, cov, rtol=1e-12, atol=1e-12)


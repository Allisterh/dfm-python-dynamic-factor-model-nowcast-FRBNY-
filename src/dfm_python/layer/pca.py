"""PCA functions for factor initialization (NumPy; torch tensors are converted)."""

import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ..logger import get_logger
from ..numeric.stability import create_scaled_identity
from ..config.constants import DEFAULT_IDENTITY_SCALE
from ..config.types import to_numpy

_logger = get_logger(__name__)


def compute_principal_components(
    cov_matrix: Union[np.ndarray, "torch.Tensor"],
    n_components: int,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Top ``n_components`` eigenpairs of ``cov_matrix`` (N x N) by ``|eigenvalue|``, descending.

    A 1x1 matrix gives its entry (``DEFAULT_IDENTITY_SCALE`` when not finite)
    and a unit eigenvector. If the eigendecomposition fails, a warning is
    logged (naming ``block_idx + 1`` when given) and a scaled identity with
    unit eigenvalues is returned.

    Returns
    -------
    eigenvalues : np.ndarray
        ``(n_components,)``.
    eigenvectors : np.ndarray
        ``(N, n_components)``.
    """
    cov_matrix = to_numpy(cov_matrix)

    if cov_matrix.size == 1:
        eigenvector = np.array([[DEFAULT_IDENTITY_SCALE]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else DEFAULT_IDENTITY_SCALE
        return np.array([eigenvalue]), eigenvector

    n_series = cov_matrix.shape[0]

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
        eigenvalues_sorted = eigenvalues[sort_idx]
        eigenvectors_sorted = eigenvectors[:, sort_idx]
        return np.real(eigenvalues_sorted), np.real(eigenvectors_sorted)
    except (ValueError, np.linalg.LinAlgError) as e:
        if block_idx is not None:
            _logger.warning(
                f"PCA: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        else:
            _logger.warning(
                f"PCA: Eigendecomposition failed, using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = create_scaled_identity(n_series, DEFAULT_IDENTITY_SCALE)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def fit_pca(
    X: Union[np.ndarray, "torch.Tensor"],
    n_components: int,
    cov_matrix: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit PCA: by SVD of the centered ``X`` (T x N), or by eigendecomposition of ``cov_matrix`` when given.

    The SVD path returns the squared singular values as eigenvalues, the
    right singular vectors ``(N, n_components)`` and the mean ``(1, N)``, with
    ``cov_matrix`` None; fewer than ``n_components`` are returned, with a
    warning, when the rank is smaller. When the SVD fails (non-finite input
    or singular values, or a LinAlgError), the covariance of the NaN-cleaned
    centered data, regularized by ``1e-8`` on the diagonal, is
    eigendecomposed instead and returned as the fourth value. The
    ``cov_matrix`` path returns the mean as None.

    Returns
    -------
    (eigenvalues, eigenvectors, mean, cov_matrix)
    """
    if cov_matrix is not None:
        eigenvalues, eigenvectors = compute_principal_components(
            cov_matrix, n_components, block_idx=block_idx
        )
        return eigenvalues, eigenvectors, None, to_numpy(cov_matrix)

    X = to_numpy(X)
    mean_ = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean_

    try:
        if np.any(~np.isfinite(X_centered)):
            raise ValueError("Input contains non-finite values")

        col_stds = np.std(X_centered, axis=0)
        if np.any(col_stds < 1e-10):
            _logger.warning(
                f"PCA: Some columns have near-zero variance (min_std={col_stds.min():.2e}). "
                "This may cause SVD convergence issues. Consider removing constant columns."
            )

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False, hermitian=False)
        if not np.all(np.isfinite(S)):
            raise ValueError("SVD produced non-finite singular values")

        n_components_actual = min(n_components, len(S))
        if n_components_actual < n_components:
            _logger.warning(
                f"PCA: Requested {n_components} components but only {len(S)} available. "
                f"Using {n_components_actual} components."
            )

        S = S[:n_components_actual]
        Vt = Vt[:n_components_actual, :]
        return S ** 2, Vt.T, mean_, None

    except (ValueError, np.linalg.LinAlgError) as e:
        error_msg = str(e) if str(e) else type(e).__name__
        _logger.warning(
            f"PCA SVD failed ({error_msg}), falling back to eigendecomposition. "
            "This may be due to numerical instability or ill-conditioned data."
        )

        X_centered_clean = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
        T = X_centered_clean.shape[0]
        cov_raw = (X_centered_clean.T @ X_centered_clean) / max(T - 1, 1)
        reg = 1e-8 * np.eye(cov_raw.shape[0], dtype=cov_raw.dtype)
        cov_matrix_np = cov_raw + reg

        eigenvalues, eigenvectors = compute_principal_components(
            cov_matrix_np, n_components, block_idx=block_idx
        )
        return eigenvalues, eigenvectors, mean_, cov_matrix_np


def extract_pca_factors(
    data: Union[np.ndarray, "torch.Tensor"],
    n_components: int,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Center ``data`` (T x N), fit PCA on it and project it: ``(factors (T, n), eigenvectors (N, n))``."""
    data = to_numpy(data)
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_centered = data - data_mean

    _, eigenvectors, _, _ = fit_pca(
        X=data_centered,
        n_components=n_components,
        block_idx=block_idx
    )
    factors = data_centered @ eigenvectors
    return factors, eigenvectors

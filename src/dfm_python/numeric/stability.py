"""Numerical stabilization of covariance and transition matrices.

The eigendecomposition-based helpers here are O(m^3); the Kalman E-step uses
O(m^2) diagonal loading instead (see ``kalman.py``).
"""

import numpy as np
import warnings
from typing import Optional, Tuple

from ..logger import get_logger
from ..utils.errors import DataError, NumericalError
from ..utils.helper import handle_linear_algebra_error
from ..config.constants import (
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_FACTOR_VARIANCE,
    MAX_EIGENVALUE,
    DEFAULT_REGULARIZATION_SCALE,
    DEFAULT_PROCESS_NOISE,
    SYMMETRY_AVERAGE_FACTOR,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_VARIANCE_FALLBACK,
)

_logger = get_logger(__name__)

MIN_VARIANCE_COVARIANCE = MIN_FACTOR_VARIANCE


def create_scaled_identity(n: int, scale: float = DEFAULT_IDENTITY_SCALE, dtype: type = np.float32) -> np.ndarray:
    """Return ``scale * I_n`` with the given dtype."""
    return np.eye(n, dtype=dtype) * scale


def ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (M + M.T)``."""
    return SYMMETRY_AVERAGE_FACTOR * (M + M.T)


def cap_max_eigenval(
    M: np.ndarray,
    max_eigenval: float = MAX_EIGENVALUE,
    symmetric: bool = False,
    warn: bool = False
) -> np.ndarray:
    """Scale a square matrix down so its spectral radius is at most ``max_eigenval``.

    Parameters
    ----------
    M : np.ndarray
        Square matrix; returned unchanged when empty.
    max_eigenval : float, default MAX_EIGENVALUE
    symmetric : bool, default False
        Use ``eigvalsh`` (and re-symmetrize the result) instead of ``eigvals``.
    warn : bool, default False
        Log when capping happens.

    Raises
    ------
    ValueError
        If ``M`` is not a 2-D square matrix.
    """
    if M.size == 0 or M.shape[0] == 0:
        return M

    if M.ndim != 2:
        raise ValueError(
            f"cap_max_eigenval requires 2D matrix, got ndim={M.ndim}, shape={M.shape}. "
            f"This indicates a bug: eigenvalue capping should only be applied to square matrices."
        )
    if M.shape[0] != M.shape[1]:
        raise ValueError(
            f"cap_max_eigenval requires square matrix, got shape={M.shape}. "
            f"This indicates a bug: eigenvalue capping should only be applied to square matrices. "
            f"Check that you're not calling this on non-square slices or vectors."
        )
    
    def _cap_max_eigenvalue():
        if symmetric:
            eigenvals = np.linalg.eigvalsh(M)
        else:
            eigenvals = np.linalg.eigvals(M)
        max_eig = float(np.max(np.abs(eigenvals)))
        
        if max_eig > max_eigenval:
            scale_factor = max_eigenval / max_eig
            M_capped = M * scale_factor
            if symmetric:
                M_capped = ensure_symmetric(M_capped)
            if warn:
                _logger.warning(
                    f"Matrix maximum eigenvalue capped: {max_eig:.2e} -> {max_eigenval:.2e} "
                    f"(scale_factor={scale_factor:.2e})"
                )
            return M_capped
        return M
    
    return handle_linear_algebra_error(
        _cap_max_eigenvalue, "maximum eigenvalue capping",
        fallback_value=M
    )


def ensure_positive_definite(
    M: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    warn: bool = False
) -> np.ndarray:
    """Symmetrize ``M`` and shift its diagonal so the smallest eigenvalue is at least ``min_eigenval``.

    O(m^3): the shift is ``min_eigenval - min(eig(M))`` from an exact
    eigendecomposition; if that fails, ``min_eigenval`` is added instead.
    """
    M = ensure_symmetric(M)
    
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    def _apply_regularization():
        eigenvals = np.linalg.eigh(M)[0]
        min_eig = float(np.min(eigenvals))
        
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M_reg = M + create_scaled_identity(M.shape[0], reg_amount, M.dtype)
            M_reg = ensure_symmetric(M_reg)
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal."
                )
            return M_reg
        return M
    
    def _fallback_regularization():
        M_reg = M + create_scaled_identity(M.shape[0], min_eigenval, M.dtype)
        M_reg = ensure_symmetric(M_reg)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed). "
                f"Added {min_eigenval:.2e} to diagonal."
            )
        return M_reg
    
    return handle_linear_algebra_error(
        _apply_regularization, "matrix regularization",
        fallback_func=_fallback_regularization
    )


def ensure_covariance_stable(
    M: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE
) -> np.ndarray:
    """:func:`ensure_positive_definite` without warnings; empty ``M`` is returned unchanged."""
    if M.size == 0 or M.shape[0] == 0:
        return M
    return ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)


def ensure_process_noise_stable(
    Q: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    max_eigenval: float = MAX_EIGENVALUE,
    warn: bool = True,
    dtype: type = np.float32
) -> np.ndarray:
    """Bound the eigenvalues of process noise ``Q`` to ``[min_eigenval, max_eigenval]`` and cast to ``dtype``.

    Empty ``Q`` is returned unchanged (and not cast).
    """
    if Q.size == 0 or Q.shape[0] == 0:
        return Q
    Q = ensure_covariance_stable(Q, min_eigenval=min_eigenval)
    Q = cap_max_eigenval(Q, max_eigenval=max_eigenval, symmetric=True, warn=warn)
    return Q.astype(dtype)


def stabilize_innovation_covariance(
    Q: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    min_floor: Optional[float] = None,
    max_eigenval: float = MAX_EIGENVALUE,
    dtype: type = np.float32
) -> np.ndarray:
    """:func:`ensure_process_noise_stable` (silent) for VAR innovation covariances, with an optional floor.

    With ``min_floor`` the result is ``maximum(Q, min_floor * I)``: the
    diagonal is floored and negative off-diagonal entries are clipped to zero.
    """
    if Q.size == 0 or Q.shape[0] == 0:
        return Q
    Q = ensure_process_noise_stable(Q, min_eigenval=min_eigenval, max_eigenval=max_eigenval, warn=False, dtype=dtype)
    if min_floor is not None:
        Q = np.maximum(Q, create_scaled_identity(Q.shape[0], min_floor, dtype))
    return Q.astype(dtype)


def solve_regularized_ols(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = DEFAULT_REGULARIZATION_SCALE,
    use_XTX: bool = True,
    dtype: type = np.float32
) -> np.ndarray:
    """Solve ``(X'X + regularization * I) beta = X'y``, falling back to ``pinv(X) @ y``.

    With ``use_XTX=True`` ``X`` is the (T x p) design matrix and the result
    is (n x p) for 2-D ``y`` (T x n), i.e. transposed, or (p,) for 1-D ``y``.
    With ``use_XTX=False`` ``X`` is already the (p x p) Gram matrix, ``y`` is
    (p x n) or (p,), and the result is returned untransposed.
    """
    if use_XTX:
        try:
            XTX = X.T @ X
            XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype)
            if y.ndim == 1:
                beta = np.linalg.solve(XTX_reg, X.T @ y)
            else:
                beta = np.linalg.solve(XTX_reg, X.T @ y).T
            return beta.astype(dtype)
        except (np.linalg.LinAlgError, ValueError):
            if y.ndim == 1:
                beta = np.linalg.pinv(X) @ y
            else:
                beta = (np.linalg.pinv(X) @ y).T
            return beta.astype(dtype)
    try:
        X_reg = X + create_scaled_identity(X.shape[0], regularization, dtype)
        return np.linalg.solve(X_reg, y).astype(dtype)
    except (np.linalg.LinAlgError, ValueError):
        return (np.linalg.pinv(X) @ y).astype(dtype)


def compute_var_safe(
    data: np.ndarray,
    ddof: int = 0,
    min_variance: float = MIN_VARIANCE_COVARIANCE,
    default_variance: float = DEFAULT_VARIANCE_FALLBACK
) -> float:
    """NaN-ignoring variance of all elements of ``data``.

    Returns ``default_variance`` when ``data`` is empty or the variance is
    non-finite or below ``min_variance``.
    """
    if data.size == 0:
        return default_variance
    if data.ndim > 1:
        data = data.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_val = np.nanvar(data, ddof=ddof)
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    return float(var_val)


def compute_cov_safe(
    data: np.ndarray,
    rowvar: bool = True,
    pairwise_complete: bool = False,
    min_eigenval: float = MIN_EIGENVALUE,
    fallback_to_identity: bool = True
) -> np.ndarray:
    """Covariance (N x N) of ``data`` with missing-data handling and a PSD guarantee.

    ``rowvar`` follows ``numpy.cov``: True means ``data`` is (N x T), False
    (T x N). Observations with any non-finite entry are dropped (listwise),
    or with ``pairwise_complete`` each pair uses its own complete
    observations (population moments, ``ddof=0``). A 1-D input or a single
    variable gives ``[[compute_var_safe(...)]]``. Negative eigenvalues are
    lifted to ``min_eigenval``. Empty data and failures return
    ``DEFAULT_IDENTITY_SCALE * I`` when ``fallback_to_identity`` and raise
    ``DataError`` / ``NumericalError`` otherwise.
    """
    if data.size == 0:
        if fallback_to_identity:
            n = 1 if data.ndim == 1 else (data.shape[1] if rowvar else data.shape[0])
            return create_scaled_identity(n, DEFAULT_IDENTITY_SCALE)
        raise DataError(
            "Cannot compute covariance: data is empty",
            details="Input data has zero size. Provide non-empty data for covariance computation."
        )

    if data.ndim == 1:
        var_val = compute_var_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                   default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])

    n_vars = data.shape[0] if rowvar else data.shape[1]
    if n_vars == 1:
        var_val = compute_var_safe(data.flatten(), ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                   default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])

    try:
        if pairwise_complete:
            data_for_cov = data if rowvar else data.T  # (variables x observations)
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    var_i = data_for_cov[i, :]
                    var_j = data_for_cov[j, :]
                    complete_mask = np.isfinite(var_i) & np.isfinite(var_j)
                    if np.sum(complete_mask) < 2:
                        if i == j:
                            cov[i, j] = DEFAULT_VARIANCE_FALLBACK
                        else:
                            cov[i, j] = DEFAULT_ZERO_VALUE
                    else:
                        var_i_complete = var_i[complete_mask]
                        var_j_complete = var_j[complete_mask]
                        if i == j:
                            cov[i, j] = np.var(var_i_complete, ddof=0)
                        else:
                            mean_i = np.mean(var_i_complete)
                            mean_j = np.mean(var_j_complete)
                            cov[i, j] = np.mean((var_i_complete - mean_i) * (var_j_complete - mean_j))
                            cov[j, i] = cov[i, j]
            np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE_COVARIANCE))
        else:
            if rowvar:
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise DataError(
                        "Insufficient complete observations for covariance",
                        details=f"Only {np.sum(complete_cols)} complete columns available, need at least 2 for covariance computation"
                    )
                cov = np.cov(data[:, complete_cols], rowvar=True)
            else:
                complete_rows = np.all(np.isfinite(data), axis=1)
                if np.sum(complete_rows) < 2:
                    raise DataError(
                        "Insufficient complete observations for covariance",
                        details=f"Only {np.sum(complete_rows)} complete rows available, need at least 2 for covariance computation"
                    )
                cov = np.cov(data[complete_rows, :], rowvar=False)

            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                if len(cov) == n_vars:
                    cov = np.diag(cov)
                else:
                    raise NumericalError(
                        f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})",
                        details="Covariance computation returned unexpected shape. This may indicate numerical issues with input data."
                    )

        if cov.shape != (n_vars, n_vars):
            raise NumericalError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}",
                details="Covariance matrix has incorrect shape. This may indicate numerical issues or data preprocessing problems."
            )
        if np.any(~np.isfinite(cov)):
            raise NumericalError(
                "Covariance contains non-finite values",
                details="Covariance matrix contains NaN or Inf values. Check input data for missing values or numerical issues."
            )
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            cov = cov + create_scaled_identity(n_vars, reg_amount)
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return create_scaled_identity(n_vars, DEFAULT_IDENTITY_SCALE)
        raise


def convergence_checker(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
) -> Tuple[float, float]:
    """Relative change of the reconstruction MSE between two iterations.

    Returns ``(delta, loss_now)`` where the MSEs are taken over the entries
    where ``y_actual`` is not NaN and ``delta = |loss_now - loss_prev| /
    loss_prev``; below ``MIN_FACTOR_VARIANCE`` the absolute difference is used
    to avoid dividing by a near-zero loss.
    """
    mask_flat = ~np.isnan(y_actual).flatten()
    y_prev_valid = y_prev.flatten()[mask_flat]
    y_now_valid = y_now.flatten()[mask_flat]
    y_actual_valid = y_actual.flatten()[mask_flat]

    loss_prev = float(np.mean((y_actual_valid - y_prev_valid) ** 2))
    loss_now = float(np.mean((y_actual_valid - y_now_valid) ** 2))

    if loss_prev < MIN_FACTOR_VARIANCE:
        delta = float(abs(loss_now - loss_prev))
    else:
        delta = float(abs(loss_now - loss_prev) / loss_prev)
    return delta, loss_now


def stabilize_em_process_noise(
    Q: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    max_eigenval: float = MAX_EIGENVALUE,
    default_process_noise: float = None,
    dtype: type = np.float64,
    warn: bool = True
) -> np.ndarray:
    """:func:`ensure_process_noise_stable` with a finite fallback for the EM M-step.

    If the stabilized ``Q`` still has non-finite entries it is replaced by
    ``default_process_noise * I`` (``DEFAULT_PROCESS_NOISE`` when None),
    stabilized again.
    """
    if default_process_noise is None:
        default_process_noise = DEFAULT_PROCESS_NOISE

    Q_stable = ensure_process_noise_stable(Q, min_eigenval=min_eigenval, max_eigenval=max_eigenval, warn=warn, dtype=dtype)
    if np.any(~np.isfinite(Q_stable)):
        _logger.error(
            f"Q is non-finite after ensure_process_noise_stable. "
            f"Inf: {np.sum(np.isinf(Q_stable))}, NaN: {np.sum(np.isnan(Q_stable))}. "
            f"Using fallback scaled identity."
        )
        Q_stable = create_scaled_identity(Q_stable.shape[0], default_process_noise, dtype=dtype)
        Q_stable = ensure_process_noise_stable(Q_stable, min_eigenval=min_eigenval, warn=False, dtype=dtype)
    
    return Q_stable


def stabilize_em_initial_covariance(
    V_0: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    max_eigenval: float = MAX_EIGENVALUE,
    warn: bool = True
) -> np.ndarray:
    """Bound the eigenvalues of the initial state covariance ``V_0`` to ``[min_eigenval, max_eigenval]``.

    Keeps the Kalman prediction ``A V_0 A' + Q`` from overflowing.
    """
    V_0_stable = ensure_covariance_stable(V_0, min_eigenval=min_eigenval)
    return cap_max_eigenval(V_0_stable, max_eigenval=max_eigenval, symmetric=True, warn=warn)


def cap_smoothed_states(
    EZ: np.ndarray,
    max_eigenval: float = MAX_EIGENVALUE,
    warn: bool = True
) -> np.ndarray:
    """Clip smoothed states ``EZ`` (T x m) to ``±sqrt(max_eigenval * T)`` and zero non-finite entries.

    Keeps ``EZ.T @ EZ`` in the M-step bounded by ``max_eigenval * T``.
    """
    if EZ.size == 0:
        return EZ

    max_EZ = np.max(np.abs(EZ))
    EZ_safe_max = np.sqrt(max_eigenval * EZ.shape[0])

    if max_EZ > EZ_safe_max or np.any(~np.isfinite(EZ)):
        n_capped = np.sum(np.abs(EZ) > EZ_safe_max) + np.sum(~np.isfinite(EZ))
        if warn:
            _logger.warning(
                f"Smoothed states EZ have extreme values (max={max_EZ:.2e} > {EZ_safe_max:.2e}) "
                f"or non-finite values. Capping {n_capped}/{EZ.size} values to prevent overflow in M-step."
            )
        EZ = np.clip(EZ, -EZ_safe_max, EZ_safe_max)
        EZ = np.where(np.isfinite(EZ), EZ, 0.0)
    
    return EZ


def rescale_ssm_params_for_C_normalization(
    A: Optional[np.ndarray],
    Q: Optional[np.ndarray],
    V_0: Optional[np.ndarray],
    Z_0: Optional[np.ndarray],
    D_inv: np.ndarray,
    *,
    min_variance: float = MIN_DIAGONAL_VARIANCE,
    max_abs_A: float = 1e6,
    max_abs_Q: float = MAX_EIGENVALUE * 10,
    dtype: type = np.float64,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Rescale (A, Q, V_0, Z_0) so that ``C diag(D_inv)`` gives the same model.

    With ``C' = C diag(D_inv)`` the state in the new units is ``x' = D x`` where
    ``D = diag(1 / D_inv)``, and the model (likelihood, fitted values) is
    unchanged under::

        A' = D A D^-1,   Q' = D Q D,   V_0' = D V_0 D,   Z_0' = D Z_0

    A is not symmetrized. The keyword arguments are accepted for compatibility
    and unused (no guards or fallbacks: the transform is exact).
    """
    if D_inv.size == 0:
        return A, Q, V_0, Z_0

    d = 1.0 / np.asarray(D_inv, dtype=np.float64)
    A_new = None if A is None else np.asarray(A, dtype=np.float64) * d[:, np.newaxis] / d[np.newaxis, :]
    Q_new = None if Q is None else np.asarray(Q, dtype=np.float64) * d[:, np.newaxis] * d[np.newaxis, :]
    V_0_new = None if V_0 is None else np.asarray(V_0, dtype=np.float64) * d[:, np.newaxis] * d[np.newaxis, :]
    Z_0_new = None if Z_0 is None else np.asarray(Z_0, dtype=np.float64) * d
    return A_new, Q_new, V_0_new, Z_0_new


__all__ = [
    'create_scaled_identity',
    'ensure_symmetric',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_var_safe',
    'compute_cov_safe',
    'convergence_checker',
    'solve_regularized_ols',
    'stabilize_em_process_noise',
    'stabilize_em_initial_covariance',
    'cap_smoothed_states',
    'rescale_ssm_params_for_C_normalization',
]


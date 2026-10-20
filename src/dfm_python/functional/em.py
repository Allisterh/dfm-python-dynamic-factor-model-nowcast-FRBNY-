"""EM algorithm implementation for DFM.

This module provides the Expectation-Maximization algorithm for DFM parameter estimation.
The E-step is ``ssm.kalman.DFMKalmanFilter`` (Kalman filter and Durbin-Koopman
smoother with per-element missing data and the exact log-likelihood); the M-step
is the closed-form Banbura-Modugno (2014) update implemented here: per-block
VAR(p) for the factors (companion shift rows kept), AR(1) idiosyncratic states
with the lag-one smoothed covariance, per-series loadings (tent-restricted
least squares for slower-frequency series) with the fixed idiosyncratic
loadings' cross-moments subtracted, the missing-data R formula, and
``Z_0, V_0`` from the smoothed initial state. All moments are float64 and
aligned with the E-step (``EZ[t]`` goes with data row ``t``); there are no
caps, clips or ridge terms, so with the exact E-step the log-likelihood of the
iterates does not decrease.
"""

import time as time_module
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, replace
from scipy.linalg import block_diag, cho_factor, cho_solve

from ..ssm.kalman import DFMKalmanFilter
from ..logger import get_logger
from ..config.schema.block import BlockStructure
from ..config.schema.params import DFMModelState
from ..config.constants import (
    MAX_ESTEP_SECONDS,
    MAX_MSTEP_SECONDS,
    MAX_EM_ITERATION_SECONDS,
    MAX_FIRST_ITERATION_SECONDS,
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_OBSERVATION_NOISE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    VAR_STABILITY_THRESHOLD,
    DEFAULT_SLOWER_FREQ_AR_COEF,
    DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR,
    DEFAULT_EXTREME_FORECAST_THRESHOLD,
    DEFAULT_MAX_VARIANCE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_TENT_KERNEL_SIZE,
)
from ..numeric.stability import (
    create_scaled_identity,
    rescale_ssm_params_for_C_normalization,
)
from ..numeric.estimator import (
    estimate_var,
    estimate_ar1,
    apply_ar_clipping,
)
from ..utils.misc import get_config_attr
from ..utils.errors import DataValidationError, NumericalError
from ..logger.progress import log_em_iteration, log_convergence

_logger = get_logger(__name__)


@dataclass
class EMConfig:
    """Configuration for EM algorithm parameters.

    The M-step is the closed-form Banbura-Modugno (2014) update with no caps,
    clips or ridge terms. Fields kept for API compatibility that the M-step no
    longer uses: ``regularization`` (no ridge term in the moment solves),
    ``max_eigenval`` (no eigenvalue cap on A, Q or V_0), ``max_variance`` (no
    upper clip on R) and ``matrix_regularization``. ``min_variance`` is the
    floor of ``R_ii`` for series without an idiosyncratic state (series with one
    use ``MIN_OBSERVATION_NOISE``). ``normalize_C`` rescales each factor once,
    after the last EM iteration, so its current-period loading column has unit
    norm (a likelihood-invariant change of state units, see
    ``_normalize_C_with_invariance``); it is never applied inside the EM loop.
    ``ar_clip`` and ``damping_factor`` are opt-in and break the EM ascent
    property.
    """
    regularization: float = DEFAULT_REGULARIZATION  # Unused by the M-step (kept for compatibility)
    min_norm: float = MIN_EIGENVALUE  # Loading columns with a smaller norm are not normalized
    max_eigenval: float = VAR_STABILITY_THRESHOLD  # Unused by the M-step (kept for compatibility)
    min_variance: float = MIN_DIAGONAL_VARIANCE
    max_variance: float = DEFAULT_MAX_VARIANCE  # Unused by the M-step (kept for compatibility)
    min_iterations_for_convergence_check: int = 2
    convergence_log_interval: int = DEFAULT_LOG_INTERVAL
    progress_log_interval: int = DEFAULT_PROGRESS_LOG_INTERVAL
    small_loglik_threshold: float = MIN_FACTOR_VARIANCE
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    # Initialization constants (used by DFM initialization)
    default_transition_coef: float = DEFAULT_TRANSITION_COEF
    default_process_noise: float = DEFAULT_PROCESS_NOISE
    default_observation_noise: float = MIN_DIAGONAL_VARIANCE
    matrix_regularization: float = DEFAULT_REGULARIZATION
    eigenval_floor: float = MIN_EIGENVALUE
    slower_freq_ar_coef: float = DEFAULT_SLOWER_FREQ_AR_COEF  # AR coefficient for slower-frequency idiosyncratic components
    tent_kernel_size: int = DEFAULT_TENT_KERNEL_SIZE
    slower_freq_variance_denominator: float = DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR  # Variance denominator for slower-frequency series
    extreme_forecast_threshold: float = DEFAULT_EXTREME_FORECAST_THRESHOLD
    # Numerical stability parameters (from DFMConfig)
    ar_clip: Optional[Dict[str, float]] = None  # AR coefficient clipping: {"min": float, "max": float}
    damping_factor: Optional[float] = None  # Damping factor for parameter updates (0.8 = 80% new, 20% old)
    data_clip: Optional[float] = None  # Clip data values beyond this many standard deviations
    normalize_C: bool = True  # Rescale factors to unit-norm loading columns once after EM (likelihood-invariant)


_DEFAULT_EM_CONFIG = EMConfig()


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetric part of a square matrix (removes round-off asymmetry only)."""
    return 0.5 * (M + M.T)


def _initial_covariance_update(V_smooth_0: np.ndarray) -> np.ndarray:
    """M-step for V_0: ``Cov(x_0 | Y)``, symmetrized and, if needed, projected to PSD.

    The exact smoothed covariance is positive semidefinite; with an
    ill-conditioned prior (e.g. a V_0 with eigenvalues from 1e-6 to 1e5) its
    computed value can have small negative eigenvalues, which would make the
    next prior an invalid Gaussian. Those eigenvalues are set to 0 (the
    nearest PSD matrix); a PSD input is returned only symmetrized.
    """
    V = _sym(np.asarray(V_smooth_0, dtype=np.float64))
    w, U = np.linalg.eigh(V)
    if w.size and w[0] < 0.0:
        V = _sym((U * np.maximum(w, 0.0)) @ U.T)
    return V


class _MStepError(NumericalError):
    """An M-step block could not be computed at parameters whose E-step succeeded.

    ``loglik`` is the exact log-likelihood of the parameters the failed
    iteration started from, so ``run_em_algorithm`` can return them.
    """

    def __init__(self, message: str, loglik: float, details: Optional[str] = None):
        super().__init__(message, details=details)
        self.loglik = loglik


def _solve_spd(M: np.ndarray, B: np.ndarray, what: str) -> np.ndarray:
    """Solve ``M X = B`` for a symmetric positive definite moment matrix (Cholesky).

    The moment matrices of the M-step (sums of ``E[x x' | Y]``) are positive
    definite unless the data cannot identify the parameters. A matrix that is
    not numerically positive definite raises ``NumericalError``; nothing is
    regularized.
    """
    M = _sym(np.asarray(M, dtype=np.float64))
    try:
        factor = cho_factor(M, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise NumericalError(
            f"M-step: the moment matrix of the {what} is not positive definite ({e})."
        ) from e
    d = np.abs(np.diag(factor[0]))
    if d.size and d.min() ** 2 <= M.shape[0] * np.finfo(np.float64).eps * d.max() ** 2:
        raise NumericalError(
            f"M-step: the moment matrix of the {what} is singular to working precision "
            f"(Cholesky diagonal ratio {d.min() / d.max():.2e})."
        )
    X = cho_solve(factor, np.asarray(B, dtype=np.float64))
    if not np.all(np.isfinite(X)):
        raise NumericalError(f"M-step: the solve for the {what} is not finite.")
    return X


def _factor_scales(C: np.ndarray, r: np.ndarray, p_plus_one: int, min_norm: float) -> np.ndarray:
    """Per-factor scale ``d`` (length m) for the C normalization.

    ``d`` is the Euclidean norm of the column of C that loads the current value
    of each factor, repeated on every lag copy of that factor in the companion
    block; it is 1 on the idiosyncratic states and on factors whose loading
    column has norm at most ``min_norm``.
    """
    m = C.shape[1]
    d = np.ones(m, dtype=np.float64)
    offset = 0
    for r_b in np.asarray(r).astype(int):
        for f in range(r_b):
            norm = float(np.linalg.norm(C[:, offset + f]))
            if norm > min_norm:
                lag_copies = offset + f + r_b * np.arange(p_plus_one)
                d[lag_copies[lag_copies < m]] = norm
        offset += r_b * p_plus_one
    return d


def _normalize_C_with_invariance(
    C: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
    log_prefix: str = ""
) -> Tuple[np.ndarray, DFMModelState]:
    """Rescale each factor so its current-period loading column has unit norm.

    The state-space model is unchanged (same likelihood, same fitted values)
    under the change of state units ``x' = D x`` with ``D`` diagonal::

        C' = C D^-1,  A' = D A D^-1,  Q' = D Q D,  V_0' = D V_0 D,  Z_0' = D Z_0

    ``D`` holds one scale per factor (the norm of the C column of its current
    value, from ``_factor_scales``), used for every lag copy of that factor so
    the companion shift rows stay identity; idiosyncratic states keep scale 1.
    EM calls this once, after the last iteration, when ``config.normalize_C`` is
    True; it is not part of the M-step.

    Returns
    -------
    Tuple[np.ndarray, DFMModelState]
        ``(C', state')`` where ``state'`` carries C', A', Q', V_0', Z_0'.
    """
    p_plus_one = state.max_lag_size if state.max_lag_size is not None else (state.p + 1)
    d = _factor_scales(np.asarray(C, dtype=np.float64), state.r, int(p_plus_one), config.min_norm)
    D_inv = 1.0 / d
    C_new = np.asarray(C, dtype=np.float64) * D_inv[np.newaxis, :]
    A_new, Q_new, V_0_new, Z_0_new = rescale_ssm_params_for_C_normalization(
        state.A, state.Q, state.V_0, state.Z_0, D_inv
    )
    n_scaled = int(np.sum(d != 1.0))
    if n_scaled:
        _logger.debug(f"{log_prefix}Normalized the loading scale of {n_scaled}/{C.shape[1]} states "
                      f"(likelihood-invariant change of state units).")
    return C_new, replace(state, C=C_new, A=A_new, Q=Q_new, V_0=V_0_new, Z_0=Z_0_new)


def _validate_blocks_shape(blocks: np.ndarray, n_series: int) -> None:
    """Validate that blocks array has correct number of rows.
    
    Raises
    ------
    ValueError
        If blocks.shape[0] != n_series
    """
    if blocks.shape[0] != n_series:
        raise ValueError(
            f"Blocks array shape mismatch: blocks.shape[0]={blocks.shape[0]} != n_series={n_series}. "
            f"Blocks must have exactly n_series rows (one row per data series). "
            f"This indicates a configuration error - blocks should be aligned in fit() via build_dfm_blocks()."
        )


def _compute_and_cache_block_indices(block_structure: BlockStructure, N: int) -> None:
    """Compute and cache block structure indices for reuse across EM iterations.
    
    Computes unique block patterns, factor indices (bl_idxM, bl_idxQ), constraint matrices,
    and idiosyncratic component indices. Results are cached in block_structure.
    """
    # Validate cache is still valid (not just existence check)
    # Cache validity depends on: N, blocks, r, p, p_plus_one, idio_indicator, R_mat, q, n_clock_freq
    if block_structure.has_cached_indices():
        # Check if cached N matches current N
        if block_structure._cached_N == N:
            # Additional validation: check if structure parameters match
            # If blocks, r, p, etc. changed, cache is invalid
            # For now, we trust that if N matches and cache exists, it's valid
            # (blocks/r/p changes would require new BlockStructure instance)
            return  # Already cached and valid
        else:
            # N changed - invalidate cache
            block_structure.clear_cache()
    
    blocks = block_structure.blocks
    r = block_structure.r
    p_plus_one = block_structure.p_plus_one
    n_blocks = len(r)
    R_mat = block_structure.R_mat
    q = block_structure.q
    n_clock_freq = block_structure.n_clock_freq
    idio_indicator = block_structure.idio_indicator
    
    _validate_blocks_shape(blocks, N)
    
    block_tuples = [tuple(row) for row in blocks]
    unique_blocks = []
    unique_indices = []
    seen = set()
    for i, bt in enumerate(block_tuples):
        if bt not in seen:
            unique_blocks.append(blocks[i].copy())
            unique_indices.append(i)
            seen.add(bt)
    
    # Build block indices for clock-frequency and slower-frequency factors
    bl_idxM = []
    bl_idxQ = []
    R_con = None
    q_con = None
    
    # Calculate total factor state dimension
    total_factor_dim = int(np.sum(r) * p_plus_one)
    
    if R_mat is not None and q is not None:
        from scipy.linalg import block_diag
        R_con_blocks = []
        q_con_blocks = []
        
        # Build indices for each unique block pattern
        for bl_row in unique_blocks:
            bl_idxQ_row = []
            bl_idxM_row = []
            
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
            
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
            
            # Build constraint matrix for blocks used in this pattern
            pattern_blocks = [block_idx for block_idx in range(n_blocks) if bl_row[block_idx] > 0]
            if pattern_blocks:
                for block_idx in pattern_blocks:
                    R_con_blocks.append(np.kron(R_mat, create_scaled_identity(int(r[block_idx]), DEFAULT_IDENTITY_SCALE)))
                    q_con_blocks.append(np.zeros(R_mat.shape[0] * int(r[block_idx])))
        
        if R_con_blocks:
            R_con = block_diag(*R_con_blocks)
            q_con = np.concatenate(q_con_blocks)
    else:
        # No constraints - simpler indexing
        for bl_row in unique_blocks:
            bl_idxM_row = []
            bl_idxQ_row = []
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
    
    # Convert to boolean arrays
    bl_idxM = [np.array(row, dtype=bool) for row in bl_idxM] if bl_idxM else []
    bl_idxQ = [np.array(row, dtype=bool) for row in bl_idxQ] if bl_idxQ else []
    
    # Idiosyncratic component indices
    idio_indicator_M = idio_indicator[:n_clock_freq]
    n_idio_M = int(np.sum(idio_indicator_M))
    c_idio_indicator = np.cumsum(idio_indicator)
    rp1 = int(np.sum(r) * p_plus_one)  # Start of idiosyncratic components
    
    # Cache all computed indices
    block_structure._cached_unique_blocks = unique_blocks
    block_structure._cached_unique_indices = unique_indices
    block_structure._cached_bl_idxM = bl_idxM
    block_structure._cached_bl_idxQ = bl_idxQ
    block_structure._cached_R_con = R_con
    block_structure._cached_q_con = q_con
    block_structure._cached_total_factor_dim = total_factor_dim
    block_structure._cached_idio_indicator_M = idio_indicator_M
    block_structure._cached_n_idio_M = n_idio_M
    block_structure._cached_c_idio_indicator = c_idio_indicator
    block_structure._cached_rp1 = rp1
    block_structure._cached_N = N  # Store N value used for caching


def _factor_offsets(r: np.ndarray, p_plus_one: int) -> np.ndarray:
    """State index where each factor block starts (blocks are ``r_b * p_plus_one`` wide)."""
    widths = np.asarray(r).astype(int) * int(p_plus_one)
    return np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(int)


def _update_transition_matrix_blocked(
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """M-step for A and Q (Banbura-Modugno 2014), block by block.

    Timing: ``EZ[t]`` is the smoothed state of data row ``t`` (``t = 0..T-1``),
    ``VVsmooth[t] = Cov(x_t, x_{t-1} | Y)``. The transition sums run over
    ``t = 1..T-1``::

        S11 = sum E[x_t x_t'],  S00 = sum E[x_{t-1} x_{t-1}'],  S10 = sum E[x_t x_{t-1}']

    Factor block ``b`` (``r_b`` factors, companion width ``r_b * L``): only the
    ``r_b`` current-factor rows are estimated, on the first ``r_b * p`` lag
    columns (a VAR(p)): ``A_b = S10 S00^-1`` and ``Q_b = (S11 - A_b S10') / (T-1)``.
    The other columns of those rows are zero, and the companion shift rows and
    their process noise are kept as they are.

    Idiosyncratic AR(1) states (``sum(idio_indicator)`` states after the factor
    blocks): ``a_i = S10_ii / S00_ii`` and ``q_i = (S11_ii - a_i S10_ii) / (T-1)``,
    with the lag-one smoothed covariance in ``S10``. The slower-frequency
    idiosyncratic chains after them are kept fixed.

    ``V_0`` is set to ``V_smooth[0]`` (the maximizer; see ``_initial_covariance_update``).
    Everything is float64, with no caps, clips or regularization; a singular
    moment matrix raises ``NumericalError``.
    """
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    VVsmooth = np.asarray(VVsmooth, dtype=np.float64)
    T, m = EZ.shape
    A_new = np.array(state.A, dtype=np.float64, copy=True)
    Q_new = np.array(state.Q, dtype=np.float64, copy=True)
    if T < 2:
        return replace(state, A=A_new, Q=Q_new)

    r = np.asarray(state.r).astype(int)
    p = int(state.p)
    p_plus_one = int(state.max_lag_size if state.max_lag_size is not None else (p + 1))
    n_lags = min(p, p_plus_one)
    offsets = _factor_offsets(r, p_plus_one)

    for b, (r_b, start) in enumerate(zip(r, offsets)):
        cur = np.arange(start, start + r_b)
        lag = np.arange(start, start + r_b * n_lags)
        A_b, Q_b = estimate_var(
            y=EZ[1:, cur],
            x=EZ[:-1, lag],
            V_smooth=V_smooth[1:][:, cur][:, :, cur],
            V_smooth_lag=V_smooth[:-1][:, lag][:, :, lag],
            VVsmooth=VVsmooth[1:][:, cur][:, :, lag],
            dtype=np.float64,
        )
        A_new[start:start + r_b, start:start + r_b * p_plus_one] = 0.0
        A_new[np.ix_(cur, lag)] = A_b
        Q_new[np.ix_(cur, cur)] = Q_b

    n_factor_states = int(np.sum(r) * p_plus_one)
    idio_indicator = state.idio_indicator
    n_idio = int(np.sum(np.asarray(idio_indicator) > 0)) if idio_indicator is not None else 0
    n_idio = max(0, min(n_idio, m - n_factor_states))
    if n_idio > 0:
        ii = np.arange(n_factor_states, n_factor_states + n_idio)
        a_diag, q_diag = estimate_ar1(
            y=EZ[1:, ii],
            x=EZ[:-1, ii],
            V_smooth=V_smooth[1:][:, ii][:, :, ii],
            V_smooth_lag=V_smooth[:-1][:, ii][:, :, ii],
            VVsmooth=VVsmooth[1:][:, ii][:, :, ii],
            dtype=np.float64,
        )
        A_new[ii, ii] = a_diag
        Q_new[ii, ii] = q_diag

    return replace(state, A=A_new, Q=Q_new, V_0=_initial_covariance_update(V_smooth[0]))


def _second_moments(EZ: np.ndarray, V_smooth: np.ndarray) -> np.ndarray:
    """``E[x_t x_t' | Y] = EZ_t EZ_t' + V_smooth_t`` for every row (T x m x m), float64."""
    EZ = np.asarray(EZ, dtype=np.float64)
    return np.asarray(V_smooth, dtype=np.float64) + EZ[:, :, np.newaxis] * EZ[:, np.newaxis, :]


def _update_loading_row(
    j: int,
    cols: np.ndarray,
    C: np.ndarray,
    Y0: np.ndarray,
    W: np.ndarray,
    EZ: np.ndarray,
    EZZ: np.ndarray,
    R_con: Optional[np.ndarray] = None,
    q_con: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Loadings of series ``j`` on the state columns ``cols`` (BM14 C update, one row).

    The other entries of the row (idiosyncratic loadings, lag columns the
    series does not load on) are held fixed and their contribution is
    subtracted::

        den = sum_t w_t E[x_c x_c'],   nom = sum_t w_t (y_t E[x_c] - E[x_c x_o'] C[j, o]')

    with ``w_t`` the observed indicator of ``y_tj``. With a restriction
    ``R_con c = q_con`` (tent aggregation) the restricted least-squares solution
    ``c = c_u - den^-1 R' (R den^-1 R')^-1 (R c_u - q)`` is returned.
    """
    w = W[:, j].astype(np.float64)
    M = np.tensordot(w, EZZ[:, cols, :], axes=(0, 0))  # (len(cols), m)
    den = M[:, cols]
    c_other = C[j].copy()
    c_other[cols] = 0.0
    nom = (Y0[:, j] * w) @ EZ[:, cols] - M @ c_other
    what = f"loadings of series {j}"
    if R_con is None or R_con.shape[0] == 0:
        return _solve_spd(den, nom, what)
    sol = _solve_spd(den, np.column_stack([nom, R_con.T]), what)
    c_u, den_inv_Rt = sol[:, 0], sol[:, 1:]
    lam = _solve_spd(R_con @ den_inv_Rt, R_con @ c_u - q_con, f"tent restriction of series {j}")
    return c_u - den_inv_Rt @ lam


def _update_observation_matrix_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
    block_structure: Optional[BlockStructure] = None
) -> DFMModelState:
    """M-step for C (Banbura-Modugno 2014), one series at a time.

    ``EZ[t]``/``V_smooth[t]`` go with data row ``t`` (all T rows are used).
    A clock-frequency series loads on the current value of the factors of
    the blocks it belongs to; a slower-frequency series loads on the first ``k``
    lag copies of those factors (``k`` = tent length), restricted by
    ``kron(R_mat, I_r) c = kron(q, 1_r)``. All other entries of C (the
    idiosyncratic loadings, the remaining lag columns) are kept fixed and their
    cross-moments are subtracted (``_update_loading_row``). Series with no
    observation keep their row. No normalization: see ``_normalize_C_with_invariance``.
    """
    X = np.asarray(X, dtype=np.float64)
    EZ = np.asarray(EZ, dtype=np.float64)
    T, N = X.shape
    C = np.array(state.C, dtype=np.float64, copy=True)
    blocks = np.asarray(state.blocks)
    _validate_blocks_shape(blocks, N)
    r = np.asarray(state.r).astype(int)
    p_plus_one = int(state.max_lag_size if state.max_lag_size is not None else (state.p + 1))
    offsets = _factor_offsets(r, p_plus_one)
    n_slower = int(state.n_slower_freq or 0)
    n_clock = int(state.n_clock_freq) if state.n_clock_freq is not None else N - n_slower
    R_mat = state.constraint_matrix
    q = state.constraint_vector
    if R_mat is not None:
        R_mat = np.asarray(R_mat, dtype=np.float64)
        k = int(R_mat.shape[1])
        q = np.zeros(R_mat.shape[0]) if q is None else np.asarray(q, dtype=np.float64).ravel()
        if k > p_plus_one:
            raise DataValidationError(
                f"Tent length {k} exceeds the companion width {p_plus_one} of the factor blocks."
            )

    W = ~np.isnan(X)
    Y0 = np.where(W, X, 0.0)
    EZZ = _second_moments(EZ, V_smooth)

    for j in range(N):
        if not W[:, j].any():
            continue
        loaded = [b for b in range(len(r)) if blocks[j, b] > 0]
        if not loaded:
            continue
        if j >= n_clock and R_mat is not None:
            cols = np.concatenate([offsets[b] + np.arange(k * r[b]) for b in loaded])
            R_con = block_diag(*[np.kron(R_mat, np.eye(r[b])) for b in loaded])
            q_con = np.concatenate([np.kron(q, np.ones(r[b])) for b in loaded])
        else:
            cols = np.concatenate([offsets[b] + np.arange(r[b]) for b in loaded])
            R_con = q_con = None
        C[j, cols] = _update_loading_row(j, cols, C, Y0, W, EZ, EZZ, R_con, q_con)

    return replace(state, C=C)


def _observation_noise_floor(state: DFMModelState, N: int, config: EMConfig) -> np.ndarray:
    """Lower bound of each R_ii: ``MIN_OBSERVATION_NOISE`` for series with an idiosyncratic state.

    Clock-frequency series with an AR(1) idiosyncratic state and all
    slower-frequency series (tent chain) have ``R_ii >= MIN_OBSERVATION_NOISE``
    (part of the model, as in BM14); other series ``R_ii >= config.min_variance``.
    """
    floor = np.full(N, float(config.min_variance))
    n_slower = int(state.n_slower_freq or 0)
    n_clock = int(state.n_clock_freq) if state.n_clock_freq is not None else N - n_slower
    if state.idio_indicator is not None:
        idio = np.asarray(state.idio_indicator).ravel()[:N] > 0
        floor[:len(idio)][idio] = MIN_OBSERVATION_NOISE
    floor[n_clock:] = MIN_OBSERVATION_NOISE
    return floor


def _update_observation_noise_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """M-step for the diagonal R (Banbura-Modugno 2014 missing-data formula).

    ``R_ii = [sum_{t obs} ((y_ti - C_i EZ_t)^2 + C_i V_t C_i') + sum_{t missing} R_ii^old] / T``
    over all T rows (``EZ[t]`` goes with row ``t``), using the C of this M-step,
    then floored by ``_observation_noise_floor``. No upper clip.
    """
    X = np.asarray(X, dtype=np.float64)
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    T, N = X.shape
    C = np.asarray(state.C, dtype=np.float64)
    R_old = np.diag(np.asarray(state.R, dtype=np.float64))
    W = ~np.isnan(X)
    resid = np.where(W, np.where(W, X, 0.0) - EZ @ C.T, 0.0)
    CVC = np.einsum('ij,tjk,ik->ti', C, V_smooth, C, optimize=True)
    total = (np.sum(resid ** 2, axis=0) + np.sum(np.where(W, CVC, 0.0), axis=0)
             + np.sum(~W, axis=0) * R_old)
    R_diag = total / T
    if not np.all(np.isfinite(R_diag)):
        raise NumericalError("M-step: the observation noise update is not finite.")
    R_diag = np.maximum(R_diag, _observation_noise_floor(state, N, config))
    return replace(state, R=np.diag(R_diag))


def _update_unstructured(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """EM M-step without block structure: full A, Q, C, diagonal R.

    ``A = S10 S00^-1`` and ``Q = (S11 - A S10') / (T-1)`` on the whole state,
    every row of C regressed on the whole state, R by the BM14 formula.
    """
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    VVsmooth = np.asarray(VVsmooth, dtype=np.float64)
    T, m = EZ.shape
    A, Q = np.array(state.A, dtype=np.float64), np.array(state.Q, dtype=np.float64)
    if T >= 2:
        A, Q = estimate_var(
            y=EZ[1:], x=EZ[:-1], V_smooth=V_smooth[1:], V_smooth_lag=V_smooth[:-1],
            VVsmooth=VVsmooth[1:], dtype=np.float64,
        )
    X = np.asarray(X, dtype=np.float64)
    W = ~np.isnan(X)
    Y0 = np.where(W, X, 0.0)
    EZZ = _second_moments(EZ, V_smooth)
    C = np.array(state.C, dtype=np.float64, copy=True)
    cols = np.arange(m)
    for j in range(X.shape[1]):
        if W[:, j].any():
            C[j] = _update_loading_row(j, cols, C, Y0, W, EZ, EZZ)
    state = replace(state, A=A, Q=Q, C=C, V_0=_initial_covariance_update(V_smooth[0]))
    return _update_observation_noise_blocked(X, EZ, V_smooth, state, config)


def em_step(
    X: np.ndarray,
    state: DFMModelState,
    kalman_filter: Optional[DFMKalmanFilter] = None,
    config: Optional[EMConfig] = None,
    block_structure: Optional[BlockStructure] = None,
    num_iter: int = 0
) -> Tuple[DFMModelState, float, Optional[DFMKalmanFilter]]:
    """Perform one EM step: E-step (Kalman filter/smoother) + M-step (parameter updates).
    
    E-step: DFMKalmanFilter.filter_and_smooth on X with NaN (exact moments and
    loglik at ``state``). M-step uses custom constrained updates
    that preserve block structure, mixed-frequency constraints, and idiosyncratic components.
    If block_structure is provided, uses blocked updates; otherwise uses unconstrained updates.
    
    GUARDRAILS:
    - Time budgets enforced (E-step, M-step, total iteration)
    - Numerical health checks before expensive operations
    - First iteration must be fast (diagnostic check)
    - An M-step that cannot be computed (singular moment matrix) raises
      ``_MStepError`` (a ``NumericalError``) carrying the loglik of ``state``
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    A = state.A
    C = state.C
    Q = state.Q
    R = state.R
    Z_0 = state.Z_0
    V_0 = state.V_0
    if kalman_filter is None:
        kalman_filter = DFMKalmanFilter(
            transition_matrices=A, observation_matrices=C,
            transition_covariance=Q, observation_covariance=R,
            initial_state_mean=Z_0, initial_state_covariance=V_0
        )
    else:
        kalman_filter.update_parameters(A, C, Q, R, Z_0, V_0)
    
    if block_structure is None and state.blocks is not None:
        block_structure = BlockStructure(
            blocks=state.blocks,
            r=state.r,
            p=state.p,
            p_plus_one=state.max_lag_size if state.max_lag_size is not None else (state.p + 1),
            n_clock_freq=state.n_clock_freq if state.n_clock_freq is not None else (X.shape[1] - state.n_slower_freq),
            idio_indicator=state.idio_indicator,
            R_mat=state.constraint_matrix,
            q=state.constraint_vector,
            n_slower_freq=state.n_slower_freq,
            tent_weights_dict=state.tent_weights_dict
        )
    
    # E-step: Kalman filter and Durbin-Koopman smoother on X with NaN (per-element missing data)
    import time as time_module
    e_step_start = time_module.time()

    verbose_iterations = num_iter < 5
    if verbose_iterations:
        T, m, N = X.shape[0], A.shape[0], X.shape[1]
        _logger.info(f"    E-step: Running Kalman filter + smoother (T={T}, N={N}, m={m})...")

    EZ, V_smooth, VVsmooth, loglik = kalman_filter.filter_and_smooth(X)
    kalman_filter._cached_smoothed_factors = EZ.copy()

    e_step_time = time_module.time() - e_step_start
    
    # GUARDRAIL: Time budget check for E-step
    # Time-budget invariants (non-negotiable)
    # Single constraint: 10 minutes for all datasets
    max_e_step_time = MAX_FIRST_ITERATION_SECONDS if num_iter == 0 else MAX_ESTEP_SECONDS
    
    if e_step_time > max_e_step_time:
        raise NumericalError(
            f"E-step exceeded time budget ({e_step_time:.2f}s > {max_e_step_time:.2f}s). "
            f"Likely numerical instability or missing preprocessing. "
            f"Iteration {num_iter + 1}, T={X.shape[0]}, m={EZ.shape[1] if EZ.shape else 0}.",
            details=f"E-step time: {e_step_time:.2f}s, budget: {max_e_step_time:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )
    
    _logger.info(f"    E-step: Completed in {e_step_time:.1f}s, log-likelihood={loglik:.2e}")
    
    m_step_start = time_module.time()
    
    if block_structure is not None and block_structure.is_valid():
        if not block_structure.has_cached_indices():
            N = X.shape[1]
            _compute_and_cache_block_indices(block_structure, N)
        n_blocks = len(block_structure.r) if block_structure.r is not None else 0
    else:
        n_blocks = 0
    
    if verbose_iterations:
        if block_structure is not None and block_structure.is_valid():
            _logger.info(f"    M-step: Updating parameters (block structure: {n_blocks} blocks)...")
        else:
            _logger.info("    M-step: Updating parameters (unconstrained)...")
    
    # M-step: closed-form conditional maximizers of the expected complete-data
    # log-likelihood (BM14), each from the moments of this E-step. A failure
    # (singular moment matrix, non-finite update) is reported with the loglik of
    # the parameters this iteration started from.
    try:
        if block_structure is not None and block_structure.is_valid():
            if verbose_iterations:
                _logger.info("      → Updating transition matrix A and process noise Q...")
            state = _update_transition_matrix_blocked(EZ, V_smooth, VVsmooth, state, config)
            if verbose_iterations:
                _logger.info("      → Updating observation matrix C...")
            state = _update_observation_matrix_blocked(X, EZ, V_smooth, state, config, block_structure=block_structure)
            if verbose_iterations:
                _logger.info("      → Updating observation noise R...")
            state = _update_observation_noise_blocked(X, EZ, V_smooth, state, config)
        else:
            state = _update_unstructured(X, EZ, V_smooth, VVsmooth, state, config)
    except (NumericalError, np.linalg.LinAlgError) as e:
        raise _MStepError(f"M-step failed at iteration {num_iter + 1}: {e}", loglik=loglik) from e

    # Initial state: Z_0 = E[x_0 | Y], V_0 = Cov(x_0 | Y) (the maximizers; V_0 projected to PSD).
    state = replace(state, Z_0=np.array(EZ[0], dtype=np.float64), V_0=_initial_covariance_update(V_smooth[0]))
    
    if config.ar_clip is not None:
        A_new, clip_stats = apply_ar_clipping(state.A, config)
        if clip_stats['n_clipped'] > 0:
            _logger.debug(f"AR clipping: {clip_stats['n_clipped']}/{clip_stats['n_total']} coefficients clipped")
        state = replace(state, A=A_new)
    
    m_step_time = time_module.time() - m_step_start
    
    # GUARDRAIL: Time budget check for M-step
    # Single constraint: 10 minutes for all datasets
    max_m_step_time = MAX_MSTEP_SECONDS
    
    if m_step_time > max_m_step_time:
        raise NumericalError(
            f"M-step exceeded time budget ({m_step_time:.2f}s > {max_m_step_time:.2f}s). "
            f"Likely numerical instability or excessive fallbacks. "
            f"Iteration {num_iter + 1}.",
            details=f"M-step time: {m_step_time:.2f}s, budget: {max_m_step_time:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )
    
    # GUARDRAIL: Total iteration time check
    # Single constraint: 20 minutes for all datasets
    total_iter_time = e_step_time + m_step_time
    max_total_time = MAX_EM_ITERATION_SECONDS
    
    if total_iter_time > max_total_time:
        raise NumericalError(
            f"EM iteration exceeded total time budget ({total_iter_time:.2f}s > {max_total_time:.2f}s). "
            f"Likely numerical instability. Iteration {num_iter + 1}.",
            details=f"Total iteration time: {total_iter_time:.2f}s, budget: {max_total_time:.2f}s. "
                   f"E-step: {e_step_time:.2f}s, M-step: {m_step_time:.2f}s."
        )
    
    if num_iter < 5:
        A_max = np.max(np.abs(state.A)) if state.A is not None and np.isfinite(state.A).all() else np.nan
        C_max = np.max(np.abs(state.C)) if state.C is not None and np.isfinite(state.C).all() else np.nan
        Q_max_elem = np.max(np.abs(state.Q)) if state.Q is not None and np.isfinite(state.Q).all() else np.nan
        try:
            if state.Q is not None and np.isfinite(state.Q).all() and state.Q.size > 0:
                Q_eigenvals = np.linalg.eigvalsh(state.Q)
                Q_max_eigval = np.max(np.abs(Q_eigenvals))
            else:
                Q_max_eigval = np.nan
        except (np.linalg.LinAlgError, ValueError):
            Q_max_eigval = np.nan
        R_diag_max = np.max(np.abs(np.diag(state.R))) if state.R is not None and np.isfinite(state.R).all() else np.nan
        _logger.info(f"    M-step: Completed in {m_step_time:.1f}s | "
                    f"Max values: |A|={A_max:.3f}, |C|={C_max:.3f}, |Q|={Q_max_elem:.3f} (max_eig={Q_max_eigval:.3f}), |R_diag|={R_diag_max:.3f}")
    
    total_time = e_step_time + m_step_time
    should_log_timing = (num_iter < 3) or (num_iter % 5 == 0) or (total_time > 30.0)
    
    if should_log_timing:
        T, m_dim = X.shape[0], EZ.shape[1] if EZ.shape else 0
        e_step_pct = 100*e_step_time/total_time if total_time > 0 else 0
        m_step_pct = 100*m_step_time/total_time if total_time > 0 else 0
        _logger.info(f"  Iteration {num_iter + 1} timing: E-step={e_step_time:.2f}s ({e_step_pct:.1f}%), "
                    f"M-step={m_step_time:.2f}s ({m_step_pct:.1f}%), "
                    f"Total={total_time:.2f}s (T={T}, m={m_dim})")
    
    return state, loglik, kalman_filter


def run_em_algorithm(
    X: np.ndarray,
    initial_state: DFMModelState,
    max_iter: int = 200,
    threshold: float = 1e-4,
    config: Optional[EMConfig] = None,
    checkpoint_callback: Optional[Callable[[int, DFMModelState], None]] = None
) -> Tuple[DFMModelState, Dict[str, Any]]:
    """Run EM algorithm until convergence.
    
    Iterates E-step (Kalman filter/smoother) and M-step (parameter updates) until
    convergence (relative log-likelihood change < threshold) or max_iter reached.
    Returns final state and training metadata (loglik, num_iter, converged, etc.).
    
    The E-step is exact (no covariance regularization). If it fails after the
    first iteration (non-finite filter at the latest parameters), EM stops and
    returns the parameters of the last E-step that succeeded, with its loglik.

    If the M-step cannot be computed (singular moment matrix) after the first
    iteration, EM stops and returns the parameters that iteration started from,
    with their loglik.

    The M-step is the closed-form Banbura-Modugno (2014) update (with the tent
    restriction for slower-frequency loadings); with the exact E-step every
    block is a conditional maximizer of the expected complete-data
    log-likelihood, so the exact log-likelihood of the iterates does not
    decrease. The opt-in ``config.damping_factor`` and ``config.ar_clip`` break
    that property. When ``config.normalize_C`` is True the returned parameters
    are rescaled once, after the last iteration, to unit-norm factor loading
    columns (``_normalize_C_with_invariance``; same likelihood).
    """

    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    if config.data_clip is not None and config.data_clip > 0:
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)  # Avoid division by zero
        X_clipped = np.clip(X, X_mean - config.data_clip * X_std, X_mean + config.data_clip * X_std)
        n_clipped = np.sum((X != X_clipped) & np.isfinite(X))
        if n_clipped > 0:
            _logger.warning(f"Data clipping applied: {n_clipped}/{X.size} values clipped to ±{config.data_clip} std devs")
        X = X_clipped
    
    state = initial_state
    previous_state = initial_state
    
    kalman_filter = DFMKalmanFilter(
        transition_matrices=state.A,
        observation_matrices=state.C,
        transition_covariance=state.Q,
        observation_covariance=state.R,
        initial_state_mean=state.Z_0,
        initial_state_covariance=state.V_0
    )
    
    block_structure = None
    if (state.blocks is not None and state.r is not None and state.p is not None 
        and state.max_lag_size is not None and state.n_clock_freq is not None 
        and state.idio_indicator is not None):
        block_structure = BlockStructure(
            blocks=state.blocks,
            r=state.r,
            p=state.p,
            p_plus_one=state.max_lag_size,
            n_clock_freq=state.n_clock_freq,
            idio_indicator=state.idio_indicator,
            R_mat=state.constraint_matrix,
            q=state.constraint_vector,
            n_slower_freq=state.n_slower_freq,
            tent_weights_dict=state.tent_weights_dict
        )
    
    previous_loglik = float('-inf')
    # Input parameters and loglik of the last iteration whose E-step succeeded.
    previous_input_state = initial_state
    previous_input_loglik = float('-inf')
    num_iter = 0
    converged = False
    loglik = float('-inf')
    change = DEFAULT_ZERO_VALUE
    
    # Track damping state for proper convergence checking
    damping_applied = False
    loglik_undamped = float('-inf')
    
    em_start_time = time_module.time()
    iteration_times = []
    
    _logger.info(f"Starting EM algorithm: max_iter={max_iter}, threshold={threshold:.2e}")
    if block_structure is not None:
        total_factors = int(np.sum(state.r)) if state.r is not None else 0
        _logger.info(f"  Block structure: {state.blocks.shape[1] if state.blocks is not None else 0} blocks, {total_factors} factors")
    if state.constraint_matrix is not None:
        _logger.info(f"  Mixed-frequency: tent kernel constraints enabled (R_mat shape: {state.constraint_matrix.shape})")
    
    while num_iter < max_iter and not converged:
        iter_start_time = time_module.time()
        # Heartbeat every 10 iterations
        if num_iter % 10 == 0:
            _logger.info(f"EM heartbeat: iteration {num_iter}/{max_iter}")
        
        if num_iter < 3:
            _logger.info(f"Starting iteration {num_iter + 1}/{max_iter}...")
        
        # Parameters this iteration starts from (and whose E-step it runs).
        # Returned instead of the update if the M-step output is not usable.
        last_finite_state = state

        try:
            state, loglik, kalman_filter_updated = em_step(
                X, state, kalman_filter=kalman_filter, config=config,
                block_structure=block_structure, num_iter=num_iter
            )
        except _MStepError as e:
            if num_iter == 0:
                _logger.error(f"EM step failed at iteration 1: {e}", exc_info=True)
                raise RuntimeError(f"EM algorithm failed at iteration 1: {e}") from e
            # The E-step at this iteration's parameters succeeded but no M-step
            # update could be computed: return these parameters and their loglik.
            _logger.error(f"EM: {e}; stopping and returning the parameters of iteration {num_iter + 1}.")
            state = last_finite_state
            loglik = e.loglik
            break
        except NumericalError as e:
            if num_iter == 0:
                _logger.error(f"EM step failed at iteration 1: {e}", exc_info=True)
                raise RuntimeError(f"EM algorithm failed at iteration 1: {e}") from e
            # The Kalman filter is not finite (or an innovation covariance is not
            # positive definite) at the parameters of the previous M-step: keep the
            # parameters of the last E-step that succeeded and its loglik.
            _logger.error(
                f"EM: the E-step failed at iteration {num_iter + 1} ({e}); stopping and returning "
                f"the parameters of iteration {num_iter}."
            )
            state = previous_input_state
            loglik = previous_input_loglik
            break
        except Exception as e:
            _logger.error(f"EM step failed at iteration {num_iter + 1}: {e}", exc_info=True)
            _logger.error(f"  Current parameters shapes - A: {state.A.shape if state.A is not None else None}, C: {state.C.shape if state.C is not None else None}, Q: {state.Q.shape if state.Q is not None else None}, R: {state.R.shape if state.R is not None else None}")
            _logger.error(f"  Data shape: {X.shape}, Block structure: {block_structure is not None}")
            raise RuntimeError(f"EM algorithm failed at iteration {num_iter + 1}: {e}") from e
        # The E-step of this iteration succeeded: its input parameters and loglik
        # are the fallback if a later E-step fails.
        previous_input_state = last_finite_state
        previous_input_loglik = loglik

        # Track undamped loglik for convergence checks
        loglik_undamped = loglik
        damping_applied = False

        if config.damping_factor is not None and 0 < config.damping_factor <= 1:
            alpha = config.damping_factor
            A_damped = alpha * state.A + (1 - alpha) * previous_state.A if previous_state.A is not None else state.A
            C_damped = alpha * state.C + (1 - alpha) * previous_state.C if previous_state.C is not None else state.C
            Q_damped = alpha * state.Q + (1 - alpha) * previous_state.Q if previous_state.Q is not None else state.Q
            R_damped = alpha * state.R + (1 - alpha) * previous_state.R if previous_state.R is not None else state.R
            Z_0_damped = alpha * state.Z_0 + (1 - alpha) * previous_state.Z_0 if previous_state.Z_0 is not None else state.Z_0
            V_0_damped = alpha * state.V_0 + (1 - alpha) * previous_state.V_0 if previous_state.V_0 is not None else state.V_0
            state = replace(state, A=A_damped, C=C_damped, Q=Q_damped, R=R_damped, Z_0=Z_0_damped, V_0=V_0_damped)
            damping_applied = True
            if num_iter < 3:
                _logger.info(f"  Applied damping: {alpha*100:.0f}% new, {(1-alpha)*100:.0f}% old")
                _logger.warning("Damping breaks the EM monotonicity guarantee.")

        previous_state = state

        params_finite = all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                            for p in [state.A, state.C, state.Q, state.R, state.Z_0, state.V_0])
        if not (params_finite and np.isfinite(loglik)):
            _logger.error(f"EM: NaN/Inf detected at iteration {num_iter + 1}, stopping")
            _logger.error(f"  Parameter shapes - A: {state.A.shape if state.A is not None else None}, C: {state.C.shape if state.C is not None else None}, Q: {state.Q.shape if state.Q is not None else None}, R: {state.R.shape if state.R is not None else None}")
            _logger.error(f"  Loglik: {loglik}, isfinite: {np.isfinite(loglik) if isinstance(loglik, (int, float, np.number)) else 'N/A'}")
            if not params_finite:
                # Only revert when the parameters themselves diverged; a non-finite
                # loglik with finite parameters keeps the M-step update.
                _logger.error("  Keeping parameters from the last finite iteration.")
                state = last_finite_state
            break
        
        if num_iter == 0 and loglik < -1e10:
            _logger.warning(f"Extremely negative initial log-likelihood: {loglik:.2e}. "
                          f"This may indicate numerical instability or data scaling issues.")
        
        kalman_filter.update_parameters(state.A, state.C, state.Q, state.R, state.Z_0, state.V_0)
        
        # Never propagate cached factors across EM steps
        # Cache should be per-parameter snapshot only, not shared across iterations
        # Damping and parameter updates invalidate caches, so propagating stale cache
        # would reintroduce incorrect state silently
        # Cache is set within em_step() after E-step completes, and invalidated on parameter changes
        # Do NOT copy cache from kalman_filter_updated as it may reflect pre-damped parameters
        
        min_iterations = get_config_attr(config, 'min_iterations_for_convergence_check', 1)
        
        # Use undamped loglik for convergence checks
        # Damping breaks monotonicity, so damped loglik is not a valid convergence metric
        loglik_for_convergence = loglik_undamped if damping_applied else loglik
        
        # Convergence criterion that handles zero/near-zero loglik
        if num_iter >= min_iterations:
            # Use absolute change if loglik is near zero to avoid division issues
            if (previous_loglik != float('-inf') and np.isfinite(previous_loglik) and 
                abs(previous_loglik) > 1e-10 and abs(loglik_for_convergence) > 1e-10):
                # Both are non-zero and finite - use relative change
                change = abs((loglik_for_convergence - previous_loglik) / previous_loglik)
            else:
                # Use absolute change when near zero or infinite
                change = abs(loglik_for_convergence - previous_loglik) if (np.isfinite(loglik_for_convergence) and np.isfinite(previous_loglik)) else float('inf')
            
            converged = change < threshold
        else:
            change = abs(loglik_for_convergence - previous_loglik) if previous_loglik != float('-inf') and np.isfinite(loglik_for_convergence) and np.isfinite(previous_loglik) else DEFAULT_ZERO_VALUE
        
        # Track undamped loglik for next iteration's convergence check
        previous_loglik = loglik_for_convergence
        num_iter += 1
        
        iter_time = time_module.time() - iter_start_time
        iteration_times.append(iter_time)
        if len(iteration_times) > 5:
            iteration_times.pop(0)
        avg_iter_time = sum(iteration_times) / len(iteration_times)
        
        elapsed_time = time_module.time() - em_start_time
        progress_pct = (num_iter / max_iter) * 100
        remaining_iters = max_iter - num_iter if not converged else 0
        estimated_remaining = avg_iter_time * remaining_iters if remaining_iters > 0 else 0
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes}m"
        
        progress_interval = get_config_attr(config, 'progress_log_interval', 1)
        should_log = (num_iter % progress_interval == 0) or converged
        
        if should_log:
            progress_msg = f"[{progress_pct:.1f}%]"
            if elapsed_time > 0:
                elapsed_str = format_time(elapsed_time)
                if remaining_iters > 0 and len(iteration_times) >= 2:
                    remaining_str = format_time(estimated_remaining)
                    progress_msg += f" Elapsed: {elapsed_str}, Est. remaining: {remaining_str}"
                else:
                    progress_msg += f" Elapsed: {elapsed_str}"
            
            # Log both damped and undamped loglik if damping was applied
            loglik_to_log = loglik
            if damping_applied and loglik_undamped != loglik:
                _logger.debug(f"  Iteration {num_iter}: Undamped loglik={loglik_undamped:.2e}, Damped loglik={loglik:.2e}")
            
            log_em_iteration(
                iteration=num_iter,
                loglik=loglik_to_log,
                delta=change if change > 0 else None,
                max_iter=max_iter,
                converged="✓" if converged else ""
            )
            
            if avg_iter_time > 0:
                _logger.info(f"  Progress: {progress_msg}, Iteration time: {iter_time:.1f}s (avg: {avg_iter_time:.1f}s)")
        
        if checkpoint_callback is not None and num_iter % 5 == 0 and num_iter > 0:
            try:
                _logger.info(f"Saving checkpoint at iteration {num_iter}...")
                checkpoint_callback(num_iter, state)
            except Exception as e:
                _logger.warning(f"Checkpoint callback failed at iteration {num_iter}: {e}", exc_info=True)
    
    # Use final undamped loglik for convergence reporting
    # If damping was never applied, loglik_undamped will still be -inf, so use loglik
    final_loglik = loglik_undamped if (damping_applied and np.isfinite(loglik_undamped)) else loglik
    
    log_convergence(
        converged=converged,
        num_iter=num_iter,
        final_loglik=final_loglik if np.isfinite(final_loglik) else None,
        reason="converged" if converged else f"max_iterations_reached (change: {change:.2e})",
        model_type="dfm"
    )
    
    # One likelihood-invariant change of factor units (never inside the loop).
    if config.normalize_C and state.C is not None and state.r is not None:
        _, state = _normalize_C_with_invariance(state.C, state, config, log_prefix="EM: ")

    # Smoothed states at the returned parameters.
    try:
        kalman_filter.update_parameters(state.A, state.C, state.Q, state.R, state.Z_0, state.V_0)
        EZ_final, _, _, _ = kalman_filter.filter_and_smooth(X)
        kalman_filter._cached_smoothed_factors = EZ_final
    except (NumericalError, ValueError) as e:
        _logger.warning(
            f"Final filter_and_smooth() failed at the returned parameters ({type(e).__name__}: {e}); "
            f"no smoothed factors are cached."
        )
        kalman_filter._cached_smoothed_factors = None

    metadata = {
        'loglik': loglik,
        'num_iter': num_iter,
        'converged': converged,
        'change': change,
        'smoothed_factors': kalman_filter._cached_smoothed_factors
    }
    
    return state, metadata


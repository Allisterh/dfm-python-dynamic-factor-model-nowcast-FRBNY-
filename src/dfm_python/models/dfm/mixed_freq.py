"""Mixed-frequency parameters of the DFM engine: tent weights, constraints and the slower-frequency chains."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ...config.constants import (
    DEFAULT_HIERARCHY_VALUE,
    FREQUENCY_HIERARCHY,
    DEFAULT_IDENTITY_SCALE,
    MAX_TENT_SIZE,
)
from ...numeric.stability import create_scaled_identity
from ...functional.state_layout import bm14_idio_indicator
from ...utils.errors import ConfigurationError, DataValidationError
from ...logger import get_logger
from .tent import get_agg_structure

_logger = get_logger(__name__)


def find_slower_frequency(
    clock: str,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
) -> Optional[str]:
    """The one frequency slower than ``clock``: from ``tent_weights_dict``, else the next in the hierarchy.

    The DFM supports a single slower frequency; more than one key of
    ``tent_weights_dict`` other than ``clock`` raises ``ConfigurationError``.
    Without tent weights the result is a best-effort guess, None if nothing
    in ``FREQUENCY_HIERARCHY`` is slower.
    """
    if tent_weights_dict:
        slower_freqs_in_dict = [freq for freq in tent_weights_dict.keys() if freq != clock]
        if len(slower_freqs_in_dict) > 1:
            raise ConfigurationError(
                f"Multiple slower frequencies found in tent_weights_dict: {slower_freqs_in_dict}. "
                f"Only one slower frequency is supported. Clock frequency: {clock}",
                details=f"Found slower frequencies: {slower_freqs_in_dict}, clock: {clock}"
            )
        if slower_freqs_in_dict:
            return slower_freqs_in_dict[0]

    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    slower_freqs = sorted(
        [freq for freq in FREQUENCY_HIERARCHY if FREQUENCY_HIERARCHY[freq] > clock_hierarchy],
        key=lambda f: FREQUENCY_HIERARCHY[f]
    )
    return slower_freqs[0] if slower_freqs else None


def setup_mixed_frequency_params(
    config: Any,
    clock: str,
    columns: Optional[List[str]],
    N: int
) -> Dict[str, Any]:
    """Mixed-frequency parameters of the engine from its config.

    Parameters
    ----------
    config : _DFMEngineConfig
        Gives ``get_frequencies(columns=...)`` and ``tent_weights``.
    clock : str
        Clock frequency.
    columns : list of str, optional
        Column names of the data; None means no per-series frequencies.
    N : int
        Number of series.

    Returns
    -------
    dict
        ``R_mat``, ``q`` (the tent constraint, None without a slower
        frequency), ``n_slower_freq``, ``n_clock_freq``, ``tent_weights_dict``,
        ``frequencies_np`` (per-series frequencies, None without them),
        ``idio_indicator`` (BM14 layout: 1 for clock series, 0 for slower),
        ``mixed_freq``, ``tent_kernel_size`` (1 without a kernel).

    Raises
    ------
    ConfigurationError
        For a clock or slower frequency outside ``FREQUENCY_HIERARCHY``, more
        than one slower frequency, or a tent kernel that does not match
        ``R_mat``.
    DataValidationError
        When the frequencies do not match ``N``, the columns are not ordered
        ``[clock series | slower series]`` (the order the engine assumes for
        block residualization, the tent constraint and the idiosyncratic
        split), or the kernel is longer than ``MAX_TENT_SIZE``.
    """
    frequencies_list = config.get_frequencies(columns=columns) if columns is not None else []
    if frequencies_list and len(frequencies_list) != N:
        raise DataValidationError(
            f"Frequency list length ({len(frequencies_list)}) does not match number of series (N={N}). "
            f"This indicates a mismatch between config frequencies and data columns.",
            details=f"frequencies_list length: {len(frequencies_list)}, N: {N}, columns: {columns[:5] if columns else None}..."
        )

    agg_structure = get_agg_structure(config, clock=clock)
    structures = agg_structure['structures']
    tent_weights_dict = agg_structure['tent_weights']
    if structures:
        (R_mat, q) = next(iter(structures.values()))
        mixed_freq = True
    else:
        R_mat = None
        q = None
        mixed_freq = False

    if frequencies_list:
        if clock not in FREQUENCY_HIERARCHY:
            raise ConfigurationError(
                f"Clock frequency '{clock}' is not in FREQUENCY_HIERARCHY. "
                f"Valid frequencies: {list(FREQUENCY_HIERARCHY.keys())}",
                details=f"Clock frequency: {clock}"
            )
        clock_hierarchy = FREQUENCY_HIERARCHY[clock]
        slower_freqs = [
            freq for freq in frequencies_list
            if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy
        ]
        unique_slower_freqs = list(set(slower_freqs))
        if len(unique_slower_freqs) > 1:
            raise ConfigurationError(
                f"Only one slower frequency is allowed, but found {len(unique_slower_freqs)}: {unique_slower_freqs}. "
                f"DFM currently supports one clock frequency and one additional slower frequency only.",
                details=f"Slower frequencies found: {unique_slower_freqs}, clock: {clock}"
            )
        for freq in unique_slower_freqs:
            if freq not in FREQUENCY_HIERARCHY:
                raise ConfigurationError(
                    f"Slower frequency '{freq}' is not in FREQUENCY_HIERARCHY. "
                    f"Valid frequencies: {list(FREQUENCY_HIERARCHY.keys())}",
                    details=f"Invalid slower frequency: {freq}, clock: {clock}"
                )
        n_slower_freq = len(slower_freqs)
    else:
        n_slower_freq = 0
    n_clock_freq = N - n_slower_freq

    if frequencies_list and n_slower_freq > 0:
        clock_freqs_in_list = [
            freq for freq in frequencies_list
            if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) <= clock_hierarchy
        ]
        last_n_freqs = frequencies_list[-n_slower_freq:]
        if last_n_freqs != slower_freqs:
            _logger.error(
                f"COLUMN ORDERING VIOLATION: Mixed-frequency model requires [clock series | slower series] ordering. "
                f"Clock: {clock} (hierarchy: {clock_hierarchy}), "
                f"Expected: {len(clock_freqs_in_list)} clock + {len(slower_freqs)} slower. "
                f"Found: last {n_slower_freq} columns = {last_n_freqs}, slower freqs = {slower_freqs}"
            )
            raise DataValidationError(
                f"Column ordering violation: Mixed-frequency model requires [clock series | slower series] ordering. "
                f"Last {n_slower_freq} columns have frequencies {last_n_freqs}, "
                f"but slower frequencies are {slower_freqs}. "
                f"Please reorder columns so all slower-frequency series are at the end.",
                details=(
                    f"Clock frequency: {clock}, Clock hierarchy: {clock_hierarchy}. "
                    f"Expected ordering: {len(clock_freqs_in_list)} clock-frequency series, "
                    f"then {len(slower_freqs)} slower-frequency series. "
                    f"Found: {frequencies_list}"
                )
            )
        _logger.debug(
            f"Column ordering validated: {len(clock_freqs_in_list)} clock-frequency series, "
            f"{len(slower_freqs)} slower-frequency series (all at end)"
        )

    frequencies_np = np.array(frequencies_list) if frequencies_list else None
    # BM14 layout: an AR(1) idiosyncratic state per clock-frequency series; a
    # slower-frequency series carries only its tent chain (functional.state_layout).
    idio_indicator = bm14_idio_indicator(n_clock_freq, n_slower_freq)

    if R_mat is not None:
        tent_kernel_size = R_mat.shape[1]
        if tent_weights_dict:
            weights_size = len(next(iter(tent_weights_dict.values())))
            if weights_size != tent_kernel_size:
                raise ConfigurationError(
                    f"Tent kernel size mismatch: R_mat.shape[1]={tent_kernel_size} but "
                    f"tent_weights_dict has weights of length {weights_size}. "
                    f"These must match for consistent model specification.",
                    details=f"R_mat shape: {R_mat.shape}, tent_weights sizes: {[len(w) for w in tent_weights_dict.values()]}"
                )
    elif tent_weights_dict:
        tent_kernel_size = len(next(iter(tent_weights_dict.values())))
    else:
        tent_kernel_size = 1
    if tent_kernel_size > MAX_TENT_SIZE:
        raise DataValidationError(
            f"Tent kernel size {tent_kernel_size} exceeds maximum allowed size {MAX_TENT_SIZE}. "
            f"For frequency gaps larger than {MAX_TENT_SIZE}, use the missing data approach instead.",
            details=f"Tent kernel size: {tent_kernel_size}, MAX_TENT_SIZE: {MAX_TENT_SIZE}"
        )

    return {
        'R_mat': R_mat,
        'q': q,
        'n_slower_freq': n_slower_freq,
        'n_clock_freq': n_clock_freq,
        'tent_weights_dict': tent_weights_dict,
        'frequencies_np': frequencies_np,
        'idio_indicator': idio_indicator,
        'mixed_freq': mixed_freq,
        'tent_kernel_size': tent_kernel_size
    }


def build_slower_freq_idiosyncratic_chain(
    n_slower_freq: int,
    chain_size: int,
    rho0: float,
    sig_e: np.ndarray,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transition, process noise and initial covariance of the slower-frequency tent chains.

    Each chain (``chain_size`` states) has an AR(1) head with coefficient
    ``rho0`` and stationary variance ``sig_e[j]`` (FRBNY: the residual
    variance of the series after the factors divided by
    ``DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR``), so the head's innovation
    variance is ``(1 - rho0^2) * sig_e[j]``; the other states are shifts
    with zero process noise.

    Returns
    -------
    BQ, SQ, initViQ : np.ndarray
        ``(chain_size * n_slower_freq,) ** 2`` each: transition, process noise
        (nonzero only on the heads) and the exact stationary covariance
        ``V = BQ V BQ' + SQ``. Empty ``(0, 0)`` when ``n_slower_freq == 0``.
    """
    if n_slower_freq == 0:
        return (
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype)
        )

    temp = np.zeros((chain_size, chain_size), dtype=dtype)
    temp[0, 0] = 1.0
    innovation_var = (1 - rho0 ** 2) * np.asarray(sig_e, dtype=dtype)
    SQ = np.kron(np.diag(innovation_var), temp)

    BQ_block = np.zeros((chain_size, chain_size), dtype=dtype)
    BQ_block[0, 0] = rho0
    BQ_block[1:, :chain_size-1] = create_scaled_identity(chain_size-1, DEFAULT_IDENTITY_SCALE, dtype=dtype)
    BQ = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), BQ_block)

    from .initialization import initial_state_covariance  # initialization imports this module
    initViQ = initial_state_covariance(BQ, SQ).astype(dtype)
    return BQ, SQ, initViQ

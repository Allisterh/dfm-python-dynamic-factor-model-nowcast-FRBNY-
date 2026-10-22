"""``BlockStructure``: the block-related parameters the DFM EM algorithm needs."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class BlockStructure:
    """Block structure of a DFM.

    Attributes
    ----------
    blocks : np.ndarray
        Block membership (N x n_blocks), 1 if the series loads on the block.
    r : np.ndarray
        Factors per block (n_blocks,).
    p : int
        VAR lag order.
    p_plus_one : int
        State dimension per factor: ``max(p + 1, tent_kernel_size)`` (called
        ``max_lag_size`` in the model code).
    n_clock_freq : int
        Number of clock-frequency series.
    idio_indicator : np.ndarray
        Idiosyncratic component indicator (N,).
    R_mat, q : np.ndarray, optional
        Tent kernel constraint matrix and vector for mixed-frequency data.
    n_slower_freq : int, optional
        Number of slower-frequency series.
    tent_weights_dict : dict, optional
        Tent weights by frequency pair.

    The ``_cached_*`` fields, ``has_cached_indices`` and ``clear_cache`` are
    an index cache slot that nothing in the package fills or reads.
    """
    blocks: np.ndarray
    r: np.ndarray
    p: int
    p_plus_one: int
    n_clock_freq: int
    idio_indicator: np.ndarray
    R_mat: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    n_slower_freq: Optional[int] = None
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None

    _cached_unique_blocks: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_unique_indices: Optional[List[int]] = field(default=None, init=False, repr=False)
    _cached_bl_idxM: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_bl_idxQ: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_R_con: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_q_con: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_total_factor_dim: Optional[int] = field(default=None, init=False, repr=False)
    _cached_idio_indicator_M: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_n_idio_M: Optional[int] = field(default=None, init=False, repr=False)
    _cached_c_idio_indicator: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_rp1: Optional[int] = field(default=None, init=False, repr=False)
    _cached_N: Optional[int] = field(default=None, init=False, repr=False)

    def is_valid(self) -> bool:
        """True when every required field is set."""
        return (
            self.blocks is not None
            and self.r is not None
            and self.p is not None
            and self.p_plus_one is not None
            and self.n_clock_freq is not None
            and self.idio_indicator is not None
        )

    def has_cached_indices(self) -> bool:
        return self._cached_unique_blocks is not None

    def clear_cache(self) -> None:
        self._cached_unique_blocks = None
        self._cached_unique_indices = None
        self._cached_bl_idxM = None
        self._cached_bl_idxQ = None
        self._cached_R_con = None
        self._cached_q_con = None
        self._cached_total_factor_dim = None
        self._cached_idio_indicator_M = None
        self._cached_n_idio_M = None
        self._cached_c_idio_indicator = None
        self._cached_rp1 = None
        self._cached_N = None

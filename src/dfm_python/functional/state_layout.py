"""State layout of the linear DFM (Banbura-Modugno 2014).

The state vector of the DFM is, in this order::

    [ factor block 1 | ... | factor block B | clock idio AR(1) | slower idio chains ]

- Factor block ``b`` holds ``r_b`` factors in companion form, ``L`` copies
  (current value and ``L - 1`` lags), so it is ``r_b * L`` wide; ``L`` is
  ``max_lag_size = max(p + 1, k)``. Only the first ``r_b`` rows of ``A`` (the
  VAR(p) on the first ``r_b * p`` columns) and the ``r_b x r_b`` block of ``Q``
  are free; the other rows are the identity shift with zero process noise.
- Every clock-frequency series has one AR(1) idiosyncratic state, loaded with
  weight 1 (``idio_indicator`` is 1 for clock-frequency series).
- Every slower-frequency series has an idiosyncratic chain of ``k`` states
  (``k`` = tent length), loaded with the (unnormalized) tent weights, and no
  AR(1) state (``idio_indicator`` is 0 for slower-frequency series). Only the
  first state of a chain (its head) is an AR(1) with estimated coefficient and
  innovation variance; the other ``k - 1`` states are pure shifts (``A`` row =
  identity shift, zero process noise), as in FRBNY's code.

Slower-frequency series are the last ``n_slower`` columns of the data
(``[clock series | slower series]``). :class:`DFMStateLayout` gives every
index the initialization and the EM M-step need, so neither recomputes offsets
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["DFMStateLayout", "bm14_idio_indicator"]


def bm14_idio_indicator(n_clock: int, n_slower: int) -> np.ndarray:
    """BM14 ``idio_indicator``: 1 for clock-frequency series, 0 for slower ones."""
    return np.concatenate([np.ones(int(n_clock), dtype=np.int32), np.zeros(int(n_slower), dtype=np.int32)])


@dataclass(frozen=True)
class DFMStateLayout:
    """Index map of the DFM state vector (see the module docstring).

    Parameters
    ----------
    r : sequence of int
        Factors per block.
    p : int
        VAR order of the factors.
    L : int
        Companion width per factor (``max_lag_size``).
    n_series : int
        Number of series ``N``.
    n_clock : int
        Number of clock-frequency series (the first ``n_clock`` columns).
    idio_series : sequence of int
        Series with an AR(1) idiosyncratic state, in state order.
    chain_length : int
        Length ``k`` of each slower-frequency idiosyncratic chain (0 if none).
    """

    r: tuple
    p: int
    L: int
    n_series: int
    n_clock: int
    idio_series: tuple
    chain_length: int

    @classmethod
    def build(cls, r: Sequence[int], p: int, L: int, n_series: int, n_clock: int,
              chain_length: int) -> DFMStateLayout:
        """BM14 layout: AR(1) states for the clock series, chains for the others."""
        n_slower = int(n_series) - int(n_clock)
        return cls(
            r=tuple(int(x) for x in np.asarray(r).ravel()),
            p=int(p),
            L=int(L),
            n_series=int(n_series),
            n_clock=int(n_clock),
            idio_series=tuple(range(int(n_clock))),
            chain_length=int(chain_length) if n_slower > 0 else 0,
        )

    @classmethod
    def from_state(cls, state: Any, n_series: Optional[int] = None) -> DFMStateLayout:
        """Layout of a ``DFMModelState`` (its structure fields and the size of ``A``).

        The AR(1) idiosyncratic states are those of ``state.idio_indicator``;
        the states after them are split into one chain per slower-frequency
        series when they divide evenly (otherwise ``chain_length`` is 0 and
        they are left alone by the M-step).
        """
        r = np.asarray(state.r).astype(int).ravel()
        p = int(state.p)
        L = int(state.max_lag_size if state.max_lag_size is not None else p + 1)
        if n_series is None:
            n_series = int(np.asarray(state.C).shape[0])
        n_slower = int(state.n_slower_freq or 0)
        n_clock = int(state.n_clock_freq) if state.n_clock_freq is not None else n_series - n_slower
        indicator = state.idio_indicator
        idio = (tuple(int(i) for i in np.flatnonzero(np.asarray(indicator).ravel()[:n_series] > 0))
                if indicator is not None else ())
        m = int(np.asarray(state.A).shape[0])
        rest = m - int(r.sum()) * L - len(idio)
        k = rest // n_slower if (n_slower > 0 and rest > 0 and rest % n_slower == 0) else 0
        return cls(r=tuple(int(x) for x in r), p=p, L=L, n_series=int(n_series), n_clock=n_clock,
                   idio_series=idio, chain_length=int(k))

    @property
    def n_slower(self) -> int:
        return self.n_series - self.n_clock

    @property
    def n_factor_states(self) -> int:
        return int(sum(self.r)) * self.L

    @property
    def n_idio(self) -> int:
        return len(self.idio_series)

    @property
    def n_chain_states(self) -> int:
        return self.n_slower * self.chain_length

    @property
    def m(self) -> int:
        """State dimension."""
        return self.n_factor_states + self.n_idio + self.n_chain_states

    @property
    def idio_indicator(self) -> np.ndarray:
        out = np.zeros(self.n_series, dtype=np.int32)
        out[list(self.idio_series)] = 1
        return out

    @property
    def factor_offsets(self) -> np.ndarray:
        """State index where each factor block starts."""
        widths = np.asarray(self.r, dtype=int) * self.L
        return np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(int)

    def factor_current(self, b: int) -> np.ndarray:
        """States of the current values of block ``b``'s factors."""
        start = int(self.factor_offsets[b])
        return np.arange(start, start + self.r[b])

    def factor_lags(self, b: int, n_lags: int) -> np.ndarray:
        """First ``n_lags`` lag copies of block ``b``'s factors (``r_b * n_lags`` states)."""
        start = int(self.factor_offsets[b])
        return np.arange(start, start + self.r[b] * int(n_lags))

    @property
    def idio_states(self) -> np.ndarray:
        """AR(1) idiosyncratic states (one per series in ``idio_series``)."""
        return self.n_factor_states + np.arange(self.n_idio)

    @property
    def chain_start(self) -> int:
        return self.n_factor_states + self.n_idio

    def chain_states(self, j: int) -> np.ndarray:
        """Chain states of slower-frequency series ``j`` (a column index ``>= n_clock``)."""
        s = int(j) - self.n_clock
        if not (0 <= s < self.n_slower) or self.chain_length == 0:
            raise IndexError(f"series {j} has no idiosyncratic chain")
        start = self.chain_start + s * self.chain_length
        return np.arange(start, start + self.chain_length)

    @property
    def chain_heads(self) -> np.ndarray:
        """First state of every chain (the AR(1) part of the chain)."""
        if self.chain_length == 0:
            return np.zeros(0, dtype=int)
        return self.chain_start + self.chain_length * np.arange(self.n_slower)

    @property
    def ar1_states(self) -> np.ndarray:
        """States with an estimated AR(1) coefficient: clock idio states, chain heads."""
        return np.concatenate([self.idio_states, self.chain_heads]).astype(int)

    def is_bm14(self) -> bool:
        """Whether exactly the clock series have AR(1) states and the others chains."""
        chains_ok = self.n_slower == 0 or self.chain_length > 0
        return self.idio_series == tuple(range(self.n_clock)) and chains_ok

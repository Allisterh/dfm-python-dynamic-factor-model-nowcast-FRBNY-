"""The engines' training-state dataclasses (structure and state-space parameters).

None of them is exported: each is its engine's ``training_state`` and, for
DDFM, the type of ``DDFMResult.fit_params``.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Any


@dataclass
class DFMModelState:
    """DFM structure, mixed-frequency layout and fitted state-space parameters.

    Attributes
    ----------
    num_factors : int
    r : np.ndarray
        Factors per block.
    p : int
        AR lag order.
    blocks : np.ndarray
        Block membership array.
    mixed_freq : bool, optional
    constraint_matrix, constraint_vector : np.ndarray, optional
        Tent kernel aggregation constraints.
    n_slower_freq : int
    n_clock_freq : int, optional
    tent_weights_dict : dict, optional
        Tent weights by frequency.
    frequencies : np.ndarray, optional
        Frequency of each series.
    idio_indicator : np.ndarray, optional
    max_lag_size : int, optional
        Lags per factor in the state.
    A, C, Q, R, Z_0, V_0 : np.ndarray, optional
        Transition (m x m), loadings (N x m), process noise (m x m),
        observation noise (N x N, diagonal), initial state mean (m,) and
        covariance (m x m).
    """
    num_factors: int
    r: np.ndarray
    p: int
    blocks: np.ndarray
    mixed_freq: Optional[bool] = None
    constraint_matrix: Optional[np.ndarray] = None
    constraint_vector: Optional[np.ndarray] = None
    n_slower_freq: int = 0
    n_clock_freq: Optional[int] = None
    tent_weights_dict: Optional[dict] = None
    frequencies: Optional[np.ndarray] = None
    idio_indicator: Optional[np.ndarray] = None
    max_lag_size: Optional[int] = None
    A: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Z_0: Optional[np.ndarray] = None
    V_0: Optional[np.ndarray] = None


@dataclass
class DDFMModelState:
    """DDFM training state and fitted state-space parameters.

    Attributes
    ----------
    num_iter : int
        MCMC iterations run.
    loss_now : float, optional
    converged : bool
    eps : np.ndarray, optional
        Idiosyncratic residuals (T x num_target_series).
    factors : np.ndarray, optional
        (n_mc_samples x T x num_factors) or (T x num_factors).
    last_neurons : np.ndarray, optional
        Last decoder layer (n_mc_samples x T x num_neurons) or (T x
        num_neurons); equals ``factors`` for the linear decoder.
    F, Q, mu_0, Sigma_0, H, R : np.ndarray, optional
        Transition (m x m, A in the paper), process noise (m x m, W in the
        original code), initial state mean (m,) and covariance (m x m),
        decoder weights (N x m, theta in the paper) and observation noise
        (N x N, diagonal).
    """
    num_iter: int = 0
    loss_now: Optional[float] = None
    converged: bool = False
    eps: Optional[np.ndarray] = None
    factors: Optional[np.ndarray] = None
    last_neurons: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    mu_0: Optional[np.ndarray] = None
    Sigma_0: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None


@dataclass
class iVDFMModelState:
    """iVDFM training state and state-space parameters.

    Attributes
    ----------
    num_iter : int
        Epochs run.
    loss_now, elbo : float, optional
    converged : bool
    factors, innovations : np.ndarray, optional
        (T x r) or (batch x T x r).
    full_state : np.ndarray, optional
        Companion-form state: (T x r*p) for p > 1, (T x r) for p = 1.
    factor_order : int
        p of the AR(p) factor dynamics.
    A : np.ndarray, optional
        Transition (r x r) for p = 1, AR coefficients (r x p) for p > 1.
    B : np.ndarray, optional
        Innovation loadings (r x r).
    f0 : np.ndarray, optional
        Initial state (r,).
    """
    num_iter: int = 0
    loss_now: Optional[float] = None
    elbo: Optional[float] = None
    converged: bool = False
    factors: Optional[np.ndarray] = None
    innovations: Optional[np.ndarray] = None
    full_state: Optional[np.ndarray] = None
    factor_order: int = 1
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    f0: Optional[np.ndarray] = None

    @classmethod
    def from_model(cls, model: Any) -> 'iVDFMModelState':
        """Snapshot of an iVDFM engine (``ssm`` parameters as numpy, ``full_state`` built from ``factors``)."""
        factors = getattr(model, 'factors', None)
        innovations = getattr(model, 'innovations', None)
        num_iter = getattr(model, '_num_iter', 0)
        loss_now = getattr(model, 'loss_now', None)
        elbo = getattr(model, '_elbo', None)
        converged = getattr(model, '_converged', False)

        factor_order = getattr(model, 'factor_order', 1)
        A = B = f0 = None
        if hasattr(model, 'ssm') and model.ssm is not None:
            ssm = model.ssm
            if factor_order == 1:
                # First order: A is diagonal, stored as (r,)
                if hasattr(ssm, 'A'):
                    A = ssm.A.data.cpu().numpy() if hasattr(ssm.A, 'data') else ssm.A
            else:
                # Higher order: AR coefficients stored as (r, p)
                if hasattr(ssm, 'ar_coeffs'):
                    A = ssm.ar_coeffs.data.cpu().numpy() if hasattr(ssm.ar_coeffs, 'data') else ssm.ar_coeffs
            if hasattr(ssm, 'B'):
                B = ssm.B.data.cpu().numpy() if hasattr(ssm.B, 'data') else ssm.B
            if hasattr(ssm, 'f0'):
                f0 = ssm.f0.data.cpu().numpy() if hasattr(ssm.f0, 'data') else ssm.f0

        if A is None:
            A = getattr(model, 'A', None)
            if A is not None and hasattr(A, 'data'):
                A = A.data.cpu().numpy()
        if B is None:
            B = getattr(model, 'B', None)
            if B is not None and hasattr(B, 'data'):
                B = B.data.cpu().numpy()
        if f0 is None:
            f0 = getattr(model, 'f0', None)
            if f0 is not None and hasattr(f0, 'data'):
                f0 = f0.data.cpu().numpy()

        full_state = None
        if factors is not None:
            factors_np = factors
            if isinstance(factors_np, np.ndarray):
                if factors_np.ndim == 3:
                    factors_np = np.mean(factors_np, axis=0)  # (T, r), averaged over the batch

                T, r = factors_np.shape
                if factor_order == 1:
                    full_state = factors_np
                else:
                    # s_t[i*p : (i+1)*p] = [f_t[i], f_{t-1}[i], ..., f_{t-p+1}[i]]; f0 before the sample
                    full_state = np.zeros((T, r * factor_order), dtype=factors_np.dtype)
                    for t in range(T):
                        for i in range(r):
                            for lag in range(factor_order):
                                idx = t - lag
                                if idx >= 0:
                                    full_state[t, i * factor_order + lag] = factors_np[idx, i]
                                elif f0 is not None and i < len(f0):
                                    full_state[t, i * factor_order + lag] = f0[i]

        return cls(
            num_iter=num_iter,
            loss_now=loss_now,
            elbo=elbo,
            converged=converged,
            factors=factors,
            innovations=innovations,
            full_state=full_state,
            factor_order=factor_order,
            A=A,
            B=B,
            f0=f0
        )


@dataclass
class AFMModelState:
    """AFM training diagnostics."""
    num_iter: int = 0
    converged: bool = False
    loss_now: Optional[float] = None
    sharpe: Optional[float] = None
    explained_variance: Optional[float] = None
    residuals: Optional[np.ndarray] = None       # (B, T, N) traded residual portfolios
    book_returns: Optional[np.ndarray] = None    # (B, n_eval) after-cost book returns

    @classmethod
    def from_model(cls, model: Any) -> 'AFMModelState':
        return cls(
            num_iter=getattr(model, '_num_iter', 0),
            converged=getattr(model, '_converged', False),
            loss_now=getattr(model, 'loss_now', None),
            sharpe=getattr(model, '_sharpe', None),
            explained_variance=getattr(model, '_ev', None),
            residuals=getattr(model, '_residuals', None),
            book_returns=getattr(model, '_book_returns', None),
        )

    def apply_to_model(self, model: Any) -> None:
        model._num_iter = self.num_iter
        model._converged = self.converged
        model.loss_now = self.loss_now
        model._sharpe = self.sharpe
        model._ev = self.explained_variance
        model._residuals = self.residuals
        model._book_returns = self.book_returns

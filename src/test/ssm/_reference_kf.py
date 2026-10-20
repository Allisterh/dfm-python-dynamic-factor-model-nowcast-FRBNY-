"""Independent dense reference Kalman filter and smoothers for the DFM oracle tests.

This module does not import ``dfm_python``: it is the oracle the package's
E-step is checked against. Everything is float64 and written for clarity, not
speed.

Model::

    x_{t+1} = A x_t + w_t,   w_t ~ N(0, Q)
    y_t     = C x_t + v_t,   v_t ~ N(0, R)

Timing (the convention of pykalman and of statsmodels' ``initialize_known``):
``x_0 ~ N(Z_0, V_0)`` is the state AT data row 0, so the prediction for row 0
is ``(Z_0, V_0)`` and the first transition is applied before row 1.

Missing values are NaN and are handled per element: at row ``t`` only the
observed rows ``o_t`` of ``C`` and the block ``R[o_t, o_t]`` enter the update,
and a row with nothing observed contributes nothing (prediction only). The
log-likelihood is the exact Gaussian log density of the observed entries,
accumulated from the predictive densities in the same forward pass.

Lag-one covariances use pykalman's ``_smooth_pair`` layout:
``pair[t] = Cov(x_t, x_{t-1} | Y)`` for ``t >= 1`` and ``pair[0] = 0``.
statsmodels' ``smoothed_state_autocov[:, :, t]`` is ``Cov(x_{t+1}, x_t | Y)``,
i.e. ``pair[t + 1]``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats

LOG2PI = float(np.log(2.0 * np.pi))


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _as64(*arrays):
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


def kalman_filter(A, C, Q, R, Z0, V0, Y) -> dict:
    """Forward pass with per-element missing data, Joseph update and exact log-likelihood.

    Returns a dict with the predicted (``pm``, ``pP``) and filtered (``fm``,
    ``fP``) moments of every row, the per-row log-likelihood ``ll_t``, its sum
    ``ll``, the number of observed entries per row ``nobs``, and, for the DK
    smoother, the observed mask ``obs``, innovations ``v``, Cholesky factors
    ``L`` of the innovation covariances and gains ``K``.
    """
    A, C, Q, R, Z0, V0, Y = _as64(A, C, Q, R, Z0, V0, Y)
    T = Y.shape[0]
    m = A.shape[0]
    I_m = np.eye(m)
    pm = np.zeros((T, m))
    pP = np.zeros((T, m, m))
    fm = np.zeros((T, m))
    fP = np.zeros((T, m, m))
    ll_t = np.zeros(T)
    nobs = np.zeros(T, dtype=int)
    obs = ~np.isnan(Y)
    v_list, L_list, K_list = [], [], []
    for t in range(T):
        if t == 0:
            a, P = Z0.copy(), _sym(V0)
        else:
            a = A @ fm[t - 1]
            P = _sym(A @ fP[t - 1] @ A.T + Q)
        pm[t], pP[t] = a, P
        o = obs[t]
        p = int(o.sum())
        nobs[t] = p
        if p == 0:
            fm[t], fP[t] = a, P
            v_list.append(None)
            L_list.append(None)
            K_list.append(None)
            continue
        Co = C[o]
        Ro = R[np.ix_(o, o)]
        e = Y[t, o] - Co @ a
        PCt = P @ Co.T
        S = _sym(Co @ PCt + Ro)
        L = linalg.cholesky(S, lower=True)
        u = linalg.solve_triangular(L, e, lower=True)
        ll_t[t] = -0.5 * (p * LOG2PI + 2.0 * np.sum(np.log(np.diag(L))) + u @ u)
        K = linalg.cho_solve((L, True), PCt.T).T  # P C_o' S^-1
        fm[t] = a + K @ e
        IKC = I_m - K @ Co
        fP[t] = _sym(IKC @ P @ IKC.T + K @ Ro @ K.T)  # Joseph form
        v_list.append(e)
        L_list.append(L)
        K_list.append(K)
    return dict(pm=pm, pP=pP, fm=fm, fP=fP, ll_t=ll_t, ll=float(ll_t.sum()), nobs=nobs,
                obs=obs, v=v_list, L=L_list, K=K_list)


def rts_smoother(A, filt: dict):
    """Rauch-Tung-Striebel smoother: ``J_t = P_f[t] A' P_pred[t+1]^-1`` (by a solve).

    Returns ``(sm, sP, pair)`` with ``pair[t] = P_s[t] J_{t-1}' = Cov(x_t, x_{t-1} | Y)``.
    """
    (A,) = _as64(A)
    pm, pP, fm, fP = filt["pm"], filt["pP"], filt["fm"], filt["fP"]
    T, m = fm.shape
    sm = fm.copy()
    sP = fP.copy()
    J = np.zeros((T, m, m))
    for t in range(T - 2, -1, -1):
        Pn = pP[t + 1]
        J[t] = linalg.solve(Pn, A @ fP[t], assume_a="sym").T  # (Pn^-1 A P_f)' = P_f A' Pn^-1
        sm[t] = fm[t] + J[t] @ (sm[t + 1] - pm[t + 1])
        sP[t] = _sym(fP[t] + J[t] @ (sP[t + 1] - Pn) @ J[t].T)
    pair = np.zeros_like(sP)
    for t in range(1, T):
        pair[t] = sP[t] @ J[t - 1].T
    return sm, sP, pair


def dk_smoother(A, C, filt: dict):
    """Durbin-Koopman (r_t, N_t) backward smoother: no inverse of the predicted covariance.

    With ``L_t = A (I - K_t C_o)`` (``L_t = A`` on a row with nothing observed)::

        r_{t-1} = C_o' S_t^-1 v_t + L_t' r_t,   N_{t-1} = C_o' S_t^-1 C_o + L_t' N_t L_t
        E[x_t|Y] = a_t + P_t r_{t-1},          Var[x_t|Y] = P_t - P_t N_{t-1} P_t
        Cov(x_t, x_{t+1}|Y) = P_t L_t' (I - N_t P_{t+1})          (DK 2012, eq. 4.69)

    Returns ``(sm, sP, pair)`` in the same layout as :func:`rts_smoother`.
    """
    A, C = _as64(A, C)
    pm, pP = filt["pm"], filt["pP"]
    obs, v_list, L_list, K_list = filt["obs"], filt["v"], filt["L"], filt["K"]
    T, m = pm.shape
    I_m = np.eye(m)
    sm = np.zeros((T, m))
    sP = np.zeros((T, m, m))
    pair = np.zeros((T, m, m))
    r = np.zeros(m)
    N = np.zeros((m, m))
    for t in range(T - 1, -1, -1):
        N_next = N  # N_t: information from rows t+1 .. T-1
        if v_list[t] is None:
            Lt = A
            r = Lt.T @ r
            N = _sym(Lt.T @ N @ Lt)
        else:
            Co = C[obs[t]]
            chol = (L_list[t], True)
            Lt = A @ (I_m - K_list[t] @ Co)
            r = Co.T @ linalg.cho_solve(chol, v_list[t]) + Lt.T @ r
            N = _sym(Co.T @ linalg.cho_solve(chol, Co) + Lt.T @ N @ Lt)
        P = pP[t]
        sm[t] = pm[t] + P @ r
        sP[t] = _sym(P - P @ N @ P)
        if t < T - 1:
            pair[t + 1] = (P @ Lt.T @ (I_m - N_next @ pP[t + 1])).T
    return sm, sP, pair


def filter_smooth(A, C, Q, R, Z0, V0, Y, smoother: str = "dk") -> dict:
    """Filter, smooth and lag-one covariances in one call.

    ``smoother`` is ``"dk"`` (default; robust when the predicted covariance is
    singular, e.g. deterministic lag states) or ``"rts"``.
    Returns the filter dict plus ``sm``, ``sP`` and ``pair``.
    """
    filt = kalman_filter(A, C, Q, R, Z0, V0, Y)
    if smoother == "rts":
        sm, sP, pair = rts_smoother(A, filt)
    elif smoother == "dk":
        sm, sP, pair = dk_smoother(A, C, filt)
    else:
        raise ValueError(f"unknown smoother {smoother!r}")
    return {**filt, "sm": sm, "sP": sP, "pair": pair}


def loglik(A, C, Q, R, Z0, V0, Y) -> float:
    """Exact Gaussian log-likelihood of the observed entries of ``Y``."""
    return kalman_filter(A, C, Q, R, Z0, V0, Y)["ll"]


def brute_force_loglik(A, C, Q, R, Z0, V0, Y) -> float:
    """The same log-likelihood from the joint covariance of all observed entries (small T only).

    Builds the stacked state moments ``E[x_t] = A^t Z_0`` and
    ``Cov(x_s, x_t) = A^(s-t) Var(x_t)`` for ``s >= t``, maps them to the
    observed entries and evaluates ``scipy.stats.multivariate_normal.logpdf``.
    It validates :func:`kalman_filter` itself.
    """
    A, C, Q, R, Z0, V0, Y = _as64(A, C, Q, R, Z0, V0, Y)
    T, N = Y.shape
    if T > 15:
        raise ValueError("brute_force_loglik is meant for T <= 15")
    m = A.shape[0]
    means = np.zeros((T, m))
    var = np.zeros((T, m, m))
    means[0], var[0] = Z0, V0
    for t in range(1, T):
        means[t] = A @ means[t - 1]
        var[t] = A @ var[t - 1] @ A.T + Q
    Sx = np.zeros((T * m, T * m))
    for t in range(T):
        block = var[t]
        for s in range(t, T):
            Sx[s * m:(s + 1) * m, t * m:(t + 1) * m] = block
            Sx[t * m:(t + 1) * m, s * m:(s + 1) * m] = block.T
            block = A @ block
    rows, mu, y = [], [], []
    noise = np.zeros((0, 0))
    for t in range(T):
        o = ~np.isnan(Y[t])
        if not o.any():
            continue
        H = np.zeros((int(o.sum()), T * m))
        H[:, t * m:(t + 1) * m] = C[o]
        rows.append(H)
        mu.append(C[o] @ means[t])
        y.append(Y[t, o])
        noise = linalg.block_diag(noise, R[np.ix_(o, o)])
    H = np.vstack(rows)
    cov = _sym(H @ Sx @ H.T + noise)
    return float(stats.multivariate_normal.logpdf(np.concatenate(y), mean=np.concatenate(mu), cov=cov))


def statsmodels_smooth(A, C, Q, R, Z0, V0, Y) -> dict:
    """statsmodels ``KalmanSmoother`` on the same model (raises ImportError without statsmodels).

    Returns ``ll``, ``sm``, ``sP``, ``fm``, ``fP`` and ``pair`` in this module's layout
    (``pair[t + 1] = autocov[:, :, t]``).
    """
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

    A, C, Q, R, Z0, V0, Y = _as64(A, C, Q, R, Z0, V0, Y)
    T, N = Y.shape
    m = A.shape[0]
    ks = KalmanSmoother(k_endog=N, k_states=m, k_posdef=m)
    ks.bind(np.ascontiguousarray(Y))
    ks["design"] = C
    ks["obs_cov"] = R
    ks["transition"] = A
    ks["selection"] = np.eye(m)
    ks["state_cov"] = Q
    ks.initialize_known(Z0, V0)
    res = ks.smooth()
    autocov = np.moveaxis(np.asarray(res.smoothed_state_autocov), 2, 0)
    pair = np.zeros((T, m, m))
    pair[1:] = autocov[:-1]
    return dict(ll=float(np.sum(res.llf_obs)), sm=np.asarray(res.smoothed_state).T,
                sP=np.moveaxis(np.asarray(res.smoothed_state_cov), 2, 0),
                fm=np.asarray(res.filtered_state).T, fP=np.moveaxis(np.asarray(res.filtered_state_cov), 2, 0),
                pair=pair)


def random_model(m: int = 4, N: int = 5, T: int = 12, nan_frac: float = 0.4, seed: int = 0):
    """A random stable model and data with a fraction ``nan_frac`` of NaN (plus one all-NaN row when T > 7).

    Returns ``(A, C, Q, R, Z0, V0, Y)``; ``A`` is not symmetric, ``R`` is diagonal.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    A *= 0.8 / max(np.abs(np.linalg.eigvals(A)))
    C = rng.standard_normal((N, m))
    B = rng.standard_normal((m, m))
    Q = B @ B.T / m + 0.1 * np.eye(m)
    R = np.diag(rng.uniform(0.2, 1.0, N))
    Z0 = rng.standard_normal(m)
    G = rng.standard_normal((m, m))
    V0 = G @ G.T / m + np.eye(m)
    x = rng.multivariate_normal(Z0, V0)
    Y = np.zeros((T, N))
    for t in range(T):
        if t > 0:
            x = A @ x + rng.multivariate_normal(np.zeros(m), Q)
        Y[t] = C @ x + rng.multivariate_normal(np.zeros(N), R)
    Y[rng.random((T, N)) < nan_frac] = np.nan
    if T > 7:
        Y[7] = np.nan
    return A, C, Q, R, Z0, V0, Y


def rel_err(a, b) -> float:
    """``max|a - b| / max|b|`` (inf when shapes differ or a value is not finite)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return float(np.max(np.abs(a - b)) / max(scale, 1e-300)) if a.size else 0.0

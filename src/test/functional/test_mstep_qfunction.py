"""Block-by-block ascent check of the DFM M-step on the expected complete-data log-likelihood.

For EM parameters ``theta_k`` of the synthetic and monthly/quarterly fits, the
smoothed moments of ``theta_k`` come from the independent reference smoother
(``src/test/ssm/_reference_kf.py``), and

    Q(theta | theta_k) = E[log p(x_0) + sum_t log p(x_t | x_{t-1}) + sum_t log p(y_t^obs | x_t) | Y, theta_k]

is computed directly (initial, transition and observation parts). Each M-step
block (factor A/Q, idiosyncratic AR(1), clock-frequency C, slower-frequency
C, R, Z_0/V_0), fed with those moments and swapped into ``theta_k`` alone,
must not decrease its part of Q (tolerance 1e-8 relative). The closed-form
blocks are also checked to be maximizers: random perturbations of the free
entries (inside the tent restriction for slower-frequency loadings) do not
increase Q.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace

import numpy as np
import pytest

from ..ssm import _reference_kf as ref

RTOL = 1e-8
LOG2PI = float(np.log(2.0 * np.pi))
PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")


# ------------------------------------------------------------------ Q-function


def _logdet(M: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(M)
    assert sign > 0, "covariance is not positive definite"
    return float(value)


def _is_pd(M: np.ndarray) -> bool:
    w = np.linalg.eigvalsh(M)
    return bool(w.min() > 1e-12 * max(1.0, w.max()))


def q_parts(theta: dict, mom: dict, Y: np.ndarray) -> np.ndarray:
    """``[initial, transition, observation]`` parts of Q(theta | moments of theta_k).

    The initial part is NaN when ``V_0`` is singular (a projected smoothed
    covariance can be): it is then not a density, and +inf as a limit.
    """
    A, C, Q, R, z0, V0 = (np.asarray(theta[k], dtype=np.float64) for k in PARAMS)
    sm, sP, pair = mom["sm"], mom["sP"], mom["pair"]
    T, m = sm.shape
    d0 = sm[0] - z0
    q_init = (-0.5 * (m * LOG2PI + _logdet(V0) + np.trace(np.linalg.solve(V0, sP[0] + np.outer(d0, d0))))
              if _is_pd(V0) else float("nan"))
    S11 = sP[1:].sum(0) + sm[1:].T @ sm[1:]
    S10 = pair[1:].sum(0) + sm[1:].T @ sm[:-1]
    S00 = sP[:-1].sum(0) + sm[:-1].T @ sm[:-1]
    E = S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T
    # Only the states with process noise have a transition density: the companion
    # lag states and the chain shift states are deterministic (zero rows of Q, fixed
    # identity rows of A), so their part is a constant.
    s = np.flatnonzero(np.any(Q != 0.0, axis=1))
    Qs, Es = Q[np.ix_(s, s)], E[np.ix_(s, s)]
    q_trans = -0.5 * ((T - 1) * (s.size * LOG2PI + _logdet(Qs)) + np.trace(np.linalg.solve(Qs, Es)))
    q_obs = 0.0
    for t in range(T):
        o = ~np.isnan(Y[t])
        if not o.any():
            continue
        Co, Ro = C[o], R[np.ix_(o, o)]
        e = Y[t, o] - Co @ sm[t]
        M = np.outer(e, e) + Co @ sP[t] @ Co.T
        q_obs += -0.5 * (o.sum() * LOG2PI + _logdet(Ro) + np.trace(np.linalg.solve(Ro, M)))
    return np.array([q_init, q_trans, q_obs])


def _as_dict(state) -> dict:
    return {k: np.array(getattr(state, k), dtype=np.float64) for k in PARAMS}


def _assert_not_lower(after: float, before: float, label: str) -> None:
    assert after >= before - RTOL * abs(before), f"{label}: Q decreased from {before!r} to {after!r}"


# ---------------------------------------------------------------- EM iterates


def _record_states(model, panel, monkeypatch):
    """Fit ``model``; return the data and the DFMModelState every EM iteration started from."""
    import dfm_python.functional.em as em

    seen = []
    original = em.em_step

    def recording_em_step(X, state, *args, **kwargs):
        seen.append((np.array(X, dtype=np.float64), state))
        return original(X, state, *args, **kwargs)

    monkeypatch.setattr(em, "em_step", recording_em_step)
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(panel)
    finally:
        logging.disable(logging.NOTSET)
    return seen[0][0], [s for _, s in seen]


def _synthetic():
    from .test_em_oracle import _synthetic as make

    return make()


def _monthly_quarterly():
    from .test_em_oracle import _monthly_quarterly as make

    return make()


_CASES = {"dfm_synthetic": _synthetic, "dfm_macro_monthly_quarterly": _monthly_quarterly}


@pytest.fixture(scope="module")
def iterates():
    """``{case: (X, [state_0, state_1, ...])}`` for 4 EM iterations of each case."""
    out = {}
    mp = pytest.MonkeyPatch()
    try:
        for name, make in _CASES.items():
            model, panel = make()
            model.set_params(max_iter=4)
            out[name] = _record_states(model, panel, mp)
            mp.undo()
    finally:
        mp.undo()
    return out


def _layout(state):
    """``(r, L, f, n_idio)`` of the BM14 state layout: factor states, then one AR(1) per clock series."""
    r = np.asarray(state.r).astype(int)
    L = int(state.max_lag_size)
    f = int(r.sum() * L)
    n_idio = int(np.sum(np.asarray(state.idio_indicator) > 0))
    return r, L, f, n_idio


def _ar1_states(state) -> np.ndarray:
    """The clock idiosyncratic AR(1) states and the head of each slower-frequency chain.

    Computed from the sizes alone: the ``m - f - n_idio`` states after the
    AR(1) states are one chain of equal length per slower-frequency series.
    """
    r, L, f, n_idio = _layout(state)
    m = np.asarray(state.A).shape[0]
    n_slower = int(state.n_slower_freq or 0)
    heads = []
    if n_slower:
        k = (m - f - n_idio) // n_slower
        assert k * n_slower == m - f - n_idio
        heads = [f + n_idio + k * s for s in range(n_slower)]
    return np.concatenate([np.arange(f, f + n_idio), heads]).astype(int)


#: EM iterations whose parameters are checked, the initial parameters (k=0)
#: included: their slower-frequency loadings satisfy the tent restriction and
#: their R is at the R floor (MIN_OBSERVATION_NOISE), so they are in the
#: M-step's parameter set.
CASE_ITERS = [(c, k) for c in _CASES for k in (0, 1, 3)]
CASE_ITERS_WITH_INIT = CASE_ITERS


def _setup(iterates, case, k):
    from dfm_python.functional.em import EMConfig

    X, states = iterates[case]
    state = states[k]
    theta = _as_dict(state)
    mom = ref.filter_smooth(*(theta[p] for p in PARAMS), X, smoother="rts")
    return X, state, theta, mom, EMConfig(normalize_C=False)


# ---------------------------------------------------------------------- tests


@pytest.mark.parametrize("case,k", CASE_ITERS_WITH_INIT)
def test_transition_blocks_increase_q(iterates, case, k):
    """Factor A/Q and the idiosyncratic AR(1) update each raise the transition part of Q."""
    from dfm_python.functional.em import _update_transition_matrix_blocked

    X, state, theta, mom, config = _setup(iterates, case, k)
    new = _update_transition_matrix_blocked(mom["sm"], mom["sP"], mom["pair"], state, config)
    r, L, f, n_idio = _layout(state)
    q0 = q_parts(theta, mom, X)[1]

    fac = dict(theta, A=theta["A"].copy(), Q=theta["Q"].copy())
    fac["A"][:f] = np.asarray(new.A)[:f]
    fac["Q"][:f, :f] = np.asarray(new.Q)[:f, :f]
    _assert_not_lower(q_parts(fac, mom, X)[1], q0, f"{case} k={k} factor A/Q")

    ii = _ar1_states(state)
    idio = dict(theta, A=theta["A"].copy(), Q=theta["Q"].copy())
    idio["A"][ii, ii] = np.asarray(new.A)[ii, ii]
    idio["Q"][ii, ii] = np.asarray(new.Q)[ii, ii]
    _assert_not_lower(q_parts(idio, mom, X)[1], q0, f"{case} k={k} idiosyncratic AR(1) and chain heads")

    both = dict(theta, A=np.asarray(new.A), Q=np.asarray(new.Q))
    _assert_not_lower(q_parts(both, mom, X)[1], q0, f"{case} k={k} A/Q")

    # Companion shift rows and the chain shift states are untouched (zero Q).
    rows_fixed = np.ones(theta["A"].shape[0], dtype=bool)
    offset = 0
    for r_b in r:
        rows_fixed[offset:offset + r_b] = False
        offset += r_b * L
    rows_fixed[ii] = False
    np.testing.assert_array_equal(np.asarray(new.A)[rows_fixed], theta["A"][rows_fixed])
    np.testing.assert_array_equal(np.asarray(new.Q)[rows_fixed], theta["Q"][rows_fixed])
    np.testing.assert_array_equal(np.asarray(new.Q)[rows_fixed], 0.0)


@pytest.mark.parametrize("case,k", CASE_ITERS_WITH_INIT)
def test_transition_update_is_the_maximizer(iterates, case, k):
    """Perturbing the estimated VAR(p) coefficients or the idio AR(1) coefficients lowers Q."""
    from dfm_python.functional.em import _update_transition_matrix_blocked

    X, state, theta, mom, config = _setup(iterates, case, k)
    new = _as_dict(_update_transition_matrix_blocked(mom["sm"], mom["sP"], mom["pair"], state, config))
    r, L, f, n_idio = _layout(state)
    q_best = q_parts(new, mom, X)[1]
    rng = np.random.default_rng(k)
    r_b, p = int(r[0]), int(state.p)
    for _ in range(5):
        pert = dict(new, A=new["A"].copy())
        pert["A"][:r_b, :r_b * p] += 1e-3 * rng.standard_normal((r_b, r_b * p))
        assert q_parts(pert, mom, X)[1] <= q_best + RTOL * abs(q_best)
        ii = _ar1_states(state)
        if ii.size:
            pert = dict(new, A=new["A"].copy())
            pert["A"][ii, ii] += 1e-3 * rng.standard_normal(ii.size)
            assert q_parts(pert, mom, X)[1] <= q_best + RTOL * abs(q_best)


@pytest.mark.parametrize("case,k", CASE_ITERS)
def test_observation_blocks_increase_q(iterates, case, k):
    """Clock C, slower C (tent-restricted) and R each raise the observation part of Q."""
    from dfm_python.functional.em import (
        _update_observation_matrix_blocked,
        _update_observation_noise_blocked,
    )

    X, state, theta, mom, config = _setup(iterates, case, k)
    new_C = np.asarray(_update_observation_matrix_blocked(X, mom["sm"], mom["sP"], state, config).C)
    n_clock = int(state.n_clock_freq)
    q0 = q_parts(theta, mom, X)[2]

    clock = dict(theta, C=theta["C"].copy())
    clock["C"][:n_clock] = new_C[:n_clock]
    _assert_not_lower(q_parts(clock, mom, X)[2], q0, f"{case} k={k} clock C")

    if n_clock < X.shape[1]:
        slow = dict(theta, C=theta["C"].copy())
        slow["C"][n_clock:] = new_C[n_clock:]
        _assert_not_lower(q_parts(slow, mom, X)[2], q0, f"{case} k={k} slower C")
        # The tent restriction holds for every factor of every slower series.
        r, L, f, _ = _layout(state)
        R_mat = np.asarray(state.constraint_matrix)
        kk = R_mat.shape[1]
        for j in range(n_clock, X.shape[1]):
            c = new_C[j, :kk * r[0]].reshape(kk, r[0])
            np.testing.assert_allclose(R_mat @ c, 0.0, atol=1e-10 * max(1.0, np.abs(c).max()))

    with_C = dict(theta, C=new_C)
    q_C = q_parts(with_C, mom, X)[2]
    _assert_not_lower(q_C, q0, f"{case} k={k} C")
    new_R = np.asarray(_update_observation_noise_blocked(
        X, mom["sm"], mom["sP"], replace(state, C=new_C), config).R)
    _assert_not_lower(q_parts(dict(with_C, R=new_R), mom, X)[2], q_C, f"{case} k={k} R")


@pytest.mark.parametrize("case,k", CASE_ITERS)
def test_loading_update_is_the_maximizer(iterates, case, k):
    """Perturbing a clock loading, or a slower loading inside the tent restriction, lowers Q."""
    from scipy.linalg import null_space

    from dfm_python.functional.em import _update_observation_matrix_blocked

    X, state, theta, mom, config = _setup(iterates, case, k)
    new_C = np.asarray(_update_observation_matrix_blocked(X, mom["sm"], mom["sP"], state, config).C)
    best = dict(theta, C=new_C)
    q_best = q_parts(best, mom, X)[2]
    r, L, f, _ = _layout(state)
    r_b, n_clock = int(r[0]), int(state.n_clock_freq)
    rng = np.random.default_rng(10 + k)
    for _ in range(5):
        pert = dict(best, C=new_C.copy())
        pert["C"][:n_clock, :r_b] += 1e-3 * rng.standard_normal((n_clock, r_b))
        assert q_parts(pert, mom, X)[2] <= q_best + RTOL * abs(q_best)
    if n_clock < X.shape[1]:
        R_mat = np.asarray(state.constraint_matrix)
        kk = R_mat.shape[1]
        basis = null_space(np.kron(R_mat, np.eye(r_b)))
        for _ in range(5):
            pert = dict(best, C=new_C.copy())
            for j in range(n_clock, X.shape[1]):
                pert["C"][j, :kk * r_b] += basis @ (1e-3 * rng.standard_normal(basis.shape[1]))
            assert q_parts(pert, mom, X)[2] <= q_best + RTOL * abs(q_best)


@pytest.mark.parametrize("case,k", CASE_ITERS_WITH_INIT)
def test_initial_state_update_increases_q(iterates, case, k):
    """Z_0 = E[x_0|Y], V_0 = Cov(x_0|Y) raise the initial part of Q (and V_0 is PSD).

    The comparison needs a positive definite V_0 at ``theta_k`` (the initial
    part is not defined otherwise); a singular updated V_0 makes the new
    initial part +inf, which is not lower.
    """
    from dfm_python.functional.em import _update_transition_matrix_blocked

    X, state, theta, mom, config = _setup(iterates, case, k)
    V0 = np.asarray(_update_transition_matrix_blocked(mom["sm"], mom["sP"], mom["pair"], state, config).V_0)
    w = np.linalg.eigvalsh(V0)
    assert w.min() >= -1e-14 * w.max()
    np.testing.assert_array_equal(V0, V0.T)
    np.testing.assert_allclose(V0, mom["sP"][0], rtol=0, atol=1e-8 * w.max())
    if not _is_pd(theta["V_0"]):
        pytest.skip("V_0 of theta_k is singular: its initial part of Q is not defined")
    if not _is_pd(V0):
        return
    new = dict(theta, Z_0=mom["sm"][0], V_0=V0)
    _assert_not_lower(q_parts(new, mom, X)[0], q_parts(theta, mom, X)[0], f"{case} k={k} Z_0/V_0")


@pytest.mark.parametrize("case,k", CASE_ITERS)
def test_em_step_increases_q_and_loglik(iterates, case, k):
    """The whole M-step raises Q, and the exact loglik of its output is not lower."""
    import dfm_python.functional.em as em

    X, state, theta, mom, config = _setup(iterates, case, k)
    logging.disable(logging.CRITICAL)
    try:
        new_state, ll, _ = em.em_step(X, state, config=config, num_iter=k)
    finally:
        logging.disable(logging.NOTSET)
    assert abs(ll - mom["ll"]) <= 1e-8 * abs(mom["ll"])
    new = _as_dict(new_state)
    # Transition and observation parts (the initial part is checked separately).
    q0, q1 = q_parts(theta, mom, X)[1:].sum(), q_parts(new, mom, X)[1:].sum()
    _assert_not_lower(q1, q0, f"{case} k={k} M-step")
    ll1 = ref.loglik(*(new[p] for p in PARAMS), X)
    assert ll1 >= mom["ll"] - 1e-8 * abs(mom["ll"]), (mom["ll"], ll1)


@pytest.mark.parametrize("case", list(_CASES))
def test_final_normalization_is_likelihood_invariant(iterates, case):
    """The post-EM C normalization changes state units only: same loglik, identity shift rows kept."""
    from dfm_python.functional.em import EMConfig, _normalize_C_with_invariance

    X, states = iterates[case]
    state = states[-1]
    theta = _as_dict(state)
    C = theta["C"].copy()
    r, L, f, _ = _layout(state)
    C[:, :f] *= 1.7  # make the factor columns non-unit so the normalization acts
    state = replace(state, C=C)
    C_new, scaled = _normalize_C_with_invariance(C, state, EMConfig())
    new = _as_dict(scaled)
    before = ref.loglik(*(np.asarray(getattr(state, p), dtype=np.float64) for p in PARAMS), X)
    after = ref.loglik(*(new[p] for p in PARAMS), X)
    assert abs(after - before) <= 1e-10 * abs(before), (before, after)
    offset = 0
    for r_b in r:
        np.testing.assert_allclose(np.linalg.norm(C_new[:, offset:offset + r_b], axis=0), 1.0, rtol=1e-12)
        shift = slice(offset + r_b, offset + r_b * L)
        np.testing.assert_allclose(new["A"][shift], np.asarray(state.A)[shift], rtol=1e-12, atol=1e-15)
        offset += r_b * L
    np.testing.assert_array_equal(C_new[:, f:], C[:, f:])

"""Tests of ``DFMKalmanFilter`` (per-element filter, Durbin-Koopman smoother, exact loglik).

The engine's DK smoother is checked against the independent reference's RTS
smoother (Cholesky solves on the predicted covariance) where that covariance
is non-singular, against the reference DK smoother and the brute-force joint
density where it is singular (deterministic lag states, as in the DFM
companion form), and against statsmodels when installed. Also: NaN and masked
input agree, a row with nothing observed is a pure prediction, the streaming
``filter_step`` equals the batch filter, and failures raise ``NumericalError``
instead of returning placeholders.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.utils.errors import NumericalError

from . import _reference_kf as ref


def _kf(A, C, Q, R, Z0, V0) -> DFMKalmanFilter:
    return DFMKalmanFilter(
        transition_matrices=A, observation_matrices=C, transition_covariance=Q,
        observation_covariance=R, initial_state_mean=Z0, initial_state_covariance=V0,
    )


def _companion_model(T: int = 14, seed: int = 0):
    """Two AR(2) factors in companion form: the lag states have zero process noise and V_0 is singular."""
    rng = np.random.default_rng(seed)
    m, N = 4, 5
    A = np.zeros((m, m))
    A[0, 0], A[0, 2] = 0.6, 0.2   # f1_t = 0.6 f1_{t-1} + 0.2 f1_{t-2}
    A[1, 1], A[1, 3] = 0.5, -0.1
    A[2, 0] = A[3, 1] = 1.0       # shift rows
    Q = np.diag([0.5, 0.3, 0.0, 0.0])
    C = np.zeros((N, m))
    C[:, :2] = rng.standard_normal((N, 2))
    C[0, 2] = 0.7                 # one series loads on a lag
    R = np.diag(rng.uniform(0.1, 0.5, N))
    Z0 = rng.standard_normal(m)
    V0 = np.diag([1.0, 0.8, 0.0, 0.0])
    Y = rng.standard_normal((T, N))
    Y[rng.random((T, N)) < 0.35] = np.nan
    Y[5] = np.nan
    return A, C, Q, R, Z0, V0, Y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dk_smoother_matches_reference_rts(seed):
    """Non-singular predicted covariances: DK (engine) equals RTS with solves (reference) to 1e-10."""
    model = ref.random_model(m=5, N=6, T=70, nan_frac=0.4, seed=seed)
    EZ, V, VV, ll = _kf(*model[:-1]).filter_and_smooth(model[-1])
    expected = ref.filter_smooth(*model, smoother="rts")
    assert ref.rel_err(EZ, expected["sm"]) < 1e-10
    assert ref.rel_err(V, expected["sP"]) < 1e-10
    assert ref.rel_err(VV, expected["pair"]) < 1e-10
    assert abs(ll - expected["ll"]) <= 1e-10 * abs(expected["ll"])


def test_singular_predicted_covariance_companion_form():
    """Deterministic lag states (zero Q rows, singular V_0): moments match the reference DK, loglik the brute force."""
    model = _companion_model()
    *params, Y = model
    EZ, V, VV, ll = _kf(*params).filter_and_smooth(Y)
    expected = ref.filter_smooth(*model, smoother="dk")
    assert ref.rel_err(EZ, expected["sm"]) < 1e-10
    assert ref.rel_err(V, expected["sP"]) < 1e-10
    assert ref.rel_err(VV, expected["pair"]) < 1e-10
    bf = ref.brute_force_loglik(*model)
    assert abs(ll - bf) <= 1e-10 * abs(bf)
    # The lag state is the previous factor: x_t[2] = x_{t-1}[0] exactly, so its smoothed mean is shifted.
    np.testing.assert_allclose(EZ[1:, 2], EZ[:-1, 0], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ["random_40pct_nan", "companion"])
def test_matches_statsmodels(name):
    """loglik, smoothed means and covariances and lag-one covariances equal statsmodels' KalmanSmoother."""
    pytest.importorskip("statsmodels")
    if name == "companion":
        model = _companion_model(T=40, seed=3)
    else:
        model = ref.random_model(m=4, N=5, T=60, nan_frac=0.4, seed=3)
    *params, Y = model
    EZ, V, VV, ll = _kf(*params).filter_and_smooth(Y)
    sm = ref.statsmodels_smooth(*model)
    assert abs(ll - sm["ll"]) <= 1e-9 * abs(sm["ll"])
    assert ref.rel_err(EZ, sm["sm"]) < 1e-8
    assert ref.rel_err(V, sm["sP"]) < 1e-8
    assert ref.rel_err(VV[1:], sm["pair"][1:]) < 1e-8


def test_partially_observed_rows_are_used():
    """A row with one missing entry still updates the state with its observed entries (no row dropping)."""
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=4, T=6, nan_frac=0.0, seed=8)
    Y = Y.copy()
    Y[2, 0] = np.nan
    kf = _kf(A, C, Q, R, Z0, V0)
    fm, fP = kf.filter(Y)
    expected = ref.kalman_filter(A, C, Q, R, Z0, V0, Y)
    np.testing.assert_allclose(fm, expected["fm"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fP, expected["fP"], rtol=1e-12, atol=1e-12)
    # Dropping the whole row would leave the filtered state at the prediction.
    assert not np.allclose(fm[2], expected["pm"][2])


def test_nan_and_masked_input_agree_and_loglik_is_returned_without_compute_loglik():
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=4, T=30, nan_frac=0.3, seed=9)
    kf = _kf(A, C, Q, R, Z0, V0)
    a = kf.filter_and_smooth(Y)
    masked = np.ma.masked_invalid(Y)
    b = kf.filter_and_smooth(masked, compute_loglik=False)
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    assert a[3] == b[3] == kf.loglikelihood(Y)
    # An explicit mask on finite values counts as missing too.
    Y2 = np.where(np.isnan(Y), 0.0, Y)
    np.testing.assert_array_equal(kf.filter(np.ma.MaskedArray(Y2, mask=np.isnan(Y)))[0], kf.filter(Y)[0])


def test_all_missing_rows_are_prediction_steps():
    A, C, Q, R, Z0, V0, _ = ref.random_model(m=3, N=4, T=5, nan_frac=0.0, seed=10)
    Y = np.full((5, 4), np.nan)
    EZ, V, VV, ll = _kf(A, C, Q, R, Z0, V0).filter_and_smooth(Y)
    assert ll == 0.0
    x, P = Z0.copy(), V0.copy()
    for t in range(5):
        if t:
            np.testing.assert_allclose(VV[t], A @ P, rtol=1e-12, atol=1e-12)  # Cov(x_t, x_{t-1}) = A Var(x_{t-1})
            x, P = A @ x, A @ P @ A.T + Q
        np.testing.assert_allclose(EZ[t], x, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(V[t], P, rtol=1e-12, atol=1e-14)


def test_streaming_filter_step_equals_batch_filter():
    """init_streaming() takes (Z_0, V_0) as the filtered state before the first row; each step predicts then updates."""
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=4, T=12, nan_frac=0.3, seed=11)
    kf = _kf(A, C, Q, R, Z0, V0)
    kf.init_streaming()
    np.testing.assert_array_equal(kf.streaming_state, Z0)
    states = [kf.filter_step(y)[0] for y in Y]
    batch = ref.kalman_filter(A, C, Q, R, A @ Z0, A @ V0 @ A.T + Q, Y)
    np.testing.assert_allclose(np.array(states), batch["fm"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(kf.streaming_cov, batch["fP"][-1], rtol=1e-10, atol=1e-12)
    predicted, _ = kf.filter_step(Y[0], reencode=False)
    np.testing.assert_allclose(predicted, A @ batch["fm"][-1], rtol=1e-12, atol=1e-14)


def test_non_finite_parameters_raise():
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=4, T=5, nan_frac=0.0, seed=12)
    A = A.copy()
    A[0, 0] = np.nan
    with pytest.raises(NumericalError, match="non-finite parameters"):
        _kf(A, C, Q, R, Z0, V0)
    with pytest.raises(ValueError, match="shape"):
        _kf(np.eye(3), C, Q, R, Z0, np.eye(2))


def test_singular_innovation_covariance_raises():
    """S_t = 0 (a series with no loading and no noise) raises NumericalError; no -T/2 or NaN placeholder."""
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=4, T=5, nan_frac=0.0, seed=13)
    C = C.copy()
    R = R.copy()
    C[1] = 0.0
    R[1, 1] = 0.0
    kf = _kf(A, C, Q, R, Z0, V0)
    with pytest.raises(NumericalError, match="not positive definite"):
        kf.filter_and_smooth(Y)


def test_diverging_filter_raises():
    """An explosive model whose covariances overflow raises NumericalError instead of returning inf or NaN."""
    m, N, T = 2, 3, 400
    A = np.eye(m) * 1e3
    C = np.ones((N, m))
    Q = np.eye(m)
    R = np.eye(N)
    Y = np.full((T, N), np.nan)
    Y[-1] = 1.0
    kf = _kf(A, C, Q, R, np.zeros(m), np.eye(m))
    with pytest.raises(NumericalError):
        kf.filter_and_smooth(Y)

"""In-situ -> potential temperature (Bryden 1973 / Fofonoff & Millard 1983).

``adiabatic_temperature_gradient`` (``ATG``) and ``potential_temperature``
(``THETA``) are transcriptions of a published algorithm, so the first
authority is the algorithm's own check values:

* ``ATG   = 3.255976e-4 degC/dbar``  at S=40, T=40 degC, p=10000 dbar
* ``THETA = 36.89073 degC``          at S=40, T=40 degC, p=10000 dbar, pr=0

Both are quoted in the routines' own headers (and in FESOM2's copy at
``oce_ale_pressure_bv.F90``), to 7 significant figures.

One check value per routine pins the module at ONE state, which cannot
separate a compensating pair of transcription errors from a correct
implementation, so the bulk of this file compares against constructions that
are algorithmically DIFFERENT from the module's own:

* ``_atg_expanded`` -- the same published coefficients written as a sum of
  monomials instead of nested Horner, so a misplaced parenthesis diverges;
* ``_theta_fine_rk4`` -- the same ODE integrated by CLASSICAL RK4 in 4000
  small steps, which shares none of the module's four-stage Gill weights;
* a round trip (``p -> p_ref`` then back) that a sign or ``h`` error breaks;
* ``d(theta)/d(p)`` at ``p == p_ref``, which must equal ``-ATG`` exactly.

The rest cover the p == p_ref identity, the sign and magnitude at
ocean-realistic states, jit/vmap parity, float32, and gradient finiteness
(the conversion sits in an initialisation path that must stay
differentiable).

Precision: the check values are quoted to ~1e-7 relative, so these run under
``jax_enable_x64``.  The module functions carry no ``resolve_dtype`` call --
their dtype follows their inputs -- so float64 inputs suffice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import (  # noqa: E402
    adiabatic_temperature_gradient,
    potential_temperature,
)

#: Published check state.
_S_CHK, _T_CHK, _P_CHK = 40.0, 40.0, 10000.0
_ATG_CHK = 3.255976e-4      # degC/dbar
_THETA_CHK = 36.89073       # degC


def _f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def test_atg_matches_published_check_value():
    got = float(adiabatic_temperature_gradient(
        _f64(_S_CHK), _f64(_T_CHK), _f64(_P_CHK)))
    # The check value is quoted to 7 significant figures.
    assert abs(got - _ATG_CHK) < 5e-11, (
        f"ATG={got:.7e} vs published {_ATG_CHK:.7e}")


def test_theta_matches_published_check_value():
    got = float(potential_temperature(
        _f64(_S_CHK), _f64(_T_CHK), _f64(_P_CHK), 0.0))
    # Quoted to 5 decimals; require agreement to half of the last digit.
    assert abs(got - _THETA_CHK) < 5e-6, (
        f"THETA={got:.6f} vs published {_THETA_CHK:.5f}")


def test_identity_when_reference_equals_in_situ_pressure():
    """p_ref == p means no adiabatic displacement, so theta == T exactly."""
    S = _f64([30.0, 34.5, 38.0])
    T = _f64([-1.5, 4.0, 28.0])
    p = _f64([0.0, 1500.0, 5000.0])
    theta = potential_temperature(S, T, p, 0.0)
    theta_self = jnp.stack([
        potential_temperature(S[i], T[i], p[i], float(p[i]))
        for i in range(3)
    ])
    np.testing.assert_allclose(np.asarray(theta_self), np.asarray(T),
                               rtol=0.0, atol=0.0)
    # And the p_ref=0 result differs from T wherever p > 0.
    assert float(theta[0]) == float(T[0])
    assert float(theta[2]) < float(T[2])


def test_correction_sign_and_magnitude_are_physical():
    """theta < T below the surface, growing to ~0.4-0.6 degC by 5000 m.

    Adiabatic compression warms a parcel in situ, so removing it must COOL.
    The published deep-ocean magnitude is ~0.1 degC/km; anything outside
    0.05-1.0 degC at 5000 m means a coefficient or an RK weight is wrong in a
    way the single check state (S=40, T=40) does not exercise.
    """
    S = _f64(34.7)
    T = _f64(2.0)
    for p, lo, hi in ((1000.0, 0.02, 0.20), (5000.0, 0.05, 1.0)):
        theta = float(potential_temperature(S, T, _f64(p), 0.0))
        drop = float(T) - theta
        assert lo < drop < hi, f"p={p} dbar: theta drop {drop:.4f} degC"
    # Monotone in pressure.
    p_ladder = _f64([0.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0])
    theta = np.asarray(potential_temperature(S, T, p_ladder, 0.0))
    assert np.all(np.diff(theta) < 0.0)


def test_jit_parity():
    S = _f64(np.linspace(30.0, 38.0, 7))
    T = _f64(np.linspace(-1.0, 30.0, 7))
    p = _f64(np.linspace(0.0, 6000.0, 7))
    eager = np.asarray(potential_temperature(S, T, p, 0.0))
    jitted = np.asarray(jax.jit(potential_temperature)(S, T, p, 0.0))
    np.testing.assert_allclose(jitted, eager, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("argnum", [0, 1])
def test_gradients_are_finite(argnum):
    """The conversion runs inside an initialisation path that must stay
    differentiable -- no branches, no data-dependent stopping."""
    def scalar(S, T):
        return jnp.sum(potential_temperature(S, T, _f64(3000.0), 0.0))

    g = jax.grad(scalar, argnums=argnum)(_f64(34.7), _f64(2.0))
    assert np.isfinite(float(g))
    # dtheta/dT is close to 1: the adiabatic correction is a small perturbation.
    if argnum == 1:
        assert 0.9 < float(g) < 1.1


# ---------------------------------------------------------------------------
# Independent references.  The two published check values pin the module at
# ONE state each, which cannot separate compensating or small transcription
# errors; these compare against constructions that are algorithmically
# different from the module's own.
# ---------------------------------------------------------------------------

def _atg_expanded(S, T_C, p):
    """ATG written as an explicit sum of monomials.

    Same published coefficients, DIFFERENT association than the module's
    nested Horner form, so a misplaced parenthesis diverges here while a
    single check value would not see it.
    """
    ds = S - 35.0
    t, pp = T_C, p
    return (
        3.5803e-5 + 8.5258e-6 * t - 6.836e-8 * t ** 2 + 6.6228e-10 * t ** 3
        + (1.8932e-6 - 4.2393e-8 * t) * ds
        + pp * (1.8741e-8 - 6.7795e-10 * t + 8.733e-12 * t ** 2
                - 5.4481e-14 * t ** 3)
        + pp * ds * (-1.1351e-10 + 2.7759e-12 * t)
        + pp ** 2 * (-4.6206e-13 + 1.8676e-14 * t - 2.1687e-16 * t ** 2)
    )


def _theta_fine_rk4(S, T_C, p_dbar, p_ref_dbar=0.0, n_steps=4000):
    """theta by CLASSICAL RK4 in many small steps.

    The module uses the four-stage Gill arrangement of Bryden (1973) in ONE
    step over the whole pressure interval; this integrates the same ODE with
    the textbook (1, 2, 2, 1)/6 weights over thousands of steps.  Agreement
    therefore tests the module's RK weights, not just its ATG.
    """
    h = (p_ref_dbar - p_dbar) / n_steps
    t, p = float(T_C), float(p_dbar)
    for _ in range(n_steps):
        k1 = float(_atg_expanded(S, t, p))
        k2 = float(_atg_expanded(S, t + 0.5 * h * k1, p + 0.5 * h))
        k3 = float(_atg_expanded(S, t + 0.5 * h * k2, p + 0.5 * h))
        k4 = float(_atg_expanded(S, t + h * k3, p + h))
        t = t + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        p = p + h
    return t


_STATES = [
    (30.0, -1.5, 500.0), (34.0, 2.0, 2000.0), (35.0, 10.0, 4000.0),
    (36.5, 20.0, 1000.0), (38.0, 28.0, 200.0), (40.0, 40.0, 10000.0),
    (33.0, 0.0, 6000.0),
]


@pytest.mark.parametrize("S,T,p", _STATES)
def test_atg_matches_an_independent_monomial_expansion(S, T, p):
    got = float(adiabatic_temperature_gradient(_f64(S), _f64(T), _f64(p)))
    ref = float(_atg_expanded(S, T, p))
    assert abs(got - ref) <= 1e-14 * max(abs(ref), 1e-12) + 1e-22, (
        f"S={S} T={T} p={p}: {got:.12e} vs {ref:.12e}")


@pytest.mark.parametrize("S,T,p", _STATES)
def test_theta_matches_fine_step_classical_rk4(S, T, p):
    """The one-step Gill scheme must agree with a converged integration.

    TOLERANCE, and why it is not tighter: the published routine takes ONE
    RK4 step across the whole pressure interval, so it carries a real
    truncation error that grows with depth -- measured 3.3e-5 degC at the
    S=40/T=40/p=10000 check state and 5.2e-6 degC at S=33/T=0/p=6000.  The
    PUBLISHED check value itself contains that error: the module returns
    36.8907265 (matching the quoted 36.89073), while the converged
    integration gives 36.8906933.  A tighter bound here would be asserting
    that the algorithm is something other than what it is.
    """
    got = float(potential_temperature(_f64(S), _f64(T), _f64(p), 0.0))
    ref = _theta_fine_rk4(S, T, p, 0.0)
    assert abs(got - ref) < 1e-4, f"S={S} T={T} p={p}: {got} vs {ref}"


@pytest.mark.parametrize("S,T,p", _STATES)
def test_round_trip_recovers_the_in_situ_temperature(S, T, p):
    """theta(S, T, p -> 0) then theta(S, ., 0 -> p) must return T.

    A sign error or a swapped ``h`` would break this while leaving the
    forward direction superficially plausible.
    """
    theta = potential_temperature(_f64(S), _f64(T), _f64(p), 0.0)
    back = float(potential_temperature(_f64(S), theta, _f64(0.0), float(p)))
    assert abs(back - T) < 1e-5, f"S={S} T={T} p={p}: round trip {back}"


def test_nonzero_reference_pressure():
    """p_ref != 0 must displace to THAT pressure, not to the surface."""
    S, T, p, p_ref = 34.5, 6.0, 4000.0, 1500.0
    got = float(potential_temperature(_f64(S), _f64(T), _f64(p), p_ref))
    ref = _theta_fine_rk4(S, T, p, p_ref)
    assert abs(got - ref) < 2e-6
    # And it must sit BETWEEN the in-situ value and the surface-referenced one.
    theta0 = float(potential_temperature(_f64(S), _f64(T), _f64(p), 0.0))
    assert theta0 < got < T


def test_vmap_matches_elementwise():
    S = _f64([30.0, 34.0, 38.0])
    T = _f64([-1.0, 5.0, 25.0])
    p = _f64([100.0, 2500.0, 5500.0])
    batched = np.asarray(jax.vmap(potential_temperature,
                                  in_axes=(0, 0, 0, None))(S, T, p, 0.0))
    direct = np.asarray(potential_temperature(S, T, p, 0.0))
    np.testing.assert_allclose(batched, direct, rtol=0.0, atol=0.0)


def test_float32_is_finite_and_close():
    """The IC path may run under the float32 precision policy; the result
    must stay finite and within single-precision distance of the fp64 answer.
    """
    S32 = jnp.asarray([34.7, 35.0], dtype=jnp.float32)
    T32 = jnp.asarray([2.0, 12.0], dtype=jnp.float32)
    p32 = jnp.asarray([3000.0, 500.0], dtype=jnp.float32)
    got = np.asarray(potential_temperature(S32, T32, p32, 0.0))
    assert got.dtype == np.float32
    assert np.all(np.isfinite(got))
    ref = np.asarray(potential_temperature(
        _f64([34.7, 35.0]), _f64([2.0, 12.0]), _f64([3000.0, 500.0]), 0.0))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-4)


def test_derivative_wrt_pressure_equals_minus_atg_at_the_reference():
    """d(theta)/d(p_in-situ) at p == p_ref is -ATG: displacing the parcel one
    dbar further down adds one dbar of adiabatic compression.

    This is a consistency identity -- the RK stage weights must sum to 1 --
    so it holds only as well as those weights do.  They are published as
    truncated decimals (0.29289322 for 1 - 1/sqrt(2), and so on), which caps
    the agreement at ~1e-9 relative; measured 7.6e-10 here.  A tolerance
    tight enough to see roundoff would be testing the decimal truncation,
    not the implementation.
    """
    S, T, p = _f64(34.6), _f64(3.0), 1200.0

    def theta_of_p(p_in):
        return potential_temperature(S, T, p_in, p)

    d = float(jax.grad(theta_of_p)(_f64(p)))
    atg = float(adiabatic_temperature_gradient(S, T, _f64(p)))
    assert abs(d + atg) < 1e-8 * abs(atg)

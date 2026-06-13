"""Fast-SBM analytic supersaturation step (oracle JERSUPSAT_KS warm branch).

Pins the port's single expm1 formulation against BOTH oracle branches
written verbatim (|R dt| > 1e-6 closed form; small-x 5-term Taylor), the
ballistic R=0 branch, equilibrium, an independent numerical ODE solve, and
the oracle's RW psychrometric factor with its rounded constants.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.supersaturation import (
    integrate_supersaturation,
    supersat_relaxation_rate,
)

jax.config.update("jax_enable_x64", True)


def _oracle_large_x(S0, R, F, dt):
    # Oracle JERSUPSAT_KS, |RW*DT| > 1e-6 water-only branch, verbatim.
    e = np.exp(-R * dt)
    del1n = S0 * e + (F / R) * (1.0 - e)
    del1int = (-S0 * e / R + F * dt / R + F * e / (R * R)
               + S0 / R - F / (R * R))
    return del1n, del1int


def _oracle_small_x(S0, R, F, dt):
    # Oracle small-x branch with its EXPM1 statement function (5-term
    # Taylor of e^x - 1 evaluated at x = -R dt), verbatim.
    x = -R * dt
    expr = x + x**2 / 2.0 + x**3 / 6.0 + x**4 / 24.0 + x**5 / 120.0
    del1n = S0 + S0 * expr + (F / R) * (0.0 - expr)
    del1int = -S0 * expr / R + F * dt / R + F * expr / (R * R)
    return del1n, del1int


def test_matches_oracle_large_x_branch():
    S0, R, F, dt = 0.005, 2.0, 1.0e-3, 0.4
    out = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(R),
                                    jnp.asarray(F), dt)
    ref_n, ref_int = _oracle_large_x(S0, R, F, dt)
    assert float(out.S_new) == pytest.approx(ref_n, rel=1e-13)
    assert float(out.S_int) == pytest.approx(ref_int, rel=1e-13)


def test_matches_oracle_small_x_branch():
    S0, R, F, dt = 0.005, 1.0e-7, 1.0e-3, 0.4   # R dt = 4e-8 << 1e-6
    out = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(R),
                                    jnp.asarray(F), dt)
    ref_n, ref_int = _oracle_small_x(S0, R, F, dt)
    assert float(out.S_new) == pytest.approx(ref_n, rel=1e-12)
    # The oracle's S_int expression cancels two ~F·dt/R terms (~4e3) down
    # to ~2e-3 — its own float64 roundoff is ~1e-9 relative here. The
    # port's cancellation-free form is the more accurate of the two; the
    # tolerance covers the ORACLE's noise, not the port's.
    assert float(out.S_int) == pytest.approx(ref_int, rel=1e-8)


def test_ballistic_limit_R_zero():
    # Oracle IRW==0 branch: DEL1N = DEL1 + DYN1*DT, DEL1INT = DEL1*DT +
    # DYN1*DT²/2 — must come out of the same formula at R = 0 exactly.
    S0, F, dt = -0.002, 5.0e-4, 0.4
    out = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(0.0),
                                    jnp.asarray(F), dt)
    assert float(out.S_new) == pytest.approx(S0 + F * dt, rel=1e-14)
    assert float(out.S_int) == pytest.approx(S0 * dt + F * dt * dt / 2.0,
                                             rel=1e-14)


def test_equilibrium_and_decay():
    S0, R, F = 0.01, 50.0, 2.0e-2
    out = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(R),
                                    jnp.asarray(F), 10.0)
    assert float(out.S_new) == pytest.approx(F / R, rel=1e-10)
    # F = 0: pure decay, S_int -> S0/R as dt -> inf.
    out0 = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(R),
                                     jnp.asarray(0.0), 10.0)
    assert float(out0.S_new) == pytest.approx(0.0, abs=1e-12)
    assert float(out0.S_int) == pytest.approx(S0 / R, rel=1e-10)


def test_against_numerical_ode():
    S0, R, F, dt = 0.004, 3.7, -2.0e-3, 0.4
    n_sub = 200_000
    h = dt / n_sub
    s = S0
    s_int = 0.0
    for _ in range(n_sub):           # midpoint RK2 + midpoint quadrature
        k1 = -R * s + F
        sm = s + 0.5 * h * k1
        s_int += h * sm
        s = s + h * (-R * sm + F)
    out = integrate_supersaturation(jnp.asarray(S0), jnp.asarray(R),
                                    jnp.asarray(F), dt)
    assert float(out.S_new) == pytest.approx(s, rel=1e-7)
    assert float(out.S_int) == pytest.approx(s_int, rel=1e-6)


def test_relaxation_rate_matches_oracle_factor():
    # Oracle ONECOND1: RW = (OPER2(q) + B5L*AL1)*(1+S)*SFN with the
    # oracle's own rounded constants, written verbatim below.
    T, q, S, sfn = 283.0, 8.0e-3, 0.005, 0.7
    R = float(supersat_relaxation_rate(jnp.asarray(T), jnp.asarray(q),
                                       jnp.asarray(S), jnp.asarray(sfn)))
    oper2 = 0.622 / (0.622 + 0.378 * q) / q  # const-ok: oracle OPER2 statement function, verbatim fidelity reference
    rw_oracle = (oper2 + (5.42e3 / T**2) * 2500.0) * (1.0 + S) * sfn
    # Derived L_v/R_v and L_v/c_pd vs the oracle's 5.42e3/2500 roundings:
    # agreement must be sub-percent, not exact.
    assert R == pytest.approx(rw_oracle, rel=7e-3)
    # And exactly equal to the constants-derived expression.
    eps = constants.epsilon
    expect = ((eps / ((eps + (1 - eps) * q) * q))
              + constants.L_v / (constants.R_v * T**2)
              * constants.L_v / constants.c_pd) * (1 + S) * sfn
    assert R == pytest.approx(expect, rel=1e-14)


def test_differentiable_and_batched():
    S = jnp.linspace(-0.01, 0.01, 8)
    R = jnp.linspace(0.0, 5.0, 8)        # includes R = 0 exactly
    F = jnp.full((8,), 1.0e-3)
    out = integrate_supersaturation(S, R, F, 0.4)
    assert out.S_new.shape == (8,)
    assert np.all(np.isfinite(np.asarray(out.S_new)))

    def loss(params):
        S0, R0, F0 = params
        o = integrate_supersaturation(S0, R0, F0, 0.4)
        return o.S_new + o.S_int

    for R0 in (0.0, 1.0e-13, 1.0e-6, 2.0):   # straddles the small-x guard
        g = jax.grad(loss)(jnp.array([0.005, R0, 1.0e-3]))
        assert np.all(np.isfinite(np.asarray(g))), R0


def test_gradient_in_R_correct_at_zero():
    # Codex review: a constant-limit guard would zero d/dR near R=0. The
    # series guard must reproduce the analytic derivative
    #   d S_new/dR|_{R=0} = -S0 dt²/2·... — checked against central
    # finite differences far below the |x|<1e-4 switch.
    S0, F, dt = 0.005, 1.0e-3, 0.4

    def s_new(R):
        return integrate_supersaturation(jnp.asarray(S0), R,
                                         jnp.asarray(F), dt).S_new

    def s_int(R):
        return integrate_supersaturation(jnp.asarray(S0), R,
                                         jnp.asarray(F), dt).S_int

    for fn in (s_new, s_int):
        g0 = float(jax.grad(fn)(jnp.asarray(0.0)))
        h = 1.0e-7
        fd = (float(fn(jnp.asarray(h))) - float(fn(jnp.asarray(-h)))) \
            / (2.0 * h)
        assert g0 == pytest.approx(fd, rel=1e-6)
        assert g0 != 0.0

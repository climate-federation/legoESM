"""Fast-SBM warm condensation step (oracle ONECOND1) — the simple case.

Supersaturated parcel condenses with exact water/enthalpy closure and
supersaturation relaxing toward equilibrium; subsaturated cloudy parcel
evaporates; multi-step run drives S toward 0 monotonically in magnitude.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    discretize_lognormal,
    mass_density,
    mass_doubling_grid,
    number_density,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.condensation_driver import (
    warm_condensation_step,
)
from legoesm.thermo import relative_humidity, saturation_vapor_pressure

jax.config.update("jax_enable_x64", True)

P0 = 9.0e4
T0 = 283.0
DT = 0.5


def _cloudy_state(rh_target):
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)   # ~100 cm^-3, 10 um
    # Mixing ratio giving the requested saturation ratio (invert e(q)).
    e = rh_target * float(saturation_vapor_pressure(jnp.asarray(T0)))
    q = constants.epsilon * e / (P0 - e)
    return m, f, jnp.asarray(T0), jnp.asarray(q), jnp.asarray(P0), \
        jnp.asarray(1.1)


def test_supersaturated_condenses_with_exact_closure():
    m, f, T, q, p, rho = _cloudy_state(1.02)     # 2% supersaturation
    out = warm_condensation_step(f, T, q, p, rho, DT, m)
    assert float(out.dq_c) > 0.0
    assert float(out.T) > float(T)
    assert float(out.q_v) < float(q)
    # Closures to roundoff.
    lwc_q = lambda fx: float(mass_density(fx, m)) / float(rho)
    assert (float(out.q_v) + lwc_q(out.f)) == pytest.approx(
        float(q) + lwc_q(f), rel=1e-13)
    assert float(constants.c_pd * (out.T - T)) == pytest.approx(
        float(-constants.L_v * (out.q_v - q)), rel=1e-12)
    # Supersaturation decayed toward equilibrium, not past it.
    S0 = float(relative_humidity(T, p, q)) - 1.0
    assert 0.0 <= float(out.S_new) < S0
    assert float(out.min_psi) >= 0.0


def test_subsaturated_evaporates():
    m, f, T, q, p, rho = _cloudy_state(0.90)     # RH 90%
    out = warm_condensation_step(f, T, q, p, rho, DT, m)
    assert float(out.dq_c) < 0.0
    assert float(out.T) < float(T)
    assert float(out.q_v) > float(q)
    # Number cannot increase under evaporation.
    assert float(number_density(out.f, m)) <= \
        float(number_density(f, m)) * (1 + 1e-12)


def test_saturated_empty_spectrum_fixed_point():
    m, _, T, _, p, rho = _cloudy_state(1.0)
    e_s = float(saturation_vapor_pressure(jnp.asarray(T0)))
    q_sat_exact = jnp.asarray(float(constants.epsilon) * e_s / (P0 - e_s))
    f0 = jnp.zeros_like(m)
    out = warm_condensation_step(f0, T, q_sat_exact, p, rho, DT, m)
    # No spectrum → SFN = 0 → R = 0 → S frozen → no growth at S≈0.
    np.testing.assert_array_equal(np.asarray(out.f), 0.0)
    assert float(out.dq_c) == pytest.approx(0.0, abs=1e-18)
    assert float(out.T) == pytest.approx(float(T), abs=1e-12)


def test_multistep_relaxes_supersaturation():
    m, f, T, q, p, rho = _cloudy_state(1.03)

    def body(carry, _):
        f, T, q = carry
        out = warm_condensation_step(f, T, q, p, rho, DT, m)
        return (out.f, out.T, out.q_v), out.S_new

    (_, T_end, q_end), S_hist = jax.lax.scan(
        jax.jit(body), (f, T, q), None, length=40)
    S_hist = np.asarray(S_hist)
    # Magnitude decreases monotonically (no oscillation past equilibrium).
    assert np.all(np.diff(np.abs(S_hist)) <= 1e-12)
    # Ends close to saturation (S << S0).
    assert abs(S_hist[-1]) < 0.1 * 0.03
    # Sanity: warmed and dried.
    assert float(T_end) > T0


def test_driver_differentiable_in_state():
    m, f, T, q, p, rho = _cloudy_state(1.02)

    def dqc_of(x):
        T0_, q0_ = x
        out = warm_condensation_step(f, T0_, q0_, p, rho, DT, m)
        return out.dq_c

    g = jax.grad(dqc_of)(jnp.array([float(T), float(q)]))
    assert np.all(np.isfinite(np.asarray(g)))
    # More vapor → more condensation.
    assert float(g[1]) > 0.0

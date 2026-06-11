"""Unit tests for SDM sedimentation + surface rain accumulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    SuperDropletState,
    column_rainout,
    sediment_step,
    terminal_velocity_rogers_yau,
)

_PREF = 4.0 / 3.0 * np.pi * constants.rho_water


def _column(radii, xi=1.0e6, z=100.0):
    n = len(radii)
    o = jnp.ones((n,))
    state = SuperDropletState(
        multiplicity=o * xi,
        radius=jnp.asarray(radii),
        solute_mass=o * 0.0,
        active=o,
    )
    return state, jnp.full((n,), z)


def test_single_step_fall_distance_and_no_precip():
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    state, z = _column([2.0e-5, 5.0e-5], z=500.0)
    dt, area = 2.0, 1.0
    state2, z2, dp = sediment_step(state, z, 1.0, 9.0e4, 283.0, dt, area, cfg)
    v_exp = np.asarray(terminal_velocity_rogers_yau(state.radius))
    assert np.allclose(np.asarray(z - z2), v_exp * dt, rtol=1e-12)
    assert float(dp) == 0.0                       # nothing crossed yet
    assert np.all(np.asarray(state2.active) == 1.0)


def test_crossing_deposits_exact_mass_and_deactivates():
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    R = 1.0e-4
    xi = 5.0e5
    state, z = _column([R], xi=xi, z=1.0)        # 1 m up, v_t ~ 1.2 m/s
    area = 2.0
    state2, z2, dp = sediment_step(state, z, 1.0, 9.0e4, 283.0, 1.0, area, cfg)
    m_rep = xi * _PREF * R**3
    assert float(dp) == pytest.approx(m_rep / area, rel=1e-12)
    assert float(state2.active[0]) == 0.0
    assert float(z2[0]) == 0.0
    # second step: no double counting from the deactivated droplet
    state3, z3, dp2 = sediment_step(state2, z2, 1.0, 9.0e4, 283.0, 1.0, area, cfg)
    assert float(dp2) == 0.0


def test_column_rainout_conserves_water_and_orders_arrivals():
    """All drops rain out; airborne+precip conserved every step; the largest
    drop (fastest Stokes fall) arrives first."""
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    radii = [3.0e-5, 6.0e-5, 1.2e-4]             # v_t ratios 1:4:16
    xi, z0, area = 1.0e6, 50.0, 1.0
    state, z = _column(radii, xi=xi, z=z0)
    m_tot = float(jnp.sum(state.multiplicity
                          * _PREF * state.radius**3))
    dt, n_steps = 0.5, 2000                       # slowest needs z0/v_t ~ 450 s
    final, z_f, precip, hist = column_rainout(
        state, z, 1.0, 9.0e4, 283.0, dt, n_steps, area, cfg)
    # exact total rainout
    assert float(precip) * area == pytest.approx(m_tot, rel=1e-12)
    assert float(jnp.sum(final.active)) == 0.0
    # conservation at every step: airborne + precip*area = m_tot
    airborne = np.asarray(hist["airborne"])
    precip_h = np.asarray(hist["precip"])
    assert np.allclose(airborne + precip_h * area, m_tot, rtol=1e-12)
    # arrival ordering: n_active drops monotonically 3 -> 2 -> 1 -> 0, and the
    # first arrival time matches the largest drop's z0/v_t
    n_active = np.asarray(hist["n_active"])
    assert np.all(np.diff(n_active) <= 0)
    t_first = (np.argmax(n_active < 3) + 1) * dt
    v_big = float(terminal_velocity_rogers_yau(jnp.asarray(1.2e-4)))
    assert t_first == pytest.approx(z0 / v_big, abs=2 * dt)


def test_inactive_droplets_do_not_fall_or_deposit():
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    state, z = _column([1.0e-4], z=0.5)
    state = state._replace(active=jnp.zeros_like(state.active))
    state2, z2, dp = sediment_step(state, z, 1.0, 9.0e4, 283.0, 1.0, 1.0, cfg)
    assert float(dp) == 0.0
    assert float(z2[0]) == 0.5                    # height untouched


def test_jit_with_static_cfg():
    cfg = SDMConfig(terminal_velocity="cloud_rain_shima")
    state, z = _column([2.0e-5, 8.0e-5], z=200.0)
    run = jax.jit(column_rainout, static_argnames=("n_steps", "cfg"))
    final, z_f, precip, hist = run(state, z, 1.0, 9.0e4, 283.0, 1.0, 50, 1.0, cfg)
    assert jnp.isfinite(precip)
    assert np.all(np.isfinite(np.asarray(hist["airborne"])))

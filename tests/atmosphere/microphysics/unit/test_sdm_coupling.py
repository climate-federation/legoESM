"""Unit tests for SDM particle<->grid coupling (deposition, latent heat, q_t)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    SuperDropletState,
    cloud_rain_mixing_ratios,
    condensation_exchange,
    liquid_water_content,
    make_monodisperse,
)

_PREF = 4.0 / 3.0 * np.pi * constants.rho_water


def test_liquid_water_content_value():
    n_sd, R, xi, V = 10, 1.0e-5, 1.0e6, 2.0
    st = make_monodisperse(n_sd, R, xi)
    m = _PREF * R**3
    expected = n_sd * xi * m / V
    assert float(liquid_water_content(st, V)) == pytest.approx(expected, rel=1e-12)


def test_cloud_rain_split_by_threshold():
    # one cloud droplet (R<r_rain) and one rain drop (R>=r_rain)
    R_cloud, R_rain = 1.0e-5, 1.0e-3
    st = SuperDropletState(
        multiplicity=jnp.array([1.0e6, 1.0e3]),
        radius=jnp.array([R_cloud, R_rain]),
        solute_mass=jnp.array([0.0, 0.0]),
        active=jnp.array([1.0, 1.0]),
    )
    V, rho_air, r_rain = 1.0, 1.0, 4.0e-5
    q_c, q_r = cloud_rain_mixing_ratios(st, V, rho_air, r_rain)
    q_c_exp = 1.0e6 * _PREF * R_cloud**3 / (V * rho_air)
    q_r_exp = 1.0e3 * _PREF * R_rain**3 / (V * rho_air)
    assert float(q_c) == pytest.approx(q_c_exp, rel=1e-12)
    assert float(q_r) == pytest.approx(q_r_exp, rel=1e-12)
    # total deposited liquid == LWC/rho_air
    assert float(q_c + q_r) == pytest.approx(
        float(liquid_water_content(st, V)) / rho_air, rel=1e-12)


def test_inactive_droplets_not_deposited():
    st = make_monodisperse(5, 1.0e-5, 1.0e6)
    st_off = st._replace(active=jnp.zeros_like(st.active))
    assert float(liquid_water_content(st_off, 1.0)) == 0.0
    q_c, q_r = cloud_rain_mixing_ratios(st_off, 1.0, 1.0, 4.0e-5)
    assert float(q_c) == 0.0 and float(q_r) == 0.0


def test_condensation_exchange_conserves_total_water_and_heats():
    q_v, T = 1.0e-2, 290.0
    q_l0, q_l1 = 1.0e-3, 1.6e-3      # 0.6 g/kg condenses
    q_v_new, T_new = condensation_exchange(
        jnp.asarray(q_l0), jnp.asarray(q_l1), jnp.asarray(q_v), jnp.asarray(T))
    dq = q_l1 - q_l0
    # total water q_v + q_l conserved
    assert float(q_v_new) + q_l1 == pytest.approx(q_v + q_l0, rel=1e-12)
    assert float(q_v_new) == pytest.approx(q_v - dq, rel=1e-12)
    # latent heating ΔT = L_v/c_pd · Δq
    assert float(T_new) == pytest.approx(T + constants.L_v / constants.c_pd * dq, rel=1e-12)
    assert float(T_new) > T  # condensation warms


def test_evaporation_exchange_cools_and_moistens():
    q_v_new, T_new = condensation_exchange(
        jnp.asarray(2.0e-3), jnp.asarray(1.0e-3), jnp.asarray(1.0e-2), jnp.asarray(290.0))
    assert float(q_v_new) > 1.0e-2   # vapor increases
    assert float(T_new) < 290.0      # air cools


def test_coupling_jit_and_grad():
    st = make_monodisperse(16, 1.2e-5, 1.0e6)

    @jax.jit
    def lwc(radius):
        return liquid_water_content(st._replace(radius=radius), 1.5)

    g = jax.grad(lambda r: lwc(r))(st.radius)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g > 0.0)  # more liquid as radii grow

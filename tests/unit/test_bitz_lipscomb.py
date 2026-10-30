"""Unit tests for the Bitz-Lipscomb (1999) sea-ice thermodynamic core.

Pins the physics of ``ice/bitz_lipscomb.py`` (finding F-ICE-1,
the multi-layer enthalpy replacement for the Semtner-0 single skin
node), ahead of its prognostic-state integration:

* enthalpy ↔ temperature inversion is exact;
* salinity-dependent conductivity (brine lowers k) and effective heat
  capacity (brine latent storage spikes near the melting point);
* **exact column energy conservation** over a conduction step — the
  defining BL99 property;
* finite gradients (AD-safe) through the implicit solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ice.bitz_lipscomb import (
    freezing_temperature,
    ice_enthalpy,
    ice_temperature_from_enthalpy,
    ice_thermal_conductivity,
    ice_specific_heat,
    bitz_lipscomb_conduction_step,
)

_Tf = constants.T_freeze


def test_enthalpy_temperature_roundtrip():
    """T → q → T recovers the temperature (exact quadratic inverse)."""
    T = jnp.array([_Tf - 0.5, _Tf - 1.0, _Tf - 5.0, _Tf - 15.0, _Tf - 30.0])
    S = jnp.full_like(T, 4.0)
    q = ice_enthalpy(T, S)
    assert jnp.all(q < 0.0)                       # ice enthalpy is negative
    T_rt = ice_temperature_from_enthalpy(q, S)
    assert float(jnp.max(jnp.abs(T_rt - T))) < 1e-6


def test_conductivity_brine_lowers_k():
    """k(S=0) = k_0; brine (S>0) lowers conductivity (Untersteiner)."""
    T = jnp.full((2,), _Tf - 2.0)
    k = ice_thermal_conductivity(T, jnp.array([0.0, 8.0]))
    assert float(k[0]) == pytest.approx(constants.k_ice_default, rel=1e-6)
    assert float(k[1]) < float(k[0])


def test_specific_heat_brine_spikes_near_melt():
    """Effective heat capacity exceeds fresh-ice c_pi and grows toward
    the melting point (brine latent-heat storage)."""
    c = ice_specific_heat(jnp.array([_Tf - 10.0, _Tf - 1.0]), jnp.full((2,), 4.0))
    assert float(c[0]) > constants.c_pi
    assert float(c[1]) > float(c[0])             # larger near melting


def test_freezing_point_depression():
    """T_m = −μS depresses below 0 °C with salinity."""
    Tm = freezing_temperature(jnp.array([0.0, 4.0]))
    assert float(Tm[0]) == 0.0
    assert float(Tm[1]) == pytest.approx(-constants.mu_ice_freeze * 4.0)


@pytest.mark.parametrize("F_top", [[-30.0, 0.0, 50.0], [-100.0, 20.0, 5.0]])
def test_conduction_step_conserves_energy(F_top):
    """Column enthalpy change equals the net boundary flux to machine
    precision: Σ(q_new − q_old)·dz = (F_top − F_bottom)·dt."""
    n, ncol = 4, 3
    T0 = jnp.stack([jnp.linspace(_Tf - 12.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 2.0)
    q0 = ice_enthalpy(T0, S[:, None])
    F = jnp.asarray(F_top)
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)
    dt = 3600.0

    q_new, T_new, F_bottom = bitz_lipscomb_conduction_step(q0, h, S, F, T_bottom, dt)
    dz = h / n
    column_denthalpy = jnp.sum((q_new - q0) * dz[:, None], axis=1)
    net_flux = (F - F_bottom) * dt
    rel = jnp.abs(column_denthalpy - net_flux) / (jnp.abs(net_flux) + 1.0)
    assert float(jnp.max(rel)) < 1e-9
    assert jnp.all(jnp.isfinite(T_new))


def test_conduction_step_directional():
    """A warming surface flux warms the top layer; a cooling flux cools it."""
    n, ncol = 4, 2
    T0 = jnp.stack([jnp.linspace(_Tf - 12.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 2.0)
    q0 = ice_enthalpy(T0, S[:, None])
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)
    _, T_new, _ = bitz_lipscomb_conduction_step(
        q0, h, S, jnp.array([80.0, -80.0]), T_bottom, 3600.0,
    )
    assert float(T_new[0, 0]) > float(T0[0, 0])   # warmed
    assert float(T_new[1, 0]) < float(T0[1, 0])   # cooled


def test_conduction_step_ad_safe():
    """Gradients through the implicit conduction solve are finite."""
    n, ncol = 4, 2
    T0 = jnp.stack([jnp.linspace(_Tf - 10.0, _Tf - 2.0, n)] * ncol)
    S = jnp.full((ncol,), 4.0)
    h = jnp.full((ncol,), 1.5)
    q0 = ice_enthalpy(T0, S[:, None])
    T_bottom = jnp.full((ncol,), constants.T_freeze_ocean)

    def loss(flux_scalar):
        F = jnp.full((ncol,), flux_scalar)
        q_new, _, _ = bitz_lipscomb_conduction_step(q0, h, S, F, T_bottom, 3600.0)
        return jnp.sum(q_new ** 2)

    g = jax.grad(loss)(10.0)
    assert jnp.isfinite(g)


def test_si3_column_step_reverse_mode_finite_on_open_water_and_bare_ice():
    """Review F3: open water (no ice, no lead freezing) and snow-free ice hit
    0/0 in DISCARDED where-branches of the SI3 chain (new-ice thickness,
    layer-enthalpy remap); the reverse pass must stay finite there."""
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        ice_enthalpy_from_temperature,
        option2_salinity_profile,
        si3_column_step_arrays,
        snow_enthalpy_from_temperature,
    )
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as c

    def a(*v):
        return jnp.asarray(v, dtype=jnp.float64)

    # columns: open water | snow-free ice | snow-covered ice
    conc, h_i, h_s = a(0.0, 0.9, 0.9), a(0.0, 2.0, 2.0), a(0.0, 0.0, 0.2)
    bulk = a(0.0, 6.3, 6.3)
    sal = option2_salinity_profile(bulk, a(34.7, 34.7, 34.7))
    ti = jnp.stack([a(271.0, 271.0, 271.0), a(257.0, 261.0, 266.0),
                    a(257.0, 261.0, 266.0)])
    ts = jnp.stack([a(271.0, 271.0, 271.0)] * 2 + [a(255.0, 260.0, 266.0)])
    e_i = ice_enthalpy_from_temperature(ti, sal, c) * (h_i > 0)[:, None]
    e_s = snow_enthalpy_from_temperature(ts, c) * (h_s > 0)[:, None]
    state = SI3ColumnArrays(conc, h_i, h_s, a(271.0, 260.0, 260.0), e_i, e_s,
                            bulk, sal, jnp.zeros(3))
    z3 = jnp.zeros(3)
    forcing = SI3SurfaceForcing(
        qns_ice=a(-60.0, -60.0, -60.0), qsr_ice=z3, dqns_ice=a(-7.0, -7.0, -7.0),
        qtr_ice_top=z3, t_bottom=a(271.27, 271.27, 271.27),
        sss=a(34.7, 34.7, 34.7), evaporation=z3, snow_precipitation=z3,
        qprec_ice=z3, qcn_ice_bottom=z3, qsb_ice_bottom=z3, fhld=z3,
        qlead=z3)   # qlead >= 0: no lead freezing, so open water stays ice-free

    def loss(qns, h):
        out = si3_column_step_arrays(state._replace(h_ice=h),
                                     forcing._replace(qns_ice=qns),
                                     3600.0, c).exit
        return (jnp.sum(out.h_ice) + jnp.sum(out.concentration)
                + 1e-9 * jnp.sum(out.e_ice) + 1e-9 * jnp.sum(out.e_snow))

    out = si3_column_step_arrays(state, forcing, 3600.0, c).exit
    assert float(out.concentration[0]) == 0.0 and float(out.h_snow[1]) == 0.0
    g_q, g_h = jax.grad(loss, argnums=(0, 1))(forcing.qns_ice, state.h_ice)
    assert bool(jnp.all(jnp.isfinite(g_q))), g_q
    assert bool(jnp.all(jnp.isfinite(g_h))), g_h
    assert float(jnp.abs(g_h[1:]).max()) > 0.0     # ice columns stay live


def test_piecewise_remap_reverse_mode_finite_at_zero_thickness():
    """Review F3, the layer-enthalpy remap alone: a zero new thickness is
    0/0 in the discarded branch; the gradient must stay finite (zero)."""
    from legoesm.ice import bitz_lipscomb as bl

    th = jnp.asarray([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    en = jnp.asarray([[-3.0e8, -2.9e8, -2.8e8], [0.0, 0.0, 0.0]])
    g = jax.grad(lambda t: jnp.sum(bl._piecewise_remap(th, en, t)))(
        jnp.asarray([1.5, 0.0]))
    assert bool(jnp.all(jnp.isfinite(g))), g
    assert float(g[1]) == 0.0


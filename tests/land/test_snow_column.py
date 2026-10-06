"""Multi-layer snow-column (_future) conservation + physics validation.

Covers: the equal-mass remap conservation (water + enthalpy), water-mass
conservation over a run, the FULL energy budget through liquid heat capacity /
melt / refreeze / boundary fluxes / drainage (including the all-liquid,
above-freezing drainage edge case), accumulation, melt->drainage, empty-pack
safety, and JIT + differentiability.
"""

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.snow_column import (
    SnowColumnConfig,
    SnowColumnState,
    _enthalpy,
    _remap_equal_mass,
    column_enthalpy,
    initial_snow_state,
    step_snow_column,
    total_water,
)

jax.config.update("jax_enable_x64", True)

CFG = SnowColumnConfig()
_W = 3.0   # wind speed [m/s] for fresh-snow density and drift
_LF = constants.L_f
_TF = constants.T_freeze


def _packed_state(ncol=3, n=5, key=0):
    """A partly-melting pack: positive ice, some liquid, T straddling freezing."""
    k = jax.random.PRNGKey(key)
    k1, k2, k3 = jax.random.split(k, 3)
    ice = 2.0 + jax.random.uniform(k1, (ncol, n))
    liq = 0.1 * jax.random.uniform(k2, (ncol, n))
    T = _TF - 5.0 + 6.0 * jax.random.uniform(k3, (ncol, n))
    return SnowColumnState(swe_ice=ice, swe_liq=liq, T=T, density=jnp.full((ncol, n), 200.0))


def test_remap_conserves_water_and_enthalpy():
    """The equal-mass remap conserves total WATER and total ENTHALPY (the
    ice/liquid split re-equilibrates from the conserved pair)."""
    s = _packed_state()
    ice2, liq2, T2, _ = _remap_equal_mass(s.swe_ice, s.swe_liq, s.T, s.density)
    assert jnp.allclose(jnp.sum(ice2 + liq2, -1),
                        jnp.sum(s.swe_ice + s.swe_liq, -1), rtol=1e-10)
    assert jnp.allclose(jnp.sum(_enthalpy(ice2, liq2, T2), -1),
                        jnp.sum(_enthalpy(s.swe_ice, s.swe_liq, s.T), -1), rtol=1e-9)
    m = ice2 + liq2
    assert jnp.allclose(m, m[..., :1], rtol=1e-9)   # equal SWE-mass layers
    assert jnp.all((liq2 < 1e-9) | (T2 >= _TF - 1e-6))   # no liquid below freezing
    assert jnp.all((ice2 < 1e-9) | (T2 <= _TF + 1e-6))   # no ice above freezing


def test_water_mass_conservation_with_drainage():
    """total_water_final == init + sum(snowfall*dt) - sum(drainage)."""
    s = _packed_state()
    dt, ncol = 1800.0, 3
    snowfall = jnp.full((ncol,), 1e-4)
    T_air = jnp.full((ncol,), _TF - 2.0)
    Q_top = jnp.full((ncol,), 60.0)
    w0 = total_water(s)
    total_drain = jnp.zeros((ncol,))
    for _ in range(20):
        s, drain, _ = step_snow_column(s, snowfall, T_air, Q_top, jnp.zeros((ncol,)), dt, CFG, wind=_W)
        total_drain = total_drain + drain
    assert jnp.allclose(total_water(s), w0 + snowfall * dt * 20 - total_drain,
                        atol=1e-8, rtol=1e-9)
    assert jnp.all(total_drain >= 0.0)


def _run_energy(s, Q_top, G_bottom, nsteps, dt=1800.0, snowfall=None, T_air=None, cfg=CFG):
    ncol = s.swe_ice.shape[0]
    if snowfall is None:
        snowfall = jnp.zeros((ncol,))
    if T_air is None:
        T_air = jnp.full((ncol,), _TF)
    H0 = column_enthalpy(s)
    tot_drain_H = jnp.zeros((ncol,))
    for _ in range(nsteps):
        s, _, dH = step_snow_column(s, snowfall, T_air, Q_top, G_bottom, dt, cfg, wind=_W)
        tot_drain_H = tot_drain_H + dH
    dH_actual = column_enthalpy(s) - H0
    # No snowfall -> dH = (Q_top - G_bottom)*dt*nsteps - drainage_heat.
    expected = (Q_top - G_bottom) * dt * nsteps - tot_drain_H
    return dH_actual, expected


def test_energy_budget_closes_with_liquid_and_drainage():
    """Column enthalpy change == boundary heat - drainage enthalpy, for a
    liquid-bearing pack undergoing melt / refreeze / drainage."""
    dH, expected = _run_energy(_packed_state(), jnp.full((3,), 50.0), jnp.full((3,), 8.0), 15)
    assert jnp.allclose(dH, expected, rtol=1e-6, atol=1e-1)


def test_energy_budget_all_liquid_above_freezing_drainage():
    """Edge case: strong warming (Q >> L_f) makes a layer all-liquid ABOVE
    freezing, then it drains — the sensible heat of the drained water must be
    in ``drainage_heat`` for the budget to close."""
    s = SnowColumnState(swe_ice=jnp.array([[1.0]]), swe_liq=jnp.array([[0.0]]),
                        T=jnp.array([[_TF]]), density=jnp.array([[200.0]]))
    cfg = SnowColumnConfig(n_layers=1)
    Q = jnp.array([_LF + 2.0e4])            # more than enough to melt + superheat
    dH, expected = _run_energy(s, Q, jnp.zeros((1,)), 1, dt=1.0, cfg=cfg)
    assert jnp.allclose(dH, expected, rtol=1e-8, atol=1e-3)


def test_accumulation_adds_swe():
    s = initial_snow_state((2,), CFG)
    snowfall = jnp.full((2,), 5e-4)
    s, _, _ = step_snow_column(s, snowfall, jnp.full((2,), _TF - 10.0),
                               jnp.zeros((2,)), jnp.zeros((2,)), 3600.0, CFG, wind=_W)
    assert jnp.allclose(total_water(s), snowfall * 3600.0, atol=1e-8)


def test_melt_produces_drainage():
    s = _packed_state()
    ice0 = jnp.sum(s.swe_ice, -1)
    s2, drain, _ = step_snow_column(s, jnp.zeros((3,)), jnp.full((3,), _TF),
                                    jnp.full((3,), 300.0), jnp.zeros((3,)), 3600.0, CFG, wind=_W)
    assert jnp.all(jnp.sum(s2.swe_ice, -1) <= ice0 + 1e-9)   # ice melted, not created
    assert jnp.all(drain >= 0.0)


def test_empty_pack_is_safe():
    s = initial_snow_state((4,), CFG)
    s2, _, _ = step_snow_column(s, jnp.zeros((4,)), jnp.full((4,), 250.0),
                                jnp.zeros((4,)), jnp.zeros((4,)), 3600.0, CFG, wind=_W)
    assert jnp.all(jnp.isfinite(s2.T)) and jnp.all(jnp.isfinite(s2.swe_ice))
    assert jnp.allclose(total_water(s2), 0.0, atol=1e-12)


def test_jit_and_differentiable():
    """Static-N under jit (config closed over) + grad flow."""
    s = _packed_state()
    step = jax.jit(lambda st, q: step_snow_column(
        st, jnp.zeros((3,)), jnp.full((3,), _TF), q, jnp.zeros((3,)), 3600.0, CFG, wind=_W))
    out = step(s, jnp.full((3,), 40.0))[0]
    assert jnp.all(jnp.isfinite(out.T))
    g = jax.grad(lambda q: jnp.mean(step(s, jnp.full((3,), q))[0].T))(40.0)
    assert jnp.isfinite(g)

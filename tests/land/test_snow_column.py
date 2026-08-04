"""Multi-layer snow-column conservation + physics validation.

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
    apply_sublimation,
    column_enthalpy,
    initial_snow_state,
    pack_top_temperature,
    snow_base_interface_conductance,
    step_snow_column,
    total_water,
)

jax.config.update("jax_enable_x64", True)

CFG = SnowColumnConfig()
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
        s, drain, _ = step_snow_column(s, snowfall, T_air, Q_top, jnp.zeros((ncol,)), dt, CFG)
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
        s, _, dH = step_snow_column(s, snowfall, T_air, Q_top, G_bottom, dt, cfg)
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
                               jnp.zeros((2,)), jnp.zeros((2,)), 3600.0, CFG)
    assert jnp.allclose(total_water(s), snowfall * 3600.0, atol=1e-8)


def test_melt_produces_drainage():
    s = _packed_state()
    ice0 = jnp.sum(s.swe_ice, -1)
    s2, drain, _ = step_snow_column(s, jnp.zeros((3,)), jnp.full((3,), _TF),
                                    jnp.full((3,), 300.0), jnp.zeros((3,)), 3600.0, CFG)
    assert jnp.all(jnp.sum(s2.swe_ice, -1) <= ice0 + 1e-9)   # ice melted, not created
    assert jnp.all(drain >= 0.0)


def test_empty_pack_is_safe():
    s = initial_snow_state((4,), CFG)
    s2, _, _ = step_snow_column(s, jnp.zeros((4,)), jnp.full((4,), 250.0),
                                jnp.zeros((4,)), jnp.zeros((4,)), 3600.0, CFG)
    assert jnp.all(jnp.isfinite(s2.T)) and jnp.all(jnp.isfinite(s2.swe_ice))
    assert jnp.allclose(total_water(s2), 0.0, atol=1e-12)


def test_apply_sublimation_conserves_mass_and_enthalpy():
    """Sublimation removes ICE mass and exactly its sensible enthalpy; the returned
    delta_H closes the column enthalpy budget.  Deposition (negative) adds frost."""
    s = _packed_state()
    subl = jnp.array([1.0, 0.5, 0.0])            # kg/m^2 removed this step
    w0, H0 = total_water(s), column_enthalpy(s)
    s2, dH = apply_sublimation(s, subl, CFG)
    # Mass: total water drops by exactly the removed ice mass (liquid untouched).
    assert jnp.allclose(total_water(s2), w0 - subl, atol=1e-10)
    # Enthalpy: column change equals the reported delta_H (budget-closing term).
    assert jnp.allclose(column_enthalpy(s2) - H0, dH, atol=1e-6)
    # Never removes more ice than present; result stays finite/non-negative.
    assert jnp.all(s2.swe_ice >= -1e-12) and jnp.all(jnp.isfinite(s2.T))

    # Deposition (frost): negative subl adds ice to the top layer.
    dep = jnp.array([-0.3, 0.0, 0.0])
    s3, dH3 = apply_sublimation(s, dep, CFG)
    assert jnp.allclose(total_water(s3), w0 - dep, atol=1e-10)   # -dep = +0.3 added
    assert jnp.allclose(column_enthalpy(s3) - H0, dH3, atol=1e-6)
    assert s3.swe_ice[0, 0] > s.swe_ice[0, 0]                    # added to TOP layer


def test_apply_sublimation_capped_at_available_ice():
    """A sublimation demand exceeding the pack ice removes all of it, no more."""
    s = _packed_state()
    total_ice = jnp.sum(s.swe_ice, axis=-1)
    s2, _ = apply_sublimation(s, total_ice + 5.0, CFG)   # over-demand
    assert jnp.allclose(jnp.sum(s2.swe_ice, axis=-1), 0.0, atol=1e-9)
    assert jnp.all(s2.swe_ice >= -1e-12)


def test_base_conductance_and_top_temperature():
    """Base interface conductance is positive/finite and rises with density (Sturm);
    pack_top_temperature returns the top layer's T."""
    s = _packed_state()
    g = snow_base_interface_conductance(s, CFG)
    assert jnp.all(g > 0.0) and jnp.all(jnp.isfinite(g))
    # Denser pack conducts better (k ~ (rho/rho_ref)^exp, exp>0).
    s_dense = s._replace(density=s.density * 1.5)
    assert jnp.all(snow_base_interface_conductance(s_dense, CFG) > g)
    assert jnp.allclose(pack_top_temperature(s), s.T[..., 0])


def test_jit_and_differentiable():
    """Static-N under jit (config closed over) + grad flow."""
    s = _packed_state()
    step = jax.jit(lambda st, q: step_snow_column(
        st, jnp.zeros((3,)), jnp.full((3,), _TF), q, jnp.zeros((3,)), 3600.0, CFG))
    out = step(s, jnp.full((3,), 40.0))[0]
    assert jnp.all(jnp.isfinite(out.T))
    g = jax.grad(lambda q: jnp.mean(step(s, jnp.full((3,), q))[0].T))(40.0)
    assert jnp.isfinite(g)

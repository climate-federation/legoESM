"""One-layer snow thermal node (``SoilThermalConfig.snow_insulation``).

Pins: the Jordan (1991) conductivity CLM5 uses; snow-free columns bit-identical
to the plain soil solve; the snow node's flux balance with the whole-pack
resistance; the column enthalpy ledger (boundary fluxes + snow-mass change at
the node's start temperature); insulation of a warm soil under a cold surface;
water conservation of a full land step that melts a warm pack from storage;
the restart initialisation; jit parity and finite, FD-checked gradients.
Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, init_snow_temperature, land_skin_temperature,
    step_multilayer_land)
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import (
    SoilThermalConfig, compute_apparent_heat_capacity, compute_heat_capacity,
    compute_thermal_conductivity,
    melt_snow_node_excess, snow_thermal_conductivity, solve_snow_soil_thermal,
    solve_soil_thermal)
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

_TF = constants.T_freeze
_GRID = make_soil_grid(SoilGridConfig(n_layers=10, total_depth=3.0))
_HYD = SoilHydraulicsConfig()
_TH = SoilThermalConfig(snow_insulation=True)


def _column(ncol=3):
    T = jnp.linspace(268.0, 276.0, 10)[None, :] + jnp.arange(ncol)[:, None]
    theta = jnp.full((ncol, 10), 0.25)
    return T, theta


def _enthalpy(T_soil, theta, T_snow, swe):
    C = compute_heat_capacity(theta, _HYD, _TH)
    return (jnp.sum(C * _GRID.dz * (T_soil - _TF), axis=-1)
            + constants.c_pi * swe * (T_snow - _TF))


def test_jordan_conductivity_is_clm5():
    rho = 250.0
    clm5 = 0.023 + (7.75e-5 * rho + 1.105e-6 * rho * rho) * (2.290 - 0.023)
    assert abs(float(snow_thermal_conductivity(rho)) - clm5) < 1e-12
    assert 0.22 < clm5 < 0.23


def test_snow_free_columns_bit_identical_to_soil_solve():
    T, theta = _column()
    G = jnp.array([-30.0, 5.0, 40.0])
    lam = jnp.full(3, 8.0)
    ref = solve_soil_thermal(T, theta, _GRID, _HYD, _TH, G, 1800.0,
                             surface_conductance=lam)
    Ts, Tn = solve_snow_soil_thermal(T[:, 0] + 3.0, jnp.zeros(3), T, theta,
                                     _GRID, _HYD, _TH, G, 1800.0,
                                     surface_conductance=lam)
    np.testing.assert_array_equal(np.asarray(Tn), np.asarray(ref))
    np.testing.assert_array_equal(np.asarray(Ts), np.asarray(ref[:, 0]))


def test_snow_node_flux_balance_uses_whole_pack_resistance():
    """C_s (T_s' - T_s)/dt = G - g (T_s' - T_0') with g = 1/(R_s + dz0/(2 k0)),
    R_s = (swe/rho)/k_Jordan -- computed here independently of the solver."""
    T, theta = _column()
    swe = jnp.array([1.0, 20.0, 60.0])
    T_s0 = jnp.array([260.0, 255.0, 250.0])
    G = jnp.array([-20.0, -35.0, 10.0])
    dt = 1800.0
    Ts, Tn = solve_snow_soil_thermal(T_s0, swe, T, theta, _GRID, _HYD, _TH, G, dt)
    rho = _TH.snow_bulk_density_kg_m3
    R = swe / rho / snow_thermal_conductivity(rho)
    k0 = compute_thermal_conductivity(theta, _HYD, _TH)[:, 0]
    g = 1.0 / (R + 0.5 * _GRID.dz[0] / k0)
    lhs = constants.c_pi * swe * (Ts - T_s0) / dt
    rhs = G - g * (Ts - Tn[:, 0])
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-8)


@pytest.mark.parametrize("n_sub,lam,ft", [(1, None, False), (1, 6.0, False),
                                         (4, None, False), (1, 6.0, True)])
def test_column_enthalpy_ledger(n_sub, lam, ft):
    """dE = (G - lam (T_s'-T_s) + Q_geo) dt + (swe' - swe) c_ice (T_s - Tf).
    (With sub-steps the linearised surface term is integrated per sub-step,
    so lam is checked on the one-step solve.)"""
    T, theta = _column()
    swe_old = jnp.array([10.0, 40.0, 0.0])       # col 2: pack created this step
    swe_new = jnp.array([12.0, 35.0, 3.0])
    T_s0 = jnp.array([262.0, 258.0, float(T[2, 0])])   # created at the skin T
    G = jnp.array([-25.0, 15.0, -5.0])
    dt = 1800.0
    sc = None if lam is None else jnp.full(3, lam)
    th = _TH._replace(enable_freeze_thaw=ft)
    Ts, Tn = solve_snow_soil_thermal(T_s0, swe_new, T, theta, _GRID, _HYD, th,
                                     G, dt, surface_conductance=sc,
                                     n_substeps=n_sub)
    flux = G + _TH.Q_geothermal
    if lam is not None:
        flux = flux - lam * (Ts - T_s0)
    if ft:
        # Freeze/thaw: the solver's identity is in the linearised metric (the
        # apparent heat capacity at the start temperature), one step.
        C = compute_apparent_heat_capacity(T, theta, _HYD, th)
        dE = (jnp.sum(C * _GRID.dz * (Tn - T), axis=-1)
              + constants.c_pi * swe_new * (Ts - _TF)
              - constants.c_pi * swe_old * (T_s0 - _TF))
    else:
        dE = (_enthalpy(Tn, theta, Ts, swe_new)
              - _enthalpy(T, theta, T_s0, swe_old))
    expect = flux * dt + (swe_new - swe_old) * constants.c_pi * (T_s0 - _TF)
    np.testing.assert_allclose(np.asarray(dE), np.asarray(expect),
                               rtol=1e-10, atol=1e-4)


def test_deep_snow_insulates_warm_soil():
    """30 days under 240 K air with a Robin surface exchange (10 W/m2/K): under
    54 kg/m2 of snow the soil at ~17.5 cm stays far warmer than bare, and the
    snow surface sits colder than the bare soil top.  (A PRESCRIBED surface
    flux would drain the same energy from both columns -- insulation acts only
    because the exchange depends on the surface temperature.)"""
    lam, T_air = 10.0, 240.0
    theta = jnp.full((2, 10), 0.25)
    swe = jnp.array([0.0, 54.0])

    def body(_, carry):
        Ts, T = carry
        return solve_snow_soil_thermal(
            Ts, swe, T, theta, _GRID, _HYD, _TH, lam * (T_air - Ts), 1800.0,
            surface_conductance=jnp.full(2, lam))

    T0 = jnp.full((2, 10), 272.0)
    Ts, T = jax.jit(lambda c: jax.lax.fori_loop(0, 30 * 48, body, c))(
        (T0[:, 0], T0))
    k = int(np.argmin(np.abs(np.asarray(_GRID.z_node) - 0.175)))
    assert float(T[1, k] - T[0, k]) > 5.0
    assert float(Ts[1]) < float(Ts[0])


def test_restart_init_caps_under_snow_and_mirrors_elsewhere():
    cfg = MultiLayerLandConfig(thermal=_TH)
    st = init_multilayer_land_state(3, cfg, T_init=278.0)
    np.testing.assert_array_equal(np.asarray(st.T_snow), np.asarray(st.T_soil[:, 0]))
    st = st._replace(T_snow=None, snow_depth=jnp.array([0.0, 5.0, 5.0]),
                     T_soil=st.T_soil.at[2, 0].set(265.0))
    out = init_snow_temperature(st)
    np.testing.assert_allclose(np.asarray(out.T_snow), [278.0, _TF, 265.0])
    assert land_skin_temperature(out) is out.T_snow
    off = init_multilayer_land_state(3, MultiLayerLandConfig(), T_init=278.0)
    assert off.T_snow is None


def _forcing(ncol, T_air, precip, precip_snow):
    o = jnp.ones(ncol)
    p_s = 1.0e5 * o
    return AtmToSurface(
        sw_down=60.0 * o, lw_down=250.0 * o, precip_total=precip * o,
        precip_snow=precip_snow * o, T_lowest=T_air * o, q_lowest=0.002 * o,
        u_lowest=4.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.3 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


@pytest.mark.parametrize("scheme", ["seb", "two_leaf"])
def test_full_step_conserves_water_with_warm_pack(scheme):
    """Snow + soil + pond water closes over steps that melt a pack whose node
    starts above freezing (the stored-heat melt) and accumulate new snow."""
    kw = {"surface_scheme": (SimpleSEBConfig() if scheme == "seb"
                             else TwoLeafCanopyConfig())}
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH, **kw)
    ncol = 2
    st = init_multilayer_land_state(ncol, cfg, T_init=279.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0, 2.0]),
                     T_snow=jnp.array([278.0, 276.0]))
    f = _forcing(ncol, 268.0, 3.0e-5, 3.0e-5)
    dz = jnp.asarray(_GRID.dz)
    dt = 1800.0

    def W(s):
        return (jnp.sum(dz * s.theta_soil, axis=-1) + s.surface_water) * \
            constants.rho_water + s.snow_depth

    W0 = W(st)
    s = st
    out = 0.0
    melted_first = None
    step = jax.jit(lambda s_: step_multilayer_land(
        s_, f, cfg, 1.0, dt, lat=jnp.full(ncol, 1.0)))
    for i in range(6):
        s2, r, _ = step(s)
        if i == 0:
            melted_first = s.snow_depth + f.precip_snow * dt - s2.snow_depth
        out = out + (r.surface_mass_flux + s2.runoff_surface
                     + s2.runoff_subsurface) * dt
        s = s2
    P = f.precip_total * dt * 6
    resid = W(s) - W0 - (P - out)
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-6)
    # Stored heat above freezing melted snow on the first step.
    want = constants.c_pi * jnp.array([20.0, 2.0]) * jnp.array([278.0, 276.0]) \
        - constants.c_pi * jnp.array([20.0, 2.0]) * _TF
    assert np.all(np.asarray(melted_first) >= 0.9 * np.asarray(want) / constants.L_f)
    assert np.all(np.isfinite(np.asarray(s.T_snow)))


def test_full_step_jit_parity():
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH)
    st = init_multilayer_land_state(2, cfg, T_init=270.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([0.0, 30.0]),
                     T_snow=jnp.array([270.0, 262.0]))
    f = _forcing(2, 262.0, 1.0e-5, 1.0e-5)

    def run(s):
        return step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                    lat=jnp.full(2, 1.0))[0]

    e = run(st)
    j = jax.jit(run)(st)
    np.testing.assert_allclose(np.asarray(j.T_snow), np.asarray(e.T_snow), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(j.T_soil), np.asarray(e.T_soil), rtol=1e-12)


def test_solver_gradient_matches_finite_difference():
    T, theta = _column(2)

    def loss(swe):
        Ts, Tn = solve_snow_soil_thermal(jnp.array([258.0, 258.0]), swe, T,
                                         theta, _GRID, _HYD, _TH,
                                         jnp.array([-30.0, -30.0]), 1800.0,
                                         n_substeps=2)
        return jnp.sum(Tn[:, 1]) + jnp.sum(Ts)

    swe = jnp.array([15.0, 40.0])
    g = jax.jit(jax.grad(loss))(swe)
    eps = 1e-4
    fd = [(loss(swe.at[i].add(eps)) - loss(swe.at[i].add(-eps))) / (2 * eps)
          for i in range(2)]
    assert np.all(np.isfinite(np.asarray(g))) and np.all(np.abs(np.asarray(g)) > 0)
    np.testing.assert_allclose(np.asarray(g), np.asarray(fd), rtol=1e-6)


def test_surface_scheme_sees_the_snow_node_temperature():
    """The start-of-step skin handed to the surface scheme is the snow node:
    two states differing ONLY in T_snow (under snow) give different sensible
    heat.  SimpleSEB evaluates its fluxes at that skin directly (the two-leaf
    canopy relaxes its ground node toward the thermal callback instead)."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=SimpleSEBConfig())
    st = init_multilayer_land_state(1, cfg, T_init=270.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([30.0]))
    f = _forcing(1, 262.0, 0.0, 0.0)
    step = jax.jit(lambda s: step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                                  lat=jnp.full(1, 1.0))[1])
    h_cold = step(st._replace(T_snow=jnp.array([250.0]))).shflx
    h_warm = step(st._replace(T_snow=jnp.array([265.0]))).shflx
    assert float(h_warm[0] - h_cold[0]) > 10.0


def test_snow_fallen_this_step_gets_its_own_node():
    """Snow falling on bare ground forms a pack in the step's FINAL solve (end-
    of-step mass): the node then decouples from the top soil."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH)
    st = init_multilayer_land_state(1, cfg, T_init=271.0, theta_init=0.25)
    f = _forcing(1, 255.0, 2.0e-3, 2.0e-3)      # 3.6 kg/m2 of snow in 30 min
    s2 = jax.jit(lambda s: step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                                lat=jnp.full(1, 1.0))[0])(st)
    assert float(s2.snow_depth[0]) > 1.0
    assert abs(float(s2.T_snow[0] - s2.T_soil[0, 0])) > 1e-3


def test_stored_heat_melt_conserves_enthalpy_and_leaves_pack_at_freezing():
    """c_ice S (T - Tf) == m L_f + leftover; the remaining pack sits at Tf;
    leftover heat only where the whole pack melted; cold packs untouched."""
    S = jnp.array([20.0, 0.05, 10.0, 0.0])
    T = jnp.array([278.0, 290.0, 260.0, 280.0])
    S2, T2, m, heat = melt_snow_node_excess(S, T)
    E = constants.c_pi * S * jnp.maximum(T - _TF, 0.0)
    np.testing.assert_allclose(np.asarray(m * constants.L_f + heat),
                               np.asarray(E), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.asarray(S2 + m), np.asarray(S), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(T2[:2]), _TF)
    assert float(heat[0]) == 0.0 and float(heat[1]) == 0.0
    np.testing.assert_array_equal(np.asarray(T2[2:]), np.asarray(T[2:]))
    assert float(m[2]) == 0.0 and float(m[3]) == 0.0
    # Whole-pack melt with heat to spare: needs c S dT > L_f S, i.e. dT > 158 K
    S3, T3, m3, h3 = melt_snow_node_excess(jnp.array([1.0]), jnp.array([_TF + 200.0]))
    assert float(S3[0]) == 0.0 and float(h3[0]) > 0.0


def test_full_step_gradient_wrt_snow_mass_is_finite_and_matches_fd():
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH)
    st = init_multilayer_land_state(1, cfg, T_init=270.0, theta_init=0.25)
    f = _forcing(1, 258.0, 0.0, 0.0)

    def loss(swe):
        s = st._replace(snow_depth=swe, T_snow=jnp.array([262.0]))
        s2 = step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                  lat=jnp.full(1, 1.0))[0]
        return s2.T_soil[0, 1] + s2.T_snow[0]

    swe = jnp.array([30.0])
    g = jax.jit(jax.grad(loss))(swe)
    lj = jax.jit(loss)
    eps = 1e-3
    fd = (lj(swe + eps) - lj(swe - eps)) / (2 * eps)
    assert np.isfinite(float(g[0])) and abs(float(g[0])) > 0.0
    np.testing.assert_allclose(float(g[0]), float(fd), rtol=1e-4)


def test_canopy_ground_node_is_the_snow_surface():
    """Two-leaf canopy: the Picard loop's converged ground temperature is the
    snow-node solve (close to the final T_snow), not the warm top soil."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=272.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([40.0]), T_snow=jnp.array([255.0]))
    f = _forcing(1, 252.0, 0.0, 0.0)
    out = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(1, 1.0)))(st)
    s2, sfc = out[0], out[-1]
    Tg = float(sfc.Ts_solve[0])
    assert abs(Tg - float(s2.T_snow[0])) < 1.0
    assert float(s2.T_soil[0, 0]) - Tg > 5.0


def test_flux_heats_the_node_and_melt_follows_from_its_enthalpy():
    """Enthalpy-method melt: a pack starting at freezing is not melted by the
    surface flux in the same step (so it can never melt AND end below
    freezing); the heat it gained above freezing melts exactly
    c_ice*S*(T - Tf)/L_f at the next step; the exported skin stays <= Tf."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=SimpleSEBConfig())
    st = init_multilayer_land_state(1, cfg, T_init=262.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0]), T_snow=jnp.array([_TF]))
    o = jnp.ones(1)
    f = _forcing(1, 283.0, 0.0, 0.0)._replace(sw_down=500.0 * o,
                                               lw_down=330.0 * o,
                                               q_lowest=0.008 * o)
    dt = 1800.0
    step = jax.jit(lambda s: step_multilayer_land(s, f, cfg, 1.0, dt,
                                                  lat=jnp.full(1, 1.0)))
    s1, r1, _ = step(st)
    melt1 = st.snow_depth - s1.snow_depth - r1.surface_mass_flux * dt
    np.testing.assert_allclose(np.asarray(melt1), 0.0, atol=1e-9)
    assert float(r1.T_sfc[0]) <= _TF
    assert float(s1.T_snow[0]) > _TF            # flux heated the node
    s2, r2, _ = step(s1)
    melt2 = s1.snow_depth - s2.snow_depth - r2.surface_mass_flux * dt
    want = constants.c_pi * s1.snow_depth * (s1.T_snow - _TF) / constants.L_f
    np.testing.assert_allclose(np.asarray(melt2), np.asarray(want), rtol=1e-9)


def test_full_step_energy_ledger_under_melting_conditions():
    """Assembled land enthalpy over two SimpleSEB steps of warm air on a pack:
    dE_sens = (G_soil - lam*(T_s' - T_s0) + Q_geo) dt - L_f * m_store,
    E_sens = sum C dz (T - Tf) + c_ice SWE (T_snow - Tf); snow-mass changes
    other than the stored-heat melt happen at a node already at Tf (no term)."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=SimpleSEBConfig())
    st = init_multilayer_land_state(1, cfg, T_init=262.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0]), T_snow=jnp.array([_TF]))
    o = jnp.ones(1)
    f = _forcing(1, 283.0, 0.0, 0.0)._replace(sw_down=500.0 * o,
                                               lw_down=330.0 * o,
                                               q_lowest=0.008 * o)
    dt = 1800.0
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, dt, lat=jnp.full(1, 1.0)))

    def E(s):
        C = compute_heat_capacity(s.theta_soil, _HYD, cfg.thermal)
        return (jnp.sum(C * _GRID.dz * (s.T_soil - _TF), axis=-1)
                + constants.c_pi * s.snow_depth * (s.T_snow - _TF))

    s = st
    for _ in range(2):
        m_store = constants.c_pi * s.snow_depth * jnp.maximum(
            s.T_snow - _TF, 0.0) / constants.L_f
        T_s0 = jnp.minimum(s.T_snow, _TF)
        out = step(s)
        s2, sfc = out[0], out[-1]
        lam = sfc.surface_conductance
        flux = sfc.G_soil - lam * (s2.T_snow - T_s0) + cfg.thermal.Q_geothermal
        expect = flux * dt - constants.L_f * m_store
        # Water moved by Richards changes C (theta) at fixed T: compare at the
        # end-of-step theta for the soil part.
        dE = E(s2) - E(s._replace(theta_soil=s2.theta_soil))
        np.testing.assert_allclose(np.asarray(dE), np.asarray(expect),
                                   rtol=1e-9, atol=1e-3)
        s = s2

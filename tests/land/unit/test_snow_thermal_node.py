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
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

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
    ref = solve_soil_thermal(T, theta, _GRID, _HYD, _TH, G, 1800.0)
    Ts, Tn = solve_snow_soil_thermal(T[:, 0] + 3.0, jnp.zeros(3), T, theta,
                                     _GRID, _HYD, _TH, G, 1800.0)
    np.testing.assert_array_equal(np.asarray(Tn), np.asarray(ref))
    np.testing.assert_array_equal(np.asarray(Ts), np.asarray(ref[:, 0]))


def test_semi_implicit_surface_is_refused():
    T, theta = _column()
    with pytest.raises(ValueError, match="surface_conductance"):
        solve_snow_soil_thermal(T[:, 0], jnp.ones(3), T, theta, _GRID, _HYD,
                                _TH, jnp.zeros(3), 1800.0,
                                surface_conductance=jnp.ones(3))


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


@pytest.mark.parametrize("n_sub,ft", [(1, False), (4, False), (1, True)])
def test_column_enthalpy_ledger(n_sub, ft):
    """dE = (G + Q_geo) dt + (swe' - swe) c_ice (T_s - Tf)."""
    T, theta = _column()
    swe_old = jnp.array([10.0, 40.0, 0.0])       # col 2: pack created this step
    swe_new = jnp.array([12.0, 35.0, 3.0])
    T_s0 = jnp.array([262.0, 258.0, float(T[2, 0])])   # created at the skin T
    G = jnp.array([-25.0, 15.0, -5.0])
    dt = 1800.0
    th = _TH._replace(enable_freeze_thaw=ft)
    Ts, Tn = solve_snow_soil_thermal(T_s0, swe_new, T, theta, _GRID, _HYD, th,
                                     G, dt, n_substeps=n_sub)
    flux = G + _TH.Q_geothermal
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
    """30 days under 240 K air with a surface exchange of 2 W/m2/K: under
    54 kg/m2 of snow the soil at ~17.5 cm stays far warmer than bare, and the
    snow surface sits colder than the bare soil top.  (A PRESCRIBED surface
    flux would drain the same energy from both columns -- insulation acts only
    because the exchange depends on the surface temperature.)"""
    lam, T_air = 2.0, 240.0
    theta = jnp.full((2, 10), 0.25)
    swe = jnp.array([0.0, 54.0])

    def body(_, carry):
        Ts, T = carry
        return solve_snow_soil_thermal(
            Ts, swe, T, theta, _GRID, _HYD, _TH, lam * (T_air - Ts), 1800.0)

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
    np.testing.assert_array_equal(np.asarray(land_skin_temperature(out)),
                                  np.asarray(out.T_snow))
    warm = out._replace(T_snow=jnp.array([280.0, 280.0, 280.0]))
    np.testing.assert_allclose(np.asarray(land_skin_temperature(warm)),
                               [278.0, _TF, _TF])
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


def test_full_step_conserves_water_with_warm_pack():
    """Snow + soil + pond water closes over steps that melt a pack whose node
    starts above freezing (the stored-heat melt) and accumulate new snow."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
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
    c_ice*S*(T - Tf)/L_f at the next step; the exported surface humidity is
    ice saturation at the capped skin (T_freeze)."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    from legoesm.thermo import saturation_mixing_ratio_ice
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=262.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0]), T_snow=jnp.array([_TF]))
    o = jnp.ones(1)
    f = _forcing(1, 285.0, 0.0, 0.0)._replace(sw_down=700.0 * o,
                                               lw_down=340.0 * o,
                                               cos_zenith=0.8 * o,
                                               q_lowest=0.006 * o)
    dt = 1800.0
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, dt, lat=jnp.full(1, 0.8)))

    def melt(s0, out):
        s1, r1, sfc = out[0], out[1], out[-1]
        transp = sfc.lhflx - sfc.LE_soil
        assert float(transp[0]) >= 0.0          # no canopy dew onto the pack
        sublim = sfc.LE_soil / constants.L_s
        return s0.snow_depth - s1.snow_depth - sublim * dt, s1, r1

    m1, s1, r1 = melt(st, step(st))
    np.testing.assert_allclose(np.asarray(m1), 0.0, atol=1e-9)
    assert float(s1.T_snow[0]) > _TF            # flux heated the node
    np.testing.assert_allclose(
        np.asarray(r1.q_surface),
        np.asarray(saturation_mixing_ratio_ice(jnp.array([_TF]), f.p_surface)),
        rtol=1e-12)
    m2, s2, _ = melt(s1, step(s1))
    want = constants.c_pi * s1.snow_depth * (s1.T_snow - _TF) / constants.L_f
    np.testing.assert_allclose(np.asarray(m2), np.asarray(want), rtol=1e-9)


def test_full_step_energy_ledger_under_melting_conditions():
    """Assembled land enthalpy over two two-leaf steps of warm air on a pack:
    dE_sens = (G_soil + latent_excess + Q_geo) dt - L_f * m_store, with
    E_sens = sum C dz (T - Tf) + c_ice SWE (T_snow - Tf); snow-mass changes
    other than the stored-heat melt happen at a node already at Tf (no term).
    ``latent_excess`` = latent demand the reservoirs could not meet, returned
    to the ground (scheme lhflx minus the delivered lhflx)."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=262.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0]), T_snow=jnp.array([_TF]))
    o = jnp.ones(1)
    f = _forcing(1, 285.0, 0.0, 0.0)._replace(sw_down=700.0 * o,
                                               lw_down=340.0 * o,
                                               cos_zenith=0.8 * o,
                                               q_lowest=0.006 * o)
    dt = 1800.0
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, dt, lat=jnp.full(1, 0.8)))

    def E(s):
        C = compute_heat_capacity(s.theta_soil, _HYD, cfg.thermal)
        return (jnp.sum(C * _GRID.dz * (s.T_soil - _TF), axis=-1)
                + constants.c_pi * s.snow_depth * (s.T_snow - _TF))

    s = st
    for _ in range(2):
        m_store = constants.c_pi * s.snow_depth * jnp.maximum(
            s.T_snow - _TF, 0.0) / constants.L_f
        out = step(s)
        s2, r2, sfc = out[0], out[1], out[-1]
        flux = (sfc.G_soil + (sfc.lhflx - r2.lhflx)
                + cfg.thermal.Q_geothermal)
        expect = flux * dt - constants.L_f * m_store
        # Water moved by Richards changes C (theta) at fixed T: compare at the
        # end-of-step theta for the soil part.
        dE = E(s2) - E(s._replace(theta_soil=s2.theta_soil))
        np.testing.assert_allclose(np.asarray(dE), np.asarray(expect),
                                   rtol=1e-9, atol=1e-3)
        s = s2


def test_canopy_ground_node_never_above_freezing_under_snow():
    """Warm air and strong sun on a pack at freezing: the two-leaf canopy's
    converged ground temperature is capped at T_freeze (a melting surface)."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=270.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([15.0]), T_snow=jnp.array([_TF]))
    o = jnp.ones(1)
    f = _forcing(1, 285.0, 0.0, 0.0)._replace(sw_down=700.0 * o,
                                               lw_down=340.0 * o,
                                               cos_zenith=0.8 * o)
    out = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(1, 0.8)))(st)
    s2, sfc = out[0], out[-1]
    assert float(sfc.Ts_solve[0]) <= _TF + 1e-9
    assert float(s2.T_snow[0]) > _TF        # the node did take up heat


def test_substeps_equal_repeated_single_steps_with_freeze_thaw():
    """The sub-stepped solve (production: freeze/thaw on, several sub-steps)
    is exactly repeated one-step solves of dt/n, each of which closes the
    ledger tested above."""
    T, theta = _column()
    T = T - 6.0                                   # straddle freezing
    th = _TH._replace(enable_freeze_thaw=True)
    swe = jnp.array([10.0, 40.0, 0.0])
    T_s0 = jnp.array([262.0, 258.0, float(T[2, 0])])
    G = jnp.array([-25.0, 15.0, -5.0])
    n, dt = 6, 1800.0
    Ts6, Tn6 = solve_snow_soil_thermal(T_s0, swe, T, theta, _GRID, _HYD, th,
                                       G, dt, n_substeps=n)
    Ts, Tn = T_s0, T
    for _ in range(n):
        Ts, Tn = solve_snow_soil_thermal(Ts, swe, Tn, theta, _GRID, _HYD, th,
                                         G, dt / n)
    np.testing.assert_allclose(np.asarray(Tn6), np.asarray(Tn), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(Ts6), np.asarray(Ts), rtol=1e-13)


def test_node_mirrors_top_soil_once_the_pack_is_gone():
    """A trace pack that sublimates away within the step leaves no node:
    T_snow equals the top-soil temperature (the state invariant every skin
    consumer relies on)."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=272.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([1.0e-4]), T_snow=jnp.array([268.0]))
    o = jnp.ones(1)
    f = _forcing(1, 278.0, 0.0, 0.0)._replace(sw_down=600.0 * o,
                                               q_lowest=0.0005 * o)
    s2 = jax.jit(lambda s: step_multilayer_land(s, f, cfg, 1.0, 1800.0,
                                                lat=jnp.full(1, 0.8))[0])(st)
    assert float(s2.snow_depth[0]) == 0.0
    assert float(s2.T_snow[0]) == float(s2.T_soil[0, 0])


@pytest.mark.parametrize("swe", [1.0e-3, 1.0e-2, 1.0e-1])
def test_trace_pack_stays_bounded_and_canopy_converges(swe):
    """A dusting is a nearly massless node, but it is tied to the top soil by
    the conductance of its thin pack, so the node stays within a few kelvin of
    the soil top and the canopy closure converges (no held column)."""
    from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=268.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([swe]), T_snow=jnp.array([266.0]))
    o = jnp.ones(1)
    f = _forcing(1, 250.0, 0.0, 0.0)._replace(sw_down=0.0 * o,
                                               lw_down=180.0 * o)
    out = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(1, 1.0)))(st)
    s2, sfc = out[0], out[-1]
    assert int(sfc.n_held) == 0
    assert np.isfinite(float(s2.T_snow[0]))
    assert abs(float(s2.T_snow[0] - s2.T_soil[0, 0])) < 5.0
    # ...and it responds to the cooling flux through that conductance: the
    # snow surface ends colder than the soil top it drains.
    assert float(s2.T_snow[0]) < float(s2.T_soil[0, 0])


def test_held_column_reports_the_capped_skin():
    """A column the canopy fails to solve is held at its previous skin, which
    for a snow node above freezing (pending melt) is T_freeze."""
    from legoesm.land.multilayer_land import (
        _hold_unsolved_columns, step_multilayer_land_with_diagnostics)
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0), thermal=_TH,
        surface_scheme=TwoLeafCanopyConfig())
    st = init_multilayer_land_state(1, cfg, T_init=268.0, theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([10.0]), T_snow=jnp.array([276.0]))
    f = _forcing(1, 262.0, 0.0, 0.0)
    new, resp, _c, sfc = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(1, 1.0))
    _h, held_resp, _hc, mask, _n = _hold_unsolved_columns(
        st, new, resp, sfc._replace(converged=jnp.array([False])), f, cfg, 1)
    assert bool(mask[0])
    np.testing.assert_array_equal(np.asarray(_h.T_snow), [276.0])  # pending melt kept
    np.testing.assert_allclose(np.asarray(held_resp.T_sfc), [_TF])
    np.testing.assert_allclose(np.asarray(held_resp.T_rad), [_TF])


def test_skin_helper_contract():
    """Node off: top soil whatever the snow (switch-off identity). Node on:
    snow-free -> top soil (never a stale node value); snow -> node capped at
    T_freeze; a warm snow-free column is never capped."""
    off = init_multilayer_land_state(2, MultiLayerLandConfig(), T_init=265.0)
    off = off._replace(snow_depth=jnp.array([50.0, 0.0]))
    np.testing.assert_array_equal(np.asarray(land_skin_temperature(off)),
                                  np.asarray(off.T_soil[:, 0]))
    on = init_multilayer_land_state(2, MultiLayerLandConfig(thermal=_TH),
                                    T_init=300.0)
    on = on._replace(snow_depth=jnp.array([0.0, 10.0]),
                     T_snow=jnp.array([260.0, 276.0]))
    np.testing.assert_allclose(np.asarray(land_skin_temperature(on)), [300.0, _TF])

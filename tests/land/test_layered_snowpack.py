"""Layered snowpack wired into the multilayer land (``snow_scheme="layered"``).

The pack (``legoesm.land.snow_column``) and the soil are ONE implicit heat
column (``soil_thermal.solve_snow_soil_thermal``); the ground flux is split by the
snow-covered fraction; melt / refreeze is the pack's enthalpy re-equilibration;
drainage feeds the soil.  Run with ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.multilayer_land import (
    MultiLayerLandConfig, init_multilayer_land_state, land_snow_cover, seed_snow_layers,
    step_multilayer_land_with_diagnostics)
from legoesm.land.snow_bands import ElevationSnowBandConfig
from legoesm.land.snow_column import (
    SnowColumnState, column_enthalpy, seed_snow_state, snow_add_mass,
    snow_phase_and_percolate, snow_remap_compact, snow_thermal_props,
    total_water)
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import (
    SoilThermalConfig, compute_apparent_heat_capacity, compute_heat_capacity,
    solve_snow_soil_thermal, solve_soil_thermal)
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig

jax.config.update("jax_enable_x64", True)

TF = constants.T_freeze
RHO_W = constants.rho_water


def _forcing(n, *, T_air, snow=0.0, rain=0.0, sw=50.0, lw=230.0, q=0.001, wind=4.0):
    o = jnp.ones(n)
    return AtmToSurface(
        sw_down=sw * o, lw_down=lw * o, precip_total=(snow + rain) * o,
        precip_snow=snow * o, T_lowest=T_air * o, q_lowest=q * o,
        u_lowest=wind * o, v_lowest=0.0 * o, p_lowest=95000.0 * o,
        p_surface=1.0e5 * o, rho_lowest=1.0e5 / (constants.R_d * T_air) * o,
        cos_zenith=0.4 * o, co2_ppmv=400.0 * o, has_radiation=o,
        has_precipitation=o)


def _cfg(snow_scheme="layered", scheme=None, **kw):
    return MultiLayerLandConfig(surface_scheme=scheme or SimpleSEBConfig(),
                                snow_scheme=snow_scheme, **kw)


def _state(cfg, n, *, T_soil, swe):
    st = init_multilayer_land_state(n, cfg, T_init=T_soil)
    st = st._replace(snow_depth=jnp.full(n, float(swe)))
    return seed_snow_layers(st, cfg) if cfg.snow_scheme == "layered" else st


def _pack(st):
    return SnowColumnState(st.snow_ice_layers, st.snow_liq_layers, st.snow_T_layers,
                           st.snow_rho_layers)


def _run(cfg, st, forcing, n_steps, dt, lp=None, lat=0.9):
    n = st.T_soil.shape[0]
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, forcing, cfg, 1.0, dt, lat=jnp.full(n, lat), land_params=lp))
    out = []
    for _ in range(n_steps):
        st, resp, _, sfc = step(st)
        out.append((st, resp, sfc))
    return st, out


def _soil_water(cfg, st):
    dz = make_soil_grid(cfg.soil_grid).dz
    return (jnp.sum(st.theta_soil * dz, -1) + st.surface_water) * RHO_W


# ---------------------------------------------------------------------------
# combined solve
# ---------------------------------------------------------------------------

def _column_inputs(ft=False):
    grid = make_soil_grid(MultiLayerLandConfig().soil_grid)
    n = grid.dz.shape[0]
    T = jnp.linspace(262.0, 276.0, 4)[:, None] + jnp.linspace(0.0, 4.0, n)[None, :]
    theta = jnp.full((4, n), 0.3)
    return grid, T, theta, SoilHydraulicsConfig(), SoilThermalConfig(enable_freeze_thaw=ft)


def test_soil_solve_matches_dense_backward_euler():
    """The refactored soil solve IS backward Euler with Neumann G, Robin lambda
    and geothermal flux: pinned against an independently assembled dense matrix."""
    grid, T, theta, hc, tc = _column_inputs()
    G = jnp.array([-60.0, 5.0, 120.0, 0.0])
    lam = jnp.array([4.0, 10.0, 0.0, 2.0])
    dt = 1800.0
    got = solve_soil_thermal(T, theta, grid, hc, tc, G, dt, surface_conductance=lam)
    C = np.asarray(compute_heat_capacity(theta, hc, tc) * grid.dz)
    from legoesm.land.soil_thermal import compute_thermal_conductivity
    k = np.asarray(compute_thermal_conductivity(theta, hc, tc))
    dzi = np.asarray(grid.dz_interface)
    for c in range(4):
        kh = 2 * k[c, :-1] * k[c, 1:] / (k[c, :-1] + k[c, 1:] + 1e-20) / dzi
        A = np.diag(C[c] / dt)
        for i, g in enumerate(kh):
            A[i, i] += g; A[i + 1, i + 1] += g; A[i, i + 1] -= g; A[i + 1, i] -= g
        A[0, 0] += lam[c]
        b = C[c] / dt * np.asarray(T[c])
        b[0] += G[c] + lam[c] * T[c, 0]
        b[-1] += tc.Q_geothermal
        np.testing.assert_allclose(got[c], np.linalg.solve(A, b), rtol=0, atol=1e-9)


@pytest.mark.parametrize("ft", [False, True])
def test_combined_solve_conserves_energy_with_split_flux_and_robin(ft):
    """Pack + soil sensible energy (soil at its apparent heat capacity when
    freeze/thaw is on) changes by exactly the split ground flux, the split Robin
    term and the geothermal flux — deep pack included."""
    grid, T, theta, hc, tc = _column_inputs(ft)
    swe = jnp.array([0.0, 3.0, 40.0, 200.0])
    pack = seed_snow_state(swe, T[:, 0] - 8.0)
    C, coeff, rb = snow_thermal_props(pack)
    f = land_snow_cover(swe, MultiLayerLandConfig().land_albedo)
    G = jnp.array([-80.0, 10.0, 150.0, 30.0])
    lam = jnp.array([5.0, 12.0, 0.0, 3.0])
    dt = 1800.0
    Ts, Tg = solve_snow_soil_thermal(pack.T, C, coeff, rb, f, T, theta, grid, hc, tc,
                                     G, dt, surface_conductance=lam)
    Cg = (compute_apparent_heat_capacity(T, theta, hc, tc) if ft
          else compute_heat_capacity(theta, hc, tc)) * grid.dz
    dE = jnp.sum(C * (Ts - pack.T), -1) + jnp.sum(Cg * (Tg - T), -1)
    src = dt * (G - lam * (f * (Ts[:, 0] - pack.T[:, 0])
                           + (1 - f) * (Tg[:, 0] - T[:, 0])) + tc.Q_geothermal)
    np.testing.assert_allclose(dE, src, rtol=1e-11, atol=1e-5)
    # f splits the flux: an all-snow flux reaches the soil only through the pack.
    assert float(f[3]) > 0.9 and float(f[0]) == 0.0


def test_column_step_closes_energy_with_rain_sublimation_and_melt():
    """The land step's pack sequence (snowfall -> remap/compaction -> sublimation
    -> rain -> combined solve -> phase/percolation) closes pack+soil energy to
    roundoff, counting the REPORTED drainage enthalpy (incl. any liquid above
    T_freeze) and the sensible enthalpy of the sublimated ice."""
    grid, T, theta, hc, tc = _column_inputs(ft=True)
    dt = 1800.0
    pack0 = seed_snow_state(jnp.array([20.0, 10.0, 60.0, 0.5]), T[:, 0] - 3.0)
    snowfall = jnp.array([0.0, 0.5, 2.0, 0.0])
    T_air = jnp.array([280.0, 271.0, 250.0, 283.0])
    rain = jnp.array([3.0, 0.0, 0.0, 6.0])
    sub = jnp.array([0.2, -0.05, 0.1, 0.1])            # kg/m2 (negative = frost)
    G = jnp.array([1500.0, -40.0, -20.0, 300.0])      # col 0: f*G melts through the cold content
    p = snow_add_mass(pack0, snowfall, T_air)
    p = snow_remap_compact(p, dt)
    sub = jnp.minimum(sub, p.swe_ice[:, 0])
    H_sub = sub * constants.c_pi * (p.T[:, 0] - TF)
    p = p._replace(swe_ice=p.swe_ice.at[:, 0].add(-sub))
    f = land_snow_cover(total_water(p), MultiLayerLandConfig().land_albedo)
    p = snow_add_mass(p, 0.0, T_air, rain=f * rain, T_rain=T_air)
    C, coeff, rb = snow_thermal_props(p)
    Ts, Tg = solve_snow_soil_thermal(p.T, C, coeff, rb, f, T, theta, grid, hc, tc, G, dt)
    p2, drain, drain_H = snow_phase_and_percolate(p._replace(T=Ts))
    Cg = compute_apparent_heat_capacity(T, theta, hc, tc) * grid.dz
    dE = (column_enthalpy(p2) - column_enthalpy(pack0)) + jnp.sum(Cg * (Tg - T), -1)
    inputs = (snowfall * constants.c_pi * (jnp.minimum(T_air, TF) - TF)
              + f * rain * (constants.c_pw * (jnp.maximum(T_air, TF) - TF) + constants.L_f)
              - H_sub + dt * (G + tc.Q_geothermal) - drain_H)
    np.testing.assert_allclose(dE, inputs, rtol=1e-10, atol=1e-4)
    # water
    np.testing.assert_allclose(total_water(p2), total_water(pack0) + snowfall - sub
                               + f * rain - drain, atol=1e-10)
    assert float(drain[0]) > 0.0                      # the warm column drains
    ice_left = p2.swe_ice > 0.0
    assert bool(jnp.all(jnp.where(ice_left, p2.T <= TF + 1e-9, jnp.isfinite(p2.T))))


def test_rain_on_cold_pack_refreezes_and_warms():
    pack = seed_snow_state(jnp.array([40.0]), jnp.array([250.0]))
    # 0.5 kg of rain into an 8 kg top layer at 250 K: its fusion heat (1.7e5 J)
    # is well below the layer's cold content (3.9e5 J), so it all refreezes.
    p = snow_add_mass(pack, 0.0, jnp.array([276.0]), rain=jnp.array([0.5]),
                      T_rain=jnp.array([276.0]))
    assert float(p.swe_liq[0, 0]) == 0.0                # all refrozen
    assert float(p.T[0, 0]) > 250.0 + 5.0               # L_f warmed the top layer


# ---------------------------------------------------------------------------
# land step
# ---------------------------------------------------------------------------

def test_insulation_bulk_vs_layered():
    """SWE 75 (f = 75/85 = 0.88) over 270 K soil under 240 K air for 10 days: the
    layered pack's thermal resistance keeps the top soil well above the bulk
    pack's, which has no thermal body (its skin IS the soil top).  Measured
    2026-09-26: layered 248.9 K, bulk 240.3 K — the pack compacts toward 450
    kg/m3 (k ~ 0.58 W/m/K) so it is a weak insulator after 10 days, and 12% of
    the ground flux bypasses it.  Non-vacuity of the whole wiring."""
    f = _forcing(1, T_air=240.0, lw=170.0, sw=0.0, q=0.0002)
    top = {}
    for scheme in ("bulk", "layered"):
        cfg = _cfg(scheme)
        st, _ = _run(cfg, _state(cfg, 1, T_soil=270.0, swe=75.0), f, 240, 3600.0)
        top[scheme] = float(st.T_soil[0, 0])
    assert float(land_snow_cover(jnp.array(75.0), cfg.land_albedo)) == pytest.approx(
        0.9051, abs=1e-3)                     # tanh(75/50), the albedo's cover
    assert top["layered"] > top["bulk"] + 5.0, top


def test_water_closes_and_albedo_contract_holds_through_melt_and_refreeze():
    """d(SWE + soil + pond) = (P - evap - runoff) dt over cold snowfall, rain on
    snow and melt; snow_depth is the total pack water after every step."""
    cfg = _cfg()
    st = _state(cfg, 2, T_soil=271.0, swe=20.0)
    W0 = st.snow_depth + _soil_water(cfg, st)
    dt = 1800.0
    P = E = R = 0.0
    for forcing, nstep in ((_forcing(2, T_air=258.0, snow=3e-4), 24),
                           (_forcing(2, T_air=276.0, rain=4e-4, sw=300.0, lw=320.0), 24),
                           (_forcing(2, T_air=283.0, sw=600.0, lw=330.0, q=0.004), 48)):
        st, out = _run(cfg, st, forcing, nstep, dt)
        for s, resp, _ in out:
            np.testing.assert_allclose(s.snow_depth, total_water(_pack(s)), atol=1e-10)
            P = P + forcing.precip_total * dt
            E = E + resp.surface_mass_flux * dt
            R = R + resp.freshwater_flux * dt
    W1 = st.snow_depth + _soil_water(cfg, st)
    np.testing.assert_allclose(W1 - W0, P - E - R, atol=2e-3)
    assert float(st.snow_depth[0]) < 20.0             # the pack melted
    assert bool(jnp.all(jnp.isfinite(st.T_soil)))


@pytest.mark.parametrize("scheme, ft", [
    # Freeze/thaw off: the layered pack refuses it.  With it off the LAI-2 column
    # converges through the melt (measured 2026-10-04); the earlier xfail, recorded
    # with freeze/thaw on, had since been satisfied by the refusal's ValueError.
    ("two_leaf", False),
    ("seb", False)])
def test_land_step_closes_pack_plus_soil_energy(scheme, ft):
    """Full land step, both surface schemes, through cold snowfall, rain on snow
    and a warm melt: pack enthalpy + soil energy change by exactly
    dt*(applied ground flux + geothermal) + the enthalpy carried by mass
    (snowfall, frost, rain in; sublimated ice, drainage out)."""
    two = scheme == "two_leaf"
    cfg = _cfg(scheme=TwoLeafCanopyConfig() if two else None)
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=ft))
    n = 2
    lp = bare_canopy_params(n)._replace(LAI=jnp.asarray([0.3, 2.0])) if two else None
    s = _state(cfg, n, T_soil=270.0, swe=15.0)
    grid = make_soil_grid(cfg.soil_grid)
    tc, hc = cfg.thermal, cfg.hydraulics
    dt = 1800.0
    saw = {"drain": False, "rain": False}
    for forcing, nstep in ((_forcing(n, T_air=255.0, snow=2e-4, q=0.0015), 8),
                           (_forcing(n, T_air=276.0, rain=5e-4, lw=320.0), 8),
                           (_forcing(n, T_air=283.0, sw=600.0, lw=330.0, q=0.004), 16)):
        _, out = _run(cfg, s, forcing, nstep, dt, lp=lp)
        for s_new, resp, sfc in out:
            Cg = (compute_apparent_heat_capacity(s.T_soil, s.theta_soil, hc, tc) if ft
                  else compute_heat_capacity(s.theta_soil, hc, tc)) * grid.dz
            dE = (column_enthalpy(_pack(s_new)) - column_enthalpy(_pack(s))
                  + jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1))
            src = dt * (sfc.snow_ground_heat_applied + tc.Q_geothermal) + sfc.snow_advected_heat
            np.testing.assert_allclose(dE, src, rtol=1e-9, atol=1e-3)
            assert int(sfc.n_held) == 0
            saw["drain"] |= bool(jnp.any(s_new.snow_depth < s.snow_depth - 1e-3))
            saw["rain"] |= float(forcing.precip_total[0]) > float(forcing.precip_snow[0])
            s = s_new
    assert all(saw.values()), saw


def test_melt_is_dt_converged_and_overshoot_is_first_order():
    """A pack under strong sun through the full two-leaf step: the first day's
    melt (pack still present) changes < 5% when dt is halved; ice-bearing layers
    stay <= T_freeze; the pre-equilibration overshoot of the top layer (sensible
    solve, melt applied after, as CLM5's post-solve phase change) is a
    time-discretisation error: it shrinks with dt.  Measured 2026-09-27: melt
    129.9 vs 130.9 kg/m2, overshoot 31.9 K vs 19.9 K."""
    cfg = _cfg(scheme=TwoLeafCanopyConfig())
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    f = _forcing(1, T_air=279.0, sw=700.0, lw=320.0, q=0.004)
    melt, over = {}, {}
    for dt in (1800.0, 900.0):
        nstep = int(86400 / dt)
        st0 = _state(cfg, 1, T_soil=273.0, swe=150.0)
        st, out = _run(cfg, st0, f, nstep, dt, lp=lp)
        melt[dt] = float(st0.snow_depth[0] - st.snow_depth[0])
        over[dt] = max(float(sfc.snow_T_top_excess[0]) for _, _, sfc in out)
        assert float(st.snow_depth[0]) > 5.0                   # the pack survives the day
        for s, _, _ in out:
            ice = s.snow_ice_layers > 0.0
            assert bool(jnp.all(jnp.where(ice, s.snow_T_layers <= TF + 1e-9,
                                          jnp.isfinite(s.snow_T_layers))))
    assert melt[1800.0] > 50.0, melt
    assert abs(melt[1800.0] - melt[900.0]) < 0.05 * melt[900.0], melt
    assert 0.0 < over[900.0] < 0.75 * over[1800.0], over


def test_melt_out_and_reaccumulation_stay_finite_and_continuous():
    """A thin pack melts out, the column runs bare, then snow re-accumulates: no
    NaN, and the skin temperature handed to the atmosphere never jumps more than
    the bulk pack's does under the same forcing (measured 2026-09-26: largest
    step change 1.3 K while melting out vs 8 K for bulk, whose whole 3 kg pack
    vanishes in one step; both ~6-8 K at the forcing switch itself)."""
    # 72 melt steps: under partial cover only f*G reaches a thin pack, so a trace
    # pack decays exponentially (f ~ SWE/snow_depth_crit at small SWE; measured
    # 1.9e-4 kg/m2 left after 72 steps) and "bare" means f < 1e-3 (SWE < 0.01).
    seq = ((_forcing(1, T_air=282.0, sw=500.0, lw=330.0, q=0.004), 72),
           (_forcing(1, T_air=266.0, snow=3e-4, sw=0.0, lw=250.0), 48))
    jumps = {}
    for scheme in ("bulk", "layered"):
        cfg = _cfg(scheme)
        st = _state(cfg, 1, T_soil=273.5, swe=3.0)
        saw_bare = False
        worst = 0.0
        for forcing, nstep in seq:
            st, out = _run(cfg, st, forcing, nstep, 1800.0)
            T = np.array([float(r.T_sfc[0]) for _, r, _ in out])
            assert np.all(np.isfinite(T))
            worst = max(worst, float(np.max(np.abs(np.diff(T)))))
            for s, _, _ in out:
                saw_bare = saw_bare or float(s.snow_depth[0]) < 1e-2
                if scheme == "layered":
                    assert bool(jnp.all(jnp.isfinite(s.snow_T_layers)))
        jumps[scheme] = worst
        assert saw_bare and float(st.snow_depth[0]) > 3.0
    assert jumps["layered"] <= jumps["bulk"] + 0.1, jumps


def test_sublimation_clamped_to_top_layer_and_water_closes():
    """Dry windy air over a thin pack at a 6-hour step: the pack's share of the
    sublimation demand exceeds the top layer's ice, is clamped to it (never a
    negative pack), the unmet latent flux is not reported to the atmosphere as
    vapour, and water closes."""
    # snow_depth_crit 2 kg/m2: a thin pack at near-full cover, so the pack's
    # share of the demand can exceed the top layer (at the default 50 kg/m2 the
    # cover of a 2 kg/m2 pack is 0.04 and the clamp cannot bind).
    cfg = _cfg()
    cfg = cfg._replace(land_albedo=cfg.land_albedo._replace(snow_depth_crit=2.0))
    dt = 21600.0
    st = _state(cfg, 1, T_soil=272.0, swe=2.0)
    f = _forcing(1, T_air=271.0, sw=400.0, lw=300.0, q=1e-5, wind=12.0)
    W0 = st.snow_depth + _soil_water(cfg, st)
    st1, out = _run(cfg, st, f, 1, dt)
    _, resp, sfc = out[0]
    p = snow_remap_compact(snow_add_mass(_pack(st), 0.0, 271.0), dt)
    frac = float(land_snow_cover(total_water(p), cfg.land_albedo)[0])
    demand = frac * float(sfc.lhflx[0]) / constants.L_s * dt
    assert demand > float(p.swe_ice[0, 0]), (demand, float(p.swe_ice[0, 0]))  # clamp binds
    assert bool(jnp.all(st1.snow_ice_layers >= 0.0))
    assert float(resp.lhflx[0]) < float(sfc.lhflx[0])
    W1 = st1.snow_depth + _soil_water(cfg, st1)
    np.testing.assert_allclose(W1 - W0, -(resp.surface_mass_flux + resp.freshwater_flux)
                               * dt, atol=1e-6)


@pytest.mark.parametrize("scheme, nstep", [("seb", 3), ("two_leaf", 1)])
def test_grad_through_snow_active_step_matches_finite_difference(scheme, nstep):
    """d(pack-top T after n steps)/d(snow emissivity) is finite, nonzero and
    agrees with a central difference, through both surface schemes.  Jitted:
    the EAGER two-leaf gradient's compile exhausts the node's memory-map limit
    (vm.max_map_count 65530) on Levante, not RAM."""
    two = scheme == "two_leaf"
    cfg0 = _cfg(scheme=TwoLeafCanopyConfig() if two else None)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.5])) if two else None
    st0 = _state(cfg0, 1, T_soil=268.0, swe=30.0)
    f = _forcing(1, T_air=258.0, lw=200.0, sw=0.0)

    @jax.jit
    def loss(eps):
        cfg = cfg0._replace(snow_column=cfg0.snow_column._replace(emissivity_snow=eps))
        s = st0
        for _ in range(nstep):
            s, _, _, _ = step_multilayer_land_with_diagnostics(
                s, f, cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9), land_params=lp)
        return jnp.sum(s.snow_T_layers[:, 0])

    g = float(jax.jit(jax.grad(loss))(0.97))
    h = 1e-4
    fd = float((loss(0.97 + h) - loss(0.97 - h)) / (2 * h))
    assert np.isfinite(g) and abs(g) > 1e-3, g
    assert g == pytest.approx(fd, rel=1e-3)


def test_jit_matches_eager():
    cfg = _cfg()
    st = _state(cfg, 2, T_soil=271.0, swe=25.0)
    f = _forcing(2, T_air=262.0, snow=1e-4, rain=5e-5)
    a = step_multilayer_land_with_diagnostics(st, f, cfg, 1.0, 1800.0, lat=jnp.full(2, 0.9))
    b = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(2, 0.9)))(st)
    for x, y in zip(jax.tree.leaves(a[0]), jax.tree.leaves(b[0])):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-10)


# ---------------------------------------------------------------------------
# dispatch, state, restart
# ---------------------------------------------------------------------------

def test_unknown_or_unsupported_snow_scheme_raises():
    f = _forcing(1, T_air=260.0)
    cfg = _cfg("lyered")
    st = init_multilayer_land_state(1, _cfg("bulk"), T_init=270.0)
    with pytest.raises(ValueError, match="snow_scheme"):
        step_multilayer_land_with_diagnostics(st, f, cfg, 1.0, 1800.0)
    with pytest.raises(ValueError, match="snow-layer state"):
        step_multilayer_land_with_diagnostics(st, f, _cfg(), 1.0, 1800.0)
    banded = _cfg(elev_bands=ElevationSnowBandConfig(band_dz=jnp.zeros((1, 3))))
    with pytest.raises(ValueError, match="elev_bands"):
        step_multilayer_land_with_diagnostics(
            init_multilayer_land_state(1, banded, T_init=270.0), f, banded, 1.0, 1800.0)
    ft = _cfg()
    ft = ft._replace(thermal=ft.thermal._replace(enable_freeze_thaw=True))
    with pytest.raises(ValueError, match="enable_freeze_thaw"):
        step_multilayer_land_with_diagnostics(
            init_multilayer_land_state(1, ft, T_init=270.0), f, ft, 1.0, 1800.0)


def test_init_allocates_layers_only_for_layered():
    assert init_multilayer_land_state(3, _cfg("bulk")).snow_T_layers is None
    st = init_multilayer_land_state(3, _cfg(), T_init=280.0)
    assert st.snow_T_layers.shape == (3, 5)
    np.testing.assert_allclose(st.snow_T_layers, TF)            # min(T_soil, Tf)
    st = seed_snow_layers(st._replace(snow_depth=jnp.array([0.0, 5.0, 50.0])), _cfg())
    np.testing.assert_allclose(total_water(_pack(st)), [0.0, 5.0, 50.0])


def test_bulk_checkpoint_into_layered_run_is_refused_clearly():
    from legoesm.driver.model_driver import ModelDriver
    bulk = init_multilayer_land_state(2, _cfg("bulk"))
    layered = init_multilayer_land_state(2, _cfg())
    aux = {f"land_ml_{k}": np.asarray(v) for k, v in bulk._asdict().items()
           if v is not None and k != "canopy_x"}
    dst = SimpleNamespace(_check_land_soil_dz=lambda dz: None, _carry_aux=aux,
                          _land_ml_state=layered)
    with pytest.raises(ValueError, match="land_snow_scheme: bulk"):
        ModelDriver._restore_land_ml_from_carry_aux(dst)


def test_land_restart_round_trips_snow_layers(tmp_path):
    from legoesm.land.restart import (
        load_land_restart, merge_land_restart_into_template, save_land_restart)
    cfg = _cfg()
    st = seed_snow_layers(init_multilayer_land_state(2, cfg, T_init=265.0)._replace(
        snow_depth=jnp.array([4.0, 30.0])), cfg)
    path = save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                             t_end_s=0.0, n_steps_completed=0)
    loaded, _ = load_land_restart(path, expected_land_mode="multilayer", expected_ncol=2)
    merged = merge_land_restart_into_template(
        loaded, init_multilayer_land_state(2, cfg, T_init=280.0))
    for k in ("snow_ice_layers", "snow_liq_layers", "snow_T_layers", "snow_rho_layers"):
        np.testing.assert_allclose(getattr(merged, k), getattr(st, k))


@pytest.mark.parametrize("drop", ["snow_ice_layers", "snow_rho_layers"])
def test_partial_snow_layer_restart_is_refused(tmp_path, drop):
    """A restart (file or in-memory state) carrying only some of the four pack
    fields is refused: the template would fill the rest with its cold-start pack
    and the next step would rebuild SWE from that mix, losing the loaded water."""
    from legoesm.land.restart import (
        load_land_restart, merge_land_restart_into_template, save_land_restart)
    cfg = _cfg()
    st = seed_snow_layers(init_multilayer_land_state(2, cfg, T_init=265.0)._replace(
        snow_depth=jnp.array([4.0, 30.0])), cfg)
    path = save_land_restart(tmp_path / "r.npz", st, land_mode="multilayer",
                             t_end_s=0.0, n_steps_completed=0)
    d = dict(np.load(path))
    d.pop(drop)
    np.savez(tmp_path / "part.npz", **d)
    with pytest.raises(ValueError, match="partial layered snowpack"):
        load_land_restart(tmp_path / "part.npz", expected_land_mode="multilayer",
                          expected_ncol=2)
    with pytest.raises(ValueError, match="partial layered snowpack"):
        merge_land_restart_into_template(
            st._replace(**{drop: None}), init_multilayer_land_state(2, cfg, T_init=280.0))


def test_rain_on_snow_enters_the_pack_in_the_land_step():
    """Rain on a cold pack: the snow-covered fraction of it is held by the pack
    (refrozen), only the rest reaches the soil."""
    cfg = _cfg()
    st = _state(cfg, 1, T_soil=268.0, swe=40.0)
    st = st._replace(snow_T_layers=jnp.full_like(st.snow_T_layers, 255.0))
    rain = 1e-3
    st1, out = _run(cfg, st, _forcing(1, T_air=275.0, rain=rain, lw=300.0), 1, 1800.0)
    f = float(land_snow_cover(jnp.asarray(40.0), cfg.land_albedo))
    gained = float(st1.snow_depth[0] - st.snow_depth[0])
    sub = float(out[0][1].surface_mass_flux[0]) * 1800.0     # upper bound on vapour loss
    assert gained == pytest.approx(f * rain * 1800.0, abs=abs(sub) + 1e-6)


def test_trace_pack_does_not_block_ground_evaporation():
    """Two-leaf over a trace pack (SWE 0.01, f = 0.001): only f of the ground
    latent is the pack's, so the reported latent flux matches the snow-free
    column's.  A binary split sent ALL ground latent to the pack, whose
    top-layer clamp returned it to the ground heat flux."""
    cfg = _cfg(scheme=TwoLeafCanopyConfig())
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    f = _forcing(1, T_air=285.0, sw=500.0, lw=330.0, q=0.004)
    lh = {}
    for swe in (0.0, 0.01):
        st = _state(cfg, 1, T_soil=288.0, swe=swe)
        _, out = _run(cfg, st, f, 1, 1800.0, lp=lp)
        lh[swe] = float(out[0][1].lhflx[0])
    assert lh[0.0] > 50.0, lh
    assert lh[0.01] == pytest.approx(lh[0.0], rel=0.05), lh


@pytest.mark.parametrize("q_air", [0.0005, 0.002])
@pytest.mark.parametrize("scheme", ["two_leaf", "seb"])
def test_exported_humidity_agrees_with_latent_flux_sign(scheme, q_air):
    """Cold pack under sun (canopy or bare): the exported surface humidity sits
    on the same side of the air humidity as the reported latent flux, for thin
    and deep packs, so a humidity-based consumer infers the realised exchange."""
    two = scheme == "two_leaf"
    cfg = _cfg(scheme=TwoLeafCanopyConfig() if two else None)
    n = 3
    lp = bare_canopy_params(n)._replace(LAI=jnp.asarray([0.5, 2.0, 4.0])) if two else None
    st = _state(cfg, n, T_soil=268.0, swe=40.0)
    st = st._replace(snow_depth=jnp.array([0.5, 40.0, 150.0]))
    st = seed_snow_layers(st, cfg)
    st = st._replace(snow_T_layers=jnp.full_like(st.snow_T_layers, 255.0))
    f = _forcing(n, T_air=272.0, sw=600.0, lw=260.0, q=q_air)
    _, out = _run(cfg, st, f, 2, 300.0, lp=lp)
    for _, resp, sfc in out:
        if two:   # the exported humidity IS the one the canopy solved its flux with
            np.testing.assert_allclose(resp.q_surface, sfc.q_surface, rtol=0, atol=0)
        # SimpleSEB exports the humidity of the REALISED vapour flux (its sign is
        # the vapour flux's; see the mixed-sign test below)
        lh = np.asarray(resp.lhflx if two else resp.surface_mass_flux)
        dq = np.asarray(resp.q_surface) - q_air
        big = np.abs(lh) > (0.1 if two else 0.1 / constants.L_s)   # 0.1 W/m2
        assert big.any()
        assert np.all(np.sign(lh[big]) == np.sign(dq[big])), (lh, dq)


def test_layered_step_closes_energy_against_realised_latent_on_dry_soil():
    """Trace pack (0.05 kg/m2) on a hot soil at its dry floor (two-leaf, no
    Robin term): the latent demand the soil cannot supply is only known after hydrology (the Richards
    refill).  Every step, the energy the column receives equals the scheme's
    ground flux plus the latent demand minus the latent flux REPORTED to the
    atmosphere, and pack + soil energy closes against it.  Merge of main's
    realised-evaporation fix crashed this branch (energy read before hydrology);
    dropping the post-hydrology remainder leaves the first identity open."""
    from legoesm.land.richards import psi_dry_floor
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi
    cfg = _cfg(scheme=TwoLeafCanopyConfig(),
               soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    hc, tc = cfg.hydraulics, cfg.thermal
    n = 1
    tfl = float(jnp.max(theta_from_psi(psi_dry_floor(hc), hc)))
    s = init_multilayer_land_state(n, cfg, T_init=300.0, theta_init=tfl + 5.0e-3)
    s = s._replace(psi_soil=psi_from_theta(s.theta_soil, hc),
                   snow_depth=jnp.full(n, 0.05))
    s = seed_snow_layers(s, cfg)
    lp = bare_canopy_params(n)._replace(LAI=jnp.asarray([0.3]))
    grid = make_soil_grid(cfg.soil_grid)
    dt = 1800.0
    _, out = _run(cfg, s, _forcing(n, T_air=305.0, sw=600.0, lw=380.0, q=0.002),
                  24, dt, lp=lp)
    unmet = 0.0
    for s_new, resp, sfc in out:
        assert int(sfc.n_held) == 0
        want = sfc.G_soil + sfc.lhflx - resp.lhflx
        np.testing.assert_allclose(sfc.snow_ground_heat_applied, want, rtol=0, atol=1e-6)
        Cg = compute_heat_capacity(s.theta_soil, hc, tc) * grid.dz
        dE = (column_enthalpy(_pack(s_new)) - column_enthalpy(_pack(s))
              + jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1))
        src = dt * (sfc.snow_ground_heat_applied + tc.Q_geothermal) + sfc.snow_advected_heat
        np.testing.assert_allclose(dE, src, rtol=1e-9, atol=1e-3)
        unmet = max(unmet, float(jnp.max(sfc.lhflx - resp.lhflx)))
        s = s_new
    # the soil really fell short of the demand (peak ~26 W/m2 measured).  Dropping
    # the post-hydrology remainder breaks the first identity (by 0.58 W/m2 already
    # on step 1), and the merge as first resolved raised UnboundLocalError.
    assert unmet > 5.0, unmet


@pytest.mark.parametrize("swe", [5.0, 15.0, 40.0])
def test_partial_cover_sublimation_scales_with_f_not_f_squared(swe):
    """SimpleSEB over a partial layered pack on a soil at its dry floor: the snow
    exchanges as an ice-saturated surface over the fraction f and the pack
    supplies that whole share (the dry soil gives ~nothing of its own).  The
    f-blended surface gave the pack only f of an already f-weighted demand (f^2):
    measured reported latent 1.5 / 8.7 / 39.9 W/m2 at f = 0.10 / 0.29 / 0.66,
    against pack shares of 8.9 / 26.0 / 59.2 now."""
    from legoesm.land.richards import psi_dry_floor
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi
    from legoesm.land.surface_scheme.simple_seb import compute_simple_seb_fluxes
    cfg = _cfg(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    hc = cfg.hydraulics
    tfl = float(jnp.max(theta_from_psi(psi_dry_floor(hc), hc)))
    s = init_multilayer_land_state(1, cfg, T_init=268.0, theta_init=tfl + 1.0e-3)
    s = seed_snow_layers(s._replace(psi_soil=psi_from_theta(s.theta_soil, hc),
                                    snow_depth=jnp.full(1, swe)), cfg)
    f = float(land_snow_cover(jnp.full(1, swe), cfg.land_albedo)[0])
    fo = _forcing(1, T_air=270.0, sw=200.0, lw=250.0, q=0.0005)
    _, out = _run(cfg, s, fo, 1, 1800.0)
    _, resp, sfc = out[0]
    demand, reported = float(sfc.lhflx[0]), float(resp.lhflx[0])
    le_snow = float(sfc.LE_snow[0])
    assert le_snow > 5.0 and le_snow <= demand, (f, le_snow, demand)
    assert reported >= le_snow * (1.0 - 1e-9), (f, le_snow, reported)
    # the scheme itself: snow and soil exchange separately, area-weighted
    kw = dict(T_surface=s.T_soil[:, 0], snow=s.snow_depth, snow_age=s.snow_age,
              beta_soil=jnp.zeros(1), forcing=fo, land_config=cfg, U_min=1.0, lat=None,
              carbon_state=None, dt=1800.0, land_params=None, albedo_land=0.2,
              emissivity=0.97, z0=0.01)
    lh = {c: compute_simple_seb_fluxes(**kw, snow_cover=jnp.full(1, c)) for c in (f, 1.0, 0.0)}
    np.testing.assert_allclose(lh[f].lhflx, f * lh[1.0].lhflx + (1 - f) * lh[0.0].lhflx,
                               rtol=1e-12)
    np.testing.assert_allclose(lh[f].LE_snow, f * lh[1.0].lhflx, rtol=1e-12)


@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("dry", [False, True])
def test_exported_humidity_implies_realised_vapour_flux_mixed_sign(frac, dry):
    """SimpleSEB, partial pack, air humidity BETWEEN ice and liquid saturation at
    the skin: the snow frosts while the soil evaporates, so the vapour and the
    latent-energy fluxes can differ in sign, and a dry soil can turn a net
    evaporation demand into net frost.  The exported humidity must sit on the
    side of the air humidity given by the REALISED vapour flux (the area-mean
    demand humidity did not, once the dry soil failed to supply its share)."""
    from legoesm.land.richards import psi_dry_floor
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi
    from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
    cfg = _cfg(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    hc = cfg.hydraulics
    tfl = float(jnp.max(theta_from_psi(psi_dry_floor(hc), hc)))
    s = init_multilayer_land_state(1, cfg, T_init=268.0,
                                   theta_init=(tfl + 1.0e-3) if dry else 0.3)
    s = seed_snow_layers(s._replace(psi_soil=psi_from_theta(s.theta_soil, hc),
                                    snow_depth=jnp.full(1, 15.0)), cfg)
    T_skin, p = s.T_soil[:, 0], jnp.full(1, 1.0e5)
    q_ice = float(saturation_mixing_ratio_ice(T_skin, p)[0])
    q_liq = float(saturation_mixing_ratio(T_skin, p)[0])
    q_air = q_ice + frac * (q_liq - q_ice)
    _, out = _run(cfg, s, _forcing(1, T_air=268.0, sw=0.0, lw=280.0, q=q_air), 1, 1800.0)
    _, resp, _ = out[0]
    E = float(resp.surface_mass_flux[0])
    dq = float(resp.q_surface[0]) - q_air
    assert abs(E) > 1e-9, E
    assert np.sign(dq) == np.sign(E), (frac, dry, E, dq)

"""Snow-covered ground is charged the sublimation latent heat INSIDE the surface
solve (#1875), and the driver turns every latent stream back into water with
exactly the latent heat the solve charged it.

The two-leaf canopy charges its ground latent flux ``surface_latent_heat(T,
w)`` = (1-w) L_v + w L_s, w the snow weight (bulk: the snow gate; layered: the
cover f), and reports that charge as ``SurfaceFluxOutput.L_soil``.  The ground's
snow share therefore leaves the pack as ice at the L_s it paid, and no
(L_s - L_v) correction is drawn from the ground afterwards (#1864's interim
handling).  Only canopy dew frosting the pack -- a LEAF flux charged L_v -- keeps
that ground term.  Below: (a) the snow mass and reported latent heat; (b) the
solve's soil latent charge itself; (c) layered pack + soil energy closes with no
ground correction; (d) the bulk soil too; (e) a capped pack; (f) the gradient.
"""
from __future__ import annotations

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
from legoesm.land.snow_column import SnowColumnState, column_enthalpy, total_water
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_thermal import compute_heat_capacity
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes
from legoesm.thermo import (
    latent_heat_sublimation, latent_heat_vaporization, surface_latent_heat)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="float64 budgets; JAX_ENABLE_X64=1")

_T = 260.0


def _forcing(n, T_air=_T + 3.0, q=0.0003):
    o = jnp.ones(n)
    return AtmToSurface(
        sw_down=150.0 * o, lw_down=230.0 * o, precip_total=0.0 * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=q * o,
        u_lowest=5.0 * o, v_lowest=0.0 * o, p_lowest=95000.0 * o,
        p_surface=1.0e5 * o, rho_lowest=1.0e5 / (constants.R_d * T_air) * o,
        cos_zenith=0.4 * o, co2_ppmv=400.0 * o, has_radiation=o,
        has_precipitation=o)


def _two_leaf_step(snow_scheme, n_steps=1, swe=60.0):
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(),
                               snow_scheme=snow_scheme)
    st = init_multilayer_land_state(1, cfg, T_init=_T)
    st = st._replace(snow_depth=jnp.full(1, swe))
    if snow_scheme == "layered":
        st = seed_snow_layers(st, cfg)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, _forcing(1), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9), land_params=lp))
    out = []
    for _ in range(n_steps):
        st_new, resp, _, sfc = step(st)
        out.append((st, st_new, resp, sfc))
        st = st_new
    return cfg, out


def _pack(st):
    return SnowColumnState(st.snow_ice_layers, st.snow_liq_layers, st.snow_T_layers,
                           st.snow_rho_layers)


def test_two_leaf_charges_snow_ground_sublimation_inside_solve():
    """(a) Bulk pack under the two-leaf canopy at 260 K: the ground latent flux
    is charged L_s(T_surface) by the solve (``L_soil``), the leaves L_v.  The
    vapour mass is LE_soil / L_s + LE_canopy / L_v, and the latent heat reported
    to the atmosphere equals the solve's own LE except for canopy dew frosting
    the pack (charged L_v, leaves as ice at L_s)."""
    _, [(s0, _, resp, sfc)] = _two_leaf_step("bulk")
    T_s = s0.T_soil[0, 0]
    L_v, L_s = float(latent_heat_vaporization(T_s)), float(latent_heat_sublimation(T_s))
    assert float(sfc.L_soil[0]) == L_s                     # the solve's charge
    le_soil, le_can = float(sfc.LE_soil[0]), float(sfc.lhflx[0] - sfc.LE_soil[0])
    assert le_soil > 5.0, le_soil                          # a real sublimation flux
    E = float(resp.surface_mass_flux[0])
    np.testing.assert_allclose(E, le_soil / L_s + le_can / L_v, rtol=1e-9)
    np.testing.assert_allclose(
        float(resp.lhflx[0]),
        le_soil + max(le_can, 0.0) + min(le_can, 0.0) * L_s / L_v, rtol=1e-9)
    # #1864 converted the ground share at L_v: ~11 % more mass at 260 K.
    assert (le_soil / L_v) / (le_soil / L_s) - 1.0 > 0.10


def test_two_leaf_solve_charges_ground_at_the_passed_latent_heat():
    """(b) Directly at the scheme, soil skin held fixed: the ground latent flux
    scales with the charge passed in (ratio L_s / L_v exactly, cap off), the
    leaves and the canopy state do not move, the ground heat flux pays the
    difference, and ``L_soil`` reports the charge.  Fails if the solver ignores
    the ground charge (soil LE at L_v) or the scheme drops it."""
    T0 = jnp.full(1, _T)
    kw = dict(
        T_soil_top=T0, forcing=_forcing(1),
        canopy_config=TwoLeafCanopyConfig(le_cap_mode="off"),
        land_config=MultiLayerLandConfig(),
        canopy_params=bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3])),
        w_frac_rz=jnp.full(1, 0.7), wind_speed=jnp.full(1, 5.0),
        wind_dir_x=jnp.ones(1), wind_dir_y=jnp.zeros(1),
        soil_thermal_fn=lambda G, dt: T0, dt=1800.0)
    L_v, L_s = latent_heat_vaporization(T0), latent_heat_sublimation(T0)
    liq = compute_two_leaf_canopy_fluxes(**kw)
    ice = compute_two_leaf_canopy_fluxes(**kw, ground_latent_heat=L_s)
    assert float(liq.L_soil[0]) == float(L_v[0])
    assert float(ice.L_soil[0]) == float(L_s[0])
    assert abs(float(liq.LE_soil[0])) > 1.0
    np.testing.assert_allclose(ice.LE_soil, liq.LE_soil * L_s / L_v, rtol=1e-12)
    np.testing.assert_array_equal(ice.LE_canopy, liq.LE_canopy)
    np.testing.assert_array_equal(ice.T_canopy_air, liq.T_canopy_air)
    np.testing.assert_allclose(ice.G_soil - liq.G_soil, liq.LE_soil - ice.LE_soil,
                               rtol=1e-9)


def test_simple_seb_over_snow_charges_sublimation_so_no_excess():
    """SimpleSEB charges L_s(T_surface) to snow, so its conversion is the
    identity: the latent heat reported equals the scheme's charge (no excess)
    on an uncapped deep pack.  Guards the driver's per-scheme snow charge."""
    from legoesm.land.surface_scheme import SimpleSEBConfig
    cfg = MultiLayerLandConfig(surface_scheme=SimpleSEBConfig())
    st = init_multilayer_land_state(1, cfg, T_init=_T)._replace(snow_depth=jnp.full(1, 60.0))
    _, resp, _, sfc = step_multilayer_land_with_diagnostics(
        st, _forcing(1), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9))
    assert float(sfc.lhflx[0]) > 5.0
    np.testing.assert_allclose(float(resp.lhflx[0]), float(sfc.lhflx[0]), rtol=1e-9)
    np.testing.assert_allclose(
        float(resp.surface_mass_flux[0]),
        float(sfc.lhflx[0]) / float(latent_heat_sublimation(st.T_soil[0, 0])), rtol=1e-9)


@pytest.mark.parametrize("swe", [60.0, 3.0], ids=["deep_pack", "partial_cover"])
def test_layered_pack_plus_soil_energy_closes_without_ground_correction(swe):
    """(c) Every step: the column receives the scheme's ground flux plus the
    latent demand minus the latent heat reported, and pack enthalpy + soil
    energy change by exactly that (in - out - dStorage = 0).  The snow-covered
    fraction f of the ground is charged L_s inside the solve, so with no canopy
    dew the excess is ~0 (#1864 booked (L_v - L_s) * E here: < -1 W/m2).  The
    thin pack exercises 0 < f < 1, where the ground charge lies strictly between
    L_v and L_s and only f of the ground vapour leaves the pack."""
    cfg, out = _two_leaf_step("layered", n_steps=6, swe=swe)
    grid = make_soil_grid(cfg.soil_grid)
    hc, tc = cfg.hydraulics, cfg.thermal
    dt = 1800.0
    for s, s_new, resp, sfc in out:
        assert int(sfc.n_held) == 0
        excess = sfc.lhflx - resp.lhflx
        np.testing.assert_allclose(sfc.snow_ground_heat_applied, sfc.G_soil + excess,
                                   rtol=0, atol=1e-6)
        Cg = compute_heat_capacity(s.theta_soil, hc, tc) * grid.dz
        dE = (column_enthalpy(_pack(s_new)) - column_enthalpy(_pack(s))
              + jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1))
        src = dt * (sfc.snow_ground_heat_applied + tc.Q_geothermal) + sfc.snow_advected_heat
        np.testing.assert_allclose(dE, src, rtol=1e-9, atol=1e-3)
        assert float(sfc.lhflx[0] - sfc.LE_soil[0]) >= 0.0   # no canopy dew here
        np.testing.assert_allclose(float(excess[0]), 0.0, atol=1e-9)
    # The solve's ground charge is the cover-weighted blend at the skin (260 K on
    # the seeded first step, pack and soil both at 260 K).
    s0 = out[0][0]
    f = float(land_snow_cover(total_water(_pack(s0)), cfg.land_albedo)[0])
    if swe < 10.0:
        assert 0.01 < f < 0.99, f                          # partial cover
    np.testing.assert_allclose(float(out[0][3].L_soil[0]),
                               float(surface_latent_heat(s0.T_soil[0, 0], f)), rtol=1e-12)


def test_bulk_soil_energy_closes_without_ground_correction():
    """(d) No pack enthalpy on the bulk scheme: at 260 K with no melt, no
    precipitation and freeze/thaw off, the soil gains exactly
    dt * (G + lhflx_demand - lhflx_reported + geothermal); the excess is only
    canopy dew's (L_v - L_s) frost term (none here: #1864's was
    -snow_lat * (L_s / L_v - 1))."""
    cfg, out = _two_leaf_step("bulk", n_steps=4)
    assert not cfg.thermal.enable_freeze_thaw
    grid = make_soil_grid(cfg.soil_grid)
    hc, tc = cfg.hydraulics, cfg.thermal
    dt = 1800.0
    for s, s_new, resp, sfc in out:
        excess = sfc.lhflx - resp.lhflx
        T_s = s.T_soil[0, 0]
        L_v, L_s = float(latent_heat_vaporization(T_s)), float(latent_heat_sublimation(T_s))
        le_can = float(sfc.lhflx[0] - sfc.LE_soil[0])
        assert float(sfc.LE_soil[0]) > 0.0
        np.testing.assert_allclose(float(excess[0]),
                                   -min(le_can, 0.0) * (L_s / L_v - 1.0), atol=1e-9)
        # the bulk solve uses the post-hydrology moisture
        Cg = compute_heat_capacity(s_new.theta_soil, hc, tc) * grid.dz
        dE = jnp.sum(Cg * (s_new.T_soil - s.T_soil), -1)
        np.testing.assert_allclose(
            dE, dt * (sfc.G_soil + excess + tc.Q_geothermal), rtol=1e-9, atol=1e-3)


@pytest.mark.parametrize("T_air,L_charged", [
    (262.0, constants.L_s),    # freezing air: hsub
    (275.0, constants.L_v),    # thawing air over the pack: hvap
], ids=["below_freezing_hsub", "above_freezing_hvap"])
def test_clm_ml_snow_ground_mass_is_its_charge(monkeypatch, T_air, L_charged):
    """CLM-ML charges its leaves at LatVap(tref): hsub (== constants.L_s) under
    freezing air, hvap (== constants.L_v) above; its ground over snow at the
    sublimation heat the driver passes in (#1875), reported as ``L_soil``.  The
    stub mirrors that contract; the driver must pass the snow weight 1 and
    L_s(T_surface) and invert each stream with its own charge."""
    import legoesm.land.canopy.clm_ml_interface as clm
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.surface_scheme.base import SurfaceFluxOutput
    seen = {}

    def _stub(*, T_soil_top, forcing, canopy_state, ground_snow_weight,
              ground_sublimation_heat, **_):
        seen.update(w=ground_snow_weight, L=ground_sublimation_heat)
        o = jnp.ones_like(T_soil_top)
        L_soil = ((1.0 - ground_snow_weight) * L_charged
                  + ground_snow_weight * ground_sublimation_heat)
        return SurfaceFluxOutput(
            shflx=10.0 * o, lhflx=40.0 * o, tau_x=0.0 * o, tau_y=0.0 * o,
            sw_net=100.0 * o, lw_net=-40.0 * o, lw_up=300.0 * o, G_soil=10.0 * o,
            T_surface=T_soil_top, q_surface=0.001 * o, albedo=0.7 * o,
            emissivity=0.97 * o, z0=0.01 * o, T_canopy_air=T_soil_top,
            stomatal_ratio=o, LE_soil=25.0 * o, LE_canopy=15.0 * o,
            L_soil=L_soil), canopy_state

    monkeypatch.setattr(clm, "compute_clm_ml_canopy_fluxes", _stub)
    cfg = MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=265.0, theta_init=0.30)
    s0 = s0._replace(snow_depth=jnp.full(1, 60.0))
    _, resp, _, out = step_multilayer_land_with_diagnostics(
        s0, _forcing(1, T_air=T_air), cfg, 1.0, 1800.0, lat=jnp.array([0.9]),
        carbon_state=None, doy=20.0, land_params=None)
    L_sT = float(latent_heat_sublimation(s0.T_soil[0, 0]))
    assert float(seen["w"][0]) == 1.0 and float(seen["L"][0]) == L_sT
    np.testing.assert_allclose(float(resp.surface_mass_flux[0]),
                               25.0 / L_sT + 15.0 / L_charged, rtol=1e-9)
    np.testing.assert_allclose(float(resp.lhflx[0]), 40.0, rtol=1e-9)


def test_bulk_capped_pack_sends_unmet_demand_to_sh_and_nothing_to_ground():
    """(e) Thin bulk pack under the two-leaf canopy: the sublimation demand
    exceeds the pack.  The unmet demand leaves as sensible heat; the ice that
    did leave was already charged L_s by the solve, so the ground receives
    exactly the solve's G (#1864 added pack/dt * (L_v - L_s) < 0 here)."""
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=_T)
    swe0 = 1.0e-4                                         # [kg m-2], cap binds
    s0 = s0._replace(snow_depth=jnp.full(1, swe0))
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    dt = 1800.0
    s1, resp, _, sfc = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, _forcing(1), cfg, 1.0, dt, lat=jnp.full(1, 0.9), land_params=lp))(s0)
    assert float(s1.snow_depth[0]) < 1e-12                 # the whole pack left
    assert float(sfc.lhflx[0] - sfc.LE_soil[0]) >= 0.0     # no canopy dew
    X = float(sfc.lhflx[0] - resp.lhflx[0])
    assert X > 1.0, X                                     # non-vacuity: cap binds
    np.testing.assert_allclose(float(resp.shflx[0] - sfc.shflx[0]), X,
                               rtol=0, atol=1e-9)
    grid = make_soil_grid(cfg.soil_grid)
    Cg = compute_heat_capacity(s1.theta_soil, cfg.hydraulics, cfg.thermal) * grid.dz
    dE = jnp.sum(Cg * (s1.T_soil - s0.T_soil), -1)
    np.testing.assert_allclose(
        dE, dt * (sfc.G_soil + cfg.thermal.Q_geothermal), rtol=1e-9, atol=1e-3)


def test_snow_step_gradient_is_finite():
    """(f) d(reported latent + sensible heat)/d(air temperature) through one
    snow-covered two-leaf step is finite and nonzero."""
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig())
    s0 = init_multilayer_land_state(1, cfg, T_init=_T)._replace(snow_depth=jnp.full(1, 60.0))
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))

    def f(T_air):
        _, resp, _, _ = step_multilayer_land_with_diagnostics(
            s0, _forcing(1, T_air=T_air), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9),
            land_params=lp)
        return jnp.sum(resp.lhflx + resp.shflx)

    g = float(jax.jit(jax.grad(f))(_T + 3.0))
    assert np.isfinite(g) and g != 0.0, g


def test_layered_soil_share_routes_at_its_liquid_charge(monkeypatch):
    """Partial cover: the snow-FREE (1 - f) share of the ground vapour leaves the
    soil, so the bare-soil vs root-zone split weighs its energy at L_v -- LE_soil
    less the snow share charged L_s -- not (1 - f) LE_soil, which carries the
    blended charge (#1875)."""
    import legoesm.land.multilayer_land as ml
    seen = {}
    real = ml._partition_latent_root_top

    def spy(soil_evap, has_snow, f_veg, le_canopy, le_soil):
        seen["le_soil"] = le_soil
        return real(soil_evap, has_snow, f_veg, le_canopy, le_soil)

    monkeypatch.setattr(ml, "_partition_latent_root_top", spy)
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(), snow_scheme="layered")
    st = seed_snow_layers(init_multilayer_land_state(1, cfg, T_init=_T)._replace(
        snow_depth=jnp.full(1, 3.0)), cfg)
    lp = bare_canopy_params(1)._replace(LAI=jnp.asarray([0.3]))
    _, _, _, sfc = step_multilayer_land_with_diagnostics(
        st, _forcing(1), cfg, 1.0, 1800.0, lat=jnp.full(1, 0.9), land_params=lp)
    f = float(land_snow_cover(total_water(_pack(st)), cfg.land_albedo)[0])
    assert 0.01 < f < 0.99, f
    L_v = float(latent_heat_vaporization(st.T_soil[0, 0]))
    le_soil = float(sfc.LE_soil[0])
    assert abs(le_soil) > 1.0
    np.testing.assert_allclose(float(seen["le_soil"][0]),
                               (1.0 - f) * le_soil / float(sfc.L_soil[0]) * L_v, rtol=1e-12)
